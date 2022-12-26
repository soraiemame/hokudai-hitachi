#include "worker.hpp"
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include "io.hpp"
#include "job.hpp"
#include "job_manager.hpp"
#include "lib/json.hpp"
#include "logger.hpp"
#include "worker_manager.hpp"
bool WorkerInfo::is_valid(std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    VALIDITY_CHECK_WITH_MSG(processable_types.empty(), valid, msg,
                            "This worker has no processable types.");
    for (auto t : processable_types) {
        VALIDITY_CHECK_WITH_MSG(
            t < 0 || t == INVALID_JOB_TYPE, valid, msg,
            "Processable type(internal:%d) is invalid or out of range.", t);
    }
    VALIDITY_CHECK_WITH_MSG(initial_position == INVALID_VERTEX_ID, valid, msg,
                            "This worker's initial position is invalid.");
    VALIDITY_CHECK_WITH_MSG(initial_position == INVALID_VERTEX_ID, valid, msg,
                            "This worker's initial position(internal vertex "
                            "index:%d) is out of range.",
                            initial_position);
    VALIDITY_CHECK_WITH_MSG(
        max_task <= 0, valid, msg,
        "This worker has 0 or negative max task process amount(%d)", max_task);
    return valid;
}
void WorkerInfo::read_from_stream(std::istream& is_) {
    ValueReader r(is_);
    auto ssr = get_single_line_stream(r);
    int n_job_type;
    int mt;
    read_vars(ssr,
              TAG(ValueGroup::INDEX, initial_position),
              TAG(ValueGroup::POSITIVE_COUNT, mt),
              TAG(ValueGroup::POSITIVE_COUNT,
                  n_job_type));
    THROW_RUNTIME_ERROR_IF(mt <= 0,
                           "Max task processable num is 0 or negative");
    max_task = mt;
    THROW_RUNTIME_ERROR_IF(n_job_type <= 0,
                           "Num of processable jobs is 0 or negative");
    processable_types.clear();
    read_n_vars_exact<job_type_t>(
        ssr, n_job_type, ValueGroup::TYPEID, [&](job_type_t jt) {
            processable_types.emplace(jt);
        });
    std::string msg;
    THROW_RUNTIME_ERROR_IF(
        !is_valid(&msg),
        "This worker info is still invalid after successfully read data from "
        "stream for the following reason(s):%s",
        msg.c_str());
}
bool WorkerInfo::can_process(job_type_t jt) const {
    std::string msg;
    THROW_RUNTIME_ERROR_IF(
        !is_valid(&msg),
        "This worker info is invalid for the following reason(s):%s",
        msg.c_str());
    return processable_types.find(jt) != processable_types.end();
}
void to_json(json_ref j, const WorkerInfo& wi) {
    j = json{
        {"initial_position", wi.initial_position},
        {"processable_types", wi.processable_types},
        {"max_task", wi.max_task},
        {"is_valid", wi.is_valid()},
    };
}
Position WorkerState::current_position() const {
    return current_pos_;
}
void WorkerState::initialize(const Position& p) {
    current_pos_ = p;
}
bool WorkerState::is_valid(const UndirectedGraph& graph,
                           std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    {
        std::string pmsg;
        VALIDITY_CHECK_WITH_MSG(
            !graph.is_position_valid(current_pos_, &pmsg), valid, msg,
            "Current position is invalid for the following reason(s):%s",
            pmsg.c_str());
    }
    return valid;
}
void to_json(json_ref j, const WorkerState& ws) {
    j = json{{"current_position", ws.current_position()}};
}
bool TaskExecutionRecord::operator<(const TaskExecutionRecord& e) const {
    return time < e.time;
}
void to_json(json_ref j, const TaskExecutionRecord& ter) {
    j = json{
        {"time", ter.time},
        {"job_id", ter.id},
        {"amount", ter.amount},
    };
}
bool Worker::is_valid(std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    if (always_valid_) {
        *msg = "Validity check is skipped due to 'always_valid_' flag\n";
        return true;
    }
    VALIDITY_CHECK_WITH_MSG(graph == nullptr, valid, msg, "Graph is not set.");
    VALIDITY_CHECK_WITH_MSG(man_ == nullptr, valid, msg,
                            "WorkerManager is not set.");
    VALIDITY_CHECK_WITH_MSG(id() == INVALID_WORKER_ID, valid, msg,
                            "Worker id is invalid.");
    {
        std::string info_msg;
        VALIDITY_CHECK_WITH_MSG(
            !info.is_valid(&info_msg), valid, msg,
            "Worker info is invalid for the following reason(s):%s",
            info_msg.c_str());
    }
    {
        std::string state_msg;
        VALIDITY_CHECK_WITH_MSG(
            !state.is_valid(*graph, &state_msg), valid, msg,
            "Worker state is invalid for the following reason(s):%s",
            state_msg.c_str());
    }
    return valid;
}
void Worker::initialize() {
    {
        std::string msg;
        THROW_RUNTIME_ERROR_IF(
            !info.is_valid(&msg),
            "The worker info is invalid for the following reason(s):%s",
            msg.c_str());
    }
    state.initialize(Position{info.initial_position});
}
void Worker::read_worker_info(std::istream& is) {
    info.read_from_stream(is);
}
Position Worker::current_position() const {
    return state.current_position();
}
void Worker::set_worker_manager(WorkerManager* wmp) {
    man_ = wmp;
}
void Worker::set_graph(const UndirectedGraph* gp) {
    graph = gp;
}
worker_id_t Worker::id() const {
    return id_;
}
void WorkerState::move_towards(vertex_index_t to,
                               const UndirectedGraph& graph) {
    THROW_WA_IF(WrongAnswerType::MOVE_VERTEX_DOES_NOT_EXIST,
                !graph.vertex_exists(to), "move dest does not exist");
    THROW_WA_IF(WrongAnswerType::DO_NOT_SPECIFY_NONSENSE_MOVE_ACTION,
                graph.distance(current_pos_, to) == 0, "nonsense move action");
    auto next_v =
        graph.next_vertex(current_pos_, to);
    THROW_WA_IF(WrongAnswerType::INVALID_MOVE_DESTINATION,
                next_v == INVALID_VERTEX_ID,
                "Cannot move to vertex:%d from position:%s", to,
                stream_to_string(current_pos_));
    if (current_pos_.is_exact_vertex()) {
        current_pos_.v = next_v;
        current_pos_.distance_from_u = 1;
    } else {
        THROW_LOGIC_ERROR_IF(current_pos_.u != next_v &&
                                 current_pos_.v != next_v,
                             "(INTERNAL)The next vertex is not on the edge.");
        if (current_pos_.u == next_v) {
            current_pos_.distance_from_u -= 1;
        }
        if (current_pos_.v == next_v) {
            current_pos_.distance_from_u += 1;
        }
    }
    current_pos_ = graph.canonicalize_position(current_pos_);
    DEBUG("This MOVE operation has successfully finished.");
}
void Worker::do_action(discrete_time_t current_time, const action::Action& a) {
    a.dispatch(current_time, this);
}
void Worker::do_action(discrete_time_t current_time, const action::Stay& s) {
}
void Worker::do_action(discrete_time_t current_time, const action::Move& m) {
    {
        std::string wmsg;
        THROW_RUNTIME_ERROR_IF(!is_valid(&wmsg),
                               "(before moving) This worker is invalid for the "
                               "following reason(s):%s",
                               wmsg.c_str());
    }
    state.move_towards(m.to, *graph);
}
void Worker::do_action(discrete_time_t current_time, const action::Execute& e) {
    {
        std::string wmsg;
        THROW_RUNTIME_ERROR_IF(!is_valid(&wmsg),
                               "(before executing a job) This worker is "
                               "invalid for the following reason(s):%s",
                               wmsg.c_str());
    }
    man_->execute_job(current_time, *this, e.job_id, e.amount);
    TaskExecutionRecord r;
    r.time = current_time;
    r.id = e.job_id;
    r.amount = e.amount;
    task_history_.emplace(std::move(r));
    DEBUG("Task history recorded.");
}
void Worker::move_FORCE(const Position& p) {
    state.current_pos_ = p;
    DEBUG("Forced to move to position:%s", stream_to_string(p).c_str());
}
const std::set<TaskExecutionRecord>& Worker::task_history() const {
    return task_history_;
}
bool Worker::can_process(job_type_t jt) const {
    return info.can_process(jt);
}
void Worker::set_always_valid_DO_NOT_USE_(bool valid) {
    always_valid_ = valid;
}
void Worker::log_raw_action_string(std::string a) {
    raw_action_history_.push_back(a);
}
void to_json(json_ref j, const Worker& w) {
    if (w.log_mutable_only_) {
        j = json{
            {"id", w.id()},
            {"state", w.state},
            {"last_action", w.raw_action_history_.empty()
                                ? ""
                                : w.raw_action_history_.back()}
        };
    } else {
        j = json{
            {"id", w.id()},
            {"info", w.info},
            {"state", w.state},
            {"task_history", w.task_history()},
            {"is_valid", w.is_valid()},
            {"raw_action_history", w.raw_action_history_}
        };
    }
}
