#include "worker_manager.hpp"
#include "io.hpp"
#include "lib/json.hpp"
#include "logger.hpp"
void WorkerManager::set_schedule_manager(ScheduleManager* sm) {
    sch_man_ = sm;
}
void WorkerManager::set_job_manager(JobManager* jm) {
    job_man_ = jm;
}
void WorkerManager::set_graph(const UndirectedGraph* gp) {
    graph_ = gp;
}
int WorkerManager::worker_num() const {
    return workers_.size();
}
const Worker& WorkerManager::workers(int id) const {
    THROW_RUNTIME_ERROR_IF(!exists(id), "worker id is out of range");
    return workers_.at(id);
}
bool WorkerManager::is_valid(std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    VALIDITY_CHECK_WITH_MSG(job_man_ == nullptr, valid, msg,
                            "JobManager is not set.");
    VALIDITY_CHECK_WITH_MSG(graph_ == nullptr, valid, msg, "Graph is not set.");
    VALIDITY_CHECK_WITH_MSG(workers_.empty(), valid, msg, "No worker exists.");
    for (const auto& w : workers_) {
        std::string wmsg;
        VALIDITY_CHECK_WITH_MSG(
            !w.second.is_valid(&wmsg), valid, msg,
            "Worker %d is invalid for the following reason(s):%s", w.first,
            wmsg.c_str());
    }
    return valid;
}
void WorkerManager::execute_job(discrete_time_t current_time, const Worker& w,
                                job_id_t id, int amount) {
    std::string validity_msg;
    THROW_LOGIC_ERROR_IF(!is_valid(&validity_msg),
                         "(before executing a job) WorkerManager is invalid "
                         "for the following reason(s):%s",
                         validity_msg.c_str());
    job_man_->execute_job(current_time, w, id, amount);
    sch_man_->accumulate_schedule_penalty(current_time, w.id(), id, amount);
    DEBUG("A job execution(worker_id:%d job_id:%d amount:%d) has finished.",
          w.id(), id, amount);
}
void WorkerManager::command_for_all_workers(discrete_time_t current_time,
                                            std::istream& is) {
    std::string validity_msg;
    THROW_LOGIC_ERROR_IF(!is_valid(&validity_msg),
                         "(before interpreting commands) WorkerManager is "
                         "invalid for the following reason(s):%s",
                         validity_msg.c_str());
    for (auto& w : workers_) {
        INFO("Enter an action for the worker(ID=%d):", w.first);
        std::string line;
        std::getline(is, line);
        auto a = action::Action::interpret(line);
        w.second.do_action(current_time, *a);
        w.second.log_raw_action_string(line);
        INFO("Action accepted:%s", stream_to_string(*a).c_str());
    }
    DEBUG("All the actions have been accepted.");
    job_man_->update_other_jobs(current_time);
    DEBUG("The other relevant jobs have been processed automatically.");
}
void WorkerManager::read_workers(std::istream& is) {
    ValueReader r(is);
    int n;
    readline_exact(r, TAG(ValueGroup::POSITIVE_COUNT, n));
    THROW_RUNTIME_ERROR_IF(n <= 0, "Worker count must be positive.");
    for (int i = 0; i < n; i++) {
        int id = id_gen_++;
        auto& worker = workers_[id];
        worker.id_ = id;
        worker.read_worker_info(is);
        worker.initialize();
        worker.set_worker_manager(this);
        worker.set_graph(graph_);
    }
    DEBUG("Worker data loaded successfully.");
}
void WorkerManager::log_mutable_info_only(discrete_time_t crt0b, bool e) const {
    this->LogInfoSwitcher::log_mutable_info_only(crt0b, e);
    for_each_worker([&](worker_id_t id) {
        for (auto& kv : workers_)
            kv.second.log_mutable_info_only(crt0b, e);
    });
}
bool WorkerManager::exists(worker_id_t wid) const {
    return workers_.find(wid) != workers_.end();
}
void to_json(json_ref j, const WorkerManager& wm) {
    j = json{
        {"workers", wm.workers_},
    };
}
