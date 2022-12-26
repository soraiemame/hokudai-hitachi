#include "job.hpp"
#include <algorithm>
#include <utility>
#include "error_check.hpp"
#include "io.hpp"
#include "lib/json.hpp"
#include "logger.hpp"
#include "worker.hpp"
bool JobInfo::is_valid(std::string* msg) const {
    bool invalid = false;
    if (msg != nullptr) {
        *msg = "";
    }
#define TRUE_IF_INVALID(m,...) \
    do { \
        if (__VA_ARGS__) { \
            if (msg != nullptr) { \
                *msg += m "\n"; \
            } \
            invalid = true; \
        } \
    } while (0)
    TRUE_IF_INVALID("Job type is INVALID or negative(internal)",
                    type < 0 || type == INVALID_JOB_TYPE);
    TRUE_IF_INVALID("Task num is INVALID", n_task == INVALID_COUNT);
    TRUE_IF_INVALID("Task num is 0 or negative", n_task < 1);
    TRUE_IF_INVALID("No gain function data.", gain_function_data.empty());
    TRUE_IF_INVALID("Position index is INVALID", position == INVALID_VERTEX_ID);
    TRUE_IF_INVALID("Position index is negative", position < 0);
    TRUE_IF_INVALID("Penalty coefficient is INVALID",
                    penalty_coeff == INVALID_REAL);
    TRUE_IF_INVALID("Penalty coefficient is out of range",
                    penalty_coeff < 0.0 || penalty_coeff > 1.0);
    return !invalid;
}
float_score_t JobInfo::gain(discrete_time_t t) const {
    return piecewise_linear_function(gain_function_data, t);
}
void JobInfo::read_from_stream(WorldType wt, std::istream& is__) {
    ValueReader is_(is__);
    {
        readline_exact(is_,
                       TAG(ValueGroup::INDEX, id),
                       TAG(ValueGroup::TYPEID, type),
                       TAG(ValueGroup::POSITIVE_COUNT, n_task),
                       TAG(ValueGroup::INDEX, position),
                       TAG(ValueGroup::COEFFICIENT, penalty_coeff),
                       TAG(ValueGroup::COEFFICIENT, weather_dependency),
                       TAG(ValueGroup::FLAG, mandatory));
        THROW_RUNTIME_ERROR_IF(weather_dependency > 1.0,
                               "Weather dependency is too large");
        THROW_RUNTIME_ERROR_IF(weather_dependency < 0.0,
                               "Weather dependency is too small");
        if (wt == WorldType::A) {
            mandatory = false;
            penalty_coeff = 1.0;
            weather_dependency = 0.0;
        }
    }
    {
        auto r = get_single_line_stream(is_);
        int gain_n;
        read_vars(r, TAG(ValueGroup::POSITIVE_COUNT, gain_n));
        THROW_RUNTIME_ERROR_IF(gain_n <= 0,
                               "gain function point num is 0 or negative");
        gain_function_data.resize(gain_n);
        for (int i = 0; i < gain_n; i++) {
            read_vars(r,
                      TAG(ValueGroup::TIME, gain_function_data[i].first),
                      TAG(ValueGroup::SCORE, gain_function_data[i].second));
            THROW_RUNTIME_ERROR_IF(gain_function_data[i].first < -1,
                                   "gain point time must be >= -1 (-1 is "
                                   "allowed only for the beginning point) (%d)",
                                   gain_function_data[i].first);
        }
        throw_if_nontrivial_chars_left(r);
        gain_function_data.validate();
    }
    {
        auto r = get_single_line_stream(is_);
        int dep_n;
        read_vars(r, TAG(ValueGroup::NON_NEGATIVE_COUNT, dep_n));
        THROW_RUNTIME_ERROR_IF(dep_n < 0, "Dependency num is negative.");
        dependency.clear();
        read_n_vars_exact<job_id_t>(
            r, dep_n, ValueGroup::INDEX, [&](job_id_t dep_id) {
                THROW_RUNTIME_ERROR_IF(dep_id < 0,
                                       "Invalid dependency job ID.");
                THROW_RUNTIME_ERROR_IF(
                    dependency.find(dep_id) != dependency.end(),
                    "There is a duplicate in job dependency.");
                dependency.insert(dep_id);
            });
    }
    {
        std::string msg;
        THROW_RUNTIME_ERROR_IF(!is_valid(&msg),
                               "(after loading job info from stream) Job info "
                               "is invalid for the following reason(s):%s",
                               msg.c_str());
    }
}
void to_json(json_ref j, const JobInfo& ji) {
    j = json{
        {"id", ji.id},
        {"type", ji.type},
        {"n_task", ji.n_task},
        {"gain_function_data", ji.gain_function_data},
        {"position", ji.position},
        {"penalty", ji.penalty_coeff},
        {"mandatory", ji.mandatory},
        {"dependency", ji.dependency},
        {"weather_dependency", ji.weather_dependency},
        {"is_valid", ji.is_valid()},
    };
}
std::ostream& operator<<(std::ostream& os, const JobInfo& j) {
    json js = j;
    os << js;
    return os;
}
bool JobState::is_valid(std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    VALIDITY_CHECK_WITH_MSG(
        task_cumul() == INVALID_COUNT || task_cumul() < 0, valid, msg,
        "Cumulative task amount is invalid (including negative)");
    VALIDITY_CHECK_WITH_MSG(pending_task_done_ < 0, valid, msg,
                            "Temporary cumulative task amount(to be recorded "
                            "at the next finalization) is negative");
    return valid;
}
void to_json(json_ref j, const JobState& js) {
    j = json{
        {"task_cumul", js.task_cumul()},
        {"pending_task_done", js.pending_task_done_}
    };
}
int JobState::task_cumul() const {
    return task_cumul_;
}
std::ostream& operator<<(std::ostream& os, const JobState& js) {
    json j = js;
    os << j;
    return os;
}
void JobState::initialize() {
    task_cumul_ = 0;
    pending_task_done_ = 0;
}
void JobState::task_done_NOT_FINALIZED(int task_num) {
    THROW_RUNTIME_ERROR_IF(task_cumul() == INVALID_COUNT,
                           "Cumulative task amount is invalid");
    pending_task_done_ += task_num;
}
void JobState::finalize_task_done() {
    task_cumul_ += pending_task_done_;
    pending_task_done_ = 0;
}
bool Job::is_valid(std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    VALIDITY_CHECK_WITH_MSG(id() == INVALID_JOB_ID, valid, msg,
                            "Job ID is invalid.");
    VALIDITY_CHECK_WITH_MSG(id() < 0, valid, msg, "Job ID is negative.");
    {
        std::string imsg;
        VALIDITY_CHECK_WITH_MSG(
            !info.is_valid(&imsg), valid, msg,
            "Job info is invalid for the following reason(s):%s", imsg.c_str());
    }
    {
        std::string smsg;
        VALIDITY_CHECK_WITH_MSG(
            !state.is_valid(&smsg), valid, msg,
            "Job State is invalid for the following reason(s):%s",
            smsg.c_str());
    }
    return valid;
}
job_id_t Job::id() const {
    return id_;
}
std::ostream& operator<<(std::ostream& os, const Job& j) {
    json js = j;
    os << js;
    return os;
}
void Job::read_job_info(WorldType wt, std::istream& is) {
    info.read_from_stream(wt, is);
}
void Job::initialize_state() {
    state.initialize();
}
void Job::task_done_NOT_FINALIZED(int amount) {
    int ntask = info.n_task;
    THROW_LOGIC_ERROR_IF(state.task_cumul() + amount > ntask,
                         "Total task amount done exceeds n_task");
    state.task_done_NOT_FINALIZED(amount);
}
void Job::finalize_task_done() {
    state.finalize_task_done();
}
bool Job::completed() const {
    THROW_LOGIC_ERROR_IF(
        state.task_cumul() > info.n_task,
        "Something is wrong; cumulative task amount exceeds n_task");
    return state.task_cumul() == info.n_task;
}
int Job::task_rest() const {
    return info.n_task - state.task_cumul();
}
void to_json(json_ref j, const Job& job) {
    if (job.log_mutable_only_) {
        j = json{{"id", job.id()}, {"state", job.state}};
    } else {
        j = json{{"id", job.id()}, {"info", job.info}, {"state", job.state}};
    }
}
void TaskLimitInfo::read_from_stream(std::istream& is) {
    ValueReader r(is);
    int weather_value_num;
    weather_limit_const_.clear();
    {
        auto ss = get_single_line_stream(r);
        read_vars(ss, TAG(ValueGroup::COUNT, weather_value_num));
        THROW_RUNTIME_ERROR_IF(weather_value_num <= 0,
                               "There should be at least one weather value");
        read_n_vars_exact<int>(
            ss, weather_value_num, ValueGroup::COEFFICIENT, [&](int c) {
                THROW_RUNTIME_ERROR_IF(
                    c < 0, "weather limit const must be zero or positive");
                weather_limit_const_.push_back(c);
            });
    }
    INFO("Task limit info loaded successfully.");
}
int TaskLimitInfo::task_limit(int n_max, double weather_dep,
                              weather_value_t weather) const {
    if (1.0 - weather_dep == 0.0 && weather_limit_const_.at(weather) == 0) {
        return n_max;
    }
    return n_max *
           std::pow(1.0 - weather_dep, weather_limit_const_.at(weather));
}
void to_json(json_ref j, const TaskLimitInfo& tl) {
    j = json{{"weather_limit_coeff", tl.weather_limit_const_}};
}
size_t TaskLimitInfo::limit_constant_num() const {
    return weather_limit_const_.size();
}
void TaskLimitInfo::output_limit_constants_to_contestant(
    std::ostream& os) const {
    ValueWriter w(os);
    output_line_to_contestant(
        w, TAG(ValueGroup::CONTAINER, StreamAdapter(
                                          weather_limit_const_,
                                          [&](ValueWriter& o, double d) {
                                              o << TAG(ValueGroup::COEFFICIENT,
                                                       d);
                                          },
                                          true)));
}
