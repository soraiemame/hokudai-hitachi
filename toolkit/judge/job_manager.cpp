#include "job_manager.hpp"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "error_check.hpp"
#include "io.hpp"
#include "lib/json.hpp"
#include "logger.hpp"
#include "worker.hpp"
void JobManager::set_world_type(WorldType wt) {
    world_type_ = wt;
}
WorldType JobManager::world_type() const {
    return world_type_;
}
bool JobManager::done_job_acceptance() const {
    return done_job_acceptance_;
}
void JobManager::accept_jobs(std::istream& is_) {
    ValueReader w(is_);
    auto ss = get_single_line_stream(w);
    int n;
    read_vars(ss, TAG(ValueGroup::COUNT, n));
    THROW_WA_IF(WrongAnswerType::SELECTED_NUM_IS_NEGATIVE, n < 0,
                "The number of jobs accepted is negative.");
    std::unordered_set<job_id_t> ids;
    read_n_vars_exact<job_id_t>(
        ss, n, ValueGroup::INDEX,
        [&](job_id_t id) {
            THROW_WA_IF(WrongAnswerType::SELECTED_JOB_DOES_NOT_EXIST,
                        !job_exists(id), "one of selected jobs does not exist");
            THROW_WA_IF(WrongAnswerType::DUPS_IN_SELECTED_JOBS,
                        ids.find(id) != ids.end(),
                        "duplicate in selected jobs");
            ids.emplace(id);
        },
        WrongAnswerType::SELECTED_NUM_NEQ_ACTUAL_SELECTED_ID_NUM);
    accept_jobs_(ids);
    INFO("The selected jobs have been accepted successfully.");
}
void JobManager::accept_all_jobs() {
    std::unordered_set<job_id_t> all_jobs;
    for_each_job([&](job_id_t id) { all_jobs.emplace(id); });
    accept_jobs_(all_jobs, true);
}
void JobManager::accept_jobs_(const std::unordered_set<job_id_t>& acc_jobs,
                              bool fast_all_jobs) {
    THROW_LOGIC_ERROR_IF(done_job_acceptance(),
                         "Job acceptance is already done.");
    if (fast_all_jobs) {
        THROW_LOGIC_ERROR_IF(job_num() != acc_jobs.size(),
                             "Specified fast_all_jobs but job_num is not equal "
                             "to accepted job num");
        accepted_jobs_ = acc_jobs;
        relevant_jobs_.clear();
        relevant_jobs_.insert(acc_jobs.begin(), acc_jobs.end());
    } else {
        for (const auto& jkv : jobs_) {
            if (jkv.second.info.mandatory) {
                THROW_WA_IF(WrongAnswerType::MUST_SELECT_ALL_MANDATORY_JOBS,
                            acc_jobs.find(jkv.second.id()) == acc_jobs.end(),
                            "All the mandatory jobs have to be accepted.");
            }
        }
        accepted_jobs_.clear();
        for (const auto& aid : acc_jobs) {
            THROW_WA_IF(WrongAnswerType::SELECTED_JOB_DOES_NOT_EXIST,
                        !job_exists(aid),
                        "Tried to accept jobs which do not exist.");
            for (auto dep_id : jobs(aid).info.dependency) {
                THROW_WA_IF(WrongAnswerType::MUST_SELECT_ALL_DEPENDENCIES,
                            acc_jobs.find(dep_id) == acc_jobs.end(),
                            "For every job, all jobs that it "
                            "depends on must be included.");
            }
            accepted_jobs_.insert(aid);
        }
        std::unordered_set<job_id_t> unsorted_rel;
        for (const auto& aid : accepted_jobs_) {
            if (unsorted_rel.find(aid) == unsorted_rel.end()) {
                unsorted_rel.merge(get_all_relevant_jobs_(aid));
            }
        }
        relevant_jobs_.clear();
        relevant_jobs_.insert(unsorted_rel.begin(), unsorted_rel.end());
    }
    done_job_acceptance_ = true;
}
void JobManager::execute_job(discrete_time_t current_time, const Worker& worker,
                             job_id_t id, int amount) {
    {
        std::string msg;
        THROW_RUNTIME_ERROR_IF(
            !is_valid(&msg),
            "JobManager is invalid for the following reason(s):%s",
            msg.c_str());
    }
    {
        std::string msg;
        THROW_RUNTIME_ERROR_IF(
            !worker.is_valid(&msg),
            "The worker is invalid for the following reason(s):%s",
            msg.c_str());
    }
    THROW_WA_IF(WrongAnswerType::EXECUTE_JOB_ID_DOES_NOT_EXIST,
                id == INVALID_JOB_ID, "Job id is invalid");
    THROW_WA_IF(WrongAnswerType::EXECUTE_JOB_ID_DOES_NOT_EXIST, id < 0,
                "Job id (1-based) is 0 or negative");
    THROW_WA_IF(WrongAnswerType::EXECUTE_JOB_ID_DOES_NOT_EXIST, !job_exists(id),
                "The specified job(internal:%d) does not exist.", id);
    THROW_LOGIC_ERROR_IF(!done_job_acceptance(),
                         "Job acceptance phase is not finished yet.");
    THROW_WA_IF(WrongAnswerType::EXECUTE_JOB_ID_IS_NOT_SELECTED,
                !is_job_accepted(id), "The specified job was not accepted.");
    THROW_WA_IF(WrongAnswerType::INVALID_EXECUTE_AMOUNT, amount <= 0,
                "Task amount is zero or negative.");
    THROW_WA_IF(WrongAnswerType::NO_JOB_EXISTS_AT_CURRENT_POS,
                !worker.current_position().is_exact_vertex(),
                "The worker is not on any vertex");
    auto& job = jobs(id);
    THROW_WA_IF(WrongAnswerType::NO_JOB_EXISTS_AT_CURRENT_POS,
                worker.current_position().u != job.info.position,
                "Job position(%d) != Worker position(%d)", job.info.position,
                worker.current_position().u);
    THROW_WA_IF(WrongAnswerType::WORKER_CANNOT_PROCESS_THIS_TYPE,
                !worker.can_process(job.info.type),
                "The specified job cannot be processed by this "
                "worker due to its job type restriction.");
    if (job.info.gain(current_time) <= 0) {
        std::cerr << "GAIN FUNCTION DEBUG INFO:" << std::endl;
        std::cerr << job.info.gain_function_data << std::endl;
        std::cerr << "Adjacent values:" << job.info.gain(current_time - 1)
                  << " " << job.info.gain(current_time) << " "
                  << job.info.gain(current_time + 1) << std::endl;
    }
    THROW_WA_IF(WrongAnswerType::NONPOSITIVE_REWARD_VALUE,
                job.info.gain(current_time) <= 0,
                "Cannot be processed when the gain value is zero or "
                "negative.(t=%d,v=%s)",
                current_time, job.info.gain(current_time).str().c_str());
    {
        const weather_value_t weather_value =
            weather_man_->get_weather_value(current_time);
        const int task_limit =
            tl_info_->task_limit(worker.info.max_task,
                                 job.info.weather_dependency,
                                 weather_value);
        THROW_WA_IF(WrongAnswerType::EXCEEDS_TASK_EXECUTION_LIMIT,
                    amount > task_limit,
                    "The specified task amount exceeds current task "
                    "limit (limit=%d,given=%d,weather=%d)",
                    task_limit, amount, weather_value);
    }
    THROW_WA_IF(WrongAnswerType::EXCEEDS_TASK_AMOUNT_REST,
                amount + job.state.task_cumul() > job.info.n_task,
                "Total task amount exceeds n_task of the specified job.");
    for (auto dep_id : job.info.dependency) {
        THROW_LOGIC_ERROR_IF(
            !job_exists(dep_id),
            "Some jobs on which the specifed job depends do not exist.");
        THROW_WA_IF(
            WrongAnswerType::DEPENDENCIES_NOT_DONE, !jobs(dep_id).completed(),
            "Some jobs on which the specified job depends are not completed.");
    }
    float_score_t gained_score = amount * job.info.gain(current_time);
    score_man_->add_score_jobwise(id, gained_score);
    job.task_done_NOT_FINALIZED(amount);
    executed_jobs_.emplace(id);
}
std::unordered_set<job_id_t>
JobManager::get_all_relevant_jobs_(job_id_t id) const {
    THROW_LOGIC_ERROR_IF(id == INVALID_JOB_ID, "Job ID is invalid");
    THROW_LOGIC_ERROR_IF(id < 0, "Job ID is negative (internal).");
    THROW_LOGIC_ERROR_IF(
        !job_exists(id),
        "(while finding all relevant jobs) The specified job does not exist.");
    if (relevant_jobs_memoized_.find(id) != relevant_jobs_memoized_.end()) {
        return relevant_jobs_memoized_.at(id);
    }
    std::unordered_set<job_id_t> ret{id};
    for (job_id_t dep_id : jobs(id).info.dependency) {
        ret.merge(get_all_relevant_jobs_(dep_id));
    }
    auto p = relevant_jobs_memoized_.emplace(id, std::move(ret));
    THROW_LOGIC_ERROR_IF(!p.second, "Failed to memoize dependency graph.");
    return p.first->second;
}
bool JobManager::completed_all_jobs() const {
    THROW_LOGIC_ERROR_IF(!done_job_acceptance(),
                         "Job acceptance phase is not finished yet.");
    for (auto id : accepted_jobs_) {
        if (!jobs(id).completed()) {
            return false;
        }
    }
    return true;
}
bool JobManager::is_valid(std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    VALIDITY_CHECK_WITH_MSG(world_type() == WorldType::INVALID, valid, msg,
                            "World type is invalid.");
    VALIDITY_CHECK_WITH_MSG(weather_man_ == nullptr, valid, msg,
                            "WeatherManager is not set.");
    VALIDITY_CHECK_WITH_MSG(tl_info_ == nullptr, valid, msg,
                            "TaskLimitInfo is not set.");
    VALIDITY_CHECK_WITH_MSG(score_man_ == nullptr, valid, msg,
                            "ScoreManager is not set.");
    VALIDITY_CHECK_WITH_MSG(jobs_.empty(), valid, msg, "Job set is empty.");
    return valid;
}
void JobManager::set_weather_manager(const WeatherManager* wmp) {
    weather_man_ = wmp;
}
void JobManager::set_task_limit_info(const TaskLimitInfo* tlip) {
    tl_info_ = tlip;
}
void JobManager::set_score_manager(ScoreManager* sm) {
    score_man_ = sm;
}
void JobManager::read_jobs(std::istream& is) {
    ValueReader r(is);
    int n;
    readline_exact(r, TAG(ValueGroup::COUNT, n));
    THROW_RUNTIME_ERROR_IF(n <= 0, "Job count(%d) is invalid", n);
    for (int i = 0; i < n; i++) {
        int id = id_gen++;
        auto& job = jobs_[id];
        job.id_ = id;
        job.read_job_info(world_type(), is);
        job.initialize_state();
        THROW_RUNTIME_ERROR_IF(world_type() == WorldType::A &&
                                   job.info.mandatory,
                               "Mandatody jobs should not exist in Task A");
    }
    INFO("Job data loaded successfully.");
}
const Job& JobManager::jobs(job_id_t id) const {
    return jobs_.at(id);
}
Job& JobManager::jobs(job_id_t id) {
    return jobs_.at(id);
}
bool JobManager::job_exists(job_id_t id) const {
    return jobs_.find(id) != jobs_.end();
}
bool JobManager::is_job_accepted(job_id_t id) const {
    return accepted_jobs_.find(id) != accepted_jobs_.end();
}
int JobManager::relevant_job_num() const {
    int c = 0;
    for_each_relevant_job([&](...) { c++; });
    return c;
}
int JobManager::unfinished_relevant_job_num() const {
    int c = 0;
    for_each_unfinished_relevant_job([&](...) { c++; });
    return c;
}
bool JobManager::dependency_done(job_id_t id) const {
    const Job& job = jobs(id);
    for (auto dep : job.info.dependency) {
        if (!jobs(dep).completed()) {
            return false;
        }
    }
    return true;
}
void JobManager::update_other_jobs(discrete_time_t t) {
}
int JobManager::job_num() const {
    return jobs_.size();
}
void JobManager::log_mutable_info_only(discrete_time_t crt0b, bool e) const {
    this->LogInfoSwitcher::log_mutable_info_only(crt0b, e);
    for_each_job(
        [&](job_id_t id) { jobs(id).log_mutable_info_only(crt0b, e); });
}
void to_json(json_ref j, const JobManager& jm) {
    if (jm.log_mutable_only_) {
        std::vector<std::reference_wrapper<const Job>> reljobrefs;
        jm.for_each_relevant_job(
            [&](job_id_t id) { reljobrefs.push_back(jm.jobs(id)); });
        j = json{
            {"relevant_jobs", reljobrefs},
        };
    } else {
        j = json{
            {"jobs", jm.jobs_},
            {"done_job_acceptance", jm.done_job_acceptance_},
            {"accepted_job_ids", jm.accepted_jobs_},
            {"relevant_job_ids", jm.relevant_jobs_},
        };
    }
}
