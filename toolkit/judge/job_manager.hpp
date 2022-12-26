#ifndef HEADER_2HC2022_JOB_MANAGER
#define HEADER_2HC2022_JOB_MANAGER 
#include <atomic>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "io.hpp"
#include "job.hpp"
#include "json_fwd.hpp"
#include "score.hpp"
#include "weather.hpp"
#include "worker.hpp"
class JobManager : public LogInfoSwitcher {
    const WeatherManager* weather_man_ = nullptr;
    const TaskLimitInfo* tl_info_ = nullptr;
    ScoreManager* score_man_ = nullptr;
    std::map<job_id_t, Job> jobs_;
    std::atomic<int> id_gen{0};
    bool done_job_acceptance_ = false;
    std::unordered_set<job_id_t> accepted_jobs_;
    void accept_jobs_(const std::unordered_set<job_id_t>& acc_jobs,
                      bool fast_all_jobs = false);
    mutable std::unordered_map<job_id_t, std::unordered_set<job_id_t>>
        relevant_jobs_memoized_;
    std::set<job_id_t> relevant_jobs_;
    std::unordered_set<job_id_t> get_all_relevant_jobs_(job_id_t origin) const;
    WorldType world_type_ = WorldType::INVALID;
 public:
    std::unordered_set<job_id_t> executed_jobs_;
    void log_mutable_info_only(discrete_time_t crt0b, bool e) const override;
    void set_world_type(WorldType wt);
    WorldType world_type() const;
    bool done_job_acceptance() const;
    template <class Fn> void for_each_accepted_job(const Fn& f) const {
        THROW_LOGIC_ERROR_IF(!done_job_acceptance(),
                             "Job acceptance phase is not finished yet.");
        for (auto id : accepted_jobs_) {
            f(id);
        }
    }
    template <class Fn> void for_each_job(const Fn& f) const {
        for (const auto& kv : jobs_) {
            f(kv.first);
        }
    }
    bool completed_all_jobs() const;
    bool is_valid(std::string* msg = nullptr) const;
    void set_weather_manager(const WeatherManager* wmp);
    void set_task_limit_info(const TaskLimitInfo* tlip);
    void set_score_manager(ScoreManager* sm);
    void read_jobs(std::istream& is);
    const Job& jobs(job_id_t id) const;
    Job& jobs(job_id_t id);
    bool job_exists(job_id_t id) const;
    bool is_job_accepted(job_id_t id) const;
    void execute_job(discrete_time_t current_time, const Worker& worker,
                     job_id_t id, int amount);
    void accept_jobs(std::istream& is);
    void accept_all_jobs();
    template <class Fn> void for_each_relevant_job(const Fn& callback) const {
        if (!done_job_acceptance()) {
            throw std::logic_error("Finish job acceptance phase first.");
        }
        for (auto rid : relevant_jobs_) {
            callback(rid);
        }
    }
    int relevant_job_num() const;
    template <class Fn>
    void for_each_unfinished_relevant_job(const Fn& callback) const {
        for_each_relevant_job([&](job_id_t jid) {
            if (!jobs(jid).completed()) {
                callback(jid);
            }
        });
    }
    int unfinished_relevant_job_num() const;
    bool dependency_done(job_id_t id) const;
    void update_other_jobs(discrete_time_t t);
    int job_num() const;
    friend void to_json(json_ref j, const JobManager& jm);
};
#endif
