#include "schedule_manager.hpp"
#include "io.hpp"
#include "lib/json.hpp"
#include "logger.hpp"
void ScheduleManager::output_to_contestant(std::ostream& os) const {
    ValueWriter wr(os);
    output_line_to_contestant(
        wr,
        TAG(ValueGroup::COEFFICIENT, schedule_penalty_coeff()),
        TAG(ValueGroup::COEFFICIENT, schedule_penalty_base()),
        TAG(ValueGroup::COEFFICIENT, schedule_score_scale_));
}
double ScheduleManager::schedule_penalty_base() const {
    return schedule_penalty_base_;
}
double ScheduleManager::schedule_penalty_coeff() const {
    return schedule_penalty_coeff_;
}
void ScheduleManager::read_from_stream(WorldType wt, std::istream& is) {
    ValueReader r(is);
    readline_exact(r,
                   TAG(ValueGroup::COEFFICIENT, schedule_penalty_coeff_),
                   TAG(ValueGroup::COEFFICIENT, schedule_penalty_base_),
                   TAG(ValueGroup::COEFFICIENT, schedule_score_scale_));
    if (wt == WorldType::A) {
        schedule_penalty_coeff_ = 1.0;
        schedule_penalty_base_ = 1.0;
        schedule_score_scale_ = 0.0;
    }
    THROW_RUNTIME_ERROR_IF(schedule_penalty_coeff_ < 0.0 ||
                               schedule_penalty_coeff_ > 1.0,
                           "Schedule penalty coeff must be in [0.0,1.0]");
    THROW_RUNTIME_ERROR_IF(schedule_penalty_base_ <= 0.0 ||
                               schedule_penalty_coeff_ > 1.0,
                           "Schedule penalty base must be in (0.0,1.0]");
    THROW_RUNTIME_ERROR_IF(schedule_score_scale_ < 0.0,
                           "Schedule score scale must be >=0");
}
void ScheduleManager::accumulate_schedule_penalty(discrete_time_t crt0b,
                                                  worker_id_t wid, job_id_t id,
                                                  int amount) {
    THROW_LOGIC_ERROR_IF(schedules_.find(wid) == schedules_.end(),
                         "No schedule is set for worker id %d", wid);
    if (schedule_penalties_.find(wid) == schedule_penalties_.end()) {
        schedule_penalties_[wid] = 1.0;
    }
    schedule_penalties_.at(wid) *=
        schedules_.at(wid).calc_execution_penalty(crt0b, id, amount);
}
double ScheduleManager::get_schedule_penalty(worker_id_t wid0b) const {
    return schedule_penalties_.at(wid0b);
}
void ScheduleManager::set_new_schedule(discrete_time_t crt0b, worker_id_t id,
                                       const Schedule& s) {
    THROW_RUNTIME_ERROR_IF(id < 0,
                           "Tried to set a schedule for negative worker id");
    if (schedule_penalties_.find(id) == schedule_penalties_.end()) {
        schedule_penalties_[id] = 1.0;
    }
    if (schedules_.find(id) == schedules_.end()) {
        THROW_LOGIC_ERROR_IF(crt0b != 0,
                             "Initial schedules must be set at t=0");
        THROW_LOGIC_ERROR_IF(s.update_time() != 0,
                             "An initial schedule has non-zero beginning time");
        schedules_.emplace(id, s);
        return;
    }
    schedule_penalties_.at(id) *= schedules_.at(id).calc_diff_and_apply(
        1.0, crt0b, s, schedule_penalty_coeff_, schedule_penalty_base_);
}
double ScheduleManager::calc_schedule_bonus_coefficient() const {
    double penalty = 1.0;
    for (auto [wid, p] : schedule_penalties_) {
        penalty *= get_schedule_penalty(wid);
    }
    INFO("Schedule penalty:%.17g", penalty);
    return 1.0 + schedule_score_scale_ * penalty;
}
void ScheduleManager::log_mutable_info_only(discrete_time_t crt0b,
                                            bool e) const {
    this->LogInfoSwitcher::log_mutable_info_only(crt0b, e);
}
void to_json(json_ref j, const ScheduleManager& sm) {
    if (sm.log_mutable_only_) {
        std::map<worker_id_t, std::reference_wrapper<const Schedule>> sch;
        for (auto& kv : sm.schedules_) {
            if (kv.second.update_time() == sm.switched_time_) {
                sch.emplace(kv.first, kv.second);
            }
        }
        j = json{
            {"updated_schedules", sch},
            {"schedule_penalties", sm.schedule_penalties_}
        };
    } else {
        j = json{
            {"schedules", sm.schedules_},
            {"schedule_penalties", sm.schedule_penalties_}
        };
    }
}
