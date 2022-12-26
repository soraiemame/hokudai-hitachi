#ifndef HEADER_2HC2022_SCHEDULE_MANAGER
#define HEADER_2HC2022_SCHEDULE_MANAGER 
#include <map>
#include "defines.hpp"
#include "io.hpp"
#include "schedule.hpp"
class ScheduleManager : public LogInfoSwitcher {
    std::map<worker_id_t, Schedule> schedules_;
    std::map<worker_id_t, double> schedule_penalties_;
    double schedule_penalty_base_ = INVALID_REAL;
    double schedule_penalty_coeff_ = INVALID_REAL;
    double schedule_score_scale_ = INVALID_REAL;
 public:
    void log_mutable_info_only(discrete_time_t crt0b, bool e) const override;
    double schedule_penalty_base() const;
    double schedule_penalty_coeff() const;
    void accumulate_schedule_penalty(discrete_time_t crt0b, worker_id_t wid,
                                     job_id_t jid, int amount);
    double get_schedule_penalty(worker_id_t wid0b) const;
    void set_new_schedule(discrete_time_t crt0b, worker_id_t id,
                          const Schedule& s);
    void read_from_stream(WorldType wt, std::istream& is);
    void output_to_contestant(std::ostream& os) const;
    double calc_schedule_bonus_coefficient() const;
    friend void to_json(json_ref j, const ScheduleManager& sm);
};
#endif
