#include "schedule.hpp"
#include <algorithm>
#include <string>
#include "error_check.hpp"
#include "io.hpp"
#include "lib/json.hpp"
#include "logger.hpp"
void to_json(json_ref j, const ScheduleAtom& a) {
    j = json{{"job_id", a.job_id}};
}
discrete_time_t Schedule::update_time() const {
    return update_time_;
}
void Schedule::set_update_time(discrete_time_t t) {
    update_time_ = t;
}
Schedule::iterator Schedule::begin() {
    return elems_.begin();
}
Schedule::const_iterator Schedule::begin() const {
    return elems_.begin();
}
Schedule::iterator Schedule::end() {
    return elems_.end();
}
Schedule::const_iterator Schedule::end() const {
    return elems_.end();
}
void Schedule::read_from_stream(discrete_time_t crt_0b_ ,
                                discrete_time_t t_max, std::istream& is) {
    ValueReader r(is);
    set_update_time(crt_0b_);
    t_max_ = t_max;
    discrete_time_t len = (t_max - 1) - crt_0b_ + 1;
    auto ssr = get_single_line_stream(r);
    elems_.clear();
    read_n_vars_exact<job_id_t>(
        ssr, len, ValueGroup::INDEX,
        [&](job_id_t jid) {
            ScheduleAtom a;
            a.job_id = jid;
            elems_.push_back(a);
        },
        WrongAnswerType::SCHEDULE_JOB_NUM_MISMATCH);
    throw_if_invalid();
}
void Schedule::set_empty_schedule(discrete_time_t crt_0b_ ,
                                  discrete_time_t t_max) {
    set_update_time(crt_0b_);
    t_max_ = t_max;
    discrete_time_t len = (t_max - 1) - crt_0b_ + 1;
    elems_.clear();
    for (discrete_time_t t = 0; t < len; t++) {
        ScheduleAtom a;
        a.job_id = INVALID_JOB_ID;
        elems_.push_back(a);
    }
    throw_if_invalid();
}
size_t Schedule::size() const {
    return elems_.size();
}
bool Schedule::is_valid(std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    VALIDITY_CHECK_WITH_MSG((t_max_ - 1) - update_time() + 1 != size(), valid,
                            msg, "Schedule length is invalid");
    return valid;
}
void Schedule::throw_if_invalid() const {
    std::string msg;
    THROW_RUNTIME_ERROR_IF(
        !is_valid(&msg),
        "This schedule is invalid for the following reason(s):%s", msg.c_str());
}
double Schedule::calc_diff_and_apply(double init_pen, discrete_time_t crt_0b,
                                     const Schedule& sch_modify,
                                     double p_change, double p_base) {
    THROW_LOGIC_ERROR_IF(sch_modify.t_max_ != t_max_, "T_max is different");
    THROW_LOGIC_ERROR_IF(sch_modify.update_time() < update_time(),
                         "modified schedule is ealier than base");
    discrete_time_t latest = sch_modify.update_time();
    THROW_LOGIC_ERROR_IF(crt_0b < latest,
                         "current time is ealier than schedule beginning time");
    for (discrete_time_t rt = 0; (t_max_ - 1) - rt >= crt_0b; rt++) {
        discrete_time_t t = (t_max_ - 1) - rt;
        discrete_time_t dt = t - crt_0b;
        ScheduleAtom a_ch =
            sch_modify.elems_[sch_modify.elems_.size() - rt - 1];
        ScheduleAtom a_base = elems_[elems_.size() - rt - 1];
        if (a_ch.job_id == a_base.job_id) {
        } else {
            double p = 1.0 - p_change * std::pow(p_base, dt);
            init_pen *= p;
        }
        elems_[elems_.size() - rt - 1] = a_ch;
    }
    set_update_time(crt_0b);
    return init_pen;
}
double Schedule::calc_execution_penalty(discrete_time_t t, job_id_t id,
                                        int amount) const {
    THROW_RUNTIME_ERROR_IF(id < 0, "Executed invalid id");
    THROW_RUNTIME_ERROR_IF(amount <= 0, "Executed invalid amount");
    ScheduleAtom a = elems_.at(t);
    if (a.job_id != id) {
        return 0.0;
    }
    return 1.0;
}
void to_json(json_ref j, const Schedule& s) {
    j = json{
        {"t_max", s.t_max_},
        {"update_time", s.update_time()},
        {"elements", s.elems_},
        {"is_valid", s.is_valid()},
    };
}
