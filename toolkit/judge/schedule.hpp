#ifndef HEADER_2HC2022_SCHEDULE
#define HEADER_2HC2022_SCHEDULE 
#include <string>
#include <vector>
#include "defines.hpp"
#include "error_check.hpp"
#include "json_fwd.hpp"
struct ScheduleAtom {
    job_id_t job_id = INVALID_JOB_ID;
    friend void to_json(json_ref j, const ScheduleAtom& e);
};
class Schedule {
    using container_type = std::vector<ScheduleAtom>;
    container_type elems_;
    discrete_time_t update_time_ = INVALID_TIME;
    discrete_time_t t_max_ = INVALID_TIME;
    void set_update_time(discrete_time_t t);
 public:
    using size_type = container_type::size_type;
    using iterator = container_type::iterator;
    using const_iterator = container_type::const_iterator;
    discrete_time_t update_time() const;
    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;
    void read_from_stream(discrete_time_t crt_0b_, discrete_time_t t_max,
                          std::istream& is);
    void set_empty_schedule(discrete_time_t crt_0b_, discrete_time_t t_max);
    size_t size() const;
    discrete_time_t T_max() const;
    bool is_valid(std::string* msg = nullptr) const;
    void throw_if_invalid() const;
    template <class Fn>
    void for_each_in_range(discrete_time_t st, discrete_time_t en,
                           const Fn& f) const {
        THROW_LOGIC_ERROR_IF(
            st > en, "Schedule foreach range:start time is later than end");
        THROW_LOGIC_ERROR_IF(st < update_time() || en >= t_max_,
                             "out of range");
        for (discrete_time_t t = st; t <= en; t++) {
            f(elems_[t]);
        }
    }
    double calc_diff_and_apply(double init_pen, discrete_time_t crt_0b,
                               const Schedule& sch_modify, double p_change,
                               double p_base);
    double calc_execution_penalty(discrete_time_t t, job_id_t id,
                                  int amount) const;
    friend void to_json(json_ref j, const Schedule& s);
};
#endif
