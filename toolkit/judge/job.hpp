#ifndef HEADER_2HC2022_JOB
#define HEADER_2HC2022_JOB 
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "defines.hpp"
#include "io.hpp"
#include "json_fwd.hpp"
#include "piecewise_function.hpp"
#include "weather.hpp"
class JobManager;
struct JobInfo {
    job_type_t type = INVALID_JOB_TYPE;
    int n_task = INVALID_COUNT;
    sorted_point_sequence gain_function_data = {};
    vertex_index_t position = INVALID_VERTEX_ID;
    double penalty_coeff = INVALID_REAL;
    bool mandatory = false;
    std::unordered_set<job_id_t> dependency;
    job_id_t id = INVALID_JOB_ID;
    double weather_dependency = INVALID_REAL;
    bool is_valid(std::string* msg = nullptr) const;
    float_score_t gain(discrete_time_t t) const;
    void read_from_stream(WorldType wt, std::istream& is);
    friend void to_json(json_ref j, const JobInfo& ji);
    friend std::ostream& operator<<(std::ostream& os, const JobInfo& j);
};
class JobState {
    int task_cumul_ = INVALID_COUNT;
    int pending_task_done_ = 0;
 public:
    int task_cumul() const;
    bool is_valid(std::string* msg = nullptr) const;
    friend void to_json(json_ref j, const JobState& js);
    friend std::ostream& operator<<(std::ostream& os, const JobState& j);
    void initialize();
    void task_done_NOT_FINALIZED(int task_num);
    void finalize_task_done();
};
class Job : public LogInfoSwitcher {
    job_id_t id_ = INVALID_JOB_ID;
 public:
    job_id_t id() const;
    JobInfo info;
    JobState state;
    bool is_valid(std::string* msg = nullptr) const;
    friend void to_json(json_ref j, const Job& jb);
    friend std::ostream& operator<<(std::ostream& os, const Job& j);
    void read_job_info(WorldType wt, std::istream& is);
    void initialize_state();
    void task_done_NOT_FINALIZED(int amount);
    void finalize_task_done();
    bool completed() const;
    int task_rest() const;
    friend JobManager;
};
class TaskLimitInfo {
    std::vector<int> weather_limit_const_;
 public:
    void read_from_stream(std::istream& is);
    int task_limit(int n_max, double weather_dep,
                   weather_value_t weather) const;
    size_t limit_constant_num() const;
    void output_limit_constants_to_contestant(std::ostream& os) const;
    friend void to_json(json_ref j, const TaskLimitInfo& tl);
};
#endif
