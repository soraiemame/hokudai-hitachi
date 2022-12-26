#ifndef HEADER_2HC2022_WORLD
#define HEADER_2HC2022_WORLD 
#include <string>
#include "defines.hpp"
#include "graph.hpp"
#include "io.hpp"
#include "job_manager.hpp"
#include "json_fwd.hpp"
#include "schedule_manager.hpp"
#include "worker_manager.hpp"
class World : public LogInfoSwitcher {
    discrete_time_t T_MAX_ = INVALID_TIME;
    bool initialized_ = false;
    bool loaded_ = false;
    std::ostream* json_log_ofs = nullptr;
    void begin_json_log_output() const;
    mutable std::streamsize log_prec_backup_;
    void begin_turn_json_log_output() const;
    void output_turn_data_into_log(discrete_time_t t_0b) const;
    void end_turn_json_log_output() const;
    void end_json_log_output(std::string final_score_str_,
                             double unfinished_penalty_) const;
    WorldType type_ = WorldType::A;
    void set_world_type(WorldType ty_);
    void output_forecast_to_contestant_(std::ostream& os,
                                        discrete_time_t t_0b) const;
 public:
    void log_mutable_info_only(discrete_time_t crt0b, bool e) const override;
    void set_json_log_output_stream(std::ostream* optr);
    discrete_time_t T_MAX() const;
    UndirectedGraph graph;
    JobManager job_manager;
    ScoreManager score_manager;
    TaskLimitInfo task_limit_info;
    WeatherManager weather_manager;
    WorkerManager worker_manager;
    ScheduleManager schedule_manager;
    bool is_initialized() const;
    void initialize();
    bool is_loaded() const;
    void read_from_stream(std::istream& is);
    final_result_t interact(std::istream& is, std::ostream& os);
    void output_graph_data_to_contestant(std::ostream& os) const;
    void output_worker_initial_data_to_contestant(std::ostream& os) const;
    void output_all_jobs_initial_data_to_contestant(std::ostream& os) const;
    void output_weather_initial_data_to_contestant(std::ostream& os) const;
    void output_schedule_score_info_to_contestant(std::ostream& os) const;
    void update_turn(discrete_time_t t_0b);
    void output_turn_data_to_contestant(discrete_time_t t_0b,
                                        std::ostream& os) const;
    void input_turn_data_from_contestant(discrete_time_t t_0b,
                                         std::istream& is);
    WorldType world_type() const;
    friend void to_json(json_ref j, const World& w);
};
#endif
