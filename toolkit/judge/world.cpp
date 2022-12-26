#include "world.hpp"
#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include "error_check.hpp"
#include "io.hpp"
#include "lib/json.hpp"
#include "logger.hpp"
void World::set_json_log_output_stream(std::ostream* optr) {
    json_log_ofs = optr;
}
void World::begin_json_log_output() const {
    if (json_log_ofs) {
        log_prec_backup_ =
            json_log_ofs->precision(std::numeric_limits<double>::max_digits10);
        *json_log_ofs << R"({"world_type":")"
                      << (world_type() == WorldType::A ? "A" : "B")
                      << R"(","begin":)" << json(*this) << "," << std::endl;
    }
}
void World::begin_turn_json_log_output() const {
    if (json_log_ofs)
        *json_log_ofs << R"("turn_log":[)";
}
void World::output_turn_data_into_log(discrete_time_t t_0b) const {
    if (json_log_ofs) {
        log_mutable_info_only(t_0b, true);
        *json_log_ofs << json(*this);
        log_mutable_info_only(t_0b, false);
        if (t_0b < T_MAX() - 1) {
            *json_log_ofs << ",";
        }
        *json_log_ofs << std::endl;
    }
}
void World::end_turn_json_log_output() const {
    if (json_log_ofs) {
        *json_log_ofs << "],";
    }
}
void World::end_json_log_output(std::string final_score_str_,
                                double unfinished_penalty_) const {
    if (json_log_ofs) {
        *json_log_ofs << "\"score\":" << final_score_str_
                      << ",\"unfinished_penalty\":" << unfinished_penalty_
                      << ",\"executed_jobs\":[";
        if (job_manager.executed_jobs_.size() == 0) {
        } else {
            auto it = job_manager.executed_jobs_.begin();
            *json_log_ofs << *it;
            it++;
            for (; it != job_manager.executed_jobs_.end(); it++) {
                *json_log_ofs << "," << *it;
            }
        }
        *json_log_ofs << "]" << std::endl;
        *json_log_ofs << "}";
        json_log_ofs->precision(log_prec_backup_);
    }
}
discrete_time_t World::T_MAX() const {
    return T_MAX_;
}
bool World::is_initialized() const {
    return initialized_;
}
void World::initialize() {
    score_manager.initialize();
    worker_manager.set_graph(&graph);
    worker_manager.set_job_manager(&job_manager);
    worker_manager.set_schedule_manager(&schedule_manager);
    job_manager.set_score_manager(&score_manager);
    job_manager.set_task_limit_info(&task_limit_info);
    job_manager.set_weather_manager(&weather_manager);
    initialized_ = true;
}
bool World::is_loaded() const {
    return loaded_;
}
void World::set_world_type(WorldType ty) {
    type_ = ty;
}
WorldType World::world_type() const {
    return type_;
}
void World::read_from_stream(std::istream& is_raw) {
    THROW_LOGIC_ERROR_IF(!is_initialized(),
                         "Initialize world before read data from stream.");
    ValueReader is(is_raw);
    std::string wtstr;
    readline_exact(is, TAG(ValueGroup::STRING, wtstr));
    WorldType wt = to_world_type(wtstr);
    THROW_RUNTIME_ERROR_IF(wt == WorldType::INVALID, "World type is invalid.");
    set_world_type(wt);
    job_manager.set_world_type(wt);
    readline_exact(is,
                   TAG(ValueGroup::POSITIVE_COUNT, T_MAX_));
    THROW_RUNTIME_ERROR_IF(T_MAX_ < 1, "T_max has to be positive.");
    graph.read_graph(is_raw);
    graph.compute_distance_info();
    worker_manager.read_workers(is_raw);
    job_manager.read_jobs(is_raw);
    weather_manager.read_from_stream(is_raw, T_MAX_);
    task_limit_info.read_from_stream(is_raw);
    schedule_manager.read_from_stream(world_type(), is_raw);
    loaded_ = true;
}
void World::output_graph_data_to_contestant(std::ostream& os) const {
    ValueWriter w(os);
    output_line_to_contestant(
        w,
        TAG(ValueGroup::POSITIVE_COUNT, graph.vertex_num()),
        TAG(ValueGroup::NON_NEGATIVE_COUNT, graph.edge_num()));
    graph.for_each_edge(
        [&](vertex_index_t from, vertex_index_t to, distance_t d) {
            output_line_to_contestant_no_flush(
                w,
                TAG(ValueGroup::INDEX, from),
                TAG(ValueGroup::INDEX, to),
                TAG(ValueGroup::DISTANCE, d));
        });
    os.flush();
}
void World::output_worker_initial_data_to_contestant(std::ostream& os) const {
    ValueWriter wr(os);
    output_line_to_contestant(wr,
                              TAG(ValueGroup::POSITIVE_COUNT,
                                  worker_manager.worker_num()));
    auto type_output = [](ValueWriter& wr_inner, job_type_t jt) {
        wr_inner << TAG(ValueGroup::TYPEID, jt);
    };
    worker_manager.for_each_worker([&](worker_id_t wid) {
        const auto& w = worker_manager.workers(wid);
        output_line_to_contestant_no_flush(
            wr,
            TAG(ValueGroup::INDEX, w.info.initial_position),
            TAG(ValueGroup::POSITIVE_COUNT,
                w.info.max_task),
            TAG(ValueGroup::CONTAINER,
                StreamAdapter(
                    w.info.processable_types,
                    type_output)));
    });
    os.flush();
}
void World::output_all_jobs_initial_data_to_contestant(std::ostream& os) const {
    ValueWriter wr(os);
    output_line_to_contestant(
        wr, TAG(ValueGroup::POSITIVE_COUNT, job_manager.job_num()));
    job_manager.for_each_job([&](job_id_t jid) {
        const auto& job = job_manager.jobs(jid);
        output_line_to_contestant_no_flush(
            wr,
            TAG(ValueGroup::INDEX, job.id()),
            TAG(ValueGroup::TYPEID, job.info.type),
            TAG(ValueGroup::POSITIVE_COUNT, job.info.n_task),
            TAG(ValueGroup::INDEX, job.info.position),
            TAG(world_type() == WorldType::B ? ValueGroup::COEFFICIENT
                                             : ValueGroup::NONE,
                job.info.penalty_coeff),
            TAG(world_type() == WorldType::B ? ValueGroup::COEFFICIENT
                                             : ValueGroup::NONE,
                job.info.weather_dependency),
            TAG(world_type() == WorldType::B ? ValueGroup::FLAG
                                             : ValueGroup::NONE,
                job.info.mandatory));
        using tv_pair = decltype(job.info.gain_function_data)::
            value_type;
        auto tv_pair_output = [](ValueWriter& wr_inner, const tv_pair& p) {
            concat_for_contestant(
                wr_inner,
                TAG(ValueGroup::TIME, p.first),
                TAG(ValueGroup::SCORE, p.second));
        };
        output_line_to_contestant_no_flush(
            wr,
            TAG(ValueGroup::CONTAINER,
                StreamAdapter(job.info.gain_function_data, tv_pair_output)));
        auto depend_output = [](ValueWriter& wr_inner, const job_id_t id) {
            wr_inner << TAG(ValueGroup::INDEX, id);
        };
        output_line_to_contestant_no_flush(
            wr, TAG(ValueGroup::CONTAINER,
                    StreamAdapter(job.info.dependency, depend_output)));
    });
    os.flush();
}
void World::output_weather_initial_data_to_contestant(std::ostream& os) const {
    ValueWriter wr(os);
    output_line_to_contestant(
        wr,
        TAG(ValueGroup::POSITIVE_COUNT,
            weather_manager.weather_division_length()),
        TAG(ValueGroup::POSITIVE_COUNT,
            weather_manager.weather_value_num()));
    weather_manager
        .output_transition_matrix_to_contestant_WITH_TRAILING_NEWLINE(os);
    task_limit_info.output_limit_constants_to_contestant(os);
}
void World::output_schedule_score_info_to_contestant(std::ostream& os) const {
    schedule_manager.output_to_contestant(os);
}
void World::update_turn(discrete_time_t t_0b) {
    if (t_0b > 0) {
        weather_manager.update(t_0b);
    }
}
void World::output_forecast_to_contestant_(std::ostream& os,
                                           discrete_time_t t_0b) const {
    ValueWriter wr(os);
    int weather_div_len = weather_manager.weather_division_length();
    int rest_forecast_num = (T_MAX_ - t_0b) / weather_div_len;
#ifndef ONLINE_JUDGE
    auto forecast = weather_manager.probability_forecast(t_0b);
#else
    auto forecast_str = weather_manager.probability_forecast_string(t_0b);
#endif
    for (int k = 0; k < rest_forecast_num; k++) {
        int forecast_time_0b = t_0b + k * weather_div_len;
#ifndef ONLINE_JUDGE
        const auto& probs = forecast[forecast_time_0b - t_0b];
        output_line_to_contestant_no_flush(
            wr, TAG(ValueGroup::TIME, forecast_time_0b),
            TAG(ValueGroup::CONTAINER,
                StreamAdapter(
                    probs,
                    [](ValueWriter& wr_inner, double p) {
                        wr_inner << TAG(ValueGroup::PROBABILITY, p);
                    },
                    true)));
#else
        os << to_1b(forecast_time_0b) << " "
           << *(forecast_str[forecast_time_0b - t_0b]);
#endif
    }
#ifdef ONLINE_JUDGE
#endif
    os.flush();
}
void World::output_turn_data_to_contestant(discrete_time_t t_0b,
                                           std::ostream& os) const {
    ValueWriter wr(os);
    output_line_to_contestant(
        wr, TAG(ValueGroup::WEATHER_VALUE,
                weather_manager.get_weather_value(t_0b)));
    output_line_to_contestant(
        wr, TAG(ValueGroup::NON_NEGATIVE_COUNT,
                job_manager.relevant_job_num()));
    job_manager.for_each_relevant_job([&](job_id_t rel_job_id) {
        const auto& job = job_manager.jobs(rel_job_id);
        {
            output_line_to_contestant_no_flush(
                wr,
                TAG(ValueGroup::INDEX, job.id()),
                TAG(ValueGroup::NON_NEGATIVE_COUNT,
                    job.task_rest())
            );
        }
    });
    os.flush();
    {
        worker_manager.for_each_worker([&](worker_id_t id) {
            const auto& w = worker_manager.workers(id);
            auto pos = w.state.current_position();
            output_line_to_contestant_no_flush(
                wr,
                TAG(ValueGroup::INDEX, w.id()),
                TAG(ValueGroup::INDEX, pos.u),
                TAG(ValueGroup::INDEX, pos.v),
                TAG(ValueGroup::DISTANCE,
                    pos.distance_from_u));
        });
        os.flush();
    }
    {
        int weather_div_len = weather_manager.weather_division_length();
        if (t_0b % weather_div_len == 0) {
            output_forecast_to_contestant_(os, t_0b);
        }
    }
}
void World::input_turn_data_from_contestant(discrete_time_t t_0b,
                                            std::istream& is) {
    ValueReader r(is);
    if (world_type() == WorldType::B) {
        int n_change;
        readline_exact(r,
                       TAGWA(WrongAnswerType::NEGATIVE_WORKER_NUM_FOR_SCHEDULES,
                             ValueGroup::NON_NEGATIVE_COUNT,
                             n_change));
        THROW_WA_IF(WrongAnswerType::TOO_LARGE_WORKER_NUM_FOR_SCHEDULES,
                    n_change > worker_manager.worker_num(),
                    "too large worker num");
        if (t_0b == 0) {
            THROW_WA_IF(WrongAnswerType::FORCE_SCHEDULE_AT_TURN_1_NUM_WORKER,
                        n_change != worker_manager.worker_num(),
                        "Initial schedules must be submitted "
                        "for all the workers.(Invalid number)");
        }
        auto idssr = get_single_line_stream(r);
        std::vector<worker_id_t> change_ids;
        read_n_vars_exact<worker_id_t>(
            idssr, n_change, ValueGroup::INDEX,
            [&](worker_id_t wid_0b) {
                THROW_WA_IF(WrongAnswerType::WORKER_DOES_NOT_EXIST_FOR_SCHEDULE,
                            !worker_manager.exists(wid_0b),
                            "worker does not exist");
                change_ids.push_back(wid_0b);
            },
            WrongAnswerType::N_CHANGE_NEQ_ACTUAL_WORKER_ID_NUM);
        {
            auto sorted = change_ids;
            std::sort(sorted.begin(), sorted.end());
            THROW_WA_IF(WrongAnswerType::DUPS_IN_SCHEDULE_WORKERS,
                        std::adjacent_find(sorted.begin(), sorted.end()) !=
                            sorted.end(),
                        "duplicate in schedule worker ids");
            if (t_0b == 0) {
                for (int i = 0; i < sorted.size(); i++) {
                    THROW_WA_IF(
                        WrongAnswerType::FORCE_SCHEDULE_AT_TURN_1_ID_WORKER,
                        sorted.at(i) != i,
                        "Initial schedules must be submitted for all the "
                        "workers.(Invalid id set)");
                }
            }
        }
        for (auto wid_0b : change_ids) {
            Schedule s;
            s.read_from_stream(t_0b, T_MAX(), is);
            schedule_manager.set_new_schedule(t_0b, wid_0b,
                                              s);
        }
        worker_manager.for_each_worker([&](worker_id_t wid) {
            DEBUG("Penalty for worker %d:%.17g", wid,
                  schedule_manager.get_schedule_penalty(wid));
        });
    } else {
        if (t_0b == 0) {
            worker_manager.for_each_worker([&](worker_id_t wid) {
                Schedule s;
                s.set_empty_schedule(t_0b, T_MAX());
                schedule_manager.set_new_schedule(t_0b, wid, s);
            });
        }
    }
    worker_manager.command_for_all_workers(t_0b,
                                           is);
}
final_result_t World::interact(std::istream& is, std::ostream& os) {
    THROW_LOGIC_ERROR_IF(!is_loaded(), "Load world info before interaction");
    auto os_prec_old = os.precision(std::numeric_limits<double>::max_digits10);
    begin_json_log_output();
    ValueWriter wr(os);
    output_line_to_contestant(
        wr, TAG(ValueGroup::POSITIVE_COUNT, T_MAX()));
    output_graph_data_to_contestant(os);
    output_worker_initial_data_to_contestant(os);
    output_all_jobs_initial_data_to_contestant(os);
    if (world_type() == WorldType::B) {
        output_weather_initial_data_to_contestant(os);
        output_schedule_score_info_to_contestant(
            os);
        output_forecast_to_contestant_(os, 0);
    }
    if (world_type() == WorldType::B) {
        job_manager.accept_jobs(is);
    } else if (world_type() == WorldType::A) {
        job_manager.accept_all_jobs();
    } else {
        THROW_LOGIC_ERROR_IF(true, "Invalid world type");
    }
    begin_turn_json_log_output();
    for (discrete_time_t t_0b = 0; t_0b < T_MAX(); t_0b++) {
        update_turn(t_0b);
        if (world_type() == WorldType::B) {
            output_turn_data_to_contestant(t_0b, os);
        }
        input_turn_data_from_contestant(t_0b, is);
        job_manager.for_each_relevant_job([&](job_id_t id) {
            job_manager.jobs(id).finalize_task_done();
            if (job_manager.jobs(id).completed()) {
                score_manager.join_jobwise_score(id);
            }
        });
        INFO("Score(intermediate):%s", score_manager.score().str().c_str());
        output_turn_data_into_log(t_0b);
    }
    end_turn_json_log_output();
    float_score_t score = 0;
    double unfinished_penalty = 1.0;
    if (world_type() == WorldType::A) {
        score = score_manager.score();
    } else if (world_type() == WorldType::B) {
        double penalty_coeff = 1.0;
        job_manager.for_each_accepted_job([&](job_id_t aid) {
            const auto& job = job_manager.jobs(aid);
            if (!job.completed()) {
                INFO("Unfinished job:%d (Penalty:%.17g)", aid,
                     job.info.penalty_coeff);
                penalty_coeff *=
                    job.info.penalty_coeff;
            }
        });
        INFO("Total unfinished job penalty:%.17g", penalty_coeff);
        unfinished_penalty = penalty_coeff;
        penalty_coeff *= schedule_manager.calc_schedule_bonus_coefficient();
        score = score_manager.score() * penalty_coeff;
    } else {
        THROW_LOGIC_ERROR_IF(true, "Unsupported world type");
    }
    using std::floor;
    int64_t score_integer = static_cast<int64_t>(floor(score));
    output_line_to_contestant(
        wr, TAG(ValueGroup::SCORE, score_integer));
    end_json_log_output(std::to_string(score_integer), unfinished_penalty);
    os.precision(os_prec_old);
    INFO("Final score:%" PRId64, score_integer);
    return score_integer;
}
void World::log_mutable_info_only(discrete_time_t crt0b, bool e) const {
    this->LogInfoSwitcher::log_mutable_info_only(crt0b, e);
    job_manager.log_mutable_info_only(crt0b, e);
    weather_manager.log_mutable_info_only(crt0b, e);
    worker_manager.log_mutable_info_only(crt0b, e);
    schedule_manager.log_mutable_info_only(crt0b, e);
}
void to_json(json_ref j, const World& w) {
    if (w.log_mutable_only_) {
        j = json{
            {"job_manager", w.job_manager},
            {"score_manager", w.score_manager},
            {"weather_manager", w.weather_manager},
            {"worker_manager", w.worker_manager},
            {"schedule_manager", w.schedule_manager},
        };
    } else {
        j = json{
            {"T_max", w.T_MAX_},
            {"initialized", w.initialized_},
            {"loaded", w.loaded_},
            {"graph", w.graph},
            {"job_manager", w.job_manager},
            {"score_manager", w.score_manager},
            {"task_limit_info", w.task_limit_info},
            {"weather_manager", w.weather_manager},
            {"worker_manager", w.worker_manager},
            {"schedule_manager", w.schedule_manager}
        };
    }
}
