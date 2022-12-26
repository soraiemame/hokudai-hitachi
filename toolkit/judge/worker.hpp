#ifndef HEADER_2HC2022_WORKER
#define HEADER_2HC2022_WORKER 
#include <set>
#include <string>
#include <vector>
#include "action.hpp"
#include "defines.hpp"
#include "graph.hpp"
#include "io.hpp"
#include "job_manager.hpp"
#include "json_fwd.hpp"
#include "schedule.hpp"
class WorkerManager;
struct WorkerInfo {
    vertex_index_t initial_position = INVALID_VERTEX_ID;
    std::set<job_type_t> processable_types;
    int max_task = -1;
    bool is_valid(std::string* msg = nullptr) const;
    void read_from_stream(std::istream& is_);
    bool can_process(job_type_t jt) const;
    friend void to_json(json_ref j, const WorkerInfo& wi);
};
class WorkerState {
    Position current_pos_;
 public:
    void move_towards(vertex_index_t target, const UndirectedGraph& graph);
    Position current_position() const;
    void initialize(const Position& p);
    bool is_valid(const UndirectedGraph& graph,
                  std::string* msg = nullptr) const;
    friend void to_json(json_ref j, const WorkerState& ws);
    friend Worker;
};
struct TaskExecutionRecord {
    discrete_time_t time = INVALID_TIME;
    job_id_t id = INVALID_JOB_ID;
    int amount = INVALID_COUNT;
    bool operator<(const TaskExecutionRecord& e) const;
    friend void to_json(json_ref j, const TaskExecutionRecord& ter);
};
class Worker : public LogInfoSwitcher {
    worker_id_t id_ = INVALID_WORKER_ID;
    WorkerManager* man_ = nullptr;
    std::set<TaskExecutionRecord> task_history_;
    std::vector<std::string> raw_action_history_;
    bool always_valid_ = false;
 public:
    void log_raw_action_string(std::string a);
    worker_id_t id() const;
    const UndirectedGraph* graph = nullptr;
    WorkerInfo info;
    WorkerState state;
    void do_action(discrete_time_t current_time, const action::Action& a);
    void do_action(discrete_time_t current_time, const action::Stay& s);
    void do_action(discrete_time_t current_time, const action::Move& m);
    void do_action(discrete_time_t current_time, const action::Execute& e);
    void move_FORCE(const Position& p);
    void set_graph(const UndirectedGraph* gp);
    void set_worker_manager(WorkerManager* wmp);
    Position current_position() const;
    void read_worker_info(std::istream& is);
    void initialize();
    bool is_valid(std::string* msg = nullptr) const;
    bool can_process(job_type_t jt) const;
    const std::set<TaskExecutionRecord>& task_history() const;
    void set_always_valid_DO_NOT_USE_(bool valid);
    friend void to_json(json_ref j, const Worker& w);
    friend WorkerManager;
};
#endif
