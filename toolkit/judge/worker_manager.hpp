#ifndef HEADER_2HC2022_WORKER_MANAGER
#define HEADER_2HC2022_WORKER_MANAGER 
#include <atomic>
#include <map>
#include <string>
#include "defines.hpp"
#include "graph.hpp"
#include "io.hpp"
#include "job_manager.hpp"
#include "json_fwd.hpp"
#include "schedule_manager.hpp"
#include "worker.hpp"
class WorkerManager : public LogInfoSwitcher {
    std::map<worker_id_t, Worker> workers_;
    std::atomic<int> id_gen_{0};
    JobManager* job_man_ = nullptr;
    ScheduleManager* sch_man_ = nullptr;
    const UndirectedGraph* graph_ = nullptr;
 public:
    void log_mutable_info_only(discrete_time_t crt0b, bool e) const override;
    void command_for_all_workers(discrete_time_t current_time,
                                 std::istream& is);
    void execute_job(discrete_time_t current_time, const Worker& w, job_id_t id,
                     int amount);
    bool is_valid(std::string* msg = nullptr) const;
    void set_schedule_manager(ScheduleManager* sm);
    void set_job_manager(JobManager* jm);
    void set_graph(const UndirectedGraph* gp);
    void read_workers(std::istream& is);
    int worker_num() const;
    template <class Fn> void for_each_worker(const Fn& f) const {
        for (int i = 0; i < worker_num(); i++) {
            f(i);
        }
    }
    bool exists(worker_id_t wid) const;
    const Worker& workers(int id) const;
    friend void to_json(json_ref j, const WorkerManager& wm);
};
#endif
