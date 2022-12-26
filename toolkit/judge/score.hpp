#ifndef HEADER_2HC2022_SCORE
#define HEADER_2HC2022_SCORE 
#include <unordered_map>
#include "defines.hpp"
#include "json_fwd.hpp"
class ScoreManager {
    float_score_t score_ = INVALID_FLOAT_SCORE;
    std::unordered_map<job_id_t, float_score_t> score_jobwise_;
 public:
    void initialize();
    float_score_t score() const;
    void add_score_jobwise(job_id_t id, float_score_t s);
    void join_jobwise_score(job_id_t id);
    void set_score(float_score_t s);
    friend void to_json(json_ref j, const ScoreManager& sm);
};
#endif
