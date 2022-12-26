#include "score.hpp"
#include <string>
#include "lib/json.hpp"
void ScoreManager::initialize() {
    score_ = static_cast<score_t>(0);
    score_jobwise_.clear();
}
float_score_t ScoreManager::score() const {
    return score_;
}
void ScoreManager::add_score_jobwise(job_id_t id, float_score_t s) {
    if (score_jobwise_[id] == INVALID_FLOAT_SCORE) {
        throw std::logic_error("Jobwise Score is invalid");
    }
    score_jobwise_[id] += s;
}
void ScoreManager::join_jobwise_score(job_id_t id) {
    if (score_jobwise_[id] == INVALID_FLOAT_SCORE) {
        throw std::logic_error("Jobwise Score is invalid");
    }
    if (score_ == INVALID_FLOAT_SCORE) {
        throw std::logic_error("Total score is invalid");
    }
    score_ += score_jobwise_[id];
    score_jobwise_.erase(id);
}
void ScoreManager::set_score(float_score_t s) {
    if (score_ == INVALID_FLOAT_SCORE) {
        throw std::logic_error("Score is invalid(set_score)");
    }
    if (s == INVALID_FLOAT_SCORE) {
        throw std::logic_error("Tried to set invalid score");
    }
    score_ = s;
}
void to_json(json_ref j, const ScoreManager& sm) {
    std::unordered_map<job_id_t, std::string> score_jobwise_str;
    float_score_t sum = 0.0;
    for (const auto& kv : sm.score_jobwise_) {
        score_jobwise_str[kv.first] = kv.second.str();
        sum += kv.second;
    }
    j = json{{"score", sm.score().str()},
             {"score_jobwise", score_jobwise_str},
             {"score_jobwise_sum", sum.str()}};
}
