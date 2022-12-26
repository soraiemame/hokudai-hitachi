#ifndef HEADER_2HC2022_PIECEWISE_FUNCTION
#define HEADER_2HC2022_PIECEWISE_FUNCTION 
#include <string>
#include <utility>
#include <vector>
#include "defines.hpp"
#include "json_fwd.hpp"
using point_sequence_raw =
    std::vector<std::pair<discrete_time_t, simple_score_t>>;
class sorted_point_sequence {
    point_sequence_raw sq_;
    bool validated_ = false;
    void invalidate_();
 public:
    const sorted_point_sequence& as_const() const;
    bool is_valid(std::string* msg = nullptr) const;
    using size_type = point_sequence_raw::size_type;
    using value_type = point_sequence_raw::value_type;
    bool is_sorted() const;
    bool sorted_adj_eq() const;
    sorted_point_sequence();
    explicit sorted_point_sequence(const point_sequence_raw& p_);
    sorted_point_sequence(
        const std::initializer_list<std::pair<discrete_time_t, simple_score_t>>&
            ps_);
    void validate();
    template <typename... Args> auto& emplace_back(Args&&... args) {
        invalidate_();
        return sq_.emplace_back(std::forward<Args>(args)...);
    }
    template <typename... Args> void push_back(Args&&... args) {
        invalidate_();
        sq_.push_back(std::forward<Args>(args)...);
    }
    auto& operator[](size_type i) {
        invalidate_();
        return sq_[i];
    }
    auto& at(size_type i) {
        invalidate_();
        return sq_.at(i);
    }
    const auto& operator[](size_type i) const {
        return sq_[i];
    }
    const auto& at(size_type i) const {
        return sq_.at(i);
    }
    auto begin() {
        invalidate_();
        return sq_.begin();
    }
    auto begin() const {
        return sq_.begin();
    }
    auto end() {
        invalidate_();
        return sq_.end();
    }
    auto end() const {
        return sq_.end();
    }
    auto size() const {
        return sq_.size();
    }
    const auto& back() const {
        return sq_.back();
    }
    auto& back() {
        invalidate_();
        return sq_.back();
    }
    auto empty() const {
        return sq_.empty();
    }
    void resize(size_type s) {
        invalidate_();
        sq_.resize(s);
    }
    friend void to_json(json_ref j, const sorted_point_sequence& seq);
};
float_score_t
piecewise_constant_function(const sorted_point_sequence& sorted_points,
                            discrete_time_t t);
float_score_t
piecewise_linear_function(const sorted_point_sequence& sorted_points,
                          discrete_time_t t);
std::ostream& operator<<(std::ostream& os, const sorted_point_sequence& ps);
#endif
