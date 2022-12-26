#include "piecewise_function.hpp"
#include <algorithm>
#include <string>
#include "defines.hpp"
#include "error_check.hpp"
#include "lib/json.hpp"
void to_json(json_ref j, const sorted_point_sequence& seq) {
    j = json(seq.sq_);
}
void sorted_point_sequence::invalidate_() {
    validated_ = false;
}
const sorted_point_sequence& sorted_point_sequence::as_const() const {
    return std::as_const(*this);
}
bool sorted_point_sequence::is_valid(std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    VALIDITY_CHECK_WITH_MSG(!validated_, valid, msg,
                            "This sorted point sequence is not validated yet. "
                            "Validate or fix the sequence data.");
    return valid;
}
bool sorted_point_sequence::is_sorted() const {
    for (size_type i = 0; i < sq_.size() - 1; i++) {
        if (sq_.at(i).first > sq_.at(i + 1).first) {
            return false;
        }
    }
    return true;
}
bool sorted_point_sequence::sorted_adj_eq() const {
    for (size_type i = 0; i < sq_.size() - 1; i++) {
        if (sq_.at(i).first == sq_.at(i + 1).first) {
            return true;
        }
    }
    return false;
}
sorted_point_sequence::sorted_point_sequence() = default;
sorted_point_sequence::sorted_point_sequence(const point_sequence_raw& p_)
    : sq_(p_) {
    validate();
}
sorted_point_sequence::sorted_point_sequence(
    const std::initializer_list<std::pair<discrete_time_t, simple_score_t>>&
        ps_)
    : sq_(ps_) {
    validate();
}
void sorted_point_sequence::validate() {
    if (!is_sorted()) {
        std::sort(sq_.begin(), sq_.end());
    }
    THROW_RUNTIME_ERROR_IF(sorted_adj_eq(),
                           "This point sequence has duplicated elements.");
    validated_ = true;
}
float_score_t
piecewise_constant_function(const sorted_point_sequence& sorted_points,
                            discrete_time_t t) {
    size_t n = sorted_points.size();
    THROW_LOGIC_ERROR_IF(n == 0, "point sequence is empty.");
    {
        std::string msg;
        THROW_RUNTIME_ERROR_IF(
            !sorted_points.is_valid(&msg),
            "point sequence is invalid for the following reason(s):%s",
            msg.c_str());
    }
    auto it = std::upper_bound(
        sorted_points.begin(), sorted_points.end(), t,
        [](discrete_time_t t_, std::pair<discrete_time_t, simple_score_t> p) {
            return t_ < p.first;
        });
    if (it == sorted_points.begin()) {
        return sorted_points[0].second;
    }
    return (--it)->second;
}
float_score_t
piecewise_linear_function(const sorted_point_sequence& sorted_points,
                          discrete_time_t t) {
    size_t n = sorted_points.size();
    THROW_LOGIC_ERROR_IF(n == 0, "point sequence is empty.");
    {
        std::string msg;
        THROW_RUNTIME_ERROR_IF(
            !sorted_points.is_valid(&msg),
            "point sequence is invalid for the following reason(s):%s",
            msg.c_str());
    }
    auto it = std::upper_bound(
        sorted_points.begin(), sorted_points.end(), t,
        [](discrete_time_t t_, std::pair<discrete_time_t, simple_score_t> p) {
            return t_ < p.first;
        });
    if (it == sorted_points.begin()) {
        return sorted_points[0].second;
    }
    if (it == sorted_points.end()) {
        return sorted_points.back().second;
    }
    auto [t2, v2] = *it;
    auto [t1, v1] = *--it;
    return (float_score_t{v2} - float_score_t{v1}) *
               (float_score_t{t} - float_score_t{t1}) /
               (float_score_t{t2} - float_score_t{t1}) +
           float_score_t{v1};
}
std::ostream& operator<<(std::ostream& os, const sorted_point_sequence& ps) {
    json j = ps;
    os << j;
    return os;
}
