#ifndef HEADER_2HC2022_DEFINES
#define HEADER_2HC2022_DEFINES 
#include <cmath>
#include <cstdint>
#include <filesystem>
// #include <experimental/filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include "boost/multiprecision/cpp_bin_float.hpp"
template <class T,
          class U = typename std::remove_cv<
              typename std::remove_pointer<typename std::remove_reference<
                  typename std::remove_extent<T>::type>::type>::type>::type>
struct remove_all : remove_all<U> {};
template <class T> struct remove_all<T, T> {
    typedef T type;
};
template <class T> using remove_all_t = typename remove_all<T>::type;
using final_result_t = int64_t;
using distance_t = int64_t;
using vertex_index_t = int;
using discrete_time_t = int;
using float_score_t = boost::multiprecision::cpp_bin_float_quad;
using score_t = float_score_t;
using simple_score_t = int64_t;
using job_id_t = int;
using worker_id_t = int;
using job_type_t = int;
static constexpr job_type_t INVALID_JOB_TYPE = -1;
static constexpr distance_t MAX_DISTANCE =
    std::numeric_limits<distance_t>::max() >> 1;
static constexpr distance_t INVALID_DISTANCE = -1LL;
static constexpr vertex_index_t INVALID_VERTEX_ID = -1;
static constexpr discrete_time_t INVALID_TIME =
    std::numeric_limits<discrete_time_t>::lowest();
static constexpr int INVALID_COUNT = -1;
static constexpr double INVALID_REAL =
    std::numeric_limits<double>::signaling_NaN();
static constexpr job_id_t INVALID_JOB_ID = -1;
static constexpr worker_id_t INVALID_WORKER_ID = -1;
inline float_score_t INVALID_FLOAT_SCORE_() noexcept {
    return std::numeric_limits<float_score_t>::quiet_NaN();
}
#define INVALID_FLOAT_SCORE (INVALID_FLOAT_SCORE_())
#define INVALID_SCORE (INVALID_FLOAT_SCORE)
#define CONTESTANT_VALUE_DELIMITER " "
static constexpr const char* TOKEN_DELIMITER_ONLY_FOR_TESTING_ = " ";
template <class T> std::string stream_to_string(const T& v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}
template <class T> auto to_1b(const T& zero_based_value) {
    return zero_based_value + 1;
}
template <class T> auto to_0b(const T& one_based_value) {
    return one_based_value - 1;
}
enum class WorldType : int { INVALID, A, B };
inline WorldType to_world_type(std::string str) {
    if (str == "A") {
        return WorldType::A;
    } else if (str == "B") {
        return WorldType::B;
    }
    return WorldType::INVALID;
}
template <class... Args> std::string format_str(Args&&... args) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
    int len = std::snprintf(nullptr, 0, args...);
    std::vector<char> buf(len + 1);
    std::snprintf(buf.data(), buf.size(),
                  std::forward<Args>(args)...);
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
    return std::string(buf.begin(), buf.end());
}
#define BASENAME_ (std::filesystem::path{__FILE__}.filename().c_str())
#define BASENAME_LINE_STR_ \
    (format_str("[%s:%d (%s)]", BASENAME_, __LINE__, __func__))
#define BASENAME_LINE_C_STR_ (BASENAME_LINE_STR_.c_str())
static constexpr final_result_t WA_SCORE_BASE = -100000000;
enum class WrongAnswerType : final_result_t {
    INVALID = 0,
    UNKNOWN_ERROR = 1,
    OUTPUT_ERROR = 2,
    GENERIC_WA,
    NEGATIVE_WORKER_NUM_FOR_SCHEDULES,
    TOO_LARGE_WORKER_NUM_FOR_SCHEDULES,
    N_CHANGE_NEQ_ACTUAL_WORKER_ID_NUM,
    WORKER_DOES_NOT_EXIST_FOR_SCHEDULE,
    DUPS_IN_SCHEDULE_WORKERS,
    SCHEDULE_JOB_NUM_MISMATCH,
    INVALID_ACTION_NAME,
    MOVE_VERTEX_DOES_NOT_EXIST,
    DO_NOT_SPECIFY_NONSENSE_MOVE_ACTION,
    FORCE_SCHEDULE_AT_TURN_1_NUM_WORKER,
    FORCE_SCHEDULE_AT_TURN_1_ID_WORKER,
    INVALID_MOVE_DESTINATION,
    INVALID_STAY_STRING,
    INVALID_MOVE_STRING,
    INVALID_EXECUTE_STRING,
    EXECUTE_JOB_ID_DOES_NOT_EXIST,
    EXECUTE_JOB_ID_IS_NOT_SELECTED,
    INVALID_EXECUTE_AMOUNT,
    NO_JOB_EXISTS_AT_CURRENT_POS,
    WORKER_CANNOT_PROCESS_THIS_TYPE,
    NONPOSITIVE_REWARD_VALUE,
    EXCEEDS_TASK_EXECUTION_LIMIT,
    EXCEEDS_TASK_AMOUNT_REST,
    DEPENDENCIES_NOT_DONE,
    SELECTED_NUM_IS_NEGATIVE,
    SELECTED_NUM_NEQ_ACTUAL_SELECTED_ID_NUM,
    DUPS_IN_SELECTED_JOBS,
    SELECTED_JOB_DOES_NOT_EXIST,
    MUST_SELECT_ALL_MANDATORY_JOBS,
    MUST_SELECT_ALL_DEPENDENCIES,
};
inline final_result_t wa_type_to_result(WrongAnswerType wa_type) {
    return WA_SCORE_BASE - static_cast<final_result_t>(wa_type);
}
#endif
