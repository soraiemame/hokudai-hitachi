#include "weather.hpp"
#include <algorithm>
#include <utility>
#include <vector>
#include "error_check.hpp"
#include "io.hpp"
#include "lib/Eigen/Core"
#include "lib/Eigen/Eigenvalues"
#include "lib/json.hpp"
void WeatherManager::RowMajorMatrix::throw_if_invalid() const {
    THROW_RUNTIME_ERROR_IF(n_row <= 0, "Invalid row size");
    THROW_RUNTIME_ERROR_IF(n_col <= 0, "Invalid col size");
    THROW_RUNTIME_ERROR_IF(n_row * n_col != data.size(),
                           "row*col != buffer size");
}
WeatherManager::RowMajorMatrix::row_major_matrix_raw_t::reference
WeatherManager::RowMajorMatrix::operator()(int row, int col) {
    throw_if_invalid();
    return data.at(col + n_col * row);
}
WeatherManager::RowMajorMatrix::row_major_matrix_raw_t::const_reference
WeatherManager::RowMajorMatrix::operator()(int row, int col) const {
    throw_if_invalid();
    return data.at(col + n_col * row);
}
void WeatherManager::RowMajorMatrix::set_size(int r, int c) {
    n_row = r;
    n_col = c;
    data.resize(n_row * n_col);
    throw_if_invalid();
}
void WeatherManager::RowMajorMatrix::set_size_and_fill(int r, int c,
                                                       double value) {
    set_size(r, c);
    std::fill(data.begin(), data.end(), value);
}
WeatherManager::RowMajorMatrix::row_major_matrix_raw_t::iterator
WeatherManager::RowMajorMatrix::row_begin(int r) {
    return data.begin() + n_col * r;
}
WeatherManager::RowMajorMatrix::row_major_matrix_raw_t::iterator
WeatherManager::RowMajorMatrix::row_end(int r) {
    return data.begin() + n_col * (r + 1);
}
WeatherManager::RowMajorMatrix::row_major_matrix_raw_t::const_iterator
WeatherManager::RowMajorMatrix::row_begin(int r) const {
    return data.begin() + n_col * r;
}
WeatherManager::RowMajorMatrix::row_major_matrix_raw_t::const_iterator
WeatherManager::RowMajorMatrix::row_end(int r) const {
    return data.begin() + n_col * (r + 1);
}
std::vector<double> WeatherManager::RowMajorMatrix::get_row(int r) const {
    return std::vector<double>(row_begin(r), row_end(r));
}
WeatherManager::RowMajorMatrix::RowMajorMatrix(const double* dptr, int nr,
                                               int nc)
    : n_row(nr), n_col(nc), data(0) {
    THROW_RUNTIME_ERROR_IF(n_row <= 0, "Invalid row size");
    THROW_RUNTIME_ERROR_IF(n_col <= 0, "Invalid col size");
    std::copy(dptr, dptr + n_row * n_col, std::back_inserter(data));
}
WeatherManager::RowMajorMatrix::RowMajorMatrix() = default;
void WeatherManager::RowMajorMatrix::generate_row_strings() {
    constexpr int STR_INT_MAX_LEN = 12;
    constexpr int STR_DOUBLE_MAX_LEN = 24;
    std::vector<char> buf(STR_DOUBLE_MAX_LEN +
                          (1 + STR_DOUBLE_MAX_LEN) * (n_col - 1) + 1 + 1);
    row_strings.resize(n_row);
    for (int i = 0; i < n_row; i++) {
        std::memset(&buf[0], 0, buf.size());
        size_t len = 0;
        auto endit = row_end(i);
        auto it = row_begin(i);
        len += sprintf(&buf[len], "%.17g", *it);
        ++it;
        for (; it != endit; ++it) {
            len += sprintf(&buf[len], " %.17g", *it);
        }
        len += sprintf(&buf[len], "\n");
        row_strings[i].assign(&buf[0], len);
    }
}
const std::string& WeatherManager::RowMajorMatrix::get_row_string(int r) const {
    return row_strings[r];
}
void WeatherManager::set_time_local_value(weather_value_t val) {
    time_local_value = val;
}
int WeatherManager::weather_division_length() const {
    return (weather_probability_.size() - 1) /
           current_division_weathers_.size();
}
int WeatherManager::weather_value_num() const {
    weather_trans_.throw_if_invalid();
    return weather_trans_.n_col;
}
void WeatherManager::
    output_transition_matrix_to_contestant_WITH_TRAILING_NEWLINE(
        std::ostream& os) const {
    weather_trans_.throw_if_invalid();
    int n = weather_value_num();
    THROW_LOGIC_ERROR_IF(n < 1, "weather_value_num must be >=1");
    ValueWriter wr(os);
    std::vector<double> probs(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            probs[j] = weather_trans_(i, j);
        }
        output_line_to_contestant_no_flush(
            wr, TAG(ValueGroup::CONTAINER,
                    StreamAdapter(
                        probs,
                        [&](ValueWriter& wr_inner, double p) {
                            wr_inner << TAG(ValueGroup::PROBABILITY, p);
                        },
                        true)));
    }
    os.flush();
}
using RowMajorEigenMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
std::vector<double>
WeatherManager::compute_stationary_dist_(const RowMajorMatrix& mat_) {
    {
        std::vector<bool> all_0(mat_.n_col), all_1(mat_.n_col);
        for (int c = 0; c < mat_.n_col; c++) {
            all_0[c] = true;
            all_1[c] = true;
            for (int r = 0; r < mat_.n_row; r++) {
                if (mat_(r, c) != 0.0) {
                    all_0[c] = false;
                }
                if (mat_(r, c) != 1.0) {
                    all_1[c] = false;
                }
            }
        }
        int zero_count = std::count(all_0.begin(), all_0.end(), true);
        int one_count = std::count(all_1.begin(), all_1.end(), true);
        if (zero_count == mat_.n_col - 1 && one_count == 1) {
            std::vector<double> ret;
            std::transform(all_1.begin(), all_1.end(), std::back_inserter(ret),
                           [](bool b) { return b ? 1.0 : 0.0; });
            return ret;
        }
    }
    Eigen::Map<const RowMajorEigenMatrixXd> mat_map(&mat_.data[0], mat_.n_row,
                                                    mat_.n_col);
    Eigen::EigenSolver<RowMajorEigenMatrixXd> es(mat_map.transpose(), true);
    Eigen::VectorXcd eigvals = es.eigenvalues();
    int max_i = -1;
    double max_abs_eigval = -1.0;
    for (int i = 0; i < eigvals.size(); i++) {
        double abs_eigval = std::abs(eigvals[i]);
        if (max_abs_eigval < abs_eigval) {
            max_i = i;
            max_abs_eigval = abs_eigval;
        }
    }
    std::complex<double> result_eigval = eigvals[max_i];
    THROW_LOGIC_ERROR_IF(result_eigval.real() < 0.99,
                         "eigval real part is too small :%g",
                         result_eigval.real());
    THROW_LOGIC_ERROR_IF(std::abs(result_eigval.imag()) > 0.01,
                         "eigval imag part is too large :%g",
                         result_eigval.imag());
    Eigen::VectorXcd sta = es.eigenvectors().col(max_i);
    std::vector<double> ret;
    double esum = 0.0;
    for (auto v : sta) {
        esum += v.real();
    }
    for (int i = 0; i < sta.size(); i++) {
        ret.push_back(sta[i].real() / esum);
    }
    return ret;
}
void to_json(json_ref j, const WeatherManager::RowMajorMatrix& m) {
    j = json{{"rows", m.n_row}, {"cols", m.n_col}, {"data", m.data}};
}
void WeatherManager::initialize_(
    std::vector<std::vector<double>> weather_transition_prob_mat,
    int division_size, discrete_time_t t_max, uint64_t seed) {
    THROW_RUNTIME_ERROR_IF(division_size <= 0,
                           "Weather division size must be positive");
    THROW_RUNTIME_ERROR_IF(t_max <= 0, "T_max must be positive");
    THROW_RUNTIME_ERROR_IF(t_max % division_size != 0,
                           "T_max must be divisible by division size");
    THROW_RUNTIME_ERROR_IF(weather_transition_prob_mat.size() == 0,
                           "Weather transition matrix cannot be empty");
    {
        for (const auto& v : weather_transition_prob_mat) {
            THROW_RUNTIME_ERROR_IF(v.size() !=
                                       weather_transition_prob_mat.size(),
                                   "Weather transition matrix must be square");
            for (auto p : v) {
                THROW_RUNTIME_ERROR_IF(p < 0.0,
                                       "probabilities must be non-negative");
                THROW_RUNTIME_ERROR_IF(std::isnan(p),
                                       "probabilities cannot be NaN");
            }
        }
    }
    engine_.seed(seed);
    seed_ = seed;
    for (auto& v : weather_transition_prob_mat) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        std::for_each(v.begin(), v.end(), [&](double& p) { p /= sum; });
    }
    weather_trans_.set_size(weather_transition_prob_mat.size(),
                            weather_transition_prob_mat.size());
    for (int r = 0; r < weather_transition_prob_mat.size(); r++) {
        size_t len = weather_transition_prob_mat.at(r).size();
        for (int c = 0; c < len; c++) {
            weather_trans_(r, c) = weather_transition_prob_mat[r][c];
        }
    }
    {
        current_division_weathers_.resize(t_max / division_size);
        std::vector<double> sta = compute_stationary_dist_(weather_trans_);
        std::discrete_distribution<int> dist(sta.begin(), sta.end());
        for (auto& v : current_division_weathers_) {
            v = dist(engine_);
        }
        time_local_value = current_division_weathers_[0];
    }
    {
        int nr = weather_trans_.n_row;
        int nc = weather_trans_.n_col;
        RowMajorEigenMatrixXd forecast =
            RowMajorEigenMatrixXd::Identity(nr, nc);
        Eigen::Map<const RowMajorEigenMatrixXd> wp(&weather_trans_.data[0], nr,
                                                   nc);
        weather_probability_.emplace_back(forecast.data(), nr, nc);
        weather_probability_.back().generate_row_strings();
        for (int i = 0; i < t_max; i++) {
            forecast = wp * forecast;
            for (int j = 0; j < forecast.rows(); j++) {
                double sum = 0.0;
                for (int k = 0; k < forecast.cols(); k++) {
                    sum += forecast(j, k);
                }
                for (int k = 0; k < forecast.cols(); k++) {
                    forecast(j, k) /= sum;
                }
            }
            weather_probability_.emplace_back(forecast.data(), nr, nc);
            weather_probability_.back().generate_row_strings();
        }
    }
}
std::vector<std::vector<double>>
WeatherManager::probability_forecast(discrete_time_t t) const {
    size_t division_count = current_division_weathers_.size();
    size_t total_time = weather_probability_.size() - 1;
    size_t value_num = weather_trans_.n_row;
    int division_size = total_time / division_count;
    std::vector<std::vector<double>> ret(total_time - t + 1,
                                         std::vector<double>(value_num));
    for (int i = t; i < total_time; i++) {
        int c = i / division_size;
        ret[i - t] =
            weather_probability_[i - t].get_row(current_division_weathers_[c]);
    }
    return ret;
}
std::vector<const std::string*>
WeatherManager::probability_forecast_string(discrete_time_t t) const {
    size_t division_count = current_division_weathers_.size();
    size_t total_time = weather_probability_.size() - 1;
    size_t value_num = weather_trans_.n_row;
    int division_size = total_time / division_count;
    std::vector<const std::string*> ret(total_time - t + 1);
    for (int i = t; i < total_time; i++) {
        int c = i / division_size;
        ret[i - t] = &(weather_probability_[i - t].get_row_string(
            current_division_weathers_[c]));
    }
    return ret;
}
void WeatherManager::update(discrete_time_t current_time) {
    int div_size =
        (weather_probability_.size() - 1) / current_division_weathers_.size();
    if (current_time % div_size == 0) {
        for (auto& w : current_division_weathers_) {
            std::discrete_distribution<int> dist(weather_trans_.row_begin(w),
                                                 weather_trans_.row_end(w));
            w = dist(engine_);
        }
        int division_value =
            current_division_weathers_[current_time /
                                       ((weather_probability_.size() - 1) /
                                        current_division_weathers_.size())];
        set_time_local_value(division_value);
    } else {
        std::discrete_distribution<int> dist(
            weather_trans_.row_begin(time_local_value),
            weather_trans_.row_end(time_local_value));
        time_local_value = dist(engine_);
    }
}
weather_value_t WeatherManager::get_weather_value(discrete_time_t t) const {
    return time_local_value;
}
void WeatherManager::read_from_stream(std::istream& is, discrete_time_t t_max) {
    ValueReader r(is);
    int value_num, division_size;
    uint64_t seed;
    readline_exact(
        r,
        TAG(ValueGroup::POSITIVE_COUNT, value_num),
        TAG(ValueGroup::POSITIVE_DURATION, division_size),
        TAG(ValueGroup::SEED, seed));
    THROW_RUNTIME_ERROR_IF(value_num <= 0, "Invalid weather value num");
    std::vector<std::vector<double>> wp(value_num,
                                        std::vector<double>(value_num));
    for (int i = 0; i < value_num; i++) {
        auto ssr = get_single_line_stream(r);
        int j = 0;
        read_n_vars_exact<double>(ssr, value_num, ValueGroup::PROBABILITY,
                                  [&](double p) {
                                      wp[i][j++] = p;
                                  });
    }
    initialize_(std::move(wp), division_size, t_max, seed);
}
void to_json(json_ref j, const WeatherManager& wm) {
    if (wm.log_mutable_only_) {
        j = json{
            {"division_weathers", wm.current_division_weathers_},
            {"time_local_weather", wm.time_local_value},
        };
    } else {
        j = json{
            {"trans_mat", wm.weather_trans_},
            {"division_weathers", wm.current_division_weathers_},
            {"time_local_weather", wm.time_local_value},
            {"seed", wm.seed_},
        };
    }
}
