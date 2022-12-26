#ifndef HEADER_2HC2022_WEATHER
#define HEADER_2HC2022_WEATHER 
#include <random>
#include <vector>
#include "defines.hpp"
#include "io.hpp"
#include "json_fwd.hpp"
using weather_value_t = int;
class WeatherManager : public LogInfoSwitcher {
 public:
    struct RowMajorMatrix {
        using row_major_matrix_raw_t = std::vector<double>;
        int n_row = -1;
        int n_col = -1;
        row_major_matrix_raw_t data;
        std::vector<std::string> row_strings;
        void throw_if_invalid() const;
        row_major_matrix_raw_t::reference operator()(int row, int col);
        row_major_matrix_raw_t::const_reference operator()(int row,
                                                           int col) const;
        void set_size(int r, int c);
        void set_size_and_fill(int r, int c, double value);
        row_major_matrix_raw_t::iterator row_begin(int r);
        row_major_matrix_raw_t::iterator row_end(int r);
        row_major_matrix_raw_t::const_iterator row_begin(int r) const;
        row_major_matrix_raw_t::const_iterator row_end(int r) const;
        std::vector<double> get_row(int r) const;
        RowMajorMatrix(const double* dptr, int nr, int nc);
        RowMajorMatrix();
        void generate_row_strings();
        const std::string& get_row_string(int r) const;
        friend void to_json(json_ref j, const RowMajorMatrix& rmm);
    };
 private:
    std::vector<RowMajorMatrix> weather_probability_;
    std::vector<weather_value_t> current_division_weathers_;
    RowMajorMatrix weather_trans_;
    std::mt19937_64 engine_;
    weather_value_t time_local_value = -1;
    uint64_t seed_ = 0;
    void
    initialize_(std::vector<std::vector<double>> weather_transition_prob_mat,
                int division_size, discrete_time_t t_max, uint64_t seed);
    static std::vector<double>
    compute_stationary_dist_(const RowMajorMatrix& mat_);
 public:
    void set_time_local_value(weather_value_t val);
    void read_from_stream(std::istream& is, discrete_time_t t_max);
    std::vector<std::vector<double>>
    probability_forecast(discrete_time_t t) const;
    std::vector<const std::string*>
    probability_forecast_string(discrete_time_t t) const;
    void update(discrete_time_t current_time);
    weather_value_t get_weather_value(discrete_time_t current_time) const;
    int weather_division_length() const;
    int weather_value_num() const;
    void output_transition_matrix_to_contestant_WITH_TRAILING_NEWLINE(
        std::ostream& os) const;
    friend void to_json(json_ref j, const WeatherManager& wm);
};
#endif
