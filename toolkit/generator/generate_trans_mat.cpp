#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "lib/Eigen/Core"
#include "lib/cmdline.h"
using Mat =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
Eigen::VectorXd calc_stationary_dist(Mat mat, double eps = 1e-12) {
    for (int i = 0; i < mat.rows(); i++) {
        mat.row(i) /= mat.row(i).sum();
    }
    for (int i__ = 0; i__ < 64; i__++) {
        Mat mat2 = mat * mat;
        for (int i = 0; i < mat.rows(); i++) {
            mat2.row(i) /= mat2.row(i).sum();
        }
        double diff = (mat2 - mat).norm();
        mat = std::move(mat2);
        if (diff < mat.rows() * eps) {
            break;
        }
    }
    return mat.row(0) / mat.row(0).sum();
}
Mat generate_trans_prob_mat(Eigen::VectorXd stationary_dist, int adj_range,
                            double sharpness, uint64_t seed,
                            bool centralize = true, double eps = 1e-12,
                            int loop_max = 1000000) {
    std::mt19937_64 engine(seed);
    stationary_dist /= stationary_dist.sum();
    int n = stationary_dist.size();
    Mat mat = Mat::Zero(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int d = std::abs(i - j);
            if (d > adj_range) {
                mat(i, j) = 0.0;
            } else {
                double p = std::exp(-sharpness * (d * d));
                p = p < std::numeric_limits<double>::min() ? 0.0 : p;
                mat(i, j) = p;
            }
        }
        mat.row(i) /= mat.row(i).sum();
    }
    double n1 = 1.0;
    double n2 = 1.0;
    int qual = 0;
    double qual_base = n * std::numeric_limits<double>::epsilon();
    Mat mat2;
    for (int r__ = 0; r__ < loop_max; r__++) {
        auto sta = calc_stationary_dist(mat, eps);
        n1 = (stationary_dist - sta).lpNorm<Eigen::Infinity>();
        double n1_qual = std::log(n1) / std::log(qual_base);
        int n1_qual_d = std::floor(n1_qual * 10);
        if (n1_qual_d > qual) {
            qual = n1_qual_d;
            std::cerr << "Convergence:" << n1 << std::endl;
        }
        Mat pert = n * n1 * Mat::Random(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                pert(i, j) *= std::max(eps, std::abs(mat(i, j)));
                if (std::abs(i - j) > adj_range) {
                    pert(i, j) = 0.0;
                }
            }
        }
        mat2 = mat + pert;
        for (int i = 0; i < n; i++) {
            double m = mat2.row(i).minCoeff();
            if (m < 0.0) {
                for (int j = 0; j < n; j++) {
                    if (std::abs(i - j) <= adj_range) {
                        mat2(i, j) -= m;
                    }
                }
            }
            mat2.row(i) /= mat2.row(i).sum();
        }
        if (centralize) {
            bool cent = true;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j && mat2(i, i) <= mat2(i, j)) {
                        cent = false;
                        goto NONCENT;
                    }
                }
            }
        NONCENT : {}
            if (!cent) {
                continue;
            }
        }
        auto sta2 = calc_stationary_dist(mat2, eps);
        n2 = (stationary_dist - sta2).lpNorm<Eigen::Infinity>();
        if (n1 > n2) {
            mat = mat2;
            if (n2 < n * eps) {
                std::cerr << "Done." << std::endl;
                std::cerr << "Probability matrix quality:"
                          << std::log(n2) / std::log(qual_base)
                          << " stationary dist:[";
                for (int i = 0; i < n; i++) {
                    std::cerr << sta2[i] << (i == n - 1 ? "" : ",");
                }
                std::cerr << "]" << std::endl;
                break;
            }
        }
    }
    return mat;
}
int main(int argc, char** argv) {
    std::cerr.precision(std::numeric_limits<double>::max_digits10);
    cmdline::parser p;
    p.add<std::string>(
        "dist", 'd',
        R"(定常分布(ダブルクォーテーションで囲まれたスペース区切り文字列として指定))",
        true);
    p.add<int>(
        "adj-range", 'r',
        "初期行列の成分(i,j)について、|i-j|>\"adj-range\"の時0.0にする。",
        false, 1);
    p.add<double>(
        "sharpness", 'q',
        "初期行列の成分(i,j)について、exp(-sharpness*|i-j|^2)で重み付けする。",
        false, 1.0);
    p.add<double>("eps", '\0', "収束判定基準値", false, 1e-12);
    p.add<int>("centralize", 'c', "対角成分が最大になるように 0:しない 1:する ",
               false, 1);
    p.add<uint64_t>("seed", 's', "シード値", false, 0);
    p.parse_check(argc, argv);
    std::vector<double> dist;
    std::stringstream ss;
    ss << p.get<std::string>("dist");
    while (true) {
        double p;
        ss >> p;
        if (!ss)
            break;
        dist.push_back(p);
    }
    Eigen::Map<Eigen::VectorXd> distv(dist.data(), dist.size());
    Mat mat = generate_trans_prob_mat(distv,
                                      p.get<int>("adj-range"),
                                      p.get<double>("sharpness"),
                                      p.get<uint64_t>("seed"),
                                      p.get<int>("centralize") == 1,
                                      p.get<double>("eps"));
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            std::cout << mat(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout.flush();
    return 0;
}
