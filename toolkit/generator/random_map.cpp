#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/pending/disjoint_sets.hpp>
double calc_mid(double a, double b) {
    return std::isnormal(a) && std::isnormal(b) ? a / 2 + b / 2 : (a + b) / 2;
}
struct Point {
    double x, y;
    auto to_pair() const {
        return std::pair<double, double>(x, y);
    }
    bool operator==(const Point& p) const {
        return x == p.x && y == p.y;
    }
    double dist(const Point& p) const {
        return std::hypot(x - p.x, y - p.y);
    }
};
struct HashPoint {
    size_t operator()(const Point& a) const {
        size_t seed = 0;
        seed ^= reinterpret_cast<const size_t&>(a.x) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        seed ^= reinterpret_cast<const size_t&>(a.y) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        return seed;
    }
};
struct HashPointPair {
    size_t operator()(const std::pair<Point, Point>& p) const {
        auto [a, b] = p;
        size_t seed = 0;
        seed ^= reinterpret_cast<const size_t&>(a.x) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        seed ^= reinterpret_cast<const size_t&>(a.y) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        seed ^= reinterpret_cast<const size_t&>(b.x) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        seed ^= reinterpret_cast<const size_t&>(b.y) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        return seed;
    }
};
std::pair<Point, Point> make_point_pair(Point a, Point b) {
    if (a.to_pair() < b.to_pair()) {
        return {a, b};
    } else {
        return {b, a};
    }
}
struct QuadTree {
    static std::unordered_map<int, int> node_to_depth;
    static std::atomic<int> id_;
    static int gen_global_id() {
        return id_++;
    }
    static int max_depth;
    static double global_max_x;
    static double global_max_y;
    using EdgeSet = std::unordered_map<
        int, std::unordered_set<std::pair<Point, Point>, HashPointPair>>;
    double min_x, min_y;
    double max_x, max_y;
    int depth;
    int id;
    std::vector<QuadTree*> children;
    QuadTree(double min_x_, double min_y_, double max_x_, double max_y_,
             int depth_)
        : min_x(min_x_), min_y(min_y_), max_x(max_x_), max_y(max_y_),
          depth(depth_), id(gen_global_id()) {
        node_to_depth[id] = depth;
    }
    void add_square() {
        if (depth == max_depth) {
            return;
        }
        double mid_x = calc_mid(min_x, max_x);
        double mid_y = calc_mid(min_y, max_y);
        children.push_back(new QuadTree(min_x, min_y, mid_x, mid_y, depth + 1));
        children.push_back(new QuadTree(min_x, mid_y, mid_x, max_y, depth + 1));
        children.push_back(new QuadTree(mid_x, min_y, max_x, mid_y, depth + 1));
        children.push_back(new QuadTree(mid_x, mid_y, max_x, max_y, depth + 1));
    }
    void add_square_at(int nodeid) {
        std::queue<QuadTree*> q;
        q.push(this);
        std::unordered_set<QuadTree*> seen;
        while (!q.empty()) {
            QuadTree* node = q.front();
            q.pop();
            if (nodeid == node->id && node->children.empty()) {
                node->add_square();
            } else {
                if (seen.find(node) == seen.end()) {
                    seen.insert(node);
                }
                for (QuadTree* child : node->children) {
                    if (seen.find(child) == seen.end()) {
                        q.push(child);
                    }
                }
            }
        }
    }
    void collect_edges(EdgeSet* parent) {
        Point orig = {min_x, min_y};
        Point dest_y = {min_x, max_y};
        Point dest_x = {max_x, min_y};
        if (parent->find(depth) == parent->end()) {
            parent->operator[](depth) = {};
        }
        parent->at(depth).insert(make_point_pair(orig, dest_y));
        parent->at(depth).insert(make_point_pair(orig, dest_x));
        parent->at(depth).insert(
            make_point_pair({max_x, min_y}, {max_x, max_y}));
        parent->at(depth).insert(
            make_point_pair({min_x, max_y}, {max_x, max_y}));
        for (QuadTree* q : children) {
            q->collect_edges(parent);
        }
        if (depth == 0) {
            int max_d = std::max_element(parent->begin(), parent->end(),
                                         [&](const auto& a, const auto& b) {
                                             return a.first < b.first;
                                         })
                            ->first;
            for (int d = 0; d < max_d; d++) {
                for (auto it = parent->at(d).begin();
                     it != parent->at(d).end();) {
                    Point mid = {calc_mid(it->first.x, it->second.x),
                                 calc_mid(it->first.y, it->second.y)};
                    auto pair1 = make_point_pair(it->first, mid);
                    auto pair2 = make_point_pair(it->second, mid);
                    bool removed = false;
                    for (auto cd : parent->at(d + 1)) {
                        if (pair1 == cd || pair2 == cd) {
                            it = parent->at(d).erase(it);
                            removed = true;
                            break;
                        }
                    }
                    if (!removed) {
                        ++it;
                    }
                }
            }
        }
    }
    ~QuadTree() {
        for (QuadTree* q : children) {
            delete q;
        }
    }
};
std::atomic<int> QuadTree::id_{0};
std::unordered_map<int, int> QuadTree::node_to_depth;
int QuadTree::max_depth = -1;
double QuadTree::global_max_x;
double QuadTree::global_max_y;
std::atomic<int> point_id_gen{1};
std::unordered_map<Point, int, HashPoint> point_id;
int NDIV = 128;
double terr_size = 1024.0;
int terrain_apex_max;
int terrain_apex_min;
int main(int argc, char** argv) {
    if (argc <= 8) {
        std::cerr
            << "Usage:map_size max_node_count seed max_depth node_file_output "
               "edge_file_output area_ratio terrain_peak_num"
            << std::endl;
        std::exit(1);
    }
    double size = std::stod(argv[1]);
    QuadTree::global_max_x = size;
    QuadTree::global_max_y = size;
    int nodelim = std::stoi(argv[2]);
    int seed = std::stoi(argv[3]);
    double cut_area = std::stod(argv[7]);
    double cut_height = NAN;
    int terrain_a_n = std::stoi(argv[8]);
    terrain_apex_max = terrain_apex_min = terrain_a_n;
    QuadTree::max_depth = std::stoi(argv[4]);
    nodelim =
        std::min(((int)std::pow(4, QuadTree::max_depth + 1) - 1) / 3, nodelim);
    QuadTree qt{0.0, 0.0, size, size, 0};
    qt.add_square();
    std::mt19937 engine(seed);
    while (true) {
        int node = std::uniform_int_distribution<int>(
            0, static_cast<int>(QuadTree::id_) - 1)(engine);
        if (QuadTree::node_to_depth.size() >= nodelim) {
            break;
        }
        qt.add_square_at(node);
    }
    std::cerr << "Nodes:" << QuadTree::node_to_depth.size() << std::endl;
    QuadTree::EdgeSet edges;
    qt.collect_edges(&edges);
    std::vector<double> terrain(NDIV * NDIV, 0.0);
    {
        std::mt19937 terrain_rng(seed + 1);
        double dx = terr_size / NDIV;
        double dy = terr_size / NDIV;
        double dt = 0.9 * (0.5 * dx * dx * dy * dy / ((dx * dx) + (dy * dy)));
        std::cerr << dt << std::endl;
        std::vector<double> terrain2(NDIV * NDIV, 0.0), srcs(NDIV * NDIV, 0.0),
            sinks(NDIV * NDIV, 0.0);
        std::uniform_int_distribution<> dist_num_pts(terrain_apex_min,
                                                     terrain_apex_max);
        int num_pts = dist_num_pts(terrain_rng);
        std::uniform_real_distribution<double> dist_pt(0.0, 1.0);
#define TERRAIN(ix,iy) (terrain[NDIV * (iy) + (ix)])
#define TERRAIN2(ix,iy) (terrain2[NDIV * (iy) + (ix)])
#define SRCS(ix,iy) (srcs[NDIV * (iy) + (ix)])
#define SINKS(ix,iy) (sinks[NDIV * (iy) + (ix)])
        for (int i = 0; i < num_pts; i++) {
            double sx = dist_pt(terrain_rng);
            double sy = dist_pt(terrain_rng);
            int ix = std::min(static_cast<int>(sx * NDIV), NDIV - 1);
            int iy = std::min(static_cast<int>(sy * NDIV), NDIV - 1);
            SRCS(ix, iy) = 1.0 / (dx * dy);
        }
        for (int i = 0; i < num_pts; i++) {
            double sx = dist_pt(terrain_rng);
            double sy = dist_pt(terrain_rng);
            int ix = std::min(static_cast<int>(sx * NDIV), NDIV - 1);
            int iy = std::min(static_cast<int>(sy * NDIV), NDIV - 1);
            SINKS(ix, iy) = -1.0 / (dx * dy);
        }
        for (int it = 0; it * dt < 100000.0; it++) {
            for (int iy = 0; iy < NDIV; iy++) {
                for (int ix = 0; ix < NDIV; ix++) {
                    double next_x = ix < NDIV - 1 ? TERRAIN(ix + 1, iy)
                                                  : TERRAIN(ix - 1, iy);
                    double prev_x =
                        ix > 0 ? TERRAIN(ix - 1, iy) : TERRAIN(ix + 1, iy);
                    double next_y = iy < NDIV - 1 ? TERRAIN(ix, iy + 1)
                                                  : TERRAIN(ix, iy - 1);
                    double prev_y =
                        iy > 0 ? TERRAIN(ix, iy - 1) : TERRAIN(ix, iy + 1);
                    TERRAIN2(ix, iy) =
                        TERRAIN(ix, iy) +
                        dt * ((next_x + prev_x - 2.0 * TERRAIN(ix, iy)) /
                                  (dx * dx) +
                              (next_y + prev_y - 2.0 * TERRAIN(ix, iy)) /
                                  (dy * dy) +
                              SRCS(ix, iy) + TERRAIN(ix, iy) * SINKS(ix, iy));
                }
            }
            terrain = terrain2;
        }
        double max_height = *std::max_element(terrain.begin(), terrain.end());
        std::ofstream dbg_terrain_ofs("debug_terrain.txt");
        for (int iy = 0; iy < NDIV; iy++) {
            for (int ix = 0; ix < NDIV; ix++) {
                TERRAIN(ix, iy) /= max_height;
                dbg_terrain_ofs << ix * dx << " " << iy * dy << " "
                                << TERRAIN(ix, iy) << std::endl;
            }
            dbg_terrain_ofs << std::endl;
        }
    }
    std::ofstream edgeofs(argv[6]);
    std::ofstream dbgmapofs("debug_map.txt");
    boost::disjoint_sets_with_storage<> d(nodelim * 4);
    std::unordered_map<int, Point> reverse_point_map;
    std::unordered_map<int, std::unordered_set<int>> edgeconns;
    cut_height = 1.0;
    double last_cut_height = 0.0;
    for (int count = 0; count < 1024; count++) {
        double curr = (cut_height + last_cut_height) * 0.5;
        double area = 0.0;
        double dx = 1.0 / NDIV;
        double dy = 1.0 / NDIV;
        for (int iy = 0; iy < NDIV - 1; iy++) {
            for (int ix = 0; ix < NDIV - 1; ix++) {
                double height =
                    0.25 * (TERRAIN(ix, iy) + TERRAIN(ix + 1, iy) +
                            TERRAIN(ix, iy + 1) + TERRAIN(ix + 1, iy + 1));
                if (height > curr) {
                    area += dx * dy;
                }
            }
        }
        std::cerr << cut_area << " " << area << " " << dx * dy << std::endl;
        if (std::abs(cut_area - area) <= dx * dy) {
            cut_height = curr;
            break;
        }
        if (cut_area < area) {
            last_cut_height = curr;
        }
        if (cut_area > area) {
            cut_height = curr;
        }
    }
    std::cerr << "cut height:" << cut_height << std::endl;
    for (auto& kd : edges) {
        for (auto& p : kd.second) {
            auto [ox, oy] = p.first;
            auto [dx, dy] = p.second;
            int iox = NDIV * ox / size;
            iox = std::min(iox, NDIV - 1);
            int ioy = NDIV * oy / size;
            ioy = std::min(ioy, NDIV - 1);
            int idx = NDIV * dx / size;
            idx = std::min(idx, NDIV - 1);
            int idy = NDIV * dy / size;
            idy = std::min(idy, NDIV - 1);
            if (TERRAIN(iox, ioy) < cut_height &&
                TERRAIN(idx, idy) < cut_height) {
                continue;
            }
            if (point_id.find(p.first) == point_id.end()) {
                point_id[p.first] = point_id_gen++;
                reverse_point_map[point_id[p.first]] = p.first;
                d.make_set(point_id[p.first]);
            }
            if (point_id.find(p.second) == point_id.end()) {
                point_id[p.second] = point_id_gen++;
                reverse_point_map[point_id[p.second]] = p.second;
                d.make_set(point_id[p.second]);
            }
            d.union_set(point_id[p.first], point_id[p.second]);
            int v1 = std::min(point_id[p.first], point_id[p.second]);
            int v2 = std::max(point_id[p.first], point_id[p.second]);
            edgeconns[v1].insert(v2);
        }
    }
    std::unordered_map<int, std::vector<int>> conns;
    for (int i = 1; i < static_cast<int>(point_id_gen); i++) {
        int root = d.find_set(i);
        conns[root].push_back(i);
    }
    std::cerr << "components:" << conns.size() << std::endl;
    const std::vector<int>& largest =
        std::max_element(
            conns.begin(), conns.end(),
            [](const std::unordered_map<int, std::vector<int>>::value_type& a,
               const std::unordered_map<int, std::vector<int>>::value_type& b) {
                return a.second.size() < b.second.size();
            })
            ->second;
    std::unordered_map<int, int> reverse_largest;
    for (int i = 0; i < largest.size(); i++) {
        reverse_largest[largest[i]] = i + 1;
    }
    std::cerr << "P1" << std::endl;
    for (auto& kv : reverse_largest) {
        for (auto c : edgeconns[kv.first]) {
            Point p1 = reverse_point_map.at(kv.first);
            Point p2 = reverse_point_map.at(c);
            edgeofs << reverse_largest.at(kv.first) << " "
                    << reverse_largest.at(c) << " " << p1.dist(p2) << std::endl;
            auto [ox, oy] = p1;
            auto [dx, dy] = p2;
            dbgmapofs << ox << " " << oy << " " << dx << " " << dy << std::endl;
        }
    }
    std::vector<std::tuple<int, double, double>> nodes;
    for (auto pid : largest) {
        nodes.emplace_back(reverse_largest.at(pid), reverse_point_map.at(pid).x,
                           reverse_point_map.at(pid).y);
    }
    std::sort(nodes.begin(), nodes.end());
    std::ofstream nodeofs(argv[5]);
    for (auto [id, x, y] : nodes) {
        nodeofs << id << " " << x << " " << y << std::endl;
    }
    return 0;
}
