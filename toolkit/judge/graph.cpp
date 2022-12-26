#include "graph.hpp"
#include <algorithm>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include "error_check.hpp"
#include "io.hpp"
#include "lib/json.hpp"
#include "logger.hpp"
Edge::Edge() = default;
Edge::Edge(vertex_index_t from_, vertex_index_t to_, distance_t d_)
    : from(from_), to(to_), d(d_) {
}
bool Edge::operator<(const Edge& rhs) const {
    return to_tuple_() < rhs.to_tuple_();
}
bool Edge::operator==(const Edge& rhs) const {
    return from == rhs.from && to == rhs.to && d == rhs.d;
}
bool Edge::operator>(const Edge& rhs) const {
    return rhs < *this;
}
bool Edge::operator<=(const Edge& rhs) const {
    return !(*this > rhs);
}
bool Edge::operator>=(const Edge& rhs) const {
    return !(*this < rhs);
}
bool Edge::operator!=(const Edge& rhs) const {
    return !(*this == rhs);
}
std::tuple<vertex_index_t, vertex_index_t, distance_t> Edge::to_tuple_() const {
    return std::make_tuple(from, to, d);
}
void to_json(json_ref j, const Edge& e) {
    j = json{{"from", e.from}, {"to", e.to}, {"d", e.d}};
}
void output_graph_to_stream_ONLY_FOR_TESTING_(int N_V,
                                              const std::vector<Edge>& edges,
                                              std::ostream& os) {
    if (N_V < 0) {
        throw std::out_of_range("N_V is negative.");
    }
    for (auto [a, b, d] : edges) {
        if (a >= N_V || b >= N_V || a < 0 || b < 0) {
            throw std::invalid_argument("Invalid vertex indices.");
        }
    }
    os << N_V << TOKEN_DELIMITER_ONLY_FOR_TESTING_ << edges.size() << std::endl;
    for (auto [a, b, d] : edges) {
        os << a + 1 << TOKEN_DELIMITER_ONLY_FOR_TESTING_ << b + 1
           << TOKEN_DELIMITER_ONLY_FOR_TESTING_ << d << std::endl;
    }
}
bool test_if_graph_is_valid_for_contest_UNUSED_(
    int N_V, const std::vector<Edge>& edges) {
    if (N_V <= 0) {
        return false;
    }
    for (auto [a, b, d] : edges) {
        if (a >= N_V || b >= N_V || a < 0 || b < 0) {
            return false;
        }
        if (d < 0) {
            return false;
        }
    }
    return true;
}
std::vector<distance_t> compute_distances_single(
    int N_V, const std::vector<Edge>& edges, vertex_index_t origin,
    const std::vector<std::vector<std::pair<distance_t, vertex_index_t>>>*
        g_ptr) {
    THROW_LOGIC_ERROR_IF(N_V < 0, "Vertex count is negative.");
    THROW_LOGIC_ERROR_IF(N_V == 0,
                         "Null graph is not allowed in distance computation.");
    THROW_LOGIC_ERROR_IF(
        origin < 0, "In distance computation, the origin index is invalid.");
    using dv_pair = std::pair<distance_t, vertex_index_t>;
    std::vector<std::vector<dv_pair>> g_;
    if (g_ptr == nullptr) {
        g_.resize(N_V);
        for (auto [a, b, d] : edges) {
            THROW_LOGIC_ERROR_IF(
                a >= N_V
                    || b >= N_V
                    || a < 0
                    || b < 0,
                "Some vertex indices (in edge info) are out of "
                "range. (from:%d,to:%d,dist:%d)",
                a, b, d);
            THROW_LOGIC_ERROR_IF(
                d < 0,
                "Some edges have negative weights. (from:%d,to:%d,dist:%d)", a,
                b, d);
            g_[a].emplace_back(d, b);
        }
        g_ptr = &g_;
    }
    const std::vector<std::vector<dv_pair>>& g = *g_ptr;
    constexpr distance_t IMPOSSIBLE_LARGE_DISTANCE = MAX_DISTANCE + 1;
    std::vector<distance_t> d(N_V, MAX_DISTANCE + 1);
    d[origin] = 0;
    std::vector<vertex_index_t> prev(N_V, INVALID_VERTEX_ID);
    std::priority_queue<dv_pair, std::vector<dv_pair>, std::greater<dv_pair>> q;
    q.push({0, origin});
    while (!q.empty()) {
        auto [dist, u] = q.top();
        q.pop();
        if (d[u] < dist)
            continue;
        for (auto [to_dist, to] : g[u]) {
            if (d[to] > dist + to_dist) {
                d[to] = dist + to_dist;
                q.push({d[to], to});
                prev[to] = u;
            }
        }
    }
    for (auto& dist : d) {
        if (dist == IMPOSSIBLE_LARGE_DISTANCE) {
            dist = INVALID_DISTANCE;
        }
    }
    return d;
}
std::vector<std::vector<distance_t>> compute_distances_all(
    int N_V, const std::vector<Edge>& edges,
    const std::vector<std::vector<std::pair<distance_t, vertex_index_t>>>*
        g_ptr) {
    THROW_LOGIC_ERROR_IF(N_V < 0, "Vertex count is negative.");
    THROW_LOGIC_ERROR_IF(N_V == 0 && !edges.empty(), "No vertex but edges");
    std::vector<std::vector<distance_t>> ret(N_V);
    using dv_pair = std::pair<distance_t, vertex_index_t>;
    std::vector<std::vector<dv_pair>> g_;
    if (g_ptr == nullptr) {
        g_.resize(N_V);
        for (auto [a, b, d] : edges) {
            THROW_LOGIC_ERROR_IF(
                a >= N_V
                    || b >= N_V
                    || a < 0
                    || b < 0,
                "Some vertex indices (in edge info) are out of "
                "range. (from:%d,to:%d,dist:%d)",
                a, b, d);
            THROW_LOGIC_ERROR_IF(
                d < 0,
                "Some edges have negative weights. (from:%d,to:%d,dist:%d)", a,
                b, d);
            g_[a].emplace_back(d, b);
        }
        g_ptr = &g_;
    }
    for (vertex_index_t i = 0; i < N_V; i++) {
        ret[i] = compute_distances_single(N_V, edges, i, g_ptr);
    }
    return ret;
}
bool UndirectedGraph::vertex_exists(vertex_index_t vid) const {
    return 0 <= vid && vid < vertex_num();
}
void UndirectedGraph::read_graph(std::istream& is) {
    ValueReader r(is);
    graph_loaded_ = false;
    edges_.clear();
    organized_edges_.clear();
    readline_exact(r,
                   TAG(ValueGroup::POSITIVE_COUNT, N_V),
                   TAG(ValueGroup::NON_NEGATIVE_COUNT, N_E));
    THROW_RUNTIME_ERROR_IF(N_V < 1, "Vertex count is 0 or negative.");
    THROW_RUNTIME_ERROR_IF(N_E < 0, "Edge count is negative.");
    for (int i = 0; i < N_V; i++) {
        vertex_index_t id;
        double x, y;
        readline_exact(r,
                       TAG(ValueGroup::INDEX, id),
                       TAG(ValueGroup::COORD, x),
                       TAG(ValueGroup::COORD, y));
        coords_[id] = {x, y};
    }
    for (int i = 0; i < N_E; i++) {
        Edge e;
        readline_exact(r,
                       TAG(ValueGroup::INDEX, e.from),
                       TAG(ValueGroup::INDEX, e.to),
                       TAG(ValueGroup::DISTANCE, e.d));
        THROW_RUNTIME_ERROR_IF(e.from < 0
                                   || e.from >= N_V
                                   || e.to < 0
                                   || e.to >= N_V,
                               "Vertex indices (0-based) are out of range");
        THROW_RUNTIME_ERROR_IF(e.d <= 0,
                               "Edge weight cannot be 0 nor negative.");
        Edge e2{e.to, e.from, e.d};
        edges_.emplace_back(std::move(e));
        edges_.emplace_back(std::move(e2));
    }
    organized_edges_.resize(N_V);
    for (auto [a, b, d] : edges_) {
        organized_edges_[a].emplace_back(d, b);
    }
    graph_loaded_ = true;
    INFO("Graph data loaded succesfully.");
}
bool UndirectedGraph::is_graph_loaded() const {
    return graph_loaded_;
}
void UndirectedGraph::compute_distance_info() {
    THROW_LOGIC_ERROR_IF(!is_graph_loaded(), "Graph data is not loaded.");
    if (distances_computed_) {
        INFO("The distances are already computed.");
        return;
    }
    INFO("Computing distance info...");
    distances_computed_ = false;
    using dv_pair = std::pair<distance_t, vertex_index_t>;
    constexpr distance_t IMPOSSIBLE_LARGE_DISTANCE = MAX_DISTANCE + 1;
    dist_mat_.resize(N_V * N_V);
    std::fill(dist_mat_.begin(), dist_mat_.end(), IMPOSSIBLE_LARGE_DISTANCE);
    std::queue<vertex_index_t> q;
    for (vertex_index_t origin = 0; origin < N_V; origin++) {
        distance_t* dist = &dist_mat_[N_V * origin];
        dist[origin] = 0;
        q.push(origin);
        while (!q.empty()) {
            vertex_index_t t = q.front();
            q.pop();
            for (auto e : organized_edges_[t]) {
                if (dist[e.second] > dist[t] + e.first) {
                    dist[e.second] = dist[t] + e.first;
                    q.push(e.second);
                }
            }
        }
    }
    distances_computed_ = true;
    INFO("Done.");
}
bool UndirectedGraph::is_distance_info_computed() const {
    return distances_computed_;
}
bool UndirectedGraph::is_position_valid(const Position& p,
                                        std::string* msg) const {
    bool valid = true;
    if (msg)
        msg->clear();
    {
        std::string pmsg;
        VALIDITY_CHECK_WITH_MSG(
            p.is_broken(&pmsg), valid, msg,
            "The position is broken for the following reason(s):%s",
            pmsg.c_str());
    }
    VALIDITY_CHECK_WITH_MSG(
        p.u >= N_V, valid, msg,
        "u is out of range (too large: internal u(%d) >= N_V(%d)).", p.u, N_V);
    VALIDITY_CHECK_WITH_MSG(
        p.v >= N_V, valid, msg,
        "v is out of range (too large: internal v(%d) >= N_V(%d)).", p.v, N_V);
    if (!valid) {
        return valid;
    }
    VALIDITY_CHECK_WITH_MSG(distance(p.u, p.v) == INVALID_DISTANCE, valid, msg,
                            "p.u and p.v is not connected.");
    VALIDITY_CHECK_WITH_MSG(p.distance_from_u > distance(p.u, p.v), valid, msg,
                            "The distance is too large:%d > %d",
                            p.distance_from_u, distance(p.u, p.v));
    VALIDITY_CHECK_WITH_MSG(
        p.u != p.v && (distance(p.u, p.v) == p.distance_from_u ||
                       distance(p.u, p.v) == 0),
        valid, msg, "Exact vertex but p.u(internal:%d) != p.v(internal:%d)",
        p.u, p.v);
    return valid;
}
distance_t UndirectedGraph::distance(vertex_index_t u, vertex_index_t v) const {
    THROW_LOGIC_ERROR_IF(!is_distance_info_computed(),
                         "Distances are not computed.");
    THROW_LOGIC_ERROR_IF(u == INVALID_VERTEX_ID,
                         "u (in an edge) is invalid.(%d)", u);
    THROW_LOGIC_ERROR_IF(u < 0 || u >= N_V,
                         "u (in an edge) is out of range.(%d)", u);
    THROW_LOGIC_ERROR_IF(v == INVALID_VERTEX_ID,
                         "v (in an edge) is invalid.(%d)", v);
    THROW_LOGIC_ERROR_IF(v < 0 || v >= N_V,
                         "v (in an edge) is out of range.(%d)", v);
    return dist_mat_[N_V * u + v];
}
distance_t UndirectedGraph::distance(const Position& p1, const Position& p2,
                                     vertex_index_t* towards1,
                                     vertex_index_t* from2) const {
    THROW_LOGIC_ERROR_IF(
        !is_position_valid(p1) || !is_position_valid(p2),
        "Positions are invalid (in position distance calculation)");
    if (distance(p1.u, p2.u) == INVALID_DISTANCE) {
        THROW_LOGIC_ERROR_IF(distance(p1.u, p2.v) != INVALID_DISTANCE ||
                                 distance(p1.v, p2.u) != INVALID_DISTANCE ||
                                 distance(p1.v, p2.v) != INVALID_DISTANCE,
                             "Inconsistent connectivity");
        if (towards1 != nullptr)
            *towards1 = INVALID_VERTEX_ID;
        if (from2 != nullptr)
            *from2 = INVALID_VERTEX_ID;
        return INVALID_DISTANCE;
    }
    if (p1.is_exact_vertex() && p2.is_exact_vertex() && p1.u == p2.u) {
        if (towards1 != nullptr)
            *towards1 = INVALID_VERTEX_ID;
        if (from2 != nullptr)
            *from2 = INVALID_VERTEX_ID;
        return 0;
    }
    if (!p1.is_exact_vertex() && !p2.is_exact_vertex() &&
        ((p1.u == p2.u && p1.v == p2.v) || (p1.u == p2.v && p1.v == p2.u))) {
        distance_t dist1 = p1.distance_from_u;
        distance_t dist2 = INVALID_DISTANCE;
        if (p1.u == p2.u && p1.v == p2.v) {
            dist2 = p2.distance_from_u;
        }
        if (p1.u == p2.v && p1.v == p2.u) {
            dist2 = distance(p2.u, p2.v) - p2.distance_from_u;
        }
        if (dist1 <= dist2) {
            if (towards1 != nullptr)
                *towards1 = p1.v;
            if (from2 != nullptr)
                *from2 = p1.u;
        } else if (dist1 > dist2) {
            if (towards1 != nullptr)
                *towards1 = p1.u;
            if (from2 != nullptr)
                *from2 = p1.v;
        }
        return std::abs(dist2 - dist1);
    }
    if (p1.is_exact_vertex() && (p1.u == p2.u || p1.u == p2.v)) {
        if (p1.u == p2.u) {
            if (towards1 != nullptr)
                *towards1 = p2.v;
            if (from2 != nullptr)
                *from2 = p2.u;
            return p2.distance_from_u;
        }
        if (p1.u == p2.v) {
            if (towards1 != nullptr)
                *towards1 = p2.u;
            if (from2 != nullptr)
                *from2 = p2.v;
            return distance(p2.u, p2.v) - p2.distance_from_u;
        }
        THROW_LOGIC_ERROR_IF(true,
                             "Unreachable code 1 in position distance calc");
    }
    if (p2.is_exact_vertex() && (p2.u == p1.u || p2.u == p1.v)) {
        if (p2.u == p1.u) {
            if (towards1 != nullptr)
                *towards1 = p1.u;
            if (from2 != nullptr)
                *from2 = p1.v;
            return p1.distance_from_u;
        }
        if (p2.u == p1.v) {
            if (towards1 != nullptr)
                *towards1 = p1.v;
            if (from2 != nullptr)
                *from2 = p1.u;
            return distance(p1.u, p1.v) - p1.distance_from_u;
        }
        THROW_LOGIC_ERROR_IF(true,
                             "Unreachable code 2 in position distance calc");
    }
    const distance_t d1 =
        p1.distance_from_u + distance(p1.u, p2.u) + p2.distance_from_u;
    const distance_t d2 = (distance(p1.u, p1.v) - p1.distance_from_u) +
                          distance(p1.v, p2.u) + p2.distance_from_u;
    const distance_t d3 = p1.distance_from_u + distance(p1.u, p2.v) +
                          (distance(p2.u, p2.v) - p2.distance_from_u);
    const distance_t d4 = (distance(p1.u, p1.v) - p1.distance_from_u) +
                          distance(p1.v, p2.v) +
                          (distance(p2.u, p2.v) - p2.distance_from_u);
    if (towards1 == nullptr && from2 == nullptr) {
        return std::min({d1, d2, d3, d4});
    }
    const std::vector<distance_t> ds = {d1, d2, d3, d4};
    auto it = std::min_element(ds.begin(), ds.end());
    auto idx = std::distance(ds.begin(), it);
    auto via_adj = [&](vertex_index_t vertex_at, const Position& dest) {
        auto it_ =
            std::min_element(organized_edges_[vertex_at].begin(),
                             organized_edges_[vertex_at].end(),
                             [&](std::pair<distance_t, vertex_index_t> a,
                                 std::pair<distance_t, vertex_index_t> b) {
                                 return a.first + distance(a.second, dest) <
                                        b.first + distance(b.second, dest);
                             });
        return it_->second;
    };
    if (idx == 0) {
        if (towards1 != nullptr) {
            if (p1.is_exact_vertex()) {
                *towards1 = via_adj(p1.u, p2);
            } else {
                *towards1 = p1.u;
            }
        }
        if (from2 != nullptr) {
            if (p2.is_exact_vertex()) {
                *from2 = via_adj(p2.u, p1);
            } else {
                *from2 = p2.u;
            }
        }
    } else if (idx == 1) {
        if (towards1 != nullptr) {
            if (p1.is_exact_vertex()) {
                *towards1 = via_adj(p1.v, p2);
            } else {
                *towards1 = p1.v;
            }
        }
        if (from2 != nullptr) {
            if (p2.is_exact_vertex()) {
                *from2 = via_adj(p2.u, p1);
            } else {
                *from2 = p2.u;
            }
        }
    } else if (idx == 2) {
        if (towards1 != nullptr) {
            if (p1.is_exact_vertex()) {
                *towards1 = via_adj(p1.u, p2);
            } else {
                *towards1 = p1.u;
            }
        }
        if (from2 != nullptr) {
            if (p2.is_exact_vertex()) {
                *from2 = via_adj(p2.v, p1);
            } else {
                *from2 = p2.v;
            }
        }
    } else if (idx == 3) {
        if (towards1 != nullptr) {
            if (p1.is_exact_vertex()) {
                *towards1 = via_adj(p1.v, p2);
            } else {
                *towards1 = p1.v;
            }
        }
        if (from2 != nullptr) {
            if (p2.is_exact_vertex()) {
                *from2 = via_adj(p2.v, p1);
            } else {
                *from2 = p2.v;
            }
        }
    } else {
        THROW_LOGIC_ERROR_IF(true, "Unreachable code 3 in position distance "
                                   "calc: minimum distance is not found.");
    }
    return *it;
}
distance_t UndirectedGraph::distance(vertex_index_t u,
                                     const Position& p) const {
    Position p1;
    p1.u = u;
    p1.v = u;
    p1.distance_from_u = 0;
    return distance(p1, p);
}
distance_t UndirectedGraph::distance(const Position& p,
                                     vertex_index_t v) const {
    Position p2;
    p2.u = v;
    p2.v = v;
    p2.distance_from_u = 0;
    return distance(p, p2);
}
vertex_index_t UndirectedGraph::next_vertex(const Position& current,
                                            vertex_index_t target) const {
    THROW_LOGIC_ERROR_IF(!is_distance_info_computed(),
                         "Distances are not computed.");
    THROW_LOGIC_ERROR_IF(!is_position_valid(current),
                         "Current position is invalid.");
    THROW_LOGIC_ERROR_IF(target == INVALID_VERTEX_ID,
                         "Target vertex is invalid.");
    THROW_LOGIC_ERROR_IF(target < 0 || target >= N_V,
                         "Target vertex is out of range.(%d)", target);
    vertex_index_t towards1;
    distance(current, Position(target), &towards1, nullptr);
    return towards1;
}
bool UndirectedGraph::is_directly_movable_to(const Position& current,
                                             vertex_index_t target) const {
    THROW_LOGIC_ERROR_IF(!is_position_valid(current),
                         "Current position is invalid.");
    THROW_LOGIC_ERROR_IF(!is_position_valid(current),
                         "Current position is invalid.");
    THROW_LOGIC_ERROR_IF(target == INVALID_VERTEX_ID,
                         "Target vertex is invalid.");
    THROW_LOGIC_ERROR_IF(target < 0 || target >= N_V,
                         "Target vertex is out of range.(%d)", target);
    if (current.is_exact_vertex()) {
        if (current.u == target) {
            return true;
        }
        const auto& adj = organized_edges_[current.u];
        using vtype =
            std::remove_const_t<std::remove_reference_t<decltype(adj)>>;
        return std::find_if(adj.begin(), adj.end(),
                            [&](const vtype::value_type& p) {
                                return p.second == target;
                            }) != adj.end();
    }
    return current.u == target || current.v == target;
}
Position UndirectedGraph::canonicalize_position(const Position& p) const {
    if (p.distance_from_u == 0) {
        return Position(p.u);
    }
    if (p.distance_from_u == distance(p.u, p.v)) {
        return Position(p.v);
    }
    return p;
}
size_t UndirectedGraph::vertex_num() const {
    return N_V;
}
size_t UndirectedGraph::edge_num() const {
    size_t ret = 0;
    for (const auto& v : organized_edges_) {
        ret += v.size();
    }
    THROW_LOGIC_ERROR_IF(ret % 2 != 0, "Non-symmetric edges");
    return ret / 2;
}
std::pair<double, double>
UndirectedGraph::get_coordinates(vertex_index_t vid) const {
    THROW_RUNTIME_ERROR_IF(coords_.find(vid) == coords_.end(),
                           "coordinates not found for vertex id %d", vid);
    return coords_.at(vid);
}
void to_json(json_ref j, const UndirectedGraph& ug) {
    std::vector<Edge> e;
    for (const auto& v : ug.edges_) {
        if (v.from < v.to) {
            e.emplace_back(v);
        }
    }
    j = json{
        {"n_v", ug.N_V},
        {"n_e", ug.N_E},
        {"coords", ug.coords_},
        {"edges", e},
    };
}
Position::Position() = default;
Position::Position(vertex_index_t idx) : u(idx), v(idx), distance_from_u(0) {
}
bool Position::is_broken(std::string* msg) const {
    bool not_broken = true;
    if (msg)
        msg->clear();
    VALIDITY_CHECK_WITH_MSG(u == INVALID_VERTEX_ID, not_broken, msg,
                            "u is INVALID");
    VALIDITY_CHECK_WITH_MSG(u < 0, not_broken, msg, "u is negative");
    VALIDITY_CHECK_WITH_MSG(v == INVALID_VERTEX_ID, not_broken, msg,
                            "v is INVALID");
    VALIDITY_CHECK_WITH_MSG(v < 0, not_broken, msg, "v is negative");
    VALIDITY_CHECK_WITH_MSG(distance_from_u == INVALID_VERTEX_ID, not_broken,
                            msg, "distance is INVALID");
    VALIDITY_CHECK_WITH_MSG(distance_from_u < 0, not_broken, msg,
                            "distance is negative");
    VALIDITY_CHECK_WITH_MSG(u == v && distance_from_u > 0, not_broken, msg,
                            "u == v but the distance is positive (>0)");
    VALIDITY_CHECK_WITH_MSG(
        u != v && distance_from_u == 0, not_broken, msg,
        "u != v but the distance is zero (for canonical expression)");
    return !not_broken;
}
bool Position::is_exact_vertex() const {
    std::string msg;
    THROW_LOGIC_ERROR_IF(
        is_broken(&msg),
        "This position data is broken for the following reason(s):%s",
        msg.c_str());
    return u == v;
}
void to_json(json_ref j, const Position& p) {
    j = json{{"u", p.u}, {"v", p.v}, {"distance_from_u", p.distance_from_u}};
}
std::ostream& operator<<(std::ostream& os, const Position& p) {
    json j = p;
    os << j;
    return os;
}
