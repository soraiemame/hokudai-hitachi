#ifndef HEADER_2HC2022_GRAPH
#define HEADER_2HC2022_GRAPH 
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "defines.hpp"
#include "error_check.hpp"
#include "json_fwd.hpp"
class Edge {
    std::tuple<vertex_index_t, vertex_index_t, distance_t> to_tuple_() const;
 public:
    vertex_index_t from = INVALID_VERTEX_ID;
    vertex_index_t to = INVALID_VERTEX_ID;
    distance_t d = INVALID_DISTANCE;
    Edge();
    Edge(vertex_index_t from_, vertex_index_t to_, distance_t d_);
    bool operator<(const Edge& rhs) const;
    bool operator==(const Edge& rhs) const;
    bool operator>(const Edge& rhs) const;
    bool operator<=(const Edge& rhs) const;
    bool operator>=(const Edge& rhs) const;
    bool operator!=(const Edge& rhs) const;
    friend void to_json(json_ref j, const Edge& e);
};
void output_graph_to_stream_ONLY_FOR_TESTING_(int N_V,
                                              const std::vector<Edge>& edges,
                                              std::ostream& os);
bool test_if_graph_is_valid_for_contest_UNUSED_(int N_V,
                                                const std::vector<Edge>& edges);
std::vector<distance_t> compute_distances_single(
    int N_V, const std::vector<Edge>& edges, vertex_index_t origin,
    const std::vector<std::vector<std::pair<distance_t, vertex_index_t>>>*
        g_ptr = nullptr);
std::vector<std::vector<distance_t>> compute_distances_all(
    int N_V, const std::vector<Edge>& edges,
    const std::vector<std::vector<std::pair<distance_t, vertex_index_t>>>*
        g_ptr = nullptr);
struct Position;
class UndirectedGraph {
    using distance_matrix_t_ = std::vector<std::vector<distance_t>>;
    using edge_seq_t_ = std::vector<Edge>;
    int N_V = INVALID_COUNT;
    int N_E = INVALID_COUNT;
    std::vector<distance_t> dist_mat_;
    edge_seq_t_ edges_;
    std::vector<std::vector<std::pair<distance_t, vertex_index_t>>>
        organized_edges_;
    bool graph_loaded_ = false;
    bool distances_computed_ = false;
    std::map<vertex_index_t, std::pair<double, double>> coords_;
 public:
    std::pair<double, double> get_coordinates(vertex_index_t vid) const;
    void read_graph(std::istream& is);
    bool is_graph_loaded() const;
    void compute_distance_info();
    bool is_distance_info_computed() const;
    bool is_position_valid(const Position& p, std::string* msg = nullptr) const;
    distance_t distance(vertex_index_t u, vertex_index_t v) const;
    distance_t distance(const Position& p1, const Position& p2,
                        vertex_index_t* towards1 = nullptr,
                        vertex_index_t* from2 = nullptr) const;
    distance_t distance(vertex_index_t u, const Position& p) const;
    distance_t distance(const Position& p, vertex_index_t v) const;
    vertex_index_t next_vertex(const Position& current,
                               vertex_index_t target) const;
    bool is_directly_movable_to(const Position& current,
                                vertex_index_t target_vertex) const;
    Position canonicalize_position(const Position& p) const;
    size_t vertex_num() const;
    size_t edge_num() const;
    bool vertex_exists(vertex_index_t vid) const;
    template <class Fn> void for_each_edge(const Fn& f) const {
        for (int i = 0; i < organized_edges_.size(); i++) {
            const auto& v = organized_edges_[i];
            for (int j = 0; j < v.size(); j++) {
                auto [d, to] = v[j];
                THROW_LOGIC_ERROR_IF(i == to,
                                     "The graph contains self loops(%d)", i);
                if (i < to) {
                    f(i, to, d);
                }
            }
        }
    }
    friend void to_json(json_ref j, const UndirectedGraph& ug);
};
struct Position {
    vertex_index_t u = INVALID_VERTEX_ID;
    vertex_index_t v = INVALID_VERTEX_ID;
    distance_t distance_from_u = INVALID_DISTANCE;
    Position();
    explicit Position(vertex_index_t idx);
    bool is_broken(std::string* msg = nullptr) const;
    bool is_exact_vertex() const;
    friend void to_json(json_ref j, const Position& a);
    friend std::ostream& operator<<(std::ostream& os, const Position& p);
};
#endif
