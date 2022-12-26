#include <cstdio>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>
using dist_vertex_pair = std::pair<int, int>;
int main(int argc, char** argv) {
    int N_V, N_E;
    std::cin >> N_V >> N_E;
    std::vector<std::vector<dist_vertex_pair>> dvs(N_V);
    for (int i = 0; i < N_E; i++) {
        int u, v, d;
        std::cin >> u >> v >> d;
        u--;
        v--;
        dvs[u].emplace_back(d, v);
        dvs[v].emplace_back(d, u);
    }
    constexpr int EXTREME_DISTANCE = (1 << 30) + 1;
    std::vector<int> d(N_V);
    for (int origin = 0; origin < N_V; origin++) {
        std::fill(d.begin(), d.end(), EXTREME_DISTANCE);
        std::priority_queue<dist_vertex_pair, std::vector<dist_vertex_pair>,
                            std::greater<dist_vertex_pair>>
            q;
        d[origin] = 0;
        q.emplace(0, origin);
        while (!q.empty()) {
            auto [dist, u] = q.top();
            q.pop();
            if (d[u] < dist)
                continue;
            for (auto [to_dist, to] : dvs[u]) {
                if (d[to] > dist + to_dist) {
                    d[to] = dist + to_dist;
                    q.emplace(d[to], to);
                }
            }
        }
        printf("%d", d[0]);
        for (int i = 1; i < d.size(); i++) {
            printf(" %d", d[i]);
        }
        printf("\n");
    }
    std::cout.flush();
}
