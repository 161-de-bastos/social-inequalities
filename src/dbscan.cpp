#include "dbscan.hpp"

DBSCAN::DBSCAN(
    const Dataset& X,
    std::unique_ptr<NeighborSearch> ns
) : X_(X), ns_(std::move(ns)) {}

DBSCANResult DBSCAN::fit(const DBSCANParams& params) {
    const int n = ns_->size();
    DBSCANResult res;
    res.labels.assign(n, -2); // -2 = UNVISITED

    const real_t eps    = params.eps;
    const int minPts = params.minPts;

    int current_cluster = 0;
    std::vector<int> neighbors;
    std::vector<int> seed_set;

    for (int i = 0; i < n; ++i) {
        if (res.labels[i] != -2) continue;

        ns_->radius_query(i, eps, neighbors);

        if ((int)neighbors.size() + 1 < minPts) {
            res.labels[i] = -1; // noise
            continue;
        }

        // nuevo cluster
        res.labels[i] = current_cluster;
        seed_set = neighbors;

        // expandir cluster
        for (std::size_t k = 0; k < seed_set.size(); ++k) {
            int j = seed_set[k];

            // era ruido, ahora es borde del cluster
            if (res.labels[j] == -1) res.labels[j] = current_cluster;
            // ya fue asignado
            if (res.labels[j] != -2) continue;

            res.labels[j] = current_cluster;

            ns_->radius_query(j, eps, neighbors);

            if ((int)neighbors.size() + 1 >= minPts) {
                seed_set.insert(
                    seed_set.end(),
                    neighbors.begin(), 
                    neighbors.end()
                );
            }
        }

        current_cluster++;
    }

    res.n_clusters = current_cluster;
    res.n_noise    = 0;
    for (int i = 0; i < n; ++i) if (res.labels[i] == -1) res.n_noise++;

    return res;
}
