#pragma once
#include "common.hpp"
#include "neighbors/search.hpp"
#include <vector>
#include <memory>

struct DBSCANParams {
    real_t eps;
    int minPts;
};

struct  DBSCANResult {
    std::vector<label_t> labels;
    int n_clusters = 0;
    int n_noise = 0;
};

class DBSCAN {
public:
    DBSCAN(
        const Dataset& X,
        std::unique_ptr<NeighborSearch> ns
    );

    DBSCANResult fit(const DBSCANParams& params);

private:
    const Dataset& X_;
    std::unique_ptr<NeighborSearch> ns_;
};