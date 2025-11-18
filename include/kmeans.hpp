#pragma once
#include "common.hpp"
#include <vector>

struct KMeansParams {
    int k = 4;
    int max_iters = 100;
};

struct  KMeansResult {
    std::vector<label_t> labels;
};

class KMeans {
public:
    KMeans(
        const Dataset& X,
        ComputeMode mode
    );

    KMeansResult fit(const KMeansParams& params);

private:
    const Dataset& X_;
    ComputeMode mode_;
};
