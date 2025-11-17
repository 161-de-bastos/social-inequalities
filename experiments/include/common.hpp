#pragma once
#include <vector>
#include <cstddef>

using real_t = double;
using label_t = int;

using Point = std::vector<real_t>;
using Dataset = std::vector<Point>;

enum class ComputeMode {
    Serial,
    Omp
};