#pragma once
#include "common.hpp"
#include <stdexcept>
#include <cmath>

inline real_t euclidean_dist(
    const Point& a, 
    const Point& b
) {
    if (a.size() != b.size()) throw std::runtime_error("Incompatible dimensions");

    real_t sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        real_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
