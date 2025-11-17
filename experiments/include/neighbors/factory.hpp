#pragma once
#include "neighbors/search.hpp"
#include "neighbors/brute.hpp"
#include "neighbors/kdtree.hpp"
#include "neighbors/balltree.hpp"
#include <memory>
#include <string>

std::unique_ptr<NeighborSearch> make_backend(
    const Dataset& X,
    const std::string& backend_name,    // "brute" | "kdtree" | "balltree"
    ComputeMode mode                    // Serial, Omp
);