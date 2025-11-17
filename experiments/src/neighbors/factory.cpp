#include "neighbors/factory.hpp"
#include <algorithm>

std::unique_ptr<NeighborSearch> make_backend(
    const Dataset& X,
    const std::string& backend_name,    // "brute" | "kdtree" | "balltree"
    ComputeMode mode                    // Serial, Omp
) {
    std::string name = backend_name;
    std::transform(
        name.begin(),
        name.end(),
        name.begin(),
        ::tolower
    );

    if (name == "brute") {
        return std::make_unique<BruteForce>(X, mode);
    } else if (name == "kdtree") {
        return std::make_unique<KDTree>(X, mode);
    } else if (name == "balltree") {
        return std::make_unique<BallTree>(X, mode);
    } else {
        return std::make_unique<BruteForce>(X, mode);
    }
}