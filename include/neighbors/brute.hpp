#pragma once
#include "neighbors/search.hpp"
#include "distance.hpp"

class BruteForce : public NeighborSearch {
public:
    explicit BruteForce(
        const Dataset& X, 
        ComputeMode mode = ComputeMode::Serial
    );

    void radius_query(
        int idx, 
        real_t eps,
        std::vector<int>& out_idx
    ) const override;

    int size() const override { 
        return static_cast<int>(X_.size()); 
    }

private:
    const Dataset& X_;
    ComputeMode mode_;

    void serial(
        int idx, 
        real_t eps,
        std::vector<int>& out_idx
    ) const;

    void omp(
        int idx, 
        real_t eps,
        std::vector<int>& out_idx
    ) const;
};