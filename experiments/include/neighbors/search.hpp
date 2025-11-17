#pragma once
#include "common.hpp"
#include <vector>

class NeighborSearch {
public:
    virtual ~NeighborSearch() = default;

    virtual void radius_query(
        int idx,
        real_t eps,
        std::vector<int>& out_idx
    ) const = 0;

    virtual int size() const = 0;
};