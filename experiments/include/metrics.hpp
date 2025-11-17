#pragma once
#include "common.hpp"
#include <string>
#include <vector>

struct MetricsConfig;

real_t compute_silhouette(
    const Dataset& X,
    const std::vector<label_t>& labels
);

double get_memory_usage_mb();

bool metrics_wants(
    const MetricsConfig& cfg,
    const std::string& name
);
