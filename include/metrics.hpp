#pragma once
#include "common.hpp"
#include <string>
#include <vector>

struct MetricsConfig;

real_t silhouette_score(
    const Dataset& X,
    const std::vector<label_t>& labels
);

double get_memory_usage();

bool metrics_wants(
    const MetricsConfig& cfg,
    const std::string& name
);

void append_log(
    const std::string& file,
    const std::string& algorithm,
    const std::string& task,
    const std::string& backend,
    const std::string& mode,
    int num_threads,
    std::size_t n_points,
    std::size_t dim,
    bool log_time,
    double time_s,
    bool log_memory,
    double mem_mb,
    bool log_cluster,
    real_t cluster
);