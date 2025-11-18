#pragma once
#include <string>

struct ParallelConfig {
    std::string mode = "serial";
    int num_threads = 0;
};

struct DBSCANConfig {
    double eps = 0.5;
    int minPts = 5;
};

struct KMeansConfig {
    int k = 4;
    int max_iters = 100;
};

struct DataConfig {
    std::string path = "";
    int start_col = 0;
};

struct PreprocessConfig {
    bool normalize = false;
    double skew_thr = 0.5;
    double kurt_thr = 1.0;
};

struct MetricsConfig {
    bool enabled = false;
    std::string list = "";
    std::string output = "metrics.log";
};

struct Config {
    std::string task = "dbscan";
    std::string backend = "brute";

    ParallelConfig parallel;
    DBSCANConfig dbscan;
    KMeansConfig kmeans;
    DataConfig data;
    PreprocessConfig preprocess;
    MetricsConfig metrics;
};

bool load_config(
    const std::string& path,
    Config& cfg
);