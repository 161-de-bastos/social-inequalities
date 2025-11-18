#pragma once
#include "common.hpp"
#include "config.hpp"
#include <string>
#include <vector>

void preprocess_dataset(
    Dataset& X,
    const PreprocessConfig& cfg
);

bool load_csv(
    const std::string& path,
    Dataset& X,
    int start_col = 0,
    char sep = ','
);

bool save_labels_csv(
    const std::string& path,
    const std::vector<int>& labels
);

bool is_normal_column(
    const std::vector<double>& col,
    double skew_thr,
    double kurt_thr
);