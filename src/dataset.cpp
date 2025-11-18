#include "dataset.hpp"
#include "config.hpp"

#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

// detecta si una fila es numérica desde start_col
static bool is_numeric_row(const std::string& line, char sep, int start_col) {
    std::stringstream ss(line);
    std::string token;
    int col = 0;

    while (std::getline(ss, token, sep)) {
        if (col++ < start_col) continue;  // ignorar columnas previas

        auto s = token.find_first_not_of(" \t");
        auto e = token.find_last_not_of(" \t");
        if (s == std::string::npos) continue;

        token = token.substr(s, e - s + 1);

        try {
            std::stod(token);
        } catch (...) {
            return false;
        }
    }
    return true;
}

bool load_csv(
    const std::string& path,
    Dataset& X,
    int start_col,
    char sep
) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ERROR] No se pudo abrir: " << path << "\n";
        return false;
    }

    std::string line;
    X.clear();
    bool first_line = true;

    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;

        if (first_line) {
            first_line = false;
            if (!is_numeric_row(line, sep, start_col)) {
                // Header automático
                continue;
            }
        }

        std::stringstream ss(line);
        std::string token;
        Point row;
        int col = 0;

        while (std::getline(ss, token, sep)) {
            if (col++ < start_col) continue;

            auto s = token.find_first_not_of(" \t");
            auto e = token.find_last_not_of(" \t");
            if (s == std::string::npos) continue;

            token = token.substr(s, e - s + 1);

            try {
                row.push_back(std::stod(token));
            } catch (...) {
                std::cerr << "[ERROR] Valor no numérico en columna "
                          << (col - 1) << " después del start_col.\n";
                return false;
            }
        }

        if (!row.empty()) X.push_back(std::move(row));
    }

    if (X.empty()) {
        std::cerr << "[WARN] CSV sin datos (quizás header solamente).\n";
    }

    return true;
}

bool save_labels_csv(
    const std::string& path,
    const std::vector<int>& labels
) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ERROR] No se pudo escribir en: " << path << "\n";
        return false;
    }

    f << "id,cluster\n";
    for (std::size_t i = 0; i < labels.size(); ++i) {
        f << i << "," << labels[i] << "\n";
    }

    return true;
}

static double mean(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}

static double stddev(const std::vector<double>& v, double mu) {
    double s = 0;
    for (double x : v) {
        double d = x - mu;
        s += d * d;
    }
    return std::sqrt(s / v.size() + 1e-12);
}

static double skewness(const std::vector<double>& v, double mu, double sd) {
    double s = 0;
    for (double x : v)
        s += std::pow((x - mu) / sd, 3);
    return s / v.size();
}

static double kurtosis(const std::vector<double>& v, double mu, double sd) {
    double s = 0;
    for (double x : v)
        s += std::pow((x - mu) / sd, 4);
    return s / v.size() - 3.0;   // Fisher definition
}

bool is_normal_column(
    const std::vector<double>& col,
    double skew_thr,
    double kurt_thr
) {
    if (col.size() < 30) return false;

    double mu = mean(col);
    double sd = stddev(col, mu);
    if (sd < 1e-12) return false;

    double sk = skewness(col, mu, sd);
    double ku = kurtosis(col, mu, sd);

    return (std::abs(sk) < skew_thr) && (std::abs(ku) < kurt_thr);
}

void preprocess_dataset(
    Dataset& X,
    const PreprocessConfig& cfg
) {
    if (X.empty()) return;

    if (!cfg.normalize) {
        std::cout << "[Preprocess] normalize = false, se omite escalado\n";
        return;
    }

    const std::size_t n = X.size();
    const std::size_t m = X[0].size();

    std::cout << "[Preprocess] normalize = true, Filas=" << n
              << " Cols=" << m
              << " skew_thr=" << cfg.skew_thr
              << " kurt_thr=" << cfg.kurt_thr
              << "\n";

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int j_int = 0; j_int < static_cast<int>(m); ++j_int) {
        std::size_t j = static_cast<std::size_t>(j_int);

        // extraer columna
        std::vector<double> col(n);
        for (std::size_t i = 0; i < n; ++i)
            col[i] = X[i][j];

        bool normal = is_normal_column(col, cfg.skew_thr, cfg.kurt_thr);

        double mu   = mean(col);
        double sd   = stddev(col, mu);
        double cmin = *std::min_element(col.begin(), col.end());
        double cmax = *std::max_element(col.begin(), col.end());
        double range = cmax - cmin + 1e-12;

        for (std::size_t i = 0; i < n; ++i) {
            if (normal) {
                // StandardScaler
                X[i][j] = (X[i][j] - mu) / sd;
            } else {
                // MinMaxScaler
                X[i][j] = (X[i][j] - cmin) / range;
            }
        }
    }
}
