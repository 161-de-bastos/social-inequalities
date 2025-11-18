#include "metrics.hpp"
#include "config.hpp"
#include "distance.hpp"

#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>

#ifdef __linux__
#include <unistd.h>
#endif

static std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return "";
    std::size_t b = s.find_last_not_of(" \t");
    return s.substr(a, b - a + 1);
}

bool metrics_wants(
    const MetricsConfig& cfg,
    const std::string& name
) {
    if (!cfg.enabled) return false;

    std::string target = name;
    std::transform(target.begin(), target.end(), target.begin(), ::tolower);
    std::istringstream iss(cfg.list);
    std::string token;
    while (std::getline(iss, token, ',')) {
        std::string t = trim(token);
        if (t.empty()) continue;
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        if (t == target) return true;
    }
    return false;
}

real_t silhouette_score(
    const Dataset& X,
    const std::vector<label_t>& labels
) {
    const std::size_t n = X.size();
    if (n == 0 || labels.size() != n) return real_t(-1);

    std::unordered_map<label_t, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < n; ++i) {
        label_t c = labels[i];
        if (c < 0) continue;
        groups[c].push_back(i);
    }

    if (groups.size() < 2) return real_t(-1);

    auto dist = [](
        const Point& a,
        const Point& b
    ) -> real_t {
        real_t sq = euclidean_dist(a, b);
        return std::sqrt(sq);
    };

    real_t sum_s = 0;
    std::size_t count_pts = 0;

    for (const auto& kv : groups) {
        label_t c = kv.first;
        const auto& idxs = kv.second;

        for (std::size_t pos = 0; pos < idxs.size(); ++pos) {
            std::size_t i = idxs[pos];
            const Point& xi = X[i];

            real_t a_i = 0;
            if (idxs.size() > 1) {
                for (std::size_t pos2 = 0; pos2 < idxs.size(); ++pos2) {
                    if (pos2 == pos) continue;
                    std::size_t j = idxs[pos2];
                    a_i += dist(xi, X[j]);
                } a_i /= real_t(idxs.size() - 1);
            } else a_i = 0;

            real_t b_i = std::numeric_limits<real_t>::max();

            for (const auto& kv2 : groups) {
                if (kv2.first == c) continue;
                const auto& idxs2 = kv2.second;
                if (idxs2.empty()) continue;

                real_t avg = 0;
                for (std::size_t j : idxs2) {
                    avg += dist(xi, X[j]);
                }
                avg /= real_t(idxs2.size());

                if (avg < b_i) b_i = avg;
            }

            real_t s_i = 0;
            real_t max_ab = std::max(a_i, b_i);
            if (max_ab > real_t(0)) s_i = (b_i - a_i) / max_ab;

            sum_s += s_i;
            ++count_pts;
        }
    }

    if (count_pts == 0) return real_t(-1);
    return sum_s / static_cast<real_t>(count_pts);
}

double get_memory_usage() {
    #ifdef __linux__
        std::ifstream f("/proc/self/status");
        if (!f.is_open()) return -1.0;

        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("VmRSS:", 0) == 0) {
                std::istringstream iss(line);
                std::string key;
                long value_kb = 0;
                std::string unit;
                iss >> key >> value_kb >> unit;
                return static_cast<double>(value_kb) / 1024.0;
            }
        }
        return -1.0;
    #else
        return -1.0;
    #endif
}

static std::string now_iso8601() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

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
) {
    std::ofstream out(file, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[Metrics] No se pudo abrir " << file << " para escribir\n";
        return;
    }

    out << "[" << now_iso8601() << "] "
        << "algo=" << algorithm
        << " task=" << task
        << " backend=" << (backend.empty() ? "-" : backend)
        << " mode=" << mode
        << " threads=" << num_threads
        << "\n";

    out << "  n_points=" << n_points
        << " dim=" << dim << "\n";

    if (log_time) {
        out << "  time=" << time_s << "s\n";
    }
    if (log_memory) {
        if (mem_mb >= 0.0)
            out << "  memory_rss=" << mem_mb << "MB\n";
        else
            out << "  memory_rss=n/a\n";
    }
    if (log_cluster) {
        if (cluster >= real_t(-0.999))  // -1 significa "no definido"
            out << "  silhouette=" << cluster << "\n";
        else
            out << "  silhouette=n/a\n";
    }

    out << "\n";
}