#include "config.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace {
    std::string trim(const std::string& s) {
        std::size_t a = s.find_first_not_of(" \t");
        if (a == std::string::npos) return "";
        std::size_t b = s.find_last_not_of(" \t");
        return s.substr(a, b - a + 1);
    }

    bool is_comment_or_empty(const std::string& line) {
        std::string t = trim(line);
        return t.empty() || t[0] == '#';
    }
}

bool load_config(
    const std::string& path,
    Config& cfg
) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    std::string section;

    while (std::getline(f, line)) {
        if (is_comment_or_empty(line)) continue;

        // contar espacios iniciales
        int indent = 0;
        while (indent < (int)line.size() && line[indent] == ' ') indent++;
        std::string content = trim(line);
        
        // sección nueva: "parallel:" o "dbscan:"
        if (indent == 0 && !content.empty() && content.back() == ':') {
            section = content.substr(0, content.size() - 1);
            continue;
        }

        // entrada clave: valor
        auto pos = content.find(':');
        if (pos == std::string::npos) continue;

        std::string key   = trim(content.substr(0, pos));
        std::string value = trim(content.substr(pos + 1));

        // quitar comillas simples o dobles si las hubiera
        if (!value.empty() && (value.front() == '"' || value.front() == '\'')) {
            if (value.back() == value.front() && value.size() >= 2) {
                value = value.substr(1, value.size() - 2);
            }
        }

        if (indent == 0) {
            // nivel raíz: task, backend
            if (key == "task") {
                cfg.task = value;
            } else if (key == "backend") {
                cfg.backend = value;
            }
        } else if (indent >= 2) {
            // dentro de una sección
            if (section == "data") {
                if (key == "path") cfg.data.path = value;
                if (key == "start_col") cfg.data.start_col = std::stoi(value);
            
            } else if (section == "parallel") {
                if (key == "mode") cfg.parallel.mode = value;
                if (key == "num_threads") cfg.parallel.num_threads = std::stoi(value);
            
            } else if (section == "dbscan") {
                if (key == "eps") cfg.dbscan.eps = std::stod(value);
                if (key == "minPts") cfg.dbscan.minPts = std::stoi(value);
            
            } else if (section == "kmeans") {
                if (key == "k") cfg.kmeans.k = std::stod(value);
                if (key == "max_iters") cfg.kmeans.max_iters = std::stod(value);
            
            } else if (section == "preprocess") {
                if (key == "normalize") {
                    std::string v = value;
                    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                    cfg.preprocess.normalize = (v == "true") ? true : false;
                }
                if (key == "skew_thr") cfg.preprocess.skew_thr = std::stod(value);
                if (key == "kurt_thr") cfg.preprocess.kurt_thr = std::stod(value);
            
            } else if (section == "metrics") {
                if (key == "enabled") {
                    std::string v = value;
                    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                    cfg.metrics.enabled = (v == "true") ? true : false;
                }
                if (key == "list") cfg.metrics.list = value;
                if (key == "output") cfg.metrics.output = value;
            }
        }
    }

    return true;
}