#include "dataset.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

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