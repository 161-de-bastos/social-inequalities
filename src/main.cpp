#include "config.hpp"
#include "neighbors/factory.hpp"
#include "dbscan.hpp"
#include "dataset.hpp"
#include "kmeans.hpp"
#include "metrics.hpp"

#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    // 1. Leer YAML
    Config cfg;
    if (!load_config("config.yaml", cfg)) {
        std::cerr << "No se pudo cargar config.yaml. Usando defaults.\n";
    }

    // 2. Elegir modo de paralelización
    ComputeMode mode = ComputeMode::Serial;
    if (cfg.parallel.mode == "omp") {
        mode = ComputeMode::Omp;
    }

    #ifdef _OPENMP
        if (mode == ComputeMode::Omp && cfg.parallel.num_threads > 0) {
            omp_set_num_threads(cfg.parallel.num_threads);
        }
    #endif

    // 3. Cargar dataset
    Dataset X;

    if (!load_csv(cfg.data.path, X, cfg.data.start_col)) {
        std::cerr << "Dataset vacío. Carga tus datos en X.\n";
        return 1;
    }

    const std::size_t n_points = X.size();
    const std::size_t dim = X.empty() ? 0 : X[0].size();

    std::cout << "[Main] Dataset: " << n_points
              << " filas, " << dim << " columnas\n";

    preprocess_dataset(X, cfg.preprocess);

    const bool want_time = metrics_wants(cfg.metrics, "time");
    const bool want_memory = metrics_wants(cfg.metrics, "memory");
    const bool want_silhouette = metrics_wants(cfg.metrics, "silhouette");

    std::string algorithm;
    std::vector<label_t> labels;
    auto t_start = std::chrono::steady_clock::now();

    if (cfg.task == "dbscan") {
        algorithm = "dbscan";
        auto ns = make_backend(
            X,
            cfg.backend,   // brute | kdtree | balltree
            mode           // serial | omp
        );

        DBSCANParams params;
        params.eps = cfg.dbscan.eps;
        params.minPts = cfg.dbscan.minPts;

        DBSCAN dbscan(X, std::move(ns));
        DBSCANResult result = dbscan.fit(params);
        labels = result.labels;

        std::cout << "[DBSCAN] Clusters encontrados: " << result.n_clusters << "\n";
        std::cout << "[DBSCAN] Ruido: " << result.n_noise << "\n";
        std::cout << "[DBSCAN] Puntos: " << labels.size() << "\n";

        if (!save_labels_csv("clusters.csv", labels)) {
            std::cerr << "No se pudo guardar clusters.csv\n";
        }

    } else if (cfg.task == "kmeans") {
        algorithm = "kmeans";
        KMeansParams params;
        params.k = cfg.kmeans.k;
        params.max_iters = cfg.kmeans.max_iters;

        KMeans kmeans(X, mode);
        KMeansResult result = kmeans.fit(params);
        labels = result.labels;

        std::cout << "[KMeans] k=" << params.k
                  << ", iters=" << params.max_iters << "\n";
        std::cout << "[KMeans] Puntos: " << labels.size() << "\n";

        if (!save_labels_csv("clusters.csv", labels)) {
            std::cerr << "No se pudo guardar clusters.csv\n";
        }
    } else {
        std::cerr << "ERROR: tarea desconocida: " << cfg.task << "\n";
        return 1;
    }

    auto t_end = std::chrono::steady_clock::now();

    if (cfg.metrics.enabled) {
        double time_s = 0.0;
        double mem_mb = -1.0;
        real_t sil    = real_t(-1);

        if (want_time) time_s = std::chrono::duration<double>(t_end - t_start).count();
        if (want_memory) mem_mb = get_memory_usage();
        if (want_silhouette && !labels.empty()) sil = silhouette_score(X, labels);
        
        std::string mode_str = (mode == ComputeMode::Omp) ? "omp" : "serial";
        int num_threads = 0;
        #ifdef _OPENMP
            if (mode == ComputeMode::Omp) {
                num_threads = cfg.parallel.num_threads;
            }
        #endif

        append_log(
            cfg.metrics.output,
            algorithm,
            cfg.task,
            (cfg.task == "dbscan" ? cfg.backend : std::string("-")),
            mode_str,
            num_threads,
            n_points,
            dim,
            want_time,
            time_s,
            want_memory,
            mem_mb,
            want_silhouette,
            sil
        );
    }

    return 0;
}
