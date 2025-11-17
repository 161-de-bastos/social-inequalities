#include "config.hpp"
#include "neighbors/factory.hpp"
#include "dbscan.hpp"
#include "dataset.hpp"
#include "dataset.cpp"
#include "kmeans.hpp"

#include <iostream>

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

    std::cout << "Dataset cargado: "
            << X.size() << " filas, "
            << (X.empty() ? 0 : X[0].size())
            << " columnas\n";

    preprocess_dataset(X, cfg.preprocess);

    if (cfg.task == "dbscan") {
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

        std::cout << "[DBSCAN] Clusters encontrados: " << result.n_clusters << "\n";
        std::cout << "[DBSCAN] Ruido: " << result.n_noise << "\n";
        std::cout << "[DBSCAN] Puntos: " << result.labels.size() << "\n";

        if (!save_labels_csv("clusters.csv", result.labels)) {
            std::cerr << "No se pudo guardar clusters.csv\n";
        }

        return 0;

    } else if (cfg.task == "kmeans") {
        KMeansParams params;
        params.k = cfg.kmeans.k;
        params.max_iters = cfg.kmeans.max_iters;

        KMeans kmeans(X, mode);
        KMeansResult result = kmeans.fit(params);

        
    }


   
}
