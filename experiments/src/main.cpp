#include "config.hpp"
#include "neighbors/factory.hpp"
#include "dbscan.hpp"
#include "dataset.hpp"

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

    // 3. Aplicar número de hilos si corresponde
    #ifdef _OPENMP
        if (mode == ComputeMode::Omp && cfg.parallel.num_threads > 0) {
            omp_set_num_threads(cfg.parallel.num_threads);
        }
    #endif

    // 4. Cargar dataset (reemplaza con tu carga real)
    Dataset X;

    if (!load_csv(cfg.data.path, X, cfg.data.start_col)) {
        std::cerr << "Dataset vacío. Carga tus datos en X.\n";
        return 1;
    }

    std::cout << "Dataset cargado: "
            << X.size() << " filas, "
            << (X.empty() ? 0 : X[0].size())
            << " columnas\n";

    // 5. Crear backend según YAML
    auto ns = make_backend(
        X,
        cfg.backend,   // brute | kdtree | balltree
        mode           // serial | omp
    );

    // 6. Configurar DBSCAN
    DBSCANParams params;
    params.eps    = cfg.dbscan.eps;
    params.minPts = cfg.dbscan.minPts;

    DBSCAN dbscan(X, std::move(ns));

    // 7. Ejecutar
    DBSCANResult result = dbscan.fit(params);

    // 8. Mostrar salida mínima
    std::cout << "Clusters encontrados: " << result.n_clusters << "\n";
    std::cout << "Ruido: " << result.n_noise << "\n";
    std::cout << "Puntos: " << result.labels.size() << "\n";

    // 9. Exportar CSV con id y cluster
    if (!save_labels_csv("clusters.csv", result.labels)) {
        std::cerr << "No se pudo guardar clusters.csv\n";
    }

    return 0;
}
