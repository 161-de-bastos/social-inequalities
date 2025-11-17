#include "kmeans.hpp"
#include "distance.hpp"

#include <random>
#include <limits>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

KMeans::KMeans (
    const Dataset& X,
    ComputeMode mode
) : X_(X), mode_(mode) {}


KMeansResult KMeans::fit(const KMeansParams& params) {
    KMeansResult result;

    const std::size_t n = X_.size();
    if (n == 0) {
        return result;
    }

    const std::size_t m = X_[0].size();
    const int K         = params.k;
    const int max_iters = params.max_iters;

    if (K <= 0 || max_iters <= 0) {
        std::cerr << "[KMeans] Parámetros inválidos: k y max_iters deben ser > 0\n";
        return result;
    }

    // ==============================
    // Inicialización de centroides
    // ==============================
    std::vector<Point> centroids(K, Point(m, real_t(0)));

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> uid(
        0,
        static_cast<int>(n) - 1
    );

    for (int k = 0; k < K; ++k) {
        centroids[k] = X_[uid(rng)];
    }

    std::vector<label_t> labels(n, 0);

    // ==============================
    //   Iteraciones de K-Means
    // ==============================
    for (int it = 0; it < max_iters; ++it) {

        // -----------------------------------------
        // 1) Asignación de cada punto a su centroide
        // -----------------------------------------
        if (mode_ == ComputeMode::Omp) {
        #ifdef _OPENMP
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(n); ++i) {
                const Point& xi = X_[static_cast<std::size_t>(i)];

                real_t best = std::numeric_limits<real_t>::max();
                int best_k  = 0;

                for (int k = 0; k < K; ++k) {
                    real_t d = euclidean_dist(xi, centroids[k]);
                    if (d < best) {
                        best   = d;
                        best_k = k;
                    }
                }

                labels[static_cast<std::size_t>(i)] = best_k;
            }
        #else
            // Si se compila sin OpenMP, cae al modo serial
            for (std::size_t i = 0; i < n; ++i) {
                const Point& xi = X_[i];

                real_t best = std::numeric_limits<real_t>::max();
                int best_k  = 0;

                for (int k = 0; k < K; ++k) {
                    real_t d = euclidean_dist(xi, centroids[k]);
                    if (d < best) {
                        best   = d;
                        best_k = k;
                    }
                }

                labels[i] = best_k;
            }
        #endif
        } else {
            // Serial
            for (std::size_t i = 0; i < n; ++i) {
                const Point& xi = X_[i];

                real_t best = std::numeric_limits<real_t>::max();
                int best_k  = 0;

                for (int k = 0; k < K; ++k) {
                    real_t d = euclidean_dist(xi, centroids[k]);
                    if (d < best) {
                        best   = d;
                        best_k = k;
                    }
                }

                labels[i] = best_k;
            }
        }

        // -----------------------------------------
        // 2) Recalcular centroides
        // -----------------------------------------
        std::vector<Point> new_centroids(K, Point(m, real_t(0)));
        std::vector<int> counts(K, 0);

        if (mode_ == ComputeMode::Omp) {
        #ifdef _OPENMP
            #pragma omp parallel
            {
                std::vector<Point> localC(K, Point(m, real_t(0)));
                std::vector<int> localCount(K, 0);

                #pragma omp for nowait
                for (int i = 0; i < static_cast<int>(n); ++i) {
                    int k = labels[static_cast<std::size_t>(i)];
                    const Point& xi = X_[static_cast<std::size_t>(i)];

                    localCount[k]++;
                    for (std::size_t j = 0; j < m; ++j) {
                        localC[k][j] += xi[j];
                    }
                }

                #pragma omp critical
                {
                    for (int k = 0; k < K; ++k) {
                        counts[k] += localCount[k];
                        for (std::size_t j = 0; j < m; ++j) {
                            new_centroids[k][j] += localC[k][j];
                        }
                    }
                }
            }
        #else
            // Sin OpenMP -> serial
            for (std::size_t i = 0; i < n; ++i) {
                int k = labels[i];
                const Point& xi = X_[i];
                counts[k]++;
                for (std::size_t j = 0; j < m; ++j) {
                    new_centroids[k][j] += xi[j];
                }
            }
        #endif
        } else {
            // Serial
            for (std::size_t i = 0; i < n; ++i) {
                int k = labels[i];
                const Point& xi = X_[i];
                counts[k]++;
                for (std::size_t j = 0; j < m; ++j) {
                    new_centroids[k][j] += xi[j];
                }
            }
        }

        // -----------------------------------------
        // 3) Promediar acumulaciones
        // -----------------------------------------
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                real_t inv = real_t(1) / static_cast<real_t>(counts[k]);
                for (std::size_t j = 0; j < m; ++j) {
                    new_centroids[k][j] *= inv;
                }
            } else {
                // clúster vacío: re-inicializar al azar
                new_centroids[k] = X_[static_cast<std::size_t>(uid(rng))];
            }
        }

        centroids.swap(new_centroids);
    }

    result.labels = std::move(labels);
    return result;
}