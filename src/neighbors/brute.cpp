#include "neighbors/brute.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

BruteForce::BruteForce(
    const Dataset& X,
    ComputeMode mode
) : X_(X), mode_(mode) {}

/* SERIAL */
void BruteForce::serial(
    int idx, 
    real_t eps,
    std::vector<int>& out_idx
) const {
    out_idx.clear();
    const int n = static_cast<int>(X_.size());
    const Point& p = X_[idx];
    const real_t eps2 = eps * eps;
    
    for (int j = 0; j < n; ++j) {
        if (j == idx) continue;
        if (euclidean_dist(p,X_[j]) <= eps2) out_idx.push_back(j);
    }
}

/* OPENMP */
#ifdef _OPENMP
void BruteForce::omp(
    int idx, 
    real_t eps,
    std::vector<int>& out_idx
) const {
    const int n = static_cast<int>(X_.size());
    if (n == 0) { out_idx.clear(); return; }

    const Point& p = X_[idx];
    const real_t eps2 = eps * eps;

    int nthreads = 1;
    #pragma omp parallel 
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
        }
    }

    std::vector<std::vector<int>> locals(nthreads);
    for (int t = 0; t < nthreads; ++t) locals[t].reserve(n / nthreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local = locals[tid];

        #pragma omp for schedule(static)
        for (int j = 0; j < n; ++j) {
            if (j == idx) continue;
            if (euclidean_dist(p,X_[j]) <= eps2) local.push_back(j);
        }
    }

    out_idx.clear();
    std::size_t total = 0;
    for (const auto& v: locals) total += v.size();
    out_idx.reserve(total);
    for (auto& v : locals) out_idx.insert(
            out_idx.end(),
            v.begin(),
            v.end()
        );
}
#endif

/* Dispatcher */
void BruteForce::radius_query(
    int idx, 
    real_t eps,
    std::vector<int>& out_idx
) const {
    switch (mode_) {
        case ComputeMode::Serial:
            serial(idx,eps,out_idx);
            break;
        
        case ComputeMode::Omp:
            #ifndef _OPENMP
                serial(idx,eps,out_idx);
            #else
                omp(idx,eps,out_idx);
            #endif
                break;

        default:
            serial(idx,eps,out_idx);
            break;
    }
}
