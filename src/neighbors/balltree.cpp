#include "neighbors/balltree.hpp"
#include <algorithm>
#include <cmath>

BallTree::BallTree(
    const Dataset& X,
    ComputeMode mode
) : X_(X), mode_(mode) {
    const int n = static_cast<int>(X_.size());
    dim_ = (n > 0) ? static_cast<int>(X_[0].size()) : 0;

    if (n == 0 || dim_ == 0) { root_.reset(); return; }

    std::vector<int> idxs(n);
    for (int i = 0; i < n; ++i) idxs[i] = i;
    
    switch (mode_) {
        case ComputeMode::Serial:
            root_ = build_serial(idxs, 0);
            break;
        
        case ComputeMode::Omp:
            #pragma omp parallel
            {
                #pragma omp single
                {
                    root_ = build_omp(idxs, 0);
                }
            }
            break;

        default:
            root_ = build_serial(idxs, 0);
            break;
    }
}

void BallTree::compute_center(
    Node* node
) {
    node->center.assign(dim_, 0.0);
    for (int idx : node->indexi) {
        const Point& p = X_[idx];
        for (int d = 0; d < dim_; ++d) node->center[d] += p[d];
    }

    if (!node->indexi.empty()) {
        const real_t invn = 1.0 / static_cast<real_t>(node->indexi.size());
        for (int d = 0; d < dim_; ++d) node->center[d] *= invn;
    }

    node->radius = 0.0;
    for (int idx : node->indexi) {
        real_t d2 = euclidean_dist(node->center, X_[idx]);
        if (d2 > node->radius) node->radius = d2;
    }
    node->radius = std::sqrt(node->radius);
}

/* Serial */
std::unique_ptr<BallTree::Node> BallTree::build_serial(
    const std::vector<int>& idxs,
    int depth
) {
    auto node = std::make_unique<Node>();
    node->left = node->right = nullptr;
    node->indexi = idxs;
    compute_center(node.get());

    if ((int)idxs.size() <= leaf_size_) return node;

    int i0 = idxs[0];
    int far1 = i0;
    real_t maxd = 0.0;
    for (int i : idxs) {
        real_t d = euclidean_dist(X_[i0], X_[i]);
        if (d > maxd) { maxd = d; far1 = i; }
    }

    int far2 = far1;
    maxd = 0.0;
    for (int i : idxs) {
        real_t d = euclidean_dist(X_[far1], X_[i]);
        if (d > maxd) { maxd = d; far2 = i; }
    }

    std::vector<int> left_idxs, right_idxs;
    left_idxs.reserve(idxs.size());
    right_idxs.reserve(idxs.size());

    for (int i : idxs) {
        real_t d1 = euclidean_dist(X_[i], X_[far1]);
        real_t d2 = euclidean_dist(X_[i], X_[far2]);
        if (d1 < d2) left_idxs.push_back(i);
        else right_idxs.push_back(i);
    }

    if (left_idxs.empty() || right_idxs.empty()) return node;

    node->left = build_serial(left_idxs, depth + 1).release();
    node->right = build_serial(right_idxs, depth + 1).release();
    return node;
}

/* OpenMP */
std::unique_ptr<BallTree::Node> BallTree::build_omp(
    const std::vector<int>& idxs,
    int depth
) {
    auto node = std::make_unique<Node>();
    node->left = node->right = nullptr;
    node->indexi = idxs;
    compute_center(node.get());

    if ((int)idxs.size() <= leaf_size_) return node;

    int i0 = idxs[0];
    int far1 = i0;
    real_t maxd = 0.0;
    for (int i : idxs) {
        real_t d = euclidean_dist(X_[i0], X_[i]);
        if (d > maxd) { maxd = d; far1 = i; }
    }

    int far2 = far1;
    maxd = 0.0;
    for (int i : idxs) {
        real_t d = euclidean_dist(X_[far1], X_[i]);
        if (d > maxd) { maxd = d; far2 = i; }
    }

    std::vector<int> left_idxs, right_idxs;
    left_idxs.reserve(idxs.size());
    right_idxs.reserve(idxs.size());

    for (int i : idxs) {
        real_t d1 = euclidean_dist(X_[i], X_[far1]);
        real_t d2 = euclidean_dist(X_[i], X_[far2]);
        if (d1 < d2) left_idxs.push_back(i);
        else right_idxs.push_back(i);
    }

    if (left_idxs.empty() || right_idxs.empty()) return node;

    const int threshold = 1024;
    if (static_cast<int>(idxs.size()) > threshold) {
        #pragma omp task shared(node)
        {
            node->left = build_omp(left_idxs, depth + 1).release();
        }
        #pragma omp task shared(node)
        {
            node->right = build_omp(right_idxs, depth + 1).release();
        }
        #pragma omp taskwait

    } else {
        node->left = build_serial(left_idxs, depth + 1).release();
        node->right = build_serial(right_idxs, depth + 1).release();
    }

    return node;
}

/* Serial searches */
void BallTree::radius_node(
    const Node* node,
    const Point& query,
    real_t eps2,
    std::vector<int>& out
) const {
    if (!node) return;

    real_t center_dist2 = euclidean_dist(query, node->center);
    real_t max_dist = node->radius + std::sqrt(eps2);
    if (center_dist2 > max_dist * max_dist) return;

    if (!node->left && !node->right) {
        for (int idx : node->indexi) {
            real_t d2 = euclidean_dist(query, X_[idx]);
            if (d2 <= eps2) out.push_back(idx);
        }
        return;
    }

    radius_node(node->left, query, eps2, out);
    radius_node(node->right, query, eps2, out);
}

void BallTree::radius_query(
    int idx, 
    real_t eps,
    std::vector<int>& out_idx
) const {
    out_idx.clear();
    if (!root_) return;
    const Point& query = X_[idx];
    real_t eps2 = eps * eps;
    radius_node(root_.get(), query, eps2, out_idx);

    out_idx.erase(
        std::remove(
            out_idx.begin(),
            out_idx.end(),
            idx
        ),
        out_idx.end()
    );
}