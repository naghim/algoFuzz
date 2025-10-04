#include "FCM.h"
#include <iostream>
#include <limits> // For numeric_limits
#include <random> // For random_device, mt19937, uniform_int_distribution

FCM::FCM(int num_clusters, int max_iter, float m, float kappa, float noise)
    : BaseFCM(num_clusters, max_iter, m, noise), kappa(kappa) {
    // Additional initialization specific to FCM if needed
}

void FCM::setParameters(const std::unordered_map<std::string, double>& params) {
    BaseFCM::setParameters(params); // Call base class method
    if (params.find("kappa") != params.end()) {
        kappa = static_cast<float>(params.at("kappa"));
    }
}

void FCM::fit(const Eigen::MatrixXd& X_in) {
    if (!centroids_set) {
        throw std::runtime_error("Centroids must be set before calling fit. Use setCentroids() first.");
    }

    Eigen::MatrixXd X = getXWithNoise(X_in);

    int z = X.rows(); // Number of features (dimensions)
    int n = X.cols(); // Number of samples (data points)

    member.resize(num_clusters, n);
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(num_clusters, n);
    float corrected_m = -2.0f / (m - 1.0f);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Update membership matrix (u)
        for (int k = 0; k < n; ++k) { // Iterate over each data point
            int exact_match_cluster = -1;
            for (int i = 0; i < num_clusters; ++i) { // Iterate over each cluster
                if ((X.col(k) - centroids.col(i)).norm() < 0.0000001f) {
                    exact_match_cluster = i;
                    break;
                }
            }

            if (exact_match_cluster != -1) {
                u.col(k) = Eigen::VectorXd::Zero(num_clusters);
                u(exact_match_cluster, k) = 1.0f;
                continue;
            }

            float sum_val = calculate_initial_sum(); // Initialize sum with initial terms
            for (int i = 0; i < num_clusters; ++i) {
                float dist = (X.col(k) - centroids.col(i)).norm();
                if (dist < std::numeric_limits<float>::epsilon()) { // Handle division by zero
                    u(i, k) = std::numeric_limits<float>::infinity();
                } else {
                    u(i, k) = std::pow(dist, corrected_m);
                }
                sum_val += u(i, k);
            }

            for (int i = 0; i < num_clusters; ++i) {
                if (sum_val > std::numeric_limits<float>::epsilon()) {
                    u(i, k) /= sum_val;
                } else {
                    u(i, k) = 0.0f; // Or some other appropriate handling for sum_val being zero
                }
            }
        }

        // Update centroids
        for (int i = 0; i < num_clusters; ++i) {
            Eigen::VectorXd sum_up = Eigen::VectorXd::Zero(z);
            float sum_dn = 0.0f;

            for (int k = 0; k < n; ++k) {
                float sum_cur = std::pow(u(i, k), m);
                sum_up += sum_cur * X.col(k);
                sum_dn += sum_cur;
            }
            
            if (sum_dn > std::numeric_limits<float>::epsilon()) {
                centroids.col(i) = sum_up / sum_dn;
            } else {
                // Handle case where sum_dn is zero, e.g., reinitialize centroid or keep previous
                // For now, we'll keep the previous centroid value.
            }
        }
    }

    // After iterations, set member matrix and trained flag
    member = u;
    trained = true;

    // Calculate predicted labels
    predictedLabels.clear();
    predictedLabels.reserve(n);
    for (int k = 0; k < n; ++k) {
        Eigen::MatrixXf::Index max_row;
        u.col(k).maxCoeff(&max_row);
        predictedLabels.push_back(static_cast<int>(max_row));
    }
}

float FCM::calculate_initial_sum() {
    return 0;
}