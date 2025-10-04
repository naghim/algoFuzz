#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric> // For std::accumulate
#include "STPFCM.h"
#include "BaseFCM.h" // Include BaseFCM header
#include "CentroidStrategy.h"
#include "Metrics.h" // Assuming Metrics.h is needed for computeConfusionMatrix
#include "DatasetLoader.h"
#include <random>

/**
 * @brief Fits the model to the data.
 *
 * @param X The input data.
 * @return None
 */
void STPFCM::fit(Eigen::MatrixXd &X_in) {
    if (!centroids_set) {
        throw std::runtime_error("Centroids must be set before calling fit. Use setCentroids() first.");
    }

    Eigen::MatrixXd X = getXWithNoise(X_in);

    std::cout << "Starting STPFCM fit with parameters:" << std::endl;

    int z = X.rows(); // Number of features (including noise if added)
    int n = X.cols(); // Number of data points

    num_points = n;

    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(num_clusters, n);
    Eigen::MatrixXd t = Eigen::MatrixXd::Zero(num_clusters, n);

    double corrected_p = 1.0 / (p - 1.0);
    double corrected_m = -2.0 / (m - 1.0);

    Eigen::VectorXd center = X.rowwise().mean(); // Mean of each feature
    double eta_sum = 0;
    for (int i = 0; i < n; ++i) {
        eta_sum += (X.col(i) - center).squaredNorm();
    }

    double eta_val = (kappa / n) * eta_sum;
    eta = Eigen::MatrixXd::Constant(num_clusters, n, eta_val); // Initialize eta as a matrix

    alpha = Eigen::VectorXd::Constant(num_clusters, 1.0 / num_clusters);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Calculate t
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < num_clusters; ++i) {
                double norm_sq = (X.col(k) - centroids.col(i)).squaredNorm();
                double exponent_base = norm_sq / (eta_val * pow(alpha(i), (m - 1.0)));
                t(i, k) = 1.0 / (1.0 + pow(exponent_base, corrected_p));
            }
        }

        // new u
        for (int k = 0; k < n; ++k) {
            int exact = -1;
            for (int i = 0; i < num_clusters; ++i) {
                if ((X.col(k) - centroids.col(i)).norm() < 0.0000001) {
                    exact = i;
                    break;
                }
            }

            if (exact != -1) {
                u.col(k) = Eigen::VectorXd::Zero(num_clusters);
                u(exact, k) = 1.0;
                continue;
            }

            Eigen::VectorXd norm_diffs(num_clusters);
            for (int i = 0; i < num_clusters; ++i) {
                norm_diffs(i) = (X.col(k) - centroids.col(i)).norm();
            }

            Eigen::VectorXd u_col_k = alpha.array() * norm_diffs.array().pow(corrected_m);
            u.col(k) = u_col_k / u_col_k.sum();
        }

        // new alpha
        Eigen::VectorXd new_alpha_sum = Eigen::VectorXd::Zero(num_clusters);
        for (int i = 0; i < num_clusters; ++i) {
            double sum_val = 0;
            for (int k = 0; k < n; ++k) {
                sum_val += (w_prob * pow(u(i, k), m) + pow(t(i, k), p)) * (X.col(k) - centroids.col(i)).squaredNorm();
            }
            new_alpha_sum(i) = pow(sum_val, (1.0 / m));
        }
        alpha = new_alpha_sum / new_alpha_sum.sum();

        // new v (centroids)
        for (int i = 0; i < num_clusters; ++i) {
            Eigen::VectorXd sumup = Eigen::VectorXd::Zero(z);
            double sumdn = 0;

            for (int k = 0; k < n; ++k) {
                double sumcur = (w_prob * pow(u(i, k), m) + pow(t(i, k), p));
                sumup += sumcur * X.col(k);
                sumdn += sumcur;
            }
            centroids.col(i) = sumup / sumdn;
        }
    }

    eta = Eigen::MatrixXd::Constant(num_clusters, n, eta_val); // Re-assign eta based on final eta_val
    for (int i = 0; i < num_clusters; ++i) {
        for (int k = 0; k < n; ++k) {
            eta(i, k) = eta_val * pow(alpha(i), (m - 1.0));
        }
    }

    member = (w_prob * u.array().pow(m) + t.array().pow(p)).matrix();
    trained = true;

    predictedLabels.clear();

    for (int k = 0; k < n; ++k) {
        int bestCluster = 0;
        double maxMembership = 0;

        for (int i = 0; i < num_clusters; ++i) {
            if (member(i, k) > maxMembership) {
                maxMembership = member(i, k);
                bestCluster = i;
            }
        }
        predictedLabels.push_back(bestCluster);
    }
}
