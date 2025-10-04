#include "EtaFCM.h"
#include <iostream>
#include <limits> // For numeric_limits
#include <random> // For random_device, mt19937, uniform_int_distribution

EtaFCM::EtaFCM(int num_clusters, int max_iter, float m, float kappa, float noise)
    : FCM(num_clusters, max_iter, m, kappa, noise) {
    eta_values.resize(num_clusters);
    eta_values.setZero();
}

void EtaFCM::setParameters(const std::unordered_map<std::string, double>& params) {
    FCM::setParameters(params); // Call base class method
    // No additional parameters specific to EtaFCM in the Python class,
    // but if there were, they would be handled here.
}

void EtaFCM::fit(const Eigen::MatrixXd& X_in) {
    // Call the base FCM's fit method to perform the clustering
    FCM::fit(X_in);

    // After FCM.fit, member and centroids are populated.
    // Now calculate eta for each cluster.
    int n = X_in.cols(); // Number of samples

    eta_values.resize(num_clusters);

    for (int i = 0; i < num_clusters; ++i) {
        float up = 0.0f;
        float dn = 0.0f;

        for (int k = 0; k < n; ++k) {
            float membership_pow_m = std::pow(member(i, k), m);
            up += membership_pow_m * (X_in.col(k) - centroids.col(i)).squaredNorm(); // Squared Euclidean distance
            dn += membership_pow_m;
        }
        
        if (dn > std::numeric_limits<float>::epsilon()) {
            eta_values(i) = (up / dn) * kappa;
        } else {
            eta_values(i) = 0.0f; // Handle division by zero
        }
    }
}