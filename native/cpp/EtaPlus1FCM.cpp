#include "EtaPlus1FCM.h"
#include <iostream>
#include <limits> // For numeric_limits
#include <random> // For random_device, mt19937, uniform_int_distribution

EtaPlus1FCM::EtaPlus1FCM(int num_clusters, int max_iter, float m, float kappa, float noise)
    : FCPlus1M(num_clusters, max_iter, m, kappa, noise)
{
    eta_values.resize(num_clusters);
    eta_values.setZero();
}

void EtaPlus1FCM::setParameters(const std::unordered_map<std::string, double> &params)
{
    FCPlus1M::setParameters(params); // Call base class method
    // No additional parameters specific to EtaPlus1FCM in the Python class,
    // but if there were, they would be handled here.
}

void EtaPlus1FCM::fit(const Eigen::MatrixXd &X_in)
{
    // Call the base FCM's fit method to perform the clustering
    FCPlus1M::fit(X_in);

    // After FCM.fit, member and centroids are populated.
    // Now calculate eta for each cluster.
    int n = X_in.cols(); // Number of samples

    eta_values.resize(num_clusters);

    for (int i = 0; i < num_clusters; ++i)
    {
        float up = 0.0f;
        float dn = 0.0f;

        for (int k = 0; k < n; ++k)
        {
            float membership_pow_m = std::pow(member(i, k), m);
            up += membership_pow_m * (X_in.col(k) - centroids.col(i)).squaredNorm(); // Squared Euclidean distance
            dn += membership_pow_m;
        }

        if (dn > std::numeric_limits<float>::epsilon())
        {
            eta_values(i) = (up / dn) * kappa;
        }
        else
        {
            eta_values(i) = 0.0f; // Handle division by zero
        }
    }
}