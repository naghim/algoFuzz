#ifndef FCPLUS1M_H
#define FCPLUS1M_H

#include "FCM.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <cmath> // For std::pow

/**
 * @brief Partitions a numeric dataset using the F(C+1)M algorithm.
 *        This is an extension of the Fuzzy C-Means algorithm with an extra noise cluster.
 */
class FCPlus1M : public FCM
{
public:
    FCPlus1M(int num_clusters, int max_iter = 150, float m = 2.0f, float kappa = 1.0f, float eta = 2.5f, float noise = 0.0f);

    void setParameters(const std::unordered_map<std::string, double> &params) override;
    static std::vector<std::string> getParameterNames();

protected:
    float eta; ///< The penalty factor for the noise cluster. The default value is 2.5.

    // Helper function to calculate the initial sum for membership update, specific to FCPlus1M
    float calculate_initial_sum() override;
};

#endif // FCPLUS1M_H