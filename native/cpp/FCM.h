#ifndef FCM_H
#define FCM_H

#include "BaseFCM.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <cmath> // For std::pow

/**
 * @brief Partitions a numeric dataset using the Fuzzy C-Means (FCM) algorithm.
 */
class FCM : public BaseFCM {
public:
    FCM(int num_clusters, int max_iter, float m, float kappa = 1.0f, float noise = 0.0f);

    void setParameters(const std::unordered_map<std::string, double>& params) override;

    virtual void fit(const Eigen::MatrixXd &X);

protected:
    float kappa; ///< Regulates the severity of the penalty factor eta. The default value is 1.0. Must be greater than or equal to 1.

    // Helper function to calculate the initial sum for membership update, specific to FCPlus1M
    virtual float calculate_initial_sum();
};

#endif // FCM_H