#ifndef FP3CM_H
#define FP3CM_H

#include "BaseFCM.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <cmath> // For std::pow

/**
 * @brief Partitions a numeric dataset using the Fuzzy Possibilistic Product Partition C-Means (FP3CM).
 */
class FP3CM : public BaseFCM
{
public:
    FP3CM(int num_clusters, int max_iter = 150, float m = 2.0f, float p = 2.0f, float eta = 0.1f, float noise = 0.0f);

    void setParameters(const std::unordered_map<std::string, double> &params) override;
    static std::vector<std::string> getParameterNames();

    void fit(const Eigen::MatrixXd &X);

protected:
    float p;   ///< The fuzzy exponent parameter. The default value is 2.0. Must be greater than 1.
    float eta; ///< The penalty factor for the noise cluster. The default value is 0.1. Must be greater than or equal to 1e-9.
};

#endif // FP3CM_H