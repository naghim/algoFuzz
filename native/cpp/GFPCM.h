#ifndef GFPCM_H
#define GFPCM_H

#include "BaseFCM.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <cmath> // For std::pow

/**
 * @brief Partitions a numeric dataset using the Generalized Fuzzy-Possibilistic C-Means Clustering (GFPCM) algorithm.
 */
class GFPCM : public BaseFCM {
public:
    GFPCM(int num_clusters, int max_iter, float m, float p = 2.0f, float w_prob = 1.0f, float noise = 0.0f);

    void setParameters(const std::unordered_map<std::string, double>& params) override;

    void fit(const Eigen::MatrixXd& X);

protected:
    float p;      ///< The fuzzy exponent parameter. The default value is 2.0. Must be greater than 1.
    float w_prob; ///< Serves as a balancing factor: w_prob is used to weigh the influence of the second membership term t (possibilistic) relative to u (probabilistic). The default value is 1.0. Must be greater than or equal to 1.
};

#endif // GFPCM_H