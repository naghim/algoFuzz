#ifndef PFCM_H
#define PFCM_H

#include "BaseFCM.h"
#include "EtaFCM.h" // PFCM uses EtaFCM for preprocessing
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <cmath> // For std::pow

/**
 * @brief Partitions a numeric dataset using the Possibilistic Fuzzy C-Means Clustering (PFCM) algorithm.
 */
class PFCM : public BaseFCM {
public:
    PFCM(int num_clusters, int max_iter, float m, int preprocess_iter = 15, float p = 2.0f, float w_pos = 1.0f, float w_prob = 1.0f, float noise = 0.0f);

    void setParameters(const std::unordered_map<std::string, double>& params) override;

    void fit(const Eigen::MatrixXd& X);

protected:
    int preprocess_iter; ///< Number of preprocessing iterations. The default value is 15. Must be greater than or equal to 1.
    float p;             ///< The fuzzy exponent parameter. The default value is 2.0. Must be greater than 1.
    float w_pos;         ///< Balancing factor, controls the influence of the possibilistic membership t in the clustering process. The default value is 1.0. Must be greater than 0.
    float w_prob;        ///< Balancing factor, controls the influence of the probabilistic membership u in the clustering process. The default value is 1.0. Must be greater than 0.
    Eigen::VectorXd eta_values; ///< Stores the eta values calculated during preprocessing.

};

#endif // PFCM_H