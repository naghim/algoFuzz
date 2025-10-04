#ifndef STPFCM_H
#define STPFCM_H

#include "BaseFCM.h"
#include "CentroidStrategy.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <unordered_map>

/**
 * @brief This module contains the implementation of the ST-PFCM algorithm, which is a self-tuning version of the Possibilistic Fuzzy C-Means Clustering algorithm proposed by MB. Naghi in 2023.
 */
class STPFCM : public BaseFCM {
public:
    STPFCM(int num_clusters, int max_iter, float m = 2.0, float p = 2.0, float kappa = 1.0, float w_prob = 1.0)
    : BaseFCM(num_clusters, max_iter, m), p(p), kappa(kappa), w_prob(w_prob) {}

    void setParameters(const std::unordered_map<std::string, double>& params) override {
        BaseFCM::setParameters(params);
        if (params.find("p") != params.end()) p = static_cast<float>(params.at("p"));
        if (params.find("kappa") != params.end()) kappa = static_cast<float>(params.at("kappa"));
        if (params.find("w_prob") != params.end()) w_prob = static_cast<float>(params.at("w_prob"));
    }

    void fit(Eigen::MatrixXd &X) override;
    // setCentroids is inherited from BaseFCM
    // isTrained is inherited from BaseFCM
    // getCentroids is inherited from BaseFCM
    // getMember is inherited from BaseFCM

    Eigen::VectorXd getAlpha() const { return alpha; }
    Eigen::MatrixXd getEta() const { return eta; }

    // getPredictedLabels is inherited from BaseFCM
private:
    int num_points;
    float p; ///< The fuzzy exponent parameter. The default value is 2.0. Must be greater than 1.
    float kappa; ///< The penalty factor for the noise cluster. The default value is 1. Must be greater than or equal to 1e-9.
    float w_prob; ///< Balancing factor, controls the influence of the probabilistic membership u in the clustering process. A higher weight increases the importance of u in updating the centroids. The default value is 1.0. Must be greater than 0.

    Eigen::MatrixXd eta;
    Eigen::VectorXd alpha;
};

#endif // STPFCM_H