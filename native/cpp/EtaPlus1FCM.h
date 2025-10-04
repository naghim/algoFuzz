#ifndef ETAPLUS1FCM_H
#define ETAPLUS1FCM_H

#include "FCPlus1M.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>

/**
 * @brief An extension of the FCM model that includes a penalty term (eta) plus one for each cluster.
 */
class EtaPlus1FCM : public FCPlus1M
{
public:
    EtaPlus1FCM(int num_clusters, int max_iter = 150, float m = 2.0f, float kappa = 1.0f, float noise = 0.0f);

    void setParameters(const std::unordered_map<std::string, double> &params) override;

    void fit(const Eigen::MatrixXd &X) override;

    Eigen::VectorXd getEta() const { return eta_values; }

protected:
    Eigen::VectorXd eta_values; ///< The penalty factor (eta) for each cluster.
};

#endif // ETAPLUS1FCM_H