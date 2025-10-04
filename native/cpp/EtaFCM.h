#ifndef ETAFCM_H
#define ETAFCM_H

#include "FCM.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>

/**
 * @brief An extension of the FCM model that includes a penalty term (eta) for each cluster.
 */
class EtaFCM : public FCM {
public:
    EtaFCM(int num_clusters, int max_iter, float m, float kappa = 1.0f, float noise = 0.0f);

    void setParameters(const std::unordered_map<std::string, double>& params) override;

    void fit(const Eigen::MatrixXd& X) override;

    Eigen::VectorXd getEta() const { return eta_values; }

protected:
    Eigen::VectorXd eta_values; ///< The penalty factor (eta) for each cluster.
};

#endif // ETAFCM_H