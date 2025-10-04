#ifndef STPFCM_H
#define STPFCM_H

#include "CentroidStrategy.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <unordered_map>

class STPFCM {
public:
    STPFCM(int num_clusters, int max_iter, float m = 2.0, float p = 2.0, float kappa = 1.0, float w_prob = 1.0)
    : num_clusters(num_clusters), max_iter(max_iter), m(m), p(p), kappa(kappa), w_prob(w_prob), trained(false) {}

    void setParameters(const std::unordered_map<std::string, double>& params) {
        if (params.find("num_clusters") != params.end()) num_clusters = params.at("num_clusters");
        if (params.find("m") != params.end()) m = params.at("m");
        if (params.find("p") != params.end()) p = params.at("p");
        if (params.find("kappa") != params.end()) kappa = params.at("kappa");
        if (params.find("w_prob") != params.end()) w_prob = params.at("w_prob");
    }
    void fit(Eigen::MatrixXd &X);
    void setCentroids(const Eigen::MatrixXd& initial_centroids);
    bool isTrained() const { return trained; }
    Eigen::MatrixXd getCentroids() const { return centroids; }
    Eigen::MatrixXd getMember() const { return member; }
    Eigen::VectorXd getAlpha() const { return alpha; }
    Eigen::MatrixXd getEta() const { return eta; }
    std::vector<int> getPredictedLabels() const { return predictedLabels; };
private:
    int num_clusters;
    int num_points;
    int max_iter;
    float m, p, kappa, w_prob;
    bool trained;
    bool centroids_set;

    Eigen::MatrixXd centroids;
    Eigen::MatrixXd eta;
    Eigen::MatrixXd member;
    Eigen::VectorXd alpha;

    std::vector<int> predictedLabels;
};

#endif // STPFCM_H