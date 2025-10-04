#ifndef GFPCM_H
#define GFPCM_H

#include "CentroidStrategy.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

class GFPCM {
public:
    GFPCM(int num_clusters, int max_iter, float m = 2.0, float p = 2.0, float w_prob = 1.0, float noise = 0.0, CentroidStrategy centroidStrategy = CentroidStrategy::RANDOM_POINTS)
    : num_clusters(num_clusters), max_iter(max_iter), m(m), p(p), w_prob(w_prob), noise(noise), trained(false), centroidStrategy(centroidStrategy) {}

    void setParameters(const std::unordered_map<std::string, double>& params) {
        if (params.find("num_clusters") != params.end()) num_clusters = params.at("num_clusters");
        if (params.find("m") != params.end()) m = params.at("m");
        if (params.find("p") != params.end()) p = params.at("p");
        if (params.find("w_prob") != params.end()) w_prob = params.at("w_prob");
        if (params.find("noise") != params.end()) noise = params.at("noise");
    }
    void fit(Eigen::MatrixXd X, std::vector<int> &trueLabels);
    bool isTrained() const { return trained; }
    Eigen::MatrixXd getCentroids() const { return centroids; }
    Eigen::MatrixXd getMember() const { return member; }
    std::vector<int> getPredictedLabels() const { return predictedLabels; };
private:
    int num_clusters;
    int num_points;
    int max_iter;
    float m, p, w_prob, noise;
    bool trained;

    CentroidStrategy centroidStrategy;

    Eigen::MatrixXd centroids;
    Eigen::MatrixXd eta;
    Eigen::MatrixXd member;

    std::vector<int> predictedLabels;
};

#endif // GFPCM_H