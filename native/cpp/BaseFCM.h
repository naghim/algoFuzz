#ifndef BASEFCM_H
#define BASEFCM_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <unordered_map>

/**
 * @brief Base class of all FCM implementations.
 *
 * Inheriting from this class provides default implementations of:
 * - Centroid initialization
 * - Checking if the model is trained
 * - Getting information such as the membership matrix, cluster labels, and cluster eta values
 */
class BaseFCM {
public:
    BaseFCM(int num_clusters, int max_iter, float m)
    : num_clusters(num_clusters), max_iter(max_iter), m(m), trained(false), centroids_set(false) {}

    virtual ~BaseFCM() = default; // Virtual destructor

    virtual void setParameters(const std::unordered_map<std::string, double>& params) {
        if (params.find("num_clusters") != params.end()) num_clusters = static_cast<int>(params.at("num_clusters"));
        if (params.find("max_iter") != params.end()) max_iter = static_cast<int>(params.at("max_iter"));
        if (params.find("m") != params.end()) m = static_cast<float>(params.at("m"));
    }

    virtual void fit(Eigen::MatrixXd &X) {
        throw std::runtime_error("fit() must be implemented by derived classes.");
    }

    void setCentroids(const Eigen::MatrixXd& initial_centroids) {
        centroids = initial_centroids;
        centroids_set = true;
    }

    bool isTrained() const { return trained; }
    Eigen::MatrixXd getCentroids() const { return centroids; }
    Eigen::MatrixXd getMember() const { return member; }
    std::vector<int> getPredictedLabels() const { return predictedLabels; }

protected:
    int num_clusters; ///< The number of clusters to form. The default value is 5. Must be greater than 0.
    int max_iter;     ///< The maximum number of iterations to perform. The default value is 150. Must be greater than 0.
    float m;          ///< The fuzzifier parameter. A value of 1.0 corresponds to hard clustering, while a value greater than 1.0 corresponds to soft clustering. The default value is 2.0. Must be greater than 1.0.
    bool trained;     ///< A flag indicating whether the model has been trained. The default value is False.
    bool centroids_set;

    Eigen::MatrixXd centroids;
    Eigen::MatrixXd member;
    std::vector<int> predictedLabels;
};

#endif // BASEFCM_H