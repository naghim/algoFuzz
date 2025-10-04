#include "CentroidStrategy.h"
#include <iostream>

void CentroidUtility::initializeClusters(const Eigen::MatrixXd &X, Eigen::MatrixXd &centroids, int num_clusters, CentroidStrategy clusterStrategy) {
    centroids = Eigen::MatrixXd::Zero(X.rows(), num_clusters);
    //std::cout << X.rows() << "*" << num_clusters << std::endl;

    if (clusterStrategy == CentroidStrategy::MIRTILL) {
        for (int d = 0; d < num_clusters; ++d) {
            double val = (double) d / (num_clusters - 1);
            centroids.col(d).setConstant(val);
        }
    } else if (clusterStrategy == CentroidStrategy::IRIS_DIAGONAL) {
        
        /*
        for (int d = 0; d < X.rows(); ++d) {
            centroids(d, 0) = (1 - std::pow(-1, d)) / 2;
            centroids(d, 1) = 0.5;
            centroids(d, 2) = (1 + std::pow(-1, d)) / 2;
        }
        */
    } else if (clusterStrategy == CentroidStrategy::K_MEANS) {
        // nincs most arra idonk
    } else {
        throw std::runtime_error("Cluster strategy not implemented");
    }

    bool debug = false;

    if (debug) {
        std::cout << "Centroids matrix:" << std::endl;
        std::cout << centroids << std::endl;
    }
}