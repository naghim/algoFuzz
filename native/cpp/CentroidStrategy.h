#ifndef CENTROID_STRATEGY_H
#define CENTROID_STRATEGY_H

#include <Eigen/Dense>

enum CentroidStrategy {
    RANDOM_POINTS,
    DIAGONAL,
    IRIS_DIAGONAL,
    MIRTILL,
    K_MEANS
};

class CentroidUtility {
    public:
        static void initializeClusters(const Eigen::MatrixXd &X, Eigen::MatrixXd &centroids, int num_clusters, CentroidStrategy clusterStrategy);
};

#endif // CENTROID_STRATEGY_H