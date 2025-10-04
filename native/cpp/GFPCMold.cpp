#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include "GFPCM.h"
#include "CentroidStrategy.h"
#include <random>

    void GFPCM::fit(Eigen::MatrixXd X) {
        int actualN = X.cols();

        if (noise > 0.0) {
            Eigen::MatrixXd noise_vector = Eigen::MatrixXd::Constant(X.rows(), 1, noise);
            X.conservativeResize(Eigen::NoChange, X.cols() + 1);
            X.col(X.cols() - 1) = noise_vector;
        }

        int z = X.rows();
        int n = X.cols();

        num_points = n;

        Eigen::MatrixXd u = Eigen::MatrixXd::Zero(num_clusters, n);
        Eigen::MatrixXd t = Eigen::MatrixXd::Zero(num_clusters, n);
        Eigen::MatrixXd d = Eigen::MatrixXd::Zero(num_clusters, n);

        double deriv_m = -2.0 / (m - 1);
        double deriv_p = -2.0 / (p - 1);

        CentroidUtility::initializeClusters(X, centroids, z, centroidStrategy);

        for (int iter = 0; iter < max_iter; ++iter) {
            d.setZero();

            // Calculate new d
            for (int i = 0; i < num_clusters; ++i) {
                for (int k = 0; k < n; ++k) {
                    d(i, k) = (X.col(k) - centroids.col(i)).squaredNorm();
                }
            }

            // Update u
            for (int k = 0; k < n; ++k) {
                if (d.col(k).minCoeff() < 1e-7) {
                    u.col(k).setZero();
                    for (int i = 0; i < num_clusters; ++i) {
                        if (d(i, k) < 1e-7) u(i, k) = 1;
                    }
                } else {
                    double s = pow(d.col(k).array(), deriv_m).sum();
                    for (int i = 0; i < num_clusters; ++i) {
                        u(i, k) = pow(d(i, k), deriv_m) / s;
                    }
                }
            }

            // Update t
            for (int i = 0; i < num_clusters; ++i) {
                if ((d.row(i).array() < 1e-7).count() > 0) {
                    t.row(i).setZero();

                    for (int k = 0; k < n; ++k) {
                        if (d(i, k) < 1e-7) t(i, k) = 1.0 / (d.row(i).array() < 1e-7).count();
                    }
                } else {
                    double s = pow(d.row(i).array(), deriv_p).sum();
                    for (int k = 0; k < n; ++k) {
                        t(i, k) = pow(d(i, k), deriv_p) / s;
                    }
                }
            }

            // Update centroids
            for (int i = 0; i < num_clusters; ++i) {
                Eigen::VectorXd sumup = Eigen::VectorXd::Zero(z);
                double sumdn = 0.0;

                for (int k = 0; k < n; ++k) {
                    double sumcur = pow(u(i, k), m) + w_prob * pow(t(i, k), p);
                    sumup += sumcur * X.col(k);
                    sumdn += sumcur;
                }
                centroids.col(i) = sumup / sumdn;
            }
        }

/*
        std::ofstream outFile("membership_matrix_u.bin", std::ios::binary);
        if (outFile.is_open()) {
            for (int i = 0; i < u.rows(); ++i) {
                for (int j = 0; j < u.cols(); ++j) {
                    double value = u(i, j);
                    outFile.write(reinterpret_cast<const char*>(&value), sizeof(double));
                }
            }
            outFile.close();
        } else {
            std::cerr << "Unable to open file for writing." << std::endl;
        }*/

        eta = Eigen::MatrixXd::Zero(num_clusters, n);
        member = (u.array().pow(m) + w_prob * t.array().pow(p)).matrix();
        trained = true;

        predictedLabels.clear();

        for (int k = 0; k < actualN; ++k) {
            int bestCluster = 0;
            double maxMembership = 0;

            // Find the cluster with the highest membership value
            for (int i = 0; i < num_clusters; ++i) {
                if (member(i, k) > maxMembership) {
                    maxMembership = member(i, k);
                    bestCluster = i;
                }
            }

            predictedLabels.push_back(bestCluster);
    }
/*
        std::ofstream outFile2("/home/user/Desktop/eigenalgofuzz/membership_matrix_member_cpp.bin", std::ios::binary);
        if (outFile2.is_open()) {
            for (int i = 0; i < member.rows(); ++i) {
                for (int j = 0; j < member.cols(); ++j) {
                    double value = member(i, j);
                    outFile2.write(reinterpret_cast<const char*>(&value), sizeof(double));
                }
            }
            outFile2.close();
        } else {
            std::cerr << "Unable to open file for writing." << std::endl;
        }*/
    }
