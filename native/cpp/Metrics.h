#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <Eigen/Dense>

double calculatePurity(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels);
double calculateNMI(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels);
double calculateARI(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels);

Eigen::MatrixXi computeConfusionMatrix(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, int numClasses);
Eigen::MatrixXi findBestPermutationMatrix(const Eigen::MatrixXi& confusionMatrix);

#endif // METRICS_H
