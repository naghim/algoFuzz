#include "Metrics.h"
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <iostream>

using namespace Eigen;

// Function to calculate Purity
double calculatePurity(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
/*
    std::cout << "True labels: ";
    for (int label : trueLabels) {
        std::cout << label << " ";
    }
    std::cout << "\nPredicted labels: ";
    for (int label : predictedLabels) {
        std::cout << label << " ";
    }
    std::cout << "\n";
*/
    std::unordered_map<int, std::unordered_map<int, int>> contingencyTable;
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        contingencyTable[predictedLabels[i]][trueLabels[i]]++;
    }

    double totalCount = trueLabels.size();
    double correctCount = 0;

    // For each predicted cluster, find the true label that appears most frequently
    for (const auto& cluster : contingencyTable) {
        int maxCount = 0;
        for (const auto& label : cluster.second) {
            maxCount = std::max(maxCount, label.second);
        }
        correctCount += maxCount;
    }

    return correctCount / totalCount;
}

// Helper functions for NMI and ARI
double entropy(const std::vector<int>& labels) {
    std::unordered_map<int, int> labelCounts;
    for (int label : labels) {
        labelCounts[label]++;
    }
    double ent = 0.0;
    for (const auto& entry : labelCounts) {
        double p = static_cast<double>(entry.second) / labels.size();
        ent -= p * std::log2(p);
    }
    return ent;
}

double calculateMutualInformation(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
    std::unordered_map<int, std::unordered_map<int, int>> contingencyTable;

    for (size_t i = 0; i < trueLabels.size(); ++i) {
        contingencyTable[predictedLabels[i]][trueLabels[i]]++;
    }

    double mutualInformation = 0.0;
    int N = trueLabels.size();

    for (const auto& cluster : contingencyTable) {
        int clusterSize = 0;
        for (const auto& label : cluster.second) {
            clusterSize += label.second;
        }

        for (const auto& label : cluster.second) {
            int nij = label.second;
            if (nij > 0) {
                double pij = static_cast<double>(nij) / N;
                double pi = static_cast<double>(clusterSize) / N;
                double pj = static_cast<double>(std::accumulate(trueLabels.begin(), trueLabels.end(), 0,
                        [&label](int sum, int lbl) { return sum + (lbl == label.first); })) / N;

                mutualInformation += pij * std::log2(pij / (pi * pj));
            }
        }
    }
    return mutualInformation;
}

// Function to calculate NMI
double calculateNMI(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
    double hTrue = entropy(trueLabels);
    double hPred = entropy(predictedLabels);
    double mi = calculateMutualInformation(trueLabels, predictedLabels);
    return 2 * mi / (hTrue + hPred);
}

// Function to calculate ARI
double calculateARI(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
    size_t N = trueLabels.size();
    std::unordered_map<int, std::unordered_map<int, int>> contingencyTable;

    for (size_t i = 0; i < N; ++i) {
        contingencyTable[predictedLabels[i]][trueLabels[i]]++;
    }

    int sumCombinations = 0, sumRowCombinations = 0, sumColCombinations = 0;
    for (const auto& cluster : contingencyTable) {
        int rowSum = 0;
        for (const auto& label : cluster.second) {
            int nij = label.second;
            sumCombinations += nij * (nij - 1) / 2;
            rowSum += nij;
        }
        sumRowCombinations += rowSum * (rowSum - 1) / 2;
    }

    for (int label : trueLabels) {
        int colSum = std::accumulate(predictedLabels.begin(), predictedLabels.end(), 0,
            [label](int sum, int predLabel) { return sum + (predLabel == label); });
        sumColCombinations += colSum * (colSum - 1) / 2;
    }

    double expectedIndex = static_cast<double>(sumRowCombinations) * sumColCombinations / (N * (N - 1) / 2);
    double maxIndex = 0.5 * (sumRowCombinations + sumColCombinations);
    return (sumCombinations - expectedIndex) / (maxIndex - expectedIndex);
}

Eigen::MatrixXi computeConfusionMatrix(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, int numClasses) {
    Eigen::MatrixXi confusionMatrix = Eigen::MatrixXi::Zero(numClasses, numClasses);

    /*
    std::cout << "True labels: " << trueLabels.size() << std::endl;

    std::cout << "True labels: ";
    for (int label : trueLabels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;

    std::cout << "Predicted labels: ";
    for (int label : predictedLabels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;*/

    for (size_t i = 0; i < trueLabels.size(); ++i) {
        int trueLabel = trueLabels[i];
        int predictedLabel = predictedLabels[i];

        if (trueLabel >= 0 && trueLabel < numClasses && predictedLabel >= 0 && predictedLabel < numClasses) {
            confusionMatrix(trueLabel, predictedLabel)++;
        }
    }

    return confusionMatrix;
}

Eigen::MatrixXi findBestPermutationMatrix(const Eigen::MatrixXi& confusionMatrix) {
    int n = confusionMatrix.rows();
    std::vector<int> perm(n);
    std::vector<int> bestPerm(n);
    std::iota(perm.begin(), perm.end(), 0);
    
    int maxDiagonalSum = 0;
    
    do {
        int currentSum = 0;
        for (int i = 0; i < n; i++) {
            currentSum += confusionMatrix(i, perm[i]);
        }
        
        if (currentSum > maxDiagonalSum) {
            maxDiagonalSum = currentSum;
            bestPerm = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    
    MatrixXi permutedMatrix = Eigen::MatrixXi::Zero(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            permutedMatrix(i, j) = confusionMatrix(i, bestPerm[j]);
        }
    }
    
    return permutedMatrix;
}