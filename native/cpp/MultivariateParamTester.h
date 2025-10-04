#ifndef MULTIVARIATE_PARAM_TESTER_H
#define MULTIVARIATE_PARAM_TESTER_H

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <thread>
#include <mutex>
#include <future>
#include "DatasetLoader.h"      // Custom dataset loader (needs to be implemented)

class MultivariateParamTester {
public:
    MultivariateParamTester(const std::unordered_map<std::string, std::vector<double>>& paramGrid);
    void fitToCsv(const std::string& filename);
    void loadDatasets();
    double totalTime() const { return total_time; }

private:
    std::unordered_map<std::string, std::vector<double>> param_grid;
    std::vector<std::vector<double>> values;
    double total_time = 0.0;

    std::unordered_map<std::string, Dataset *> datasets;

    std::unordered_map<DatasetType, std::string> datasetMap = {
        {DatasetType::IRIS, "iris"},
        {DatasetType::BREAST_CANCER, "breast_cancer"},
        {DatasetType::WINE, "wine"},
        {DatasetType::GLASS, "glass"},
        {DatasetType::SEEDS, "seeds"}
    };

    std::vector<std::vector<double>> queueOptions();
    bool evaluate(std::vector<double> item, std::vector<double>& result);
    void writeCsv(const std::string& filename);

    std::mutex mutex_;
};

#endif // MULTIVARIATE_PARAM_TESTER_H
