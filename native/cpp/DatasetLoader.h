#ifndef DATASETLOADER_H
#define DATASETLOADER_H

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>

enum DatasetType {
    IRIS = 0,
    BREAST_CANCER = 1,
    WINE = 2,
    GLASS = 3,
    SEEDS = 4
};

class Dataset {
public:
    void loadFromCSV(const std::string& filename, bool normalize);
    Eigen::MatrixXd X;
    std::vector<int> true_labels;
    int num_clusters;
    int num_features;
    int num_samples;
private:
    // Helper function to parse CSV line
    std::vector<std::string> split(const std::string& line, char delimiter);
};

class DatasetLoader {
public:
    // Load dataset from a CSV file
    int loadFromCSV(const std::string& filename, bool normalize);
    int loadFromNumpyArray(const Eigen::MatrixXd& X, const std::vector<int>& true_labels, int num_clusters);
    Dataset *getDataset(int id);

private:
    std::unordered_map<int, Dataset*> datasets;
    int next_id = 0;
    int getNextId();
};

extern DatasetLoader dataset_loader;

#endif // DATASETLOADER_H
