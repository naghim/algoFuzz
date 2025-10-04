#include "DatasetLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

using namespace Eigen;

// Helper function to split a string by a delimiter
std::vector<std::string> Dataset::split(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Load dataset from CSV file
void Dataset::loadFromCSV(const std::string& filename, bool normalize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    std::getline(file, line);
    num_clusters = std::stoi(line);

    // Read CSV line-by-line
    while (std::getline(file, line)) {
        auto tokens = split(line, ',');
        if (tokens.empty()) continue;  // Skip empty lines

        // Parse feature values and label
        std::vector<double> features;
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            features.push_back(std::stod(tokens[i]));
        }

        // Last column is the label
        int label = std::stoi(tokens.back());
        true_labels.push_back(label);
        data.push_back(features);
    }

    num_samples = data.size();
    num_features = data[0].size();

    // Normalize features to [-1, 1] range
    if (normalize) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            // Find min and max for each feature
            double min_val = data[0][j];
            double max_val = data[0][j];

            for (size_t i = 1; i < data.size(); ++i) {
                min_val = std::min(min_val, data[i][j]);
                max_val = std::max(max_val, data[i][j]); 
            }
            
            // Normalize each feature
            double range = max_val - min_val;

            if (range > 0) {
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i][j] = (data[i][j] - min_val) / range;
                }
            }
        }
/*
        // Print normalized values
        std::cout << "Normalized values:" << std::endl;
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }*/
    }

    // Convert data to Eigen matrix
    int numRows = data.size();
    int numCols = data[0].size();

    X = Eigen::MatrixXd::Zero(numCols, numRows);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            X(j, i) = data[i][j];
        }
    }

    file.close();
}

// Load dataset from CSV file
int DatasetLoader::loadFromCSV(const std::string& filename, bool normalize) {
    Dataset *dataset = new Dataset();
    dataset->loadFromCSV(filename, normalize);
    int id = getNextId();
    datasets[id] = dataset;
    return id;
}

Dataset *DatasetLoader::getDataset(int id) {
    auto it = datasets.find(id);
    if (it != datasets.end()) {
        return it->second;
    }
    return nullptr; // Or throw an exception
}

int DatasetLoader::loadFromNumpyArray(const Eigen::MatrixXd& X, const std::vector<int>& true_labels, int num_clusters) {
    Dataset *dataset = new Dataset();
    dataset->X = X;
    dataset->true_labels = true_labels;
    dataset->num_clusters = num_clusters;
    dataset->num_samples = X.cols();
    dataset->num_features = X.rows();
    int id = getNextId();
    datasets[id] = dataset;
    return id;
}

int DatasetLoader::getNextId() {
    return next_id++;
}

DatasetLoader dataset_loader;