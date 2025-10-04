#include "MultivariateParamTester.h"
#include "Metrics.h"
#include "GFPCM.h"
#include <chrono>
#include <algorithm>
#include <random>

MultivariateParamTester::MultivariateParamTester(const std::unordered_map<std::string, std::vector<double>>& paramGrid)
    : param_grid(paramGrid) {}

std::vector<std::vector<double>> MultivariateParamTester::queueOptions() {
    std::vector<std::vector<double>> processQueue;

    // Generate all parameter combinations
    for (const auto& dataset : param_grid.at("dataset")) {
        for (const auto& m : param_grid.at("m")) {
            for (const auto& p : param_grid.at("p")) {
                for (const auto& seed : param_grid.at("seed")) {
                    for (const auto& w_prob : param_grid.at("w_prob")) {
                        for (const auto& noise : param_grid.at("noise")) {
                            for (const auto& centroid_strategy : param_grid.at("centroid_strategy")) {
                                Dataset *actual_dataset = datasets[datasetMap[static_cast<DatasetType>(static_cast<int>(dataset))]];
                                //double actual_w_prob = (actual_dataset->num_clusters * actual_dataset->num_features * actual_dataset->num_samples) / (m * 3);
                                double actual_w_prob = 1000;    
                                //double actual_w_prob = ((actual_dataset->num_samples/actual_dataset->num_clusters) * (actual_dataset->num_samples/actual_dataset->num_clusters)) / 3.0;
                                //double actual_w_prob = pow(10, (m / actual_dataset->num_clusters) + p);
                                std::cout << "Dataset: " << dataset << ", m: " << m << ", p: " << p << ", seed: " << seed << ", w_prob: " << actual_w_prob << ", noise: " << noise << ", centroid_strategy: " << centroid_strategy << std::endl;
                                processQueue.push_back({dataset, m, p, seed, actual_w_prob, noise, centroid_strategy});
                            }
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "Total number of parameter combinations: " << processQueue.size() << std::endl;
    //std::shuffle(processQueue.begin(), processQueue.end(), std::mt19937{std::random_device{}()});
    return processQueue;
}

bool MultivariateParamTester::evaluate(std::vector<double> item, std::vector<double>& result) {
        // Placeholder function to get FCM instance, similar to `get_fcm_by_type`
        auto dataset = datasets[datasetMap[static_cast<DatasetType>(static_cast<int>(item[0]))]];
        auto max_iter = 150;
        auto m = item[1];
        auto p = item[2];
        auto w_prob = item[4];
        auto noise = item[5];
        auto centroid_strategy = static_cast<CentroidStrategy>(static_cast<int>(item[6]));
        
        GFPCM fcm(dataset->num_clusters, max_iter, m, p, w_prob, noise, centroid_strategy);

        // Fit and evaluate
        auto start = std::chrono::high_resolution_clock::now();
        fcm.fit(dataset->X, dataset->true_labels);
        auto duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        // Calculate metrics
        auto labels = fcm.getPredictedLabels();
        double purity = calculatePurity(dataset->true_labels, labels);
        //double nmi = calculateNMI(dataset->true_labels, labels);
        //double ari = calculateARI(dataset->true_labels, labels);

        // Add results to result vector
        result = item;
        result.push_back(duration);
        result.push_back(purity);
        //result.push_back(nmi);
        //result.push_back(ari);

        return true;
}

void MultivariateParamTester::fitToCsv(const std::string& filename) {
    auto processQueue = queueOptions();
    total_time = 0.0;

    std::ofstream outFile(filename);
    outFile << "dataset,m,p,seed,w_prob,noise,centroid_strategy,time,purity,nmi,ari\n";

    std::vector<std::future<std::vector<double>>> futures;

    auto start = std::chrono::high_resolution_clock::now();

    size_t chunk_size = 1000;
    size_t num_chunks = (processQueue.size() + chunk_size - 1) / chunk_size;

    std::cout << "Number of chunks: " << num_chunks << std::endl;

    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        size_t start_idx = chunk * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, processQueue.size());

        for (size_t i = start_idx; i < end_idx; ++i) {
            futures.emplace_back(std::async(std::launch::async, [this, item = processQueue[i]]() {
                std::vector<double> result;
                if (evaluate(item, result)) return result;
                return std::vector<double>{};
            }));
        }

        for (auto& future : futures) {
            auto result = future.get();

            if (!result.empty()) {
                std::lock_guard<std::mutex> lock(mutex_);
                for (size_t i = 0; i < result.size() - 1; ++i) outFile << result[i] << ",";
                outFile << result.back();
                outFile << "\n";
            }
        }

        futures.clear();
        std::cout << "Finished chunk " << chunk + 1 << " of " << num_chunks << std::endl;
    }

    outFile.close();
    total_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

void MultivariateParamTester::writeCsv(const std::string& filename) {
    // Save results to a CSV file.
    std::ofstream file(filename);
    file << "dataset,m,p,seed,w_prob,noise,time,purity,nmi,ari\n";
    for (const auto& row : values) {
        for (size_t i = 0; i < row.size() - 1; ++i) {
            file << row[i] << ",";
        }
        file << row.back();
        file << "\n";
    }
    file.close();
}

void MultivariateParamTester::loadDatasets() {
    // Placeholder function to load datasets
    DatasetLoader loader;
    datasets["iris"] = loader.getDataset(loader.loadFromCSV("iris.csv", true));
    datasets["breast_cancer"] = loader.getDataset(loader.loadFromCSV("breast_cancer.csv", true));
    datasets["wine"] = loader.getDataset(loader.loadFromCSV("wine.csv", true)); // 3
    datasets["glass"] = loader.getDataset(loader.loadFromCSV("glass.csv", true)); // 6
    datasets["seeds"] = loader.getDataset(loader.loadFromCSV("seeds.csv", true));
}