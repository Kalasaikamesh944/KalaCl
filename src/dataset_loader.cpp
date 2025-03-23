//
// Created by saika on 3/23/2025.
//

#include "dataset_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

// Define a text-to-index mapping for encoding categorical features
std::unordered_map<std::string, int> text_mapping;

// Load dataset from CSV file
std::vector<DataPoint> loadDataset(const std::string& filename) {
    std::vector<DataPoint> dataset;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[Error] Could not open file: " << filename << std::endl;
        return dataset;
    }

    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        if (firstLine) {
            firstLine = false;  // Skip header
            continue;
        }

        std::stringstream ss(line);
        std::string value;
        DataPoint dp;

        // Read numerical features
        dp.features.resize(2);  // Ensure space for 2 numerical features
        std::getline(ss, value, ',');
        dp.features[0] = std::stod(value);

        std::getline(ss, value, ',');
        dp.features[1] = std::stod(value);

        // Read text feature and map to an integer
        std::getline(ss, dp.text_feature, ',');
        if (text_mapping.find(dp.text_feature) == text_mapping.end()) {
            text_mapping[dp.text_feature] = text_mapping.size();
        }
        dp.features.push_back(static_cast<double>(text_mapping[dp.text_feature]));

        // Read target value
        std::getline(ss, value, ',');
        dp.target = std::stod(value);

        // ✅ Correctly add the data point
        dataset.push_back(dp);
    }

    file.close();
    return dataset;
}

// Save model weights to a file
void saveModel(const std::vector<double>& weights, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto& w : weights) {
            file << w << "\n";
        }
        file.close();
        std::cout << "[KalaCL] Model saved to " << filename << std::endl;
    } else {
        std::cerr << "[KalaCL] Error: Unable to save model!" << std::endl;
    }
}

// Load model weights from a file
std::vector<double> loadModel(const std::string& filename) {
    std::vector<double> weights;
    std::ifstream file(filename);
    if (file.is_open()) {
        double w;
        while (file >> w) {
            weights.push_back(w);
        }
        file.close();
        std::cout << "[KalaCL] Model loaded from " << filename << std::endl;
    } else {
        std::cerr << "[KalaCL] Error: Unable to load model!" << std::endl;
    }
    return weights;
}

// Test model predictions
void testModel(const std::vector<std::vector<double>>& testData, const std::vector<double>& modelWeights) {
    std::cout << "[KalaCL] Testing model..." << std::endl;
    for (const auto& sample : testData) {
        double prediction = 0.0;
        for (size_t i = 0; i < sample.size(); ++i) {
            prediction += sample[i] * modelWeights[i];
        }
        std::cout << "Prediction: " << prediction << std::endl;
    }
}
