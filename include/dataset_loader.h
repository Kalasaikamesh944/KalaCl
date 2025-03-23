//
// Created by saika on 3/23/2025.
//

#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <vector>
#include <string>

// Define struct DataPoint
struct DataPoint {
    std::vector<double> features;  // Holds numerical + encoded text feature
    double feature1;
    double feature2;
    double feature3;
    std::string text_feature;      // Original text feature
    double target;
    double label;// Target value (dependent variable)
};

// Function declarations
std::vector<DataPoint> loadDataset(const std::string& filename);
void saveModel(const std::vector<double>& weights, const std::string& filename);
std::vector<double> loadModel(const std::string& filename);
void testModel(const std::vector<std::vector<double>>& testData, const std::vector<double>& modelWeights);

#endif // DATASET_LOADER_H
