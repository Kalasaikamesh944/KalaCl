//
// Created by saika on 3/23/2025.
//

#include "dataset_loader.h"
#include "kala_rnn.h"
#include <iostream>

int main() {
    std::string datasetFile = "large_dataset.csv";

    // ✅ Load dataset
    std::vector<DataPoint> dataset = loadDataset(datasetFile);
    if (dataset.empty()) {
        std::cerr << "[KalaRNN] Error: Dataset is empty!" << std::endl;
        return 1;
    }

    // ✅ Initialize Kala RNN
    int input_size = dataset[0].features.size();
    KalaRNN model(input_size, 8, 0.01);  // 8 hidden neurons

    // ✅ Train model
    int epochs = 20;
    model.train(dataset, epochs);

    return 0;
}
