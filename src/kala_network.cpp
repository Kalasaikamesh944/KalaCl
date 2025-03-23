//
// Created by saika on 3/23/2025.
//

#include "kala_network.h"
#include <iostream>

// ✅ Constructor: Initializes the network
KalaNetwork::KalaNetwork(int input_size, int hidden_layers, int neurons_per_layer, double lr)
    : learning_rate(lr) {
    layers.emplace_back(KalaCell(input_size));  // Input layer
    for (int i = 0; i < hidden_layers; i++) {
        layers.emplace_back(KalaCell(neurons_per_layer));  // Hidden layers
    }
}

// ✅ Forward Pass: Passes input through all layers
std::vector<double> KalaNetwork::forward(const std::vector<double>& input) {
    std::vector<double> output = input;
    for (auto& layer : layers) {
        std::vector<double> next_output;
        for (double val : output) {
            next_output.push_back(layer.forward({val}));
        }
        output = next_output;
    }
    return output;
}

// ✅ Training function using basic gradient descent
void KalaNetwork::train(std::vector<DataPoint>& dataset, int epochs) {
    for (int epoch = 1; epoch <= epochs; epoch++) {
        double total_loss = 0.0;

        for (auto& data : dataset) {
            std::vector<double> prediction = forward(data.features);
            double error = data.label - prediction[0];

            // 🔥 Compute gradient (Simple Delta Rule)
            std::vector<double> gradient = {error * learning_rate};
            layers[0].update_weights(gradient, learning_rate);

            total_loss += error * error;
        }

        std::cout << "[KalaCL] Epoch " << epoch << " Loss: " << total_loss << std::endl;
    }
}
