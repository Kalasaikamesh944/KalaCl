//
// Created by saika on 3/23/2025.
//

#include "kala_rnn.h"
#include <iostream>
#include <cmath>

// ✅ Constructor: Initializes the RNN
KalaRNN::KalaRNN(int input_size, int hidden_size, double lr)
    : learning_rate(lr) {
    neurons.resize(hidden_size, KalaCell(input_size));
    hidden_state.resize(hidden_size, 0.0);
}

// ✅ Forward Pass: Uses tanh activation for better stability
std::vector<double> KalaRNN::forward(const std::vector<double>& input) {
    std::vector<double> output(hidden_state.size());

    for (size_t i = 0; i < neurons.size(); i++) {
        double combined_input = input[i % input.size()] * 0.5 + hidden_state[i] * 0.5; // Weighted sum
        output[i] = std::tanh(neurons[i].forward({combined_input}));  // Activation function
    }

    hidden_state = output;
    return output;
}

// ✅ Training function using Backpropagation Through Time (BPTT)
void KalaRNN::train(std::vector<DataPoint>& dataset, int epochs) {
    for (int epoch = 1; epoch <= epochs; epoch++) {
        double total_loss = 0.0;

        for (auto& data : dataset) {
            std::vector<double> prediction = forward(data.features);
            double error = data.label - prediction[0];

            // 🔥 Compute gradient
            std::vector<double> gradient = {error * (1 - prediction[0] * prediction[0]) * learning_rate};

            for (auto& neuron : neurons) {
                neuron.update_weights(gradient, learning_rate);
            }

            total_loss += error * error;
        }

        std::cout << "[KalaRNN] Epoch " << epoch << " Loss: " << total_loss << std::endl;
    }
}

