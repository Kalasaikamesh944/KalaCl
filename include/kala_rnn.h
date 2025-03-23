//
// Created by saika on 3/23/2025.
//
#ifndef KALA_RNN_H
#define KALA_RNN_H

#include "kala_cell.h"
#include <vector>

class KalaRNN {
private:
    std::vector<KalaCell> neurons;
    std::vector<double> hidden_state;
    double learning_rate;

public:
    KalaRNN(int input_size, int hidden_size, double lr);

    std::vector<double> forward(const std::vector<double>& input);
    void train(std::vector<DataPoint>& dataset, int epochs);
};

#endif // KALA_RNN_H
