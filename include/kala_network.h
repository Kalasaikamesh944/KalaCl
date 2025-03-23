#ifndef KALA_NETWORK_H
#define KALA_NETWORK_H

#include <iostream>
#include <vector>
#include "kala_cell.h"

class KalaNetwork {
private:
    std::vector<KalaCell> layers;
    double learning_rate;

public:
    KalaNetwork(int input_size, int hidden_layers, int neurons_per_layer, double lr);

    std::vector<double> forward(const std::vector<double>& input);
    void train(std::vector<DataPoint>& dataset, int epochs);
};

#endif // KALA_NETWORK_H
