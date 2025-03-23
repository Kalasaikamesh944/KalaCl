//
// Created by saika on 3/23/2025.
//

#ifndef KALA_CELL_H
#define KALA_CELL_H

#include <vector>
#include <immintrin.h>
#include "dataset_loader.h"

class KalaCell {
public:
    KalaCell(int input_size);
    double forward(const std::vector<double>& input);
    void update_weights(const std::vector<double>& gradient, double learning_rate);

private:
    std::vector<double> weights;
    double curvature_factor;
};

// AVX-Optimized Matrix Multiplication
void kala_matrix_mul_avx(const double* A, const double* B, double* C, int N);

// OpenMP-Optimized Matrix Multiplication
void kala_matrix_mul_omp(const double* A, const double* B, double* C, int N);

// Prefetching Optimization for Memory
void kala_matrix_mul_prefetch(const double* A, const double* B, double* C, int N);

// Train Kala Cells with AVX
void trainKalaCells(std::vector<DataPoint>& dataset, int epochs, double learning_rate);

#endif // KALA_CELL_H
