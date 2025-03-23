//
// Created by saika on 3/23/2025.
//

#include <cmath>
#include <random>
#include  <chrono>
#include <iostream>
#include "dataset_loader.h"
#include "kala_cell.h"

// ✅ Constructor for Kala Cells
KalaCell::KalaCell(int input_size) {
    weights.resize(input_size);
    curvature_factor = 1.618; // Default phi value
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (double &w : weights) {
        w = dis(gen);
    }
}

// ✅ Forward propagation using Kala Theory
double KalaCell::forward(const std::vector<double>& input) {
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); i++) {
        sum += weights[i] * input[i];
    }
    return sum * curvature_factor; // Spacetime curvature influence
}

// ✅ Weight update for AI learning
void KalaCell::update_weights(const std::vector<double>& gradient, double learning_rate) {
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] -= learning_rate * gradient[i] * curvature_factor;
    }
}

// ✅ AVX-Optimized Matrix Multiplication
void kala_matrix_mul_avx(const double* A, const double* B, double* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m256d sum = _mm256_setzero_pd();
            for (int k = 0; k < N; k += 4) {
                __m256d a = _mm256_load_pd(&A[i * N + k]);
                __m256d b = _mm256_load_pd(&B[k * N + j]);
                sum = _mm256_fmadd_pd(a, b, sum);
            }
            double temp[4];
            _mm256_store_pd(temp, sum);
            C[i * N + j] = temp[0] + temp[1] + temp[2] + temp[3];
        }
    }
}

// ✅ OpenMP-Optimized Matrix Multiplication
void kala_matrix_mul_omp(const double* A, const double* B, double* C, int N) {
    #pragma omp parallel for collapse(2) schedule(static, 32)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ✅ Prefetching for Memory Optimization
void kala_matrix_mul_prefetch(const double* A, const double* B, double* C, int N) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                _mm_prefetch((const char*)&A[i * N + k + 16], _MM_HINT_T0);
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// AVX-Optimized Training Function
void trainKalaCells(std::vector<DataPoint>& dataset, int epochs, double learning_rate) {
    double weight1 = 0.1, weight2 = 0.1, bias = 0.1;  // Initial parameters

    for (int epoch = 1; epoch <= epochs; epoch++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        __m256d sum_grad_w1 = _mm256_setzero_pd();
        __m256d sum_grad_w2 = _mm256_setzero_pd();
        __m256d sum_grad_b  = _mm256_setzero_pd();

        double total_loss = 0.0;
        int batch_size = dataset.size();

        for (int i = 0; i < batch_size; i += 4) {
            double pred[4], error[4];

            for (int j = 0; j < 4 && (i + j) < batch_size; ++j) {
                pred[j] = weight1 * dataset[i + j].feature1 + weight2 * dataset[i + j].feature2 + bias;
                error[j] = pred[j] - dataset[i + j].target;
                total_loss += error[j] * error[j]; // MSE Loss
            }

            __m256d err_vec = _mm256_loadu_pd(error);
            __m256d feat1_vec = _mm256_set_pd(dataset[i].feature1, dataset[i + 1].feature1, dataset[i + 2].feature1, dataset[i + 3].feature1);
            __m256d feat2_vec = _mm256_set_pd(dataset[i].feature2, dataset[i + 1].feature2, dataset[i + 2].feature2, dataset[i + 3].feature2);

            sum_grad_w1 = _mm256_fmadd_pd(err_vec, feat1_vec, sum_grad_w1);
            sum_grad_w2 = _mm256_fmadd_pd(err_vec, feat2_vec, sum_grad_w2);
            sum_grad_b  = _mm256_add_pd(sum_grad_b, err_vec);
        }

        double grad_w1[4], grad_w2[4], grad_b[4];
        _mm256_storeu_pd(grad_w1, sum_grad_w1);
        _mm256_storeu_pd(grad_w2, sum_grad_w2);
        _mm256_storeu_pd(grad_b, sum_grad_b);

        weight1 -= learning_rate * (grad_w1[0] + grad_w1[1] + grad_w1[2] + grad_w1[3]) / batch_size;
        weight2 -= learning_rate * (grad_w2[0] + grad_w2[1] + grad_w2[2] + grad_w2[3]) / batch_size;
        bias    -= learning_rate * (grad_b[0] + grad_b[1] + grad_b[2] + grad_b[3]) / batch_size;

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "[KalaCL] Epoch " << epoch << " Loss: " << total_loss / batch_size << " Time: " << elapsed.count() << " sec\n";
    }

    std::cout << "Final Weights -> w1: " << weight1 << ", w2: " << weight2 << ", bias: " << bias << std::endl;
}
