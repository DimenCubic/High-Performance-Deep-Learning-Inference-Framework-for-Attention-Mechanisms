#include "cuda/cuda_gemm.cuh"
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

int main(){
    const int N = 1024;
    const int warmup_runs = 10;
    const int benchmark_runs = 100;

    const std::size_t elements = static_cast<std::size_t>(N) * N;
    const std::size_t bytes = elements * sizeof(float);

    std::vector<float> A(elements, 1.0f);
    std::vector<float> B(elements, 1.0f);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    // Warmup
    for(int i = 0; i < warmup_runs; i++)
        cuda_gemm_naive(d_A, d_B, d_C, N);
    
    cudaDeviceSynchronize();


    
}