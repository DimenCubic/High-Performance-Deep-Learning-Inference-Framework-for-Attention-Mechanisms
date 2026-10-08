#include "cuda/cuda_ops.cuh"
#include "ops/operator.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>


int main(){
    const int M = 3;
    const int K = 4;
    const int N = 2;


    std::vector<float> A = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };


    std::vector<float> B = {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    };


    std::vector<float> cpu_output(M * N);
    std::vector<float> gpu_output(M * N);




    // CPU Reference
    matmul(A.data(), B.data(), cpu_output.data(), M, K, N);




    // CUDA Memory
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    const std::size_t bytes_A = static_cast<std::size_t>(M * K) * sizeof(float);
    const std::size_t bytes_B = static_cast<std::size_t>(K * N) * sizeof(float);
    const std::size_t bytes_C = static_cast<std::size_t>(M * N) * sizeof(float);


    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes_B, cudaMemcpyHostToDevice);



    // CUDA Matmul
    cuda_matmul(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_output.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);


    // Verify
    const float tolerance = 1e-5f;
    bool correct = true;

    for(int i = 0; i < M * N; i++){
        const float diff = std::fabs(cpu_output[i] - gpu_output[i]);

        if(diff > tolerance){
            correct = false;

            std::cout << "Mismatch at index " << i << std::endl;
            std::cout << "CPU = " << cpu_output[i] << std::endl;
            std::cout << "GPU = " << gpu_output[i] << std::endl;
            std::cout << "Diff = " << diff << std::endl;

            break;
        }
    }




    std::cout << "CUDA Matmul: " << (correct ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return correct ? 0 : 1;
}