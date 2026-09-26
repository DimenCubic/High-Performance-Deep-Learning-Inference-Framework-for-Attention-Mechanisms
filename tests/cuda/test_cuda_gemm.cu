#include "cuda/cuda_gemm.cuh"
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

int main(){
    const int N = 4;
    const std::size_t bytes = N * N * sizeof(float);

    std::vector<float> A(N*N);
    std::vector<float> B(N*N);
    std::vector<float> C(N*N);
    std::vector<float> expected(N*N);

    for(int i = 0; i < N*N; i++){
        A[i] = static_cast<float>(i+1);
        B[i] = static_cast<float>(i+1);
    }

    for(int row = 0; row < N; row++){
        for(int col = 0; col < N; col++){
            float sum = 0.0f;

            for(int k = 0; k < N; k++)
                sum += A[row * N + k] * B[k * N + col];

            expected[row * N + col] = sum;
        }
    }


    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    cuda_gemm_naive(d_A, d_B, d_C, N);
    
    
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);



    bool correct = true;
    const float tolerance = 1e-5f;

    for(int i = 0; i < N * N; i++){
        if(std::fabs(C[i] - expected[i]) > tolerance){
            correct = false;

            std::cout << "Mismatch at index" << i << ": GPU = " << C[i] << ", expected = " << expected[i] << std::endl;

            break;
        }
    }


    std::cout << "CUDA Naive GEMM: " << (correct ? "PASS" : "FAIL") << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return 0;
}