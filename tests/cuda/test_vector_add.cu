#include "cuda/cuda_ops.cuh"
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

int main(){
    const int N = 1024;
    const std::size_t bytes = N * sizeof(float);

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N);

    for(int i = 0; i < N; i++){
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }



    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;


    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    
    // Copy data from CPU to GPU
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), bytes, cudaMemcpyHostToDevice);

    cuda_vector_add(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);


    bool correct = true;


    for(int i = 0; i < N; i++){
        float expected = A[i] + B[i];

        if(C[i] != expected){
            correct = false;
            std::cout << "Mismatch at index: " << i << std::endl;

            break;
        }
    }


    std::cout << (correct ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}