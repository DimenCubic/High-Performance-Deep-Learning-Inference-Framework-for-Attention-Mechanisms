#include "cuda/cuda_gemm.cuh"
#include <cuda_runtime.h>

__global__   // Convention made x as column and y as row.
void gemm_naive_kernel(const float* A, const float* B, float* C, int N){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < N && col < N){
        float sum = 0.0f;

        for(int k = 0; k < N; k++)
            sum += A[row*N + k] * B[k*N + col];

        C[row*N + col] = sum;
    }
}


void cuda_gemm_naive(const float* A, const float* B, float* C, int N){
    dim3 block(16,16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    gemm_naive_kernel <<< grid, block >>> (A, B, C, N);
}