#include "cuda/cuda_gemm.cuh"
#include <cuda_runtime.h>


__global__
void gemm_unrolled_kernel(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= N || col >= N)
        return;

    float sum = 0.0f;


    int k = 0;
    for(; k + 3 < N; k += 4){
        sum += A[row * N + k] * B[k * N + col];
        sum += A[row * N + k + 1] * B[(k + 1) * N + col];
        sum += A[row * N + k + 2] * B[(k + 2) * N + col];
        sum += A[row * N + k + 3] * B[(k + 3) * N + col];
    }


    for(; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];

    C[row * N + col] = sum;

}


void cuda_gemm_unrolled(const float* A, const float* B, float* C, int N){
    dim3 block(16,16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    gemm_unrolled_kernel <<< grid, block >>> (A, B, C, N);
}