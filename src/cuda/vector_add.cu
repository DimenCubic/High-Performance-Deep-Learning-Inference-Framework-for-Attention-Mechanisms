#include "cuda/cuda_ops.cuh"
#include <cuda_runtime.h>

__global__     // GPU start instruction.
void vector_add_kernel(const float* A, const float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
        C[i] = A[i] + B[i];
}


void cuda_vector_add(const float* A, const float* B, float* C, int N){
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;  // # of blocks

    vector_add_kernel <<< grid_size, block_size >>>(A, B, C, N);
}