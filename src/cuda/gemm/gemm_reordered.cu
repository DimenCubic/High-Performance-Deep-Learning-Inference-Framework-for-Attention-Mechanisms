#include "cuda/cuda_gemm.cuh"
#include <cuda_runtime.h>


// Every block will include a A row and every thread inside it will represent every k.
// Here 1D thread consisted the block.
// So actually it shouldn't be called as reordered, it shoule be the initial version of the tiled memory.
__global__
void gemm_reordered_kernel(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 在一个block中共用
    __shared__ float shared_a;
    

    // 这里每一个thread计算的还是C[row][col]
    float sum = 0.0f;

    for(int k = 0; k < N; k++){
        if(threadIdx.x == 0)
            shared_a = A[row * N + k];

        __syncthreads();

        if(col < N)
            sum += shared_a * B[k * N + col];
    }



    // Prevent thread 0 got into next round so quickly, therefore wait all thread finish this k to update shared memory.
    __syncthreads();


    if(col < N)
        C[row * N + col] = sum;
}



void cuda_gemm_reordered(const float* A, const float* B, float* C, int N){
    const int block_size = 256;

    dim3 block(block_size);  // one dimension 256 threads.
    dim3 grid((N + block_size - 1) / block_size, N);   // # of blocks for the total number.


    gemm_reordered_kernel<<< grid, block >>>(A, B, C, N);
}