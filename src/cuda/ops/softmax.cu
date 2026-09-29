#include "include/cuda/cuda_ops.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>


__global__
void softmax_kernel(const float* input, float* output, int size){
    extern __shared__ float shared[];   // Dynamic shared memory.

    const int tid = threadIdx.x;


    // Each thread finds its maximum
    float local_max = -FLT_MAX;   // 极小负数

    for(int i = tid; i < size; i += blockDim.x)
        
}