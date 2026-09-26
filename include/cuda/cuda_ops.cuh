#ifndef CUDA_OPS_CUH
#define CUDA_OPS_CUH

void cuda_vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
);

#endif