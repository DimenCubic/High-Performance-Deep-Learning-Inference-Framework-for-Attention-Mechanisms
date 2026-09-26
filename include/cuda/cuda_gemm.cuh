#ifndef CUDA_GEMM_CUH
#define CUDA_GEMM_CUH

void cuda_gemm_naive(
    const float* A,
    const float* B,
    float* C,
    int N
);

#endif