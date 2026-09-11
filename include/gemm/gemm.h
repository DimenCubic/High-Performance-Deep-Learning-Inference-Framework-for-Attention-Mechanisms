#ifndef GEMM_H
#define GEMM_H

void naive_gemm(
    const float* A,
    const float* B,
    float* C,
    int N     // N behaves Line number / Column Number
);

void reordered_gemm(
    const float* A,
    const float* B,
    float* C,
    int N
);

void unrolled_gemm(
    const float* A,
    const float* B,
    float* C,
    int N
);

void blocked_gemm(
    const float* A,
    const float* B,
    float* C,
    int N
);

void blocked2_gemm(
    const float* A,
    const float* B,
    float* C,
    int N
);

#endif