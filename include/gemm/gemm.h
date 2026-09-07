#ifndef GEMM_H
#define GEMM_H

void naive_gemm(
    float* A,
    float* B,
    float* C,
    int N     // N behaves Line number / Column Number
);

#endif