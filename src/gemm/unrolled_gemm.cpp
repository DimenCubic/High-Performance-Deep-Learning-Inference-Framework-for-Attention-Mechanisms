#include "gemm/gemm.h"

void unrolled_gemm(const float* A, const float* B, float* C, int N){
    for(int i = 0; i < N; i++){
        int c_base = i*N;

        for(int k = 0; k < N; k++){
            int b_base = k*N;
            float a = A[i*N + k];


            int j = 0;
            for(; j + 3 < N; j += 4){
                C[c_base + j] += a * B[k*N + j];
                C[c_base + j + 1] += a * B[b_base + j + 1];
                C[c_base + j + 2] += a * B[b_base + j + 2];
                C[c_base + j + 3] += a * B[b_base + j + 3];
            }


            for(; j < N; j++)
                C[c_base + j] += a * B[b_base + j];
        }
    }



}