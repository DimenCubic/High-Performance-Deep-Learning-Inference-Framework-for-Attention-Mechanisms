#include "gemm/gemm.h"

void reordered_gemm(const float* A, const float* B, float* C, int N){
    for(int i = 0; i < N; i++)
        for(int k = 0; k < N; k++){

            float a = A[i*N + k];

            for(int j = 0; j < N; j++){
                C[i*N+j] += a * B[k*N + j];
            }
        }
}