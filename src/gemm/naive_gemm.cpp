#include "gemm/gemm.h"

void naive_gemm(float* A, float* B, float* C, int N){

    for(int i = 0; i < N; i++){

        for(int j = 0; j < N; j++){

            float sum = 0;

            for(int k = 0; k < N; k++){
                sum += A[i*N + k] + B[k*N + j];
            }

            C[i*N + j] = sum;

        }
    }
}