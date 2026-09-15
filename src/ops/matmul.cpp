#include "ops/operator.h"



// There is a reorder optimization could be done at here.
void matmul(const float* A, const float* B, float* C, int M, int K, int N){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            float sum = 0.0f;

            for(int k = 0; k < K; k++)
                sum += A[i*K + k] * B[k*N + j];

            C[i*N+j] = sum;
        }
    }
}

