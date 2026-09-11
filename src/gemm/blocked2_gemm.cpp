#include "gemm/gemm.h"
#include <algorithm>


void blocked2_gemm(const float* A, const float* B, float* C, int N){
    const int BLOCK_SIZE = 128;


    // ii, jj will determine which block to be execute, basically, we just change thread from 1 to N to the small 64 thread once time.
    // Transfer large matrix into the small matrix.
    for(int ii = 0; ii < N; ii += BLOCK_SIZE){
        for(int jj = 0; jj < N; jj += BLOCK_SIZE){
            for(int kk = 0; kk < N; kk += BLOCK_SIZE){

                    int i_end = std::min(ii + BLOCK_SIZE, N);
                    int j_end = std::min(jj + BLOCK_SIZE, N);
                    int k_end = std::min(kk + BLOCK_SIZE, N);


                    for(int i = ii; i < i_end; i++){
                        for(int k = kk; k < k_end; k++){
                            float a = A[i*N + k];
                            
                            for(int j = jj; j < j_end; j++){

                                C[i*N + j] += a * B[k*N + j];
                            }
                        }
                    }

            }
        }
    }
}