#include "gemm/gemm.h"
#include <arm_neon.h>

void simd_gemm(const float* A, const float* B, float* C, int N){
    
    for(int i = 0; i < N; i++){
        for(int k = 0; k < N; k++){
            float a = A[i*N + k];

            float32x4_t a_vec = vdupq_n_f32(a);   // float32x4_t代表一个a_vec装了四个float 32bit的向量

            int j = 0;

            for(; j + 3 < N; j += 4){
                float32x4_t b_vec = vld1q_f32(&B[k*N + j]);  // Based on the current B position to load 4 float numbers.
                float32x4_t c_vec = vld1q_f32(&C[i*N + j]);  // 

                c_vec = vfmaq_f32(c_vec, a_vec, b_vec); // FMA: Fused Multiply-Add, matrix multiplication. c = c + a*b

                vst1q_f32(&C[i*N+j], c_vec);

            }

            for(; j < N; j++){
                C[i * N + j] += a * B[k*N + j];
            }

        }
    }
}