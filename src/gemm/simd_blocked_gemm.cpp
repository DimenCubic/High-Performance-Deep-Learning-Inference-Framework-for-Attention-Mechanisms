#include "gemm/gemm.h"
#include <arm_neon.h>

void simd_blocked_gemm(const float* A, const float* B, float* C, int N){
    for(int i = 0; i < N; i++){
        int j = 0;

        for(; j+3 < N; j += 4){
            float32x4_t c_vec = vld1q_f32(&C[i*N + j]);

            for(int k = 0; k < N; k++){
                float a = A[i*N + k];

                float32x4_t a_vec = vdupq_n_f32(a);
                float32x4_t b_vec = vld1q_f32(&B[k*N + j]);

                c_vec = vfmaq_f32(c_vec, a_vec, b_vec);
            }

            vst1q_f32(&C[i*N+j], c_vec);




        }

        for(; j < N; j++){
            float sum = C[i*N + j];

            for(int k = 0; k < N; k++){
                sum += A[i*N + k] * B[k*N + j];
            }

            C[i*N + j] = sum;
        }
    }
}