#include "gemm/gemm.h"

#include <algorithm>
#include <arm_neon.h>
#include <omp.h>

void optimized_gemm(const float* A, const float* B, float* C, int N){
    const int BLOCK_SIZE = 128;

    // OpenMP: different ii represent different blocks which are operated on different rows of C
    #pragma omp parallel for schedule(static)
    for(int ii = 0; ii < N; ii += BLOCK_SIZE){
        for(int kk = 0; kk < N; kk += BLOCK_SIZE){
            for(int jj = 0; jj < N; jj += BLOCK_SIZE){
                const int i_end = std::min(ii + BLOCK_SIZE, N);
                const int k_end = std::min(kk + BLOCK_SIZE, N);
                const int j_end = std::min(jj + BLOCK_SIZE, N);

                for(int i = ii; i < i_end; i++){
                    const int c_base = i * N;
                    for(int k = kk; k < k_end; k++){
                        const int b_base = k * N;
                        const float a = A[i*N + k];

                        const float32x4_t a_vec = vdupq_n_f32(a); // Broadcast one a value for 4 SIMD lanes

                        int j = jj;

                        // SIMD + 4x loop unrolling at here
                        // 4 vecotrs x 4 floats = 16 float per loop iteration

                        for(; j + 15 < j_end; j += 16){
                            float32x4_t b0 = vld1q_f32(&B[b_base + j]);
                            float32x4_t b1 = vld1q_f32(&B[b_base + j + 4]);
                            float32x4_t b2 = vld1q_f32(&B[b_base + j + 8]);
                            float32x4_t b3 = vld1q_f32(&B[b_base + j + 12]);

                            float32x4_t c0 = vld1q_f32(&C[c_base + j]);
                            float32x4_t c1 = vld1q_f32(&C[c_base + j + 4]);
                            float32x4_t c2 = vld1q_f32(&C[c_base + j + 8]);
                            float32x4_t c3 = vld1q_f32(&C[c_base + j + 12]);

                            c0 = vfmaq_f32(c0, a_vec, b0);
                            c1 = vfmaq_f32(c1, a_vec, b1);
                            c2 = vfmaq_f32(c2, a_vec, b2);
                            c3 = vfmaq_f32(c3, a_vec, b3);

                            vst1q_f32(&C[c_base + j], c0);
                            vst1q_f32(&C[c_base + j + 4], c1);
                            vst1q_f32(&C[c_base + j + 8], c2);
                            vst1q_f32(&C[c_base + j + 12], c3);


                            // remaining SIMD Chunks
                            for(; j + 3 < j_end; j += 4){
                                float32x4_t b_vec = vld1q_f32(&B[b_base + j]);
                                float32x4_t c_vec = vld1q_f32(&C[c_base + j]);

                                c_vec = vfmaq_f32(c_vec, a_vec, b_vec);
                                vst1q_f32(&C[c_base + j], c_vec);
                            }


                            // Scalar Cleanup
                            for(; j < j_end; j++)
                                C[c_base + j] += a * B[b_base + j];
                        }
                    }
                }
            }
        }
    }
}