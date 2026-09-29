#ifndef CUDA_OPS_CUH
#define CUDA_OPS_CUH

void cuda_vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
);


void cuda_gelu(
    const float* input,
    float* output,
    int size
);


void cuda_softmax(
    const float* input,
    float* output,
    int size
);

#endif