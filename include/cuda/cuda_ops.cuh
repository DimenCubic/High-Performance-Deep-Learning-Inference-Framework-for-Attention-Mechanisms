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

void cuda_layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int size,
    float epsilon
);


void cuda_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
);

#endif