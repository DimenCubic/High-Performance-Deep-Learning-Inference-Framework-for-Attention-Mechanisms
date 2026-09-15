#ifndef OPS_H
#define OPS_H

void softmax(
    const float* input,
    float* output,
    int size
);


void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int size,
    float epsilon
);


void gelu(
    const float* input,
    float* output,
    int size
);


#endif
