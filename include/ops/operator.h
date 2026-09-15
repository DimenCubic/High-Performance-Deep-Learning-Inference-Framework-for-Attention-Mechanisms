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



// Define: 
// A = M * K
// B = K * N
// C = M * N
void matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K, 
    int N
);

/*
input:
seq_len × hidden_dim

Wq:
hidden_dim × head_dim

Wk:
hidden_dim × head_dim

Wv:
hidden_dim × head_dim

Q/K/V:
seq_len × head_dim

*/
void qkv_projection(
    const float* input,
    const float* W_q,
    const float* W_k,
    const float* W_v,
    
    float* Q,
    float* K,
    float* V,

    int seq_len,   // # of token
    int hidden_dim,   // Dimension of every token
    int head_dim   //  类似于每个观察角度把hidden-dim浓缩为head-dim,最后观察完了再拼回去
);



void attention_scores(
    const float* Q,
    const float* K,
    float* scores,
    int seq_len,
    int head_dim
);



void attention_softmax(
    const float* scores, 
    float* weights, 
    int seq_len
);


void scaled_dot_product_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim
);


#endif
