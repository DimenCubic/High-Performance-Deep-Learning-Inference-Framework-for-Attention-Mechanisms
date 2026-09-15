#include "ops/operator.h"

void qkv_projection(const float* input, const float* W_q, const float* W_k, const float* W_v, float* Q, float* K, float* V, int seq_len, int hidden_dim, int head_dim){
    
    // Calculate Q, K, V
    matmul(input, W_q, Q, seq_len, hidden_dim, head_dim);
    matmul(input, W_k, K, seq_len, hidden_dim, head_dim);
    matmul(input, W_v, V, seq_len, hidden_dim, head_dim);
}    