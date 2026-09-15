#include "ops/operator.h"

#include <cmath>
#include <vector>


void scaled_dot_product_attention(const float* Q, const float* K, const float* V, float* output, int seq_len, int head_dim){
    std::vector<float> scores(seq_len * seq_len);
    std::vector<float> weights(seq_len * seq_len);

    // Calculate scores first
    attention_scores(Q, K, scores.data(), seq_len, head_dim);


    // Attension softmax
    attention_softmax(scores.data(), weights.data(), seq_len);

    // Attention Weights * V, calculate specific value
    matmul(weights.data(), V, output, seq_len, seq_len, head_dim);
}