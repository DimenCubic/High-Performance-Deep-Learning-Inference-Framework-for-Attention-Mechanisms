#include "ops/operator.h"
#include <cmath>


// QKT / root(dk)
void attention_scores(const float* Q, const float* K, float* scores, int seq_len, int head_dim){
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < seq_len; j++){
            
            if(j > i){
                scores[i * seq_len + j] = -INFINITY;
                continue;
            }
            
            float sum = 0.0f;

            for(int k = 0; k < head_dim; k++)
                sum += Q[i * head_dim + k] * K[j * head_dim + k];

            scores[i * seq_len + j] = sum * scale;
        }
        
        
    }

}