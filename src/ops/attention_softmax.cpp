#include "ops/operator.h"

void attention_softmax(const float* scores, float* weights, int seq_len){
    for(int i = 0; i < seq_len; i++)
        softmax(&scores[i * seq_len], &weights[i*seq_len], seq_len);
}