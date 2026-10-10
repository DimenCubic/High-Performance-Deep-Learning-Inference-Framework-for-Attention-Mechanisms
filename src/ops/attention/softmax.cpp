#include "ops/operator.h"
#include <algorithm>
#include <cmath>

// If we don't substract max, then float will not big enough to contain num.
void softmax(const float* input, float* output, int size){
    float max_num = input[0];

    for(int i = 1; i < size; i++){
        max_num = std::max(max_num, input[i]);
    }

    float sum = 0.0f;

    for(int i = 0; i < size; i++){
        output[i] = std::exp(input[i] - max_num);
        sum += output[i];
    }

    for(int i = 0; i < size; i++){
        output[i] /= sum;
    }
}