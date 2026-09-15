#include "ops/operator.h"
#include <cmath>

void gelu(const float* input, float* output, int size){
    const float div_sqrt_2 = 1.0f / std::sqrt(2.0f);

    for(int i = 0; i < size; i++){
        float x = input[i];

        output[i] = 0.5f * x * (1 + std::erf(x * div_sqrt_2));
    }
    

}