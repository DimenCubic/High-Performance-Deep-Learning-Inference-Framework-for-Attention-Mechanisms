#include "ops/operator.h"
#include <cmath>

void layer_norm(const float* input, const float* gamma, const float* beta, float* output, int size, float epsilon){
    // Calculate data mean
    float mean = 0.0f;

    for(int i = 0; i < size; i++) mean += input[i];

    mean /= size;



    // Calculate Variance
    float variance = 0.0f;

    for(int i = 0; i < size; i++){
        float diff = input[i] - mean;

        variance += diff*diff;
    }

    variance /= size;


    
    // Normalization factor
    float nor_factor = 1.0f / std::sqrt(variance + epsilon);



    // Normalization + Scale + Shift
    for(int i = 0; i < size; i++){
        float normalized = (input[i] - mean) * nor_factor;

        output[i] = gamma[i] * normalized + beta[i];
    } 
}