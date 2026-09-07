#include "utils/matrix.h"

#include <cmath>
#include <random>

void fill_random(float* matrix, int size){
    static std::mt19937 generator(42);   // Mersenne Twister random number genrator
                                         // Static matters here for static will keep generator between different call.

    static std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for(int i = 0; i < size; i++)
        matrix[i] = distribution(generator);
}


void fill_zero(float* matrix, int size){
    for(int i = 0; i < size; i++)
        matrix[i] = 0.0f;
}


// Due to the float number in computer will be something 
bool compare_matrices(const float* A, const float* B, int size, float tolerance){

    for(int i = 0; i < size; i++){
        if(std::fabs(A[i] - B[i]) > tolerance) return false;
    }

    return true;
}