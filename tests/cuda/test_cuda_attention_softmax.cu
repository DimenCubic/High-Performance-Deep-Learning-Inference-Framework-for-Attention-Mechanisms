#include "cuda/cuda_ops.cuh"
#include "ops/operator.h"

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>


int main(){
    const int seq_len = 4;
    const float tolerance = 1e-5f;

    std::vector<float> scores = {
        2.0f,     -INFINITY, -INFINITY, -INFINITY,

        1.0f,      3.0f,     -INFINITY, -INFINITY,

        0.5f,      1.0f,      2.0f,     -INFINITY,

        1.0f,      2.0f,      3.0f,      4.0f
    };


    std::vector<float> cpu_weights(seq_len * seq_len);
    std::vector<float> gpu_weights(seq_len * seq_len);


    // CPU Reference
    attention_softmax(scores.data(), cpu_weights.data(), seq_len);



    // CUDA Allocation
    float* d_scores = nullptr;
    float* d_weights = nullptr;

    const std::size_t bytes = static_cast<std::size_t>(seq_len * seq_len) * sizeof(float);

    cudaMalloc(&d_scores, bytes);
    cudaMalloc(&d_weights, bytes);

    cudaMemcpy(d_scores, scores.data(), bytes, cudaMemcpyHostToDevice);




    // CUDA Attention softmax
    cuda_attention_softmax(d_scores, d_weights, seq_len);

    cudaDeviceSynchronize();

    cudaMemcpy(gpu_weights.data(), d_weights, bytes, cudaMemcpyDeviceToHost);



    // Comapare
    bool correct = true;

    for(int i = 0; i < seq_len * seq_len; i++){
        const float cpu = cpu_weights[i];
        const float gpu = gpu_weights[i];

        if(std::isinf(cpu) && std::isinf(gpu))
            continue;


        const float diff = std::fabs(cpu - gpu);

        if(diff > tolerance){
            correct = false;

            std::cout << "Mismatch at index" << i << std::endl;
            std::cout << "CPU = " << cpu << std::endl;
            std::cout << "GPU = " << gpu << std::endl;
            std::cout << "Diff = " << diff << std::endl;

            break;
        }
    }



    std::cout << "CUDA Attention Softmax: " << (correct ? "PASS" : "FAIL") << std::endl;


    cudaFree(d_scores);
    cudaFree(d_weights);
    

    return correct ? 0 : 1;

}