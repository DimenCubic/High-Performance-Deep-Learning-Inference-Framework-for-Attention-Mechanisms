#include "ops/operator.h"
#include "cuda/cuda_ops.cuh"

#include <iostream>
#include <vector>

int main(){
    const int seq_len = 3;
    const int head_dim = 2;
    const float tolerance = 1e-5f;

    std::vector<float> Q = {
        4, 6,
        12, 14,
        20, 22
    };

    std::vector<float> K = {
        4, 7,
        12, 19, 
        20, 31
    };

    std::vector<float> cpu_scores(seq_len * seq_len);
    std::vector<float> gpu_scores(seq_len * seq_len);


    // CPU Reference
    attention_scores(Q.data(), K.data(), cpu_scores.data(), seq_len, head_dim);



    // CUDA Allocation
    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_scores = nullptr;

    const std::size_t qk_bytes = static_cast<std::size_t>(seq_len * head_dim) * sizeof(float);
    const std::size_t score_bytes = static_cast<std::size_t>(seq_len * seq_len) * sizeof(float);

    cudaMalloc(&d_Q, qk_bytes);
    cudaMalloc(&d_K, qk_bytes);
    cudaMalloc(&d_scores, score_bytes);

    cudaMemcpy(d_Q, Q.data(), qk_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K.data(), qk_bytes, cudaMemcpyHostToDevice);



    // CUDA Attention Scores
    cuda_attention_scores(d_Q, d_K, d_scores, seq_len, head_dim);

    cudaDeviceSynchronize();

    cudaMemcpy(gpu_scores.data(), d_scores, score_bytes, cudaMemcpyDeviceToHost);



    // Comapare
    bool correct = true;

    for(int i = 0; i < seq_len * seq_len; i++){
        const float cpu = cpu_scores[i];
        const float gpu = gpu_scores[i];

        if(std::isinf(cpu) && std::isinf(gpu))
            continue;


        const float diff = std::fabs(cpu - gpu);

        if(diff > tolerance){
            correct = false;

            std::cout << "Mismatcg at index" << i << std::endl;
            std::cout << "CPU = " << cpu << std::endl;
            std::cout << "GPU = " << gpu << std::endl;
            std::cout << "Diff = " << diff << std::endl;

            break;
        }
    }



    std::cout << "CUDA Attention Scores: " << (correct ? "PASS" : "FAIL") << std::endl;


    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_scores);

    return correct ? 0 : 1;
    
}