#include "cuda/cuda_ops.cuh"
#include "ops/operator.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>



void test_layer_norm(){
    const int size = 513;
    const float epsilon = 1e-5f;
    const float tolerance = 1e-5f;

    std::vector<float> input(size);
    std::vector<float> gamma(size);
    std::vector<float> beta(size);

    for(int i = 0; i < size; i++){
        input[i] = static_cast<float>((i % 23) -11) * 0.25f;
        gamma[i] = 1.0f + static_cast<float>(i % 7) * 0.01f;
        beta[i] = static_cast<float>((i % 5) - 2) * 0.02f;
    }

    std::vector<float> cpu_output(size);
    std::vector<float> gpu_output(size);


    // CPU Reference
    layer_norm(input.data(), gamma.data(), beta.data(), cpu_output.data(), size, epsilon);


    // GPU Reference
    const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(float);

    float* d_input = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_gamma, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_beta, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, gpu_output.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta.data(), bytes, cudaMemcpyHostToDevice);


    cuda_layer_norm(d_input, d_gamma, d_beta, d_output, size, epsilon);
    cudaDeviceSynchronize(); // Wait finish all GPU before push CPU.

    cudaMemcpy(gpu_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);



    // Compare
    bool correct = true;
    float max_diff = 0.0f;
    int max_diff_index = -1;

    for(int i = 0; i < size; i++){
        const float diff = std::fabs(cpu_output[i] - gpu_output[i]);

        if(diff > max_diff){
            max_diff = diff;
            max_diff_index = i;
        }
            

        if(diff > tolerance)
            correct  = false;
    }



    // Display
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Max difference index: " << max_diff_index << std::endl;


    if(!correct){
        std::cout  << "CPU: " << cpu_output[max_diff_index] << std::endl;
        std::cout  << "GPU: " << gpu_output[max_diff_index] << std::endl;
    }


    std::cout << "CUDA LayerNorm: " << (correct ? "PASS" : "FAIL") << std::endl;

    
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);

}






void test_softmax(){
    const int size = 513;
    const float tolerance = 1e-5f;

    std::vector<float> input(size);
    for(int i = 0; i < size; i++)
        input[i] = static_cast<float>(i % 17 - 8) * 0.25f;

    std::vector<float> cpu_output(size);
    std::vector<float> gpu_output(size);

    
    // CPU Reference
    softmax(input.data(), cpu_output.data(), size);

    // GPU Reference
    float* d_input = nullptr;
    float* d_output = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    cuda_softmax(d_input, d_output, size);
    cudaDeviceSynchronize();

    cudaMemcpy(gpu_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);



    // Comaprision
    bool correct = true;

    float gpu_sum = 0.0f;

    for(int i = 0; i < size; i++){
        gpu_sum += gpu_output[i];

        float diff = std::fabs(cpu_output[i] - gpu_output[i]);

        if(diff > tolerance){
            correct = false;

            std::cout << "Mismatch at index " << i << std::endl;
            std::cout << "CPU: " << cpu_output[i] << std::endl;
            std::cout << "GPU: " << gpu_output[i] << std::endl;
            std::cout << "Diff: " << diff << std::endl;

            break;
        }
    }


    std::cout << "GPU Softmax sum: " << gpu_sum << std::endl;
    std::cout << "CUDA Softmax: " << (correct ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

 
}






void test_gelu(){
    const int size = 8;
    const float tolerance = 1e-5f;

    std::vector<float> input = { 
        -3.0f,
        -2.0f,
        -1.0f,
        0.0f,
        0.5f,
        1.0f,
        2.0f,
        3.0f
    };

    std::vector<float> cpu_output(size);
    std::vector<float> gpu_output(size);


    // CPU reference.
    gelu(input.data(), cpu_output.data(), size);


    // GPU Reference
    float* d_input = nullptr;
    float* d_output = nullptr;

    const std::size_t bytes = size * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);


    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    cuda_gelu(d_input, d_output, size);

    cudaDeviceSynchronize();
    cudaMemcpy(gpu_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;


    // Comparision
    for(int i = 0; i < size; i++){
        float diff = std::fabs(cpu_output[i] - gpu_output[i]);

        if(diff > tolerance){
            correct = false;
            std::cout << "Mismatch at index " << i << ": CPU = " << cpu_output[i] << ", GPU = " << gpu_output[i] << ", diff = " << diff << std::endl;

            break;
        }
    }


    std::cout << "CUDA GELU: " << (correct ? "PASS" : "FAIL") << std::endl;


    cudaFree(d_input);
    cudaFree(d_output);

}




int main(){
    //test_gelu();
    //test_softmax();
    test_layer_norm();


    return 0;
}