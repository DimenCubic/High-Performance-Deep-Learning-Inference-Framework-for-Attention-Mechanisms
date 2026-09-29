#include "cuda/cuda_ops.cuh"
#include "ops/operator.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>



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
    test_softmax();


    return 0;
}