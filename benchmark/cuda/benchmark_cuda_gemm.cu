#include "cuda/cuda_gemm.cuh"
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

int main(){
    const int N = 4096;
    const int warmup_runs = 10;
    const int benchmark_runs = 100;

    const std::size_t elements = static_cast<std::size_t>(N) * N;
    const std::size_t bytes = elements * sizeof(float);

    std::vector<float> A(elements, 1.0f);
    std::vector<float> B(elements, 1.0f);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    // Warmup
    for(int i = 0; i < warmup_runs; i++){
        //cuda_gemm_naive(d_A, d_B, d_C, N);
        //cuda_gemm_reordered(d_A, d_B, d_C, N);
        cuda_gemm_unrolled(d_A, d_B, d_C, N);
    }

        
    
    cudaDeviceSynchronize();  // Finish all warm up before start record the time.


    // Start formal process
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // 在GPU这个stream里面加入一个打点事件。

    for(int i = 0; i < benchmark_runs; i++){
        //cuda_gemm_naive(d_A, d_B, d_C, N);
        //cuda_gemm_reordered(d_A, d_B, d_C, N);
        cuda_gemm_unrolled(d_A, d_B, d_C, N);
    }
        

    cudaEventRecord(stop);  // Add another record spot. The place is on the end list of the benchmark kernels. 当GPU运行玩前面的所有实例后才会运行stop。
    cudaEventSynchronize(stop);  // 在执行下面的语句前等待全部执行完。


    
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    double average_ms = elapsed_ms / benchmark_runs;
    double average_secs = average_ms / 1000.0;


    double operations = static_cast<double>(N) * N *(2.0 * N - 1.0);
    double gflops = operations / average_secs / 1e9;

    //std::cout << "CUDA Naive GEMM Benchmark" << std::endl;
    std::cout<< "Matrix size: " << N << " x" << N << std::endl;
    std::cout << "Warmup  runs: " << warmup_runs << std::endl;
    std::cout << "Benchmark runs: " << benchmark_runs << std::endl;
    std::cout << "Average time:" << average_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    

    return 0;
}