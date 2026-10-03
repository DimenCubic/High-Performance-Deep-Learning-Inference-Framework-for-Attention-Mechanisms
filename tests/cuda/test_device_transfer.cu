#include "runtime/cuda_memory_pool.h"
#include "runtime/device_transfer.h"
#include "tensor/tensor.h"

#include <iostream>

int main(){
    const std::vector<int> shape = {8,4};

    // CPU Tensor
    Tensor host_input("host_input", shape);

    for(std::size_t i = 0; i < host_input.size(); i++)
        host_input[i] = static_cast<float>(i+1);

    
    // CUDA Allocation
    CudaMemoryPool cuda_pool;

    float* device_buffer = cuda_pool.allocate(host_input.size());

    Tensor device_tensor("device_tensor", shape, device_buffer, DeviceType::CUDA);


    
    // H2D
    copy_host_to_device(host_input, device_tensor);


    // D2H
    Tensor host_output("host_output", shape);

    copy_device_to_host(device_tensor, host_output);




    // Validate
    bool correct = true;

    for(std::size_t i = 0; i < host_input.size(); i++)
        if(host_input[i] != host_output[i]){
            correct = false;
            break;
        }

    
    std::cout << "H2D / D2H Transfer: " << (correct ? "PASS" : "FAILED") << std::endl;

    cuda_pool.release(device_buffer);
}