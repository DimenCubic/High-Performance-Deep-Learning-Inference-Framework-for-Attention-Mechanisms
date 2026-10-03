#include "runtime/device_transfer.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>


void copy_host_to_device(const Tensor& host, Tensor& device){
    if(host.device() != DeviceType::CPU)
        throw std::runtime_error("Source tensor must be on CPU");

    if(device.device() != DeviceType::CUDA)
        throw std::runtime_error("Destination tensor must be on GPU");


    if(host.shape() != device.shape())
        throw std::runtime_error("H2D tensor shape mismatch");

    
    
    cudaError_t status = cudaMemcpy(device.data(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    if(status != cudaSuccess)
        throw std::runtime_error(std::string("H2D cudaMemcpy failed: ") + cudaGetErrorString(status));
}




void copy_device_to_host(const Tensor& device, Tensor& host){
    if(host.device() != DeviceType::CPU)
        throw std::runtime_error("Source tensor must be on CPU");

    if(device.device() != DeviceType::CUDA)
        throw std::runtime_error("Destination tensor must be on GPU");


    if(host.shape() != device.shape())
        throw std::runtime_error("D2H tensor shape mismatch");

    
    cudaError_t status = cudaMemcpy(host.data(), device.data(), device.size() * sizeof(float), cudaMemcpyDeviceToHost);

    if(status != cudaSuccess)
        throw std::runtime_error(std::string("D2H cudaMemcpy failed: ") + cudaGetErrorString(status));
}