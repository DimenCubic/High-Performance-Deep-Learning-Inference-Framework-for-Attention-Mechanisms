#include "runtime/cuda_memory_pool.h"
#include <cuda_runtime.h>
#include <stdexcept>   // contains errors such as runtime error

float* CudaMemoryPool::allocate(std::size_t size){
    if(size == 0)
        throw std::invalid_argument("CudaMemoryPool cannot allocate 0 element");
    
    // Try to reuse an existing free block.
    for(CudaMemoryBlock& block : blocks_){
        if(!block.in_use && block.capacity >= size){
            block.in_use = true;
            return block.data;
        }
    }


    // Can't find a qualified block, create one.
    CudaMemoryBlock block;
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&block.data), size * sizeof(float));

    if(status != cudaSuccess)
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(status));

    block.capacity = size;
    block.in_use = true;

    blocks_.push_back(block);

    return block.data;
}




// Release Operation
void CudaMemoryPool::release(float* ptr){
    if(ptr == nullptr)
        throw std::invalid_argument("cannot release null pointer");

    for(CudaMemoryBlock& block : blocks_){
        if(block.data == ptr){
            if(!block.in_use)
                throw std::runtime_error("CUDA Memory Block is already free");

            block.in_use = false;

            return;
        }
    }


    throw std::runtime_error("Pointer does not belong to CudaMemoryPool");
}



// Deconstructor
CudaMemoryPool::~CudaMemoryPool(){
    for(CudaMemoryBlock& block : blocks_){
        if(block.data != nullptr){
            cudaFree(block.data);

            block.data = nullptr;
        }
    }
}



// State Function
std::size_t CudaMemoryPool::block_count() const{
    return blocks_.size();
}


std::size_t CudaMemoryPool::total_capacity() const{
    std::size_t total = 0;

    for(const CudaMemoryBlock block : blocks_)
        total += block.capacity;

    return total;
}