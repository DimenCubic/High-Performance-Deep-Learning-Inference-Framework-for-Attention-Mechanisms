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
    cudaError_t status = 
}