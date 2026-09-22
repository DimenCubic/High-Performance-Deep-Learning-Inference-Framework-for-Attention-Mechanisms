#include "runtime/memory_pool.h"

#include <stdexcept>


float* MemoryPool::allocate(std::size_t size){
    if(size == 0)
        throw std::invalid_argument("MemoryPool cannot allocate 0 elements");

    
    // Try to reuse an existing free block
    // Now will only implment a first come first use, later we can do sort.
    for(MemoryBlock& block : blocks_){
        if(!block.in_use && block.capacity >= size){
            block.in_use = true;
            return block.data;
        }
    }


    // If no existed block, then reallocate a new block.
    MemoryBlock block;
    block.data = new float[size];
    block.capacity = size;
    block.in_use = true;
    blocks_.push_back(block);
    

    return block.data;
}



// Here is diffrent from delete memory, we only free the memory blocks already existed
void MemoryPool::release(float* ptr){
    if(ptr == nullptr)
        throw std::invalid_argument("Cannot release null pointer.");

    for(MemoryBlock& block : blocks_){
        if(block.data == ptr){
            if(!block.in_use)
                throw std::runtime_error("Memory block is already free.");
        

            block.in_use = false;

            return;
        }

    }


    throw std::runtime_error("Pointer does not belong to Memory Pool.");
}




MemoryPool::~MemoryPool(){
    for(MemoryBlock& block : blocks_){
        delete[] block.data;
        block.data = nullptr;
    }
}





// For test, also for a quick way to check status of the memory pool.
std::size_t MemoryPool::block_count() const{
    return blocks_.size();
}


std::size_t MemoryPool::total_capacity() const{
    std::size_t total = 0;

    for(const MemoryBlock& block : blocks_)
        total += block.capacity;

    return total;
}