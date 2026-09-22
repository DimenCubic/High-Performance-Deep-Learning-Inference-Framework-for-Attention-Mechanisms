#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cstddef>
#include <vector>

struct MemoryBlock {
    float* data = nullptr;
    std::size_t capacity = 0;
    bool in_use = false;
};

class MemoryPool {
    
    public:
        MemoryPool() = default;   // default constructor, means don't need to have any initialization when instance be created;

        // Used to destroy every memory space it declared;
        ~MemoryPool();


        float* allocate(std::size_t size);
        void release(float* ptr);
        
        std::size_t block_count() const;
        std::size_t total_capacity() const;


    private:
        std::vector<MemoryBlock> blocks_;
};


#endif