#ifndef CUDA_MEMORY_POOL_H
#define CUDA_MEMORY_POOL_H

#include <cstddef>   // 提供诸如size_t 指针差之类的类型
#include <vector>



struct CudaMemoryBlock{
    float* data = nullptr;
    std::size_t capacity = 0;
    bool in_use = false;
};


class CudaMemoryPool{
    public:
        CudaMemoryPool() = default;
        ~CudaMemoryPool();


        float* allocate(std::size_t size);
        void release(float* ptr);

        std::size_t block_count() const;
        std::size_t total_capacity() const;


    private:
        std::vector<CudaMemoryBlock> blocks_;

};


#endif