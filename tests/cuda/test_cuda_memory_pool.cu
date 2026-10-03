#include "runtime/cuda_memory_pool.h"
#include <iostream>

int main(){
    CudaMemoryPool pool;


    // Allocate first A.
    float* A = pool.allocate(100);
    
    std::cout << "After A:" << std::endl;
    std::cout << "Blocks: " << pool.block_count() << std::endl;
    std::cout << "Capacity: " << pool.total_capacity() << std::endl;

    pool.release(A);

    std::cout << std::endl;



    // Allocate reused B.
    float* B = pool.allocate(80);

    std::cout << std::boolalpha;
    std::cout << "A reused by B: " << (A == B) << std::endl;
    std::cout << "After B:" << std::endl;
    std::cout << "Blocks: " << pool.block_count() << std::endl;
    std::cout << "Capacity: " << pool.total_capacity() << std::endl;
    std::cout << std::endl;


    // Create new block C
    float* C = pool.allocate(200);

    std::cout << "After C:" << std::endl;
    std::cout << "Blocks: " << pool.block_count() << std::endl;
    std::cout << "Capacity: " << pool.total_capacity() << std::endl;
    std::cout << std::endl;

    pool.release(B);
    pool.release(C);



    // Reuse C for D block.
    float* D = pool.allocate(150);
    std::cout << "C reused by D: " << (C == D) << std::endl;
    std::cout << "After D:" << std::endl;
    std::cout << "Blocks: " << pool.block_count() << std::endl;
    std::cout << "Capacity: " << pool.total_capacity() << std::endl;
}