#include <iostream>

#include "runtime/memory_pool.h"


int main()
{
    MemoryPool pool;

    // A
    float* A = pool.allocate(100);


    std::cout << "After allocating A:"  << std::endl;
    std::cout << "Blocks: " << pool.block_count() << std::endl;
    std::cout << "Capacity: " << pool.total_capacity()  << std::endl;
    
    pool.release(A);



    // Test reuse B
    float* B = pool.allocate(80);


    std::cout << std::endl << "After allocating B:" << std::endl;
    std::cout << "Blocks: " << pool.block_count() << std::endl;
    std::cout << "Capacity: " << pool.total_capacity() << std::endl;
    std::cout << std::boolalpha;

    std::cout << "A and B use same memory: " << (A == B) << std::endl;



    // Test create new C
    float* C = pool.allocate(200);

    std::cout << std::endl << "After C:" << std::endl;
    std::cout << "Blocks = " << pool.block_count() << std::endl;
    std::cout << "Capacity = " << pool.total_capacity() << std::endl;


    pool.release(B);
    pool.release(C);




    // D expects reuse C.
    float* D = pool.allocate(150);

    std::cout << std::endl << "C/D same block: " << (C == D) << std::endl;
    pool.release(D);



    return 0;
}