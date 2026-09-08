#include <iostream>
#include <vector>

#include "gemm/gemm.h"
#include "utils/matrix.h"
#include "utils/timer.h"

int main(){
    int N = 512;

    int matrix_size = N * N;

    std::vector<float> A(matrix_size);
    std::vector<float> B(matrix_size);
    std::vector<float> C(matrix_size);

    fill_random(A.data(), matrix_size); // .data() return the address of the first element of the matrix.
    fill_random(B.data(), matrix_size);
    fill_zero(C.data(), matrix_size);

    
    Timer timer;
    timer.start();
    naive_gemm(A.data(), B.data(), C.data(), N);
    double elapsed = timer.stop();

    std::cout << "Matrix Size: " << N << " x" << N << std::endl;
    std::cout << "Execution time: " << elapsed * 1000 << " ms" << std::endl;

    return 0;
}