#include <iostream>
#include <vector>

#include "gemm/gemm.h"
#include "utils/matrix.h"
#include "utils/timer.h"

int main(){
    int N = 2048;

    int matrix_size = N * N;

    std::vector<float> A(matrix_size);
    std::vector<float> B(matrix_size);
    std::vector<float> C(matrix_size);

    fill_random(A.data(), matrix_size); // .data() return the address of the first element of the matrix.
    fill_random(B.data(), matrix_size);
    fill_zero(C.data(), matrix_size);


/*
    Timer timer;
    timer.start();
    naive_gemm(A.data(), B.data(), C.data(), N);
    double elapsed = timer.stop();
*/

    // Wrap average version
    const int runs = 5;
    double total_time = 0.0;

    for(int i = 0; i < runs; i++){
        fill_zero(C.data(), matrix_size);

        Timer timer;
        timer.start();

        //naive_gemm(A.data(), B.data(), C.data(), N);
        //reordered_gemm(A.data(), B.data(), C.data(), N);
        //unrolled_gemm(A.data(), B.data(), C.data(), N);
        //blocked_gemm(A.data(), B.data(), C.data(), N);
        blocked2_gemm(A.data(), B.data(), C.data(), N);

        double  elapsed = timer.stop();
        total_time += elapsed;

    }

    double avg_time = total_time / runs;


    // Add GFLOPS Cal at here.
    double operations = 2.0 * N * N * N - 1.0 * N * N;
    double gflops = operations / avg_time / 1e9;



    std::cout << "Matrix Size: " << N << " x " << N << std::endl;
    std::cout << "Average Execution time: " << avg_time * 1000 << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    return 0;
}