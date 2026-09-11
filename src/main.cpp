#include <iostream>
#include "gemm/gemm.h"
#include "utils/matrix.h"

int main(){
    int N = 70;
    int size = N*N;

    float A[size];
    float B[size];
    float C_base[size];
    float C_test[size];

    for(int i = 0; i < 5; i++){
        fill_random(A, size);
        fill_random(B, size);
        fill_zero(C_base, size);
        fill_zero(C_test, size);
        
        naive_gemm(A, B, C_base, N);
        //reordered_gemm(A, B, C_test, N);
        //unrolled_gemm(A, B, C_test, N);
        //blocked_gemm(A, B, C_test, N);
        blocked2_gemm(A,B,C_test, N);

        /*for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                std::cout << C[i*N + j] << " ";
            }

            std::cout<<std::endl;
        }*/

        std::cout<<std::boolalpha<<compare_matrices(C_base, C_test, size, 1e-4)<<std::endl;

    }

    

    return 0;
}