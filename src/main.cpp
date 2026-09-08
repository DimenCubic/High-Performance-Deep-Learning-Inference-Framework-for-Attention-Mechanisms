#include <iostream>
#include "gemm/gemm.h"

int main(){
    int N = 4;
    
    float A[16];
    float B[16];
    float C[16];

    for(int i = 0; i < 16; i++){
        A[i] = 1.0;
        B[i] = 3.0;
        C[i] = 0;
    }

    //naive_gemm(A, B, C, N);
    reordered_gemm(A, B, C, N);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            std::cout << C[i*N + j] << " ";
        }

        std::cout<<std::endl;
    }

    return 0;
}