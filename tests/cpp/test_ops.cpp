#include <fstream>  // Used for file read and write.
#include <iostream>
#include <vector>
#include <iomanip>  // For number accuracy.

#include "ops/operator.h"


void test_layernorm(){
    std::ifstream input_file("tests/data/layernorm_input.txt");
    std::ifstream gamma_file("tests/data/layernorm_gamma.txt");
    std::ifstream beta_file("tests/data/layernorm_beta.txt");

    if(!input_file || !gamma_file || !beta_file){
        std::cerr << "Failed to open LayerNorm input file" << std::endl;

        return;
    }

    std::vector<float> input;
    std::vector<float> gamma;
    std::vector<float> beta;

    float value;
    while(input_file >> value) input.push_back(value);
    while(gamma_file >> value) gamma.push_back(value);
    while(beta_file >> value) beta.push_back(value);



    if(input.size() != gamma.size() || input.size() != beta.size()){
        std::cerr << "LayerNorm input sizes do not match." << std::endl;

        return;
    }


    // Output
    std::vector<float> output(input.size());

    layer_norm(input.data(), gamma.data(), beta.data(), output.data(), static_cast<int>(input.size()), 1e-5f);


    std::ofstream output_file("tests/data/layernorm_cpp_output.txt");
    output_file << std::setprecision(9);

    for(float x : output){
        output_file << x << "\n";
    }


    std::cout << "LayerNorm completed for " << input.size() << " values." << std::endl;

}





void test_softmax(){
    std::ifstream input_file("tests/data/softmax_input.txt");

    if(!input_file){
        std::cerr << "Failed to open softmax_input.txt" << std::endl;

        return;
    }


    std::vector<float> input;
    float value;


    while(input_file >> value){
        input.push_back(value);
    }

    std::vector<float> output(input.size());


    // Test function gather.
    softmax(input.data(), output.data(), static_cast<int>(input.size())); // static_cast<int>  used to transfer size_t -> int


    std::ofstream output_file("tests/data/softmax_cpp_output.txt");


    if(!output_file){
        std::cerr << "Failed to create softmax_cpp_output.txt"
                  << std::endl;

        return;
    }


    output_file << std::setprecision(9);

    for(float x : output)
        output_file << x << "\n";


    std::cout << "Softmax completed for " << input.size() << "values." << std::endl;
}




void test_gelu(){
    std::ifstream input_file("tests/data/gelu_input.txt");

    if(!input_file){
        std::cerr << "Failed to open GELU input file." << std::endl;
        return;
    }

    std::vector<float> input;
    float value;

    while(input_file>>value)
        input.push_back(value);


    std::vector<float> output(input.size());

    gelu(input.data(), output.data(), static_cast<int>(input.size()));


    /*std::cout<<"DELU result:"<<std::endl;
    for(float value : output)
        std::cout<<value<<" ";


    std::cout<<std::endl;*/


    std::ofstream output_file("tests/data/gelu_cpp_output.txt");
    output_file << std::setprecision(9);

    for(float x : output)
        output_file << x <<"\n";


    std::cout<< "GELU completed for " << input.size() << " values." << std::endl;


    
}



void test_matmul(){
    const int M = 3;
    const int K = 4;
    const int N = 2;

    std::vector<float> A = {
        1,2,3,4,
        5,6,7,8
        ,9,10,11,12
    };

    std::vector<float> B = {
        1,2,
        3,4,
        5,6,
        7,8
    };

    std::vector<float>C(M*N, 0.0f);


    matmul(A.data(), B.data(), C.data(), M, K, N);

    std::cout<< "Matmul result:"<< std::endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            std::cout<< C[i*N + j] << " ";
        }

        std::cout << std::endl;
    }
    
}









int main(){
    
    //test_softmax();
    //test_layernorm();
    //est_gelu();
    test_matmul();

    return 0;

}