#include <fstream>  // Used for file read and write.
#include <iostream>
#include <vector>
#include <iomanip>  // For number accuracy.

#include "ops/operator.h"

int main(){
    std::ifstream input_file("tests/data/softmax_input.txt");

    if(!input_file){
        std::cerr << "Failed to open softmax_input.txt" << std::endl;

        return 1;
    }


    std::vector<float> input;
    float value;


    while(input_file >> value){
        input.push_back(value);
    }

    std::vector<float> output(input.size());

    softmax(input.data(), output.data(), static_cast<int>(input.size())); // static_cast<int>  used to transfer size_t -> int


    std::ofstream output_file("tests/data/softmax_cpp_output.txt");


    if(!output_file){
        std::cerr << "Failed to create softmax_cpp_output.txt"
                  << std::endl;

        return 1;
    }


    output_file << std::setprecision(9);

    for(float x : output)
        output_file << x << "\n";


    std::cout << "Softmax completed for " << input.size() << "values." << std::endl;



}