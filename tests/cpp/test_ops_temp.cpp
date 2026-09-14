#include <fstream>  // Used for file read and write.
#include <iostream>
#include <vector>
#include <iomanip>  // For number accuracy.

#include "ops/operator.h"

int main(){
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<float> beta = {0.0f, 0.0f, 0.0f, 0.0f};

    std::vector<float> output(input.size());

    layer_norm(input.data(), gamma.data(), beta.data(), output.data(), static_cast<int>(input.size()), 1e-5f);

    std::cout << "LaterNorm result: " << std::endl;

    for(float value : output)
        std::cout << value << " ";

    std::cout<<std::endl;

    return 0;

}