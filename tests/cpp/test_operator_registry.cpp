#include "ops/operator_registry.h"
#include "ops/cpu/gelu_operator.h"

#include "tensor/tensor.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>


int main(){
    try{

        // Craete a registry.
        OperatorRegistry registry;




        // Record all implementation way.
        registry.registry_operator(
            "GELU",
            DeviceType::CPU,
            []() -> std::unique_ptr <Operator>{
                return std::make_unique<CpuGeluOperator>();
            }
        );

        if(!registry.contains("GELU", DeviceType::CPU))
            throw std::runtime_error("CPU GELU registration failed.");

        if(registry.contains("GELU", DeviceType::CUDA))
            throw std::runtime_error("Unexepected CUDA GELU registration.");

        


        // Create operator dynamically.
        std::unique_ptr<Operator> op = registry.create("GELU", DeviceType::CPU);
        



        // Prepare data
        Tensor input("input", {4});
        Tensor output("output", {4});

        input[0] = -1.0f;
        input[1] = 0.0f;
        input[2] = 1.0f;
        input[3] = 2.0f;

        std::vector<const Tensor*> inputs = {&input};
        std::vector<Tensor*> outputs = {&output};




        // Execute by using base pointer
        op->execute(inputs, outputs);





        // Validate results.
        constexpr float tolerance = 1e-5f;

        for(std::size_t i = 0; i < input.size(); i++){
            const float x = input[i];
            const float expected = 0.5f * x * (1.0f + std::erf(x/std::sqrt(2.0f)));


            if(std::fabs(output[i] - expected) > tolerance)
                throw std::runtime_error("GELU output mismatch");
        }

        
        
        std::cout << "Operator Registry: PASS" << std::endl;
        std::cout << "CPU GELU Polymorphism: PASS" << std::endl;

        return 0;

    } catch(const std::exception& e){
        std::cerr << "Operator Registry Test: FAIL \n" << e.what() << std::endl;

        return 1;
    }
}