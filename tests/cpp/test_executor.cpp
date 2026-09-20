#include <iostream>

#include "graph/parser.h"
#include "runtime/executor.h"
#include "tensor/tensor.h"




void softmax_test(){
    Graph graph = GraphParser::parse("models/tests/test_softmax_graph.json");

    Tensor input("scores", {3,3});
    input[0] = 41.0122f;
    input[1] = -INFINITY;
    input[2] = -INFINITY;

    input[3] = 3.2376f;
    input[4] = 2.9138f;
    input[5] = -INFINITY;

    input[6] = 1.4630f;
    input[7] = 0.2763f;
    input[8] = 1.0896f;

    Tensor output("weights", {3,3});



    Executor executor;
    executor.add_tensor(input);
    executor.add_tensor(output);

    executor.run(graph);

    const Tensor& result = executor.get_tensor("weights");

    std::cout<<"Softmax output:"<<std::endl;

    for(std::size_t i = 0; i < result.size(); i++){
        std::cout<<result[i] << " ";

        if((i+1) % 3 == 0) 
            std::cout<<"\n";

    }
        

    std::cout<<std::endl;


}


int main(){
    softmax_test();

    return 0;
}


