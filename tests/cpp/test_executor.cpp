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

void test_executor_matmul(){
    Graph graph = GraphParser::parse("models/tests/test_matmul_graph.json");

    Tensor A("A", {3,4});
    Tensor B("B", {4,2});
    Tensor C("C", {3,2});

    float A_data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    float B_data[] = {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    };

    for(std::size_t i = 0; i < A.size(); i++)
        A[i] = A_data[i];

    for(std::size_t i = 0; i < B.size(); i++)
        B[i] = B_data[i];

    
    Executor executor;
    executor.add_tensor(A);
    executor.add_tensor(B);
    executor.add_tensor(C);

    executor.run(graph);

    const Tensor& result = executor.get_tensor("C");

    std::cout << "MatMul result:" << std::endl;

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 2; j++)
            std::cout << result[i * 2 + j] << " ";

        std::cout << std::endl;
    }

}


int main(){
    //softmax_test();
    test_executor_matmul();

    return 0;
}


