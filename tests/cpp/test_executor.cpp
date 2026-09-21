#include <iostream>

#include "graph/parser.h"
#include "runtime/executor.h"
#include "tensor/tensor.h"




void test_attention_softmax(){
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


// Combination of the matmul and attention softmax.
void test_executor_matmul_softmax(){
    Graph graph = GraphParser::parse("models/tests/test_matmul_softmax_graph.json");


    Tensor A("A",{3, 2});
    Tensor B("B",{2, 3});
    Tensor scores("scores",{3, 3});
    Tensor weights("weights",{3, 3});


    float A_data[] = {
        1, 0,
        0, 1,
        1, 1
    };

    float B_data[] = {
        1, 2, 3,
        4, 5, 6
    };


    for(std::size_t i = 0; i < A.size(); i++)
        A[i] = A_data[i];
    

    for(std::size_t i = 0; i < B.size(); i++)
         B[i] = B_data[i];


    Executor executor;

    executor.add_tensor(A);
    executor.add_tensor(B);
    executor.add_tensor(scores);
    executor.add_tensor(weights);


    executor.run(graph);


    const Tensor& result = executor.get_tensor("weights");


    std::cout << "Attention Softmax result:" << std::endl;


    for(int i = 0; i < 3; i++)
    {
        float row_sum = 0.0f;

        for(int j = 0; j < 3; j++)
        {
            float value = result[i * 3 + j]; 
            std::cout << value << " ";
            row_sum += value;
        }

        std::cout << "| sum = " << row_sum << std::endl;
    }
}



void test_executor_gelu()
{
    Graph graph =GraphParser::parse("models/tests/test_gelu_graph.json");

    Tensor input("input",{5});
    Tensor output("output",{5});


    float input_data[] = {
        -3.0f,
        -1.0f,
        0.0f,
        1.0f,
        3.0f
    };


    for(std::size_t i = 0; i < input.size(); i++)
        input[i] = input_data[i];
    


    Executor executor;

    executor.add_tensor(input);
    executor.add_tensor(output);


    executor.run(graph);


    const Tensor& result = executor.get_tensor("output");


    std::cout << "GELU result:" << std::endl;


    for(std::size_t i = 0; i < result.size(); i++)
        std::cout << result[i] << " ";
    

    std::cout << std::endl;
}



void test_executor_layernorm()
{
    Graph graph = GraphParser::parse("models/tests/test_layernorm_graph.json");


    Tensor input("input",{4});
    Tensor gamma("gamma", {4});
    Tensor beta("beta",{4});
    Tensor output("output",{4});


    float input_data[] = {
        1.0f,
        2.0f,
        3.0f,
        4.0f
    };


    for(std::size_t i = 0; i < input.size(); i++)
    {
        input[i] = input_data[i];
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
    }


    Executor executor;

    executor.add_tensor(input);
    executor.add_tensor(gamma);
    executor.add_tensor(beta);
    executor.add_tensor(output);


    executor.run(graph);


    const Tensor& result = executor.get_tensor( "output");


    std::cout << "LayerNorm result:" << std::endl;


    for(std::size_t i = 0; i < result.size(); i++)
        std::cout << result[i] << " ";
    

    std::cout << std::endl;
}



void test_executor_attention_scores()
{
    Graph graph = GraphParser::parse("models/tests/test_attention_scores_graph.json");

    Tensor Q("Q",{3, 2});
    Tensor K("K", {3, 2});
    Tensor scores("scores",{3, 3});


    float Q_data[] = {
        4, 6,
        12, 14,
        20, 22
    };

    float K_data[] = {
        4, 7,
        12, 19,
        20, 31
    };


    for(std::size_t i = 0; i < Q.size(); i++)
    {
        Q[i] = Q_data[i];
        K[i] = K_data[i];
    }


    Executor executor;

    executor.add_tensor(Q);
    executor.add_tensor(K);
    executor.add_tensor(scores);


    executor.run(graph);


    const Tensor& result = executor.get_tensor("scores");


    std::cout << "Attention Scores:" << std::endl;


    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++)
            std::cout << result[i * 3 + j] << " ";
        
        std::cout << std::endl;
    }
}




int main(){
    //test_attention_softmax();
    //test_executor_matmul();
    //test_executor_matmul_softmax();
    //test_executor_gelu();
    //test_executor_layernorm();
    test_executor_attention_scores();

    return 0;
}


