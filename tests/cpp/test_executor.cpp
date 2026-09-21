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



void test_executor_full_attention()
{
    Graph graph = GraphParser::parse("models/tests/test_full_attention_graph.json");


    Tensor X("X",{3, 4});
    Tensor W_q("W_q",{4, 2});
    Tensor W_k("W_k",{4, 2});
    Tensor W_v("W_v",{4, 2});
    Tensor Q( "Q", {3, 2});
    Tensor K("K",{3, 2});
    Tensor V( "V",{3, 2});
    Tensor scores("scores",{3, 3});
    Tensor weights("weights",{3, 3});
    Tensor output("output",{3, 2});

    float X_data[] = {
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 1, 1, 1
    };


    float W_q_data[] = {
        1, 0,
        0, 1,
        1, 0,
        0, 1
    };


    float W_k_data[] = {
        1, 0,
        0, 1,
        0, 1,
        1, 0
    };


    float W_v_data[] = {
        1, 0,
        0, 1,
        1, 1,
        1, -1
    };


    for(std::size_t i = 0; i < X.size(); i++)
        X[i] = X_data[i];
    

    for(std::size_t i = 0; i < W_q.size(); i++)
    {
        W_q[i] = W_q_data[i];
        W_k[i] = W_k_data[i];
        W_v[i] = W_v_data[i];
    }


    Executor executor;


    executor.add_tensor(X);

    executor.add_tensor(W_q);
    executor.add_tensor(W_k);
    executor.add_tensor(W_v);

    executor.add_tensor(Q);
    executor.add_tensor(K);
    executor.add_tensor(V);

    executor.add_tensor(scores);
    executor.add_tensor(weights);

    executor.add_tensor(output);


    executor.run(graph);


    const Tensor& Q_result = executor.get_tensor("Q");
    const Tensor& K_result = executor.get_tensor("K");
    const Tensor& V_result = executor.get_tensor("V");
    const Tensor& score_result = executor.get_tensor("scores");
    const Tensor& weight_result = executor.get_tensor("weights");
    const Tensor& output_result = executor.get_tensor("output");


    std::cout << "Q:" << std::endl;


    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 2; j++)
            std::cout << Q_result[i * 2 + j] << " ";
        

        std::cout << std::endl;
    }


    std::cout << std::endl << "K:" << std::endl;


    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 2; j++)
            std::cout << K_result[i * 2 + j] << " ";
        

        std::cout << std::endl;
    }


    std::cout << std::endl << "V:" << std::endl;


    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 2; j++)
            std::cout << V_result[i * 2 + j] << " ";
        

        std::cout << std::endl;
    }


    std::cout << std::endl << "Scores:" << std::endl;

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++)
            std::cout << score_result[i * 3 + j] << " ";
        

        std::cout << std::endl;
    }


    std::cout << std::endl << "Weights:" << std::endl;

    for(int i = 0; i < 3; i++){
        float row_sum = 0.0f;

        for(int j = 0; j < 3; j++) {
            float value = weight_result[i * 3 + j];

            std::cout << value << " ";
            row_sum += value;
        }

        std::cout << "| sum = " << row_sum << std::endl;
    }


    std::cout << std::endl << "Output:" << std::endl;

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 2; j++)
            std::cout << output_result[i * 2 + j] << " ";
        

        std::cout << std::endl;
    }
}



int main(){
    //test_attention_softmax();
    //test_executor_matmul();
    //test_executor_matmul_softmax();
    //test_executor_gelu();
    //test_executor_layernorm();
    //test_executor_attention_scores();
    test_executor_full_attention();

    return 0;
}


