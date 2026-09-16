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



void test_qkv_projection(){
    const int seq_len = 3;
    const int hidden_dim = 4;
    const int head_dim = 2;

    std::vector<float> X = {
        1,2,3,4,
        5,6,7,8,
        9,10,11,12
    };

    std::vector<float> W_q = {
        1,0,
        0,1,
        1,0,
        0,1
    };


    std::vector<float> W_k = {
        1,1,
        0,1,
        1,0,
        0,1
    };


    std::vector<float> W_v = {
        1,0,
        1,0,
        0,1,
        0,1
    };


    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    std::vector<float> V(seq_len * head_dim);

    qkv_projection(X.data(), W_q.data(), W_k.data(), W_v.data(), Q.data(), K.data(), V.data(), seq_len, hidden_dim, head_dim);

    std::cout<<"Q:"<<std::endl;

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < head_dim; j++)
            std::cout<<Q[i*head_dim+j] << " ";

        std::cout<<std::endl;
    }

    std::cout<<std::endl;
    std::cout<<"K:"<<std::endl;

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < head_dim; j++)
            std::cout<<K[i*head_dim+j] << " ";

        std::cout<<std::endl;
    }

    std::cout<<std::endl;


    std::cout<<"V:"<<std::endl;

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < head_dim; j++)
            std::cout<<V[i*head_dim+j] << " ";

        std::cout<<std::endl;
    }



}



void test_attention_scores(){
    const int seq_len = 3;
    const int head_dim = 2;

    std::vector<float> Q = {
        4, 6,
        12, 14,
        20, 22
    };

    std::vector<float> K = {
        4, 7,
        12, 19,
        20, 31
    };

    std::vector<float> scores(
        seq_len * seq_len
    );


     attention_scores(
        Q.data(),
        K.data(),
        scores.data(),
        seq_len,
        head_dim
    );

    std::cout << "Attention Scores:" << std::endl;

    for(int i = 0; i < seq_len; i++)
    {
        for(int j = 0; j < seq_len; j++)
            std::cout << scores[i * seq_len + j] << " ";
        
        std::cout << std::endl;
    }
}



void test_attention_softmax(){
    const int seq_len = 3;

    std::vector<float> scores = {
        41.0122f, -INFINITY, -INFINITY,
        3.2376f, 2.9138f, -INFINITY,
        1.4630f, 0.2763f, 1.0896f
    };

    std::vector<float> weights(seq_len*seq_len);

     
    attention_softmax(scores.data(), weights.data(), seq_len);


    std::cout<< "Attention Weights:" << std::endl;

    for(int i = 0; i < seq_len; i++){
        float row_sum = 0.0f;

        for(int j = 0; j < seq_len; j++){
            float value = weights[i * seq_len + j];

            std::cout << value << " ";

            row_sum += value;
        }

        std::cout << " | sum = " << row_sum << std::endl;
    }


}



// Two helpers at here.
std::vector<float> read_values(const std::string& filename){ // & here means refernece outside file instead of copying one.
    std::ifstream file(filename);

    if(!file){
        std::cerr << "Failed to open "<<filename << std::endl;
        return {};
    }


    std::vector<float> values;
    float value;
    while(file >> value)
        values.push_back(value);

    return values;
}

void write_values(const std::string& filename, const std::vector<float>& values){
    std::ofstream file(filename);

    if(!file){
        std::cerr << "Failed to write: "<<filename<<std::endl;

        return;
    }

    file << std::setprecision(9);

    for(float value: values)
        file << value<< "\n";

}


void test_attention_end_to_end(){
    const int seq_len = 8;
    const int hiddden_dim = 16;
    const int head_dim = 8;

    // Load input data
    std::vector<float> X = read_values("tests/data/attention_input.txt");
    std::vector<float> W_q = read_values("tests/data/attention_wq.txt");
    std::vector<float> W_k = read_values("tests/data/attention_wk.txt");
    std::vector<float> W_v = read_values("tests/data/attention_wv.txt");


    // Validate Size
    if(X.size() != static_cast<size_t>(seq_len * hiddden_dim)){   // X is a vector, which .size() number is a size_t type, so we also need to transfer it to size_t.
        std::cerr << "Invalid attention input size." << std::endl;

        return;
    }


    if(W_q.size() != static_cast<size_t>(seq_len * hiddden_dim)||
       W_k.size() != static_cast<size_t>(seq_len * hiddden_dim)|| 
       W_v.size() != static_cast<size_t>(seq_len * hiddden_dim)){

        std::cerr << "Invalid attention weight size." << std::endl;

        return;
    }


    // Allocate Q, K, V
    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    std::vector<float> V(seq_len * head_dim);


    // QKV Projection
    qkv_projection(X.data(), W_q.data(), W_k.data(), W_v.data(), Q.data(), K.data(), V.data(), seq_len, hiddden_dim, head_dim);




    // Attention scores
    std::vector<float> scores(seq_len * seq_len);
    attention_scores(Q.data(), K.data(), scores.data(), seq_len, head_dim);


    // Attention softmax
    std::vector<float> weights(seq_len * seq_len);
    attention_softmax(scores.data(), weights.data(), seq_len);

    
    // final hidden values
    std::vector<float> output(seq_len * head_dim);
    scaled_dot_product_attention(Q.data(), K.data(), V.data(), output.data(), seq_len, head_dim);

    
    
    // Save all intermediate and final results.
    write_values("tests/data/attention_q_cpp.txt", Q);
    write_values("tests/data/attention_k_cpp.txt", K);
    write_values("tests/data/attention_v_cpp.txt", V);

    write_values("tests/data/attention_scores_cpp.txt", scores);
    write_values("tests/data/attention_weights_cpp.txt", weights);
    write_values("tests/data/attention_output_cpp.txt", output);

    std::cout << "End-to-End attention completed" << std::endl;

}


int main(){
    
    //test_softmax();
    //test_layernorm();
    //est_gelu();
    //test_matmul();
    //test_qkv_projection();
    //test_attention_scores();
    //test_attention_softmax();
    test_attention_end_to_end();

    return 0;

}