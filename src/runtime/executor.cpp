#include "runtime/executor.h"
#include "ops/operator.h"
#include <stdexcept>

void Executor::add_tensor(const Tensor& tensor){
    tensors_.insert_or_assign(tensor.name(), tensor);  // C++ dunction for hash map instance.
}


Tensor& Executor::get_tensor(const std::string& name){
    auto it = tensors_.find(name);  // If directly use tensor_[name] at here, if name doesn't exists, map will create an empty instance immediately.

    if(it == tensors_.end())
        throw std::runtime_error("Tensor not found: " + name);

    return it -> second;
}
 


const Tensor& Executor::get_tensor(const std::string& name) const{
    auto it = tensors_.find(name);

    if(it == tensors_.end())
        throw std::runtime_error("Tensor not found: " + name);

    return it -> second;
}



void Executor::run(const Graph& graph){
    std::vector<int > order = graph.topological_sort();

    for(int index : order){
        const Node& node = graph.nodes()[index];

        execute_node(node);
    };
}


// Next will be large execute node part.
void Executor::execute_node(const Node& node){
    if(node.type() == "AttentionSoftmax"){
        execute_node_attention_softmax(node);
        return;
    }
        
    
    if(node.type() == "MatMul"){
        execute_matmul(node);
        return;
    }
    
    
    if(node.type() == "GELU"){
        execute_gelu(node);
        return;
    }
    
    
    
    if(node.type() == "LayerNorm"){
        execute_layernorm(node);
        return;
    }


    if(node.type() == "AttentionScores"){
        execute_attention_scores(node);
        return;
    }
    
    
    throw std::runtime_error("Unsupported operator type: " + node.type());


}


void Executor::execute_attention_scores(const Node& node){
    if(node.inputs().size() != 2 || node.outputs().size() != 1)
        throw std::runtime_error("AttentionScores expects 2 inputs and 1 output.");
    

    Tensor& Q = get_tensor(node.inputs()[0]);
    Tensor& K = get_tensor(node.inputs()[1]);
    Tensor& scores = get_tensor(node.outputs()[0]);


    // Q and K must be 2D
    if(Q.shape().size() != 2 || K.shape().size() != 2 || scores.shape().size() != 2)
        throw std::runtime_error("AttentionScores requires 2D tensors.");
    


    int seq_len = Q.shape()[0];
    int head_dim = Q.shape()[1];


    // Q and K should have same shape
    if(K.shape()[0] != seq_len || K.shape()[1] != head_dim)
        throw std::runtime_error("Q and K shape mismatch.");
    


    // scores should be seq_len × seq_len
    if(scores.shape()[0] != seq_len || scores.shape()[1] != seq_len)
        throw std::runtime_error("Attention scores output shape mismatch.");
    


    attention_scores(
        Q.data(),
        K.data(),
        scores.data(),
        seq_len,
        head_dim
    );

}



void Executor::execute_layernorm(const Node& node){
    if(node.inputs().size() != 3 || node.outputs().size() != 1)
        throw std::runtime_error("LayerNorm expects input, gamma, beta and one output");

    Tensor& input = get_tensor(node.inputs()[0]);
    Tensor& gamma = get_tensor(node.inputs()[1]);
    Tensor& beta = get_tensor(node.inputs()[2]);
    Tensor& output = get_tensor(node.outputs()[0]);

    if(input.size() != gamma.size() || input.size() != beta.size() || input.size() != output.size())
        throw std::runtime_error("LayerNorm tensor size mismatch.");

    constexpr float epsilon = 1e-5;  // constexpr means fix value on the compile state and never change later.

    layer_norm(input.data(), gamma.data(), beta.data(), output.data(), static_cast<int>(input.size()), epsilon);

}



void Executor::execute_gelu(const Node& node){
    if(node.inputs().size() != 1 || node.outputs().size() != 1)
        throw std::runtime_error("GELU exoects 1 input and 1 output.");


    Tensor& input = get_tensor(node.inputs()[0]);
    Tensor& output = get_tensor(node.outputs()[0]);

    
    if(input.shape() != output.shape())
        throw std::runtime_error("GRLU input and output mismatch.");

    gelu(input.data(), output.data(), static_cast<int>(input.size()));
}




void Executor::execute_matmul(const Node& node){
    if(node.inputs().size() != 2 || node.outputs().size() != 1)
        throw std::runtime_error("Matmul expects 2 inputs and 1 output.");

    Tensor& A = get_tensor(node.inputs()[0]);
    Tensor& B = get_tensor(node.inputs()[1]);
    Tensor& C = get_tensor(node.outputs()[0]);


    // Cuurrently, out matmul only works for 2D.
    if(A.shape().size() !=2 || B.shape().size() != 2 || C.shape().size() != 2)
        throw std::runtime_error("Matmul requires 2D Tensors.");

    int M = A.shape()[0];
    int K = A.shape()[1];
    int B_K = B.shape()[0];
    int N = B.shape()[1];

    // A: M x K   B: K x N
    if(K != B_K)
        throw std::runtime_error("matmul input shape mismatch");
    
    // C: M x N
    if(C.shape()[0] != M || C.shape()[1] != N)
        throw std::runtime_error("matmul output shape mismatch");

    matmul(A.data(), B.data(), C.data(), M, K, N);
}




void Executor::execute_node_attention_softmax(const Node& node){
    if(node.inputs().size() != 1 || node.outputs().size() != 1)
        throw std::runtime_error("Softmax node Exepects 1 input and 1 output");

    Tensor& input = get_tensor(node.inputs()[0]);
    Tensor& output = get_tensor(node.outputs()[0]);

    if(input.size() != output.size())
        throw std::runtime_error("Softmax input/output size mismatch.");

     if(input.shape().size() != 2)
        throw std::runtime_error("Attention softmax must be square.");


    if(input.shape()[0] != input.shape()[1])
        throw std::runtime_error("Attention softmax input must be square.");

   

    int seq_len = input.shape()[0];
    
    attention_softmax(input.data(), output.data(), static_cast<int>(seq_len));

}


 





