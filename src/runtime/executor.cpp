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
    
    
    
    throw std::runtime_error("Unsupported operator type: " + node.type());


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


 





