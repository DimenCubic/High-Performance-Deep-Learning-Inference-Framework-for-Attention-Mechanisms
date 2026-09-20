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
    if(node.type() == "AttentionSoftmax")
        execute_node_attention_softmax(node);
    else  
        throw std::runtime_error("Unsupported operator type: " + node.type());


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


 





