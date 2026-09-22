#include "runtime/executor.h"
#include "ops/operator.h"
#include <stdexcept>

void Executor::add_tensor(const Tensor& tensor){
    tensors_.insert_or_assign(tensor.name(), tensor);  // C++ dunction for hash map instance.
    tensor_shapes_.insert_or_assign(tensor.name(), tensor.shape());
}

void Executor::register_tensor(const std::string& name, const std::vector<int>& shape){
    tensor_shapes_.insert_or_assign(name, shape);
}


void Executor::allocate_tensor_if_needed(const std::string& name){
    // Already exists.
    if(tensors_.find(name) != tensors_.end()) return;



    auto shape_it = tensor_shapes_.find(name);
    // This tensor not been registered.
    if(shape_it == tensor_shapes_.end())
        throw std::runtime_error("Tensor shape not registered: " + name);

    const std::vector<int>& shape = shape_it -> second;

    std::size_t size = 1;
    for(int dim : shape)
        size *= static_cast<std::size_t>(dim);



    float* buffer = memory_pool_.allocate(size);
    Tensor tensor(name, shape, buffer);
    tensors_.insert_or_assign(name, tensor);
    
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






// Running Helper
void Executor::release_tensor_if_dead(const std::string& name, int step){
    if(!memory_planner_.is_intermediate(name))
        return;

    const TensorLifetime& lifetime = memory_planner_.lifetime(name);

    if(lifetime.last_use != step)
        return;

    
    auto it = tensors_.find(name);
    if(it == tensors_.end())
        return;

    
    Tensor& tensor = it -> second;
    if(tensor.owns_memory())
        return;


    memory_pool_.release(tensor.data());

    // Delete this map in the hash maps.
    tensors_.erase(it);

}





void Executor::run(const Graph& graph){
    memory_planner_.analyze(graph);
    std::vector<int> order = graph.topological_sort();

    for(int step = 0; step < static_cast<int>(order.size()); step ++){
        int node_index = order[step];
        const Node& node = graph.nodes()[node_index];

        // Allocate output tensors. (all inputs 肯定被处理好了)
        for(const std::string& output : node.outputs())
            allocate_tensor_if_needed(output);


        // Execute operator
        execute_node(node);


        
        // Release dead input tensors
        for(const std::string& input : node.inputs())
            release_tensor_if_dead(input, step);



    }
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


 





