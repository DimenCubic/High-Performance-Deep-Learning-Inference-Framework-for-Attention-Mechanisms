#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "graph/graph.h"
#include "tensor/tensor.h"

#include <string>
#include <unordered_map>

class Executor{
    
    public:

        void add_tensor(const Tensor& tensor);

        Tensor& get_tensor(const std::string& name);
        const Tensor& get_tensor(const std::string& name) const;

        void run(const Graph& graph);



    private:
        std::unordered_map<std::string, Tensor> tensors_;

        void execute_node(const Node& node);

        // sub function of the execute_node.
        void execute_node_attention_softmax(const Node& node);
        void execute_matmul(const Node& node);
        void execute_gelu(const Node& node);
        void execute_layernorm(const Node& node);
        void execute_attention_scores(const Node& node);

};

#endif