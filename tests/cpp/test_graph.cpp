#include <iostream>
#include "graph/graph.h"


// Will do a QK to attention to softmax value 的 test case。
int main(){
    Graph graph;

    graph.add_node(
        Node(
            "q_projection",
            "Matmul", 
            {"input", "W_q"},
            {"Q"}
        )
    );


    graph.add_node(
        Node(
            "k_projection",
            "Matmul", 
            {"input", "W_k"},
            {"K"}
        )
    );


    graph.add_node(
        Node(
            "attention_scores",
            "AttentionScores", 
            {"Q", "K"},
            {"scores"}
        )
    );


    graph.add_node(
        Node(
            "softmax",
            "Softmax", 
            {"scores"},
            {"weights"}
        )
    );


    graph.build_dependencies();

    const auto& nodes = graph.nodes();
    const auto& edges = graph.edges();
    const auto& indegrees = graph.indegrees();


    for(std::size_t i = 0; i < nodes.size(); i++){
        std::cout<<"Node "<< i << ": "<<nodes[i].name()<<std::endl;

        std::cout << " Indegree: " << indegrees[i] << std::endl;

        std::cout<<" Edges to: ";

        for(int next : edges[i]){
            std::cout << nodes[next].name() << " ";
        }

        std::cout << std::endl;
    }


    return 0;
}