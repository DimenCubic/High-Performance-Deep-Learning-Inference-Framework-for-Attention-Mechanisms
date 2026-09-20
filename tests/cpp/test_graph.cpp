#include <iostream>
#include "graph/graph.h"



void basic_test(){
    Graph graph;


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




    // Test topological sort
    std::vector<int> order = graph.topological_sort();

    std::cout << "Execution Order:" << std::endl;
    for(int index : order)
        std::cout<<graph.nodes()[index].name() << std::endl;

}




void cycle_test(){
    Graph graph;

    graph.add_node(
    Node(
        "A",
        "Test",
        {"C_out"},
        {"A_out"}
    )
    );

    graph.add_node(
        Node(
            "B",
            "Test",
            {"A_out"},
            {"B_out"}
        )
    );

    graph.add_node(
        Node(
            "C",
            "Test",
            {"B_out"},
            {"C_out"}
        )
    );

    graph.build_dependencies();

    try{
        auto order = graph.topological_sort();
    }catch(const std::runtime_error& e){
        std::cout<<e.what()<<std::endl;
    }


}

// Will do a QK to attention to softmax value 的 test case。
int main(){
    basic_test();

    std::cout<<std::endl;
    
    cycle_test();
    
    
    return 0;
}