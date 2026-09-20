#include <iostream>
#include "graph/parser.h"

int main(){
    try{
        Graph graph = GraphParser::parse("models/tests/test_graph.json");

        std::cout << "Parsed nodes:" << std::endl;
        for(const Node& node : graph.nodes())
            std::cout << node.name() << "(" << node.type() << ")" << std::endl;
        

        
        
        
        std::cout << std::endl << "Execution Order:" << std::endl;
        std::vector<int> order = graph.topological_sort();

        for(int index : order)
            std::cout << graph.nodes()[index].name() << std::endl;

    }catch(const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;

        return 1;
    }


    return 0;
}