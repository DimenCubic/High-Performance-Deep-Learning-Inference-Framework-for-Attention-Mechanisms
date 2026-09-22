#include <iostream>

#include "graph/parser.h"
#include "runtime/memory_planner.h"

int main(){
    Graph graph = GraphParser::parse("models/tests/test_full_attention_graph.json");

    MemoryPlanner planner;

    planner.analyze(graph);
    /*const auto& last_uses = planner.last_uses();
    const auto& births = planner.births();

    for(const auto& entry : last_uses)
        std::cout <<
         "Tensor " << entry.first << " last use: " << entry.second << std::endl;*/

    const std::string tensors[] = {
        "Q",
        "K",
        "V",
        "scores",
        "weights",
    };



    for(const std::string& name :tensors){
        std::cout
            << name
            << ": birth = "
            << planner.birth(name)
            << ", last use = "
            << planner.last_use(name)
            << ", intermediate = "
            << std::boolalpha
            << planner.is_intermediate(name)
            << std::endl;
    }


    return 0;
}