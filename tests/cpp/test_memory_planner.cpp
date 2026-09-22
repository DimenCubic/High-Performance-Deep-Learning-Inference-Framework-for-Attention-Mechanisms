#include <iostream>

#include "graph/parser.h"
#include "runtime/memory_planner.h"

int main(){
    Graph graph = GraphParser::parse("models/tests/test_full_attention_graph.json");

    MemoryPlanner planner;

    planner.analyze(graph);
    const auto& last_uses = planner.last_uses();

    for(const auto& entry : last_uses)
        std::cout << "Tensor " << entry.first << " last use: " << entry.second << std::endl;

    return 0;
}