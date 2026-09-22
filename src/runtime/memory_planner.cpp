#include "runtime/memory_planner.h"
#include <stdexcept>

void MemoryPlanner::analyze(const Graph& graph){
    last_use_.clear();

    std::vector<int> order = graph.topological_sort();

    for(int step = 0; step < static_cast<int>(order.size()); step++){
        int node_index = order[step];
        const Node& node = graph.nodes()[node_index];

        for(const std::string& input : node.inputs())
            last_use_[input] = step;
    }
}


int MemoryPlanner::last_use(const std::string& tensor_name) const{
    auto it = last_use_.find(tensor_name);

    if(it == last_use_.end())
        throw std::runtime_error("No last_use information for tensor: " + tensor_name);

    return it -> second;
}



const std::unordered_map <std::string, int>& MemoryPlanner::last_uses() const{
    return last_use_;
}