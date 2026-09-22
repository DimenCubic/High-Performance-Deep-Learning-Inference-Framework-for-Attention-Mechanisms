#include "runtime/memory_planner.h"
#include <stdexcept>

void MemoryPlanner::analyze(const Graph& graph){
    
    birth_.clear();
    last_use_.clear();

    std::vector<int> order = graph.topological_sort();

    for(int step = 0; step < static_cast<int>(order.size()); step++){
        int node_index = order[step];
        const Node& node = graph.nodes()[node_index];


        for(const std::string& input : node.inputs())
            last_use_[input] = step;

        for(const std::string& output : node.outputs())
            birth_[output] = step;
    }

 


}


int MemoryPlanner::last_use(const std::string& tensor_name) const{
    auto it = last_use_.find(tensor_name);

    if(it == last_use_.end())
        throw std::runtime_error("No last_use information for tensor: " + tensor_name);

    return it -> second;
}


int MemoryPlanner::birth(const std::string& tensor_name) const{
    auto it = birth_.find(tensor_name);

    if(it == birth_.end())
        throw std::runtime_error("No birth information for tensor: " + tensor_name);
    
    return it -> second;
}


const std::unordered_map <std::string, int>& MemoryPlanner::last_uses() const{
    return last_use_;
}


const std::unordered_map <std::string, int>& MemoryPlanner::births() const{
    return birth_;
} 


// Actually I thought we don't need to keep focusing on the intermediate but maybe it's the convention that 
// we always want to keep the input and output not losed.
bool MemoryPlanner::is_intermediate(const std::string& tensor_name) const{
    bool has_birth = birth_.find(tensor_name) != birth_.end();
    bool has_last_use = last_use_.find(tensor_name) != last_use_.end();

    return has_birth && has_last_use;
}