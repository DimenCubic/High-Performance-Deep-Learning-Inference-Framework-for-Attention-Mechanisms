#include "runtime/memory_planner.h"
#include <stdexcept>

void MemoryPlanner::analyze(const Graph& graph){
    

    /*
    birth_.clear();
    last_use_.clear();
    */

    lifetimes_.clear();

    std::vector<int> order = graph.topological_sort();

    for(int step = 0; step < static_cast<int>(order.size()); step++){
        int node_index = order[step];
        const Node& node = graph.nodes()[node_index];


        for(const std::string& input : node.inputs())
            lifetimes_[input].last_use = step;

        for(const std::string& output : node.outputs())
            lifetimes_[output].birth = step;
    }

}


const TensorLifetime& MemoryPlanner::lifetime(const std::string& tensor_name) const{
    auto it = lifetimes_.find(tensor_name);

    if(it == lifetimes_.end())
        throw std::runtime_error("No lifetime information for tensor: " + tensor_name);

    return it->second;
}



// Actually I thought we don't need to keep focusing on the intermediate but maybe it's the convention that 
// we always want to keep the input and output not losed.
bool MemoryPlanner::is_intermediate(const std::string& tensor_name) const{
    /*
    bool has_birth = birth_.find(tensor_name) != birth_.end();
    bool has_last_use = last_use_.find(tensor_name) != last_use_.end();
    */

    const TensorLifetime& info = lifetime(tensor_name);


    return info.birth >=0 && info.last_use >= 0;
    
}


bool MemoryPlanner::lifetimes_overlap(const std::string& tensor_a, const std::string& tensor_b) const{
    const TensorLifetime& a = lifetime(tensor_a);
    const TensorLifetime& b = lifetime(tensor_b);

    if(a.birth < 0 || a.last_use < 0 || b.birth < 0|| b.last_use < 0)
        throw std::runtime_error("Lifetime overlap requires two intermediate tensors.");

    bool non_overlap = a.last_use < b.birth || b.last_use < a.birth;

    return !non_overlap;  
}


// We will do a safe judge at here.
bool MemoryPlanner::can_reuse(const std::string& tensor_a, const std::string& tensor_b) const{
    if(!is_intermediate(tensor_a) || !is_intermediate(tensor_b))
        return false;

    return  !lifetimes_overlap(tensor_a, tensor_b);
}




/*
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
*/

/*
const std::unordered_map <std::string, int>& MemoryPlanner::last_uses() const{
    return last_use_;
}


const std::unordered_map <std::string, int>& MemoryPlanner::births() const{
    return birth_;
} 
*/


