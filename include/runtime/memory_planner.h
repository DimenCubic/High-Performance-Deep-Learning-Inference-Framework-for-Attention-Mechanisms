#ifndef MEMORY_PLANNER_H
#define MEMORY_PLANNER_H

#include "graph/graph.h"

#include <string>
#include <unordered_map>

class MemoryPlanner{
    
    public: 
        
        void analyze(const Graph& graph);

        int last_use(const std::string& tensor_name) const;

        const std::unordered_map <std::string, int>& last_uses() const;

    
    private:
        std::unordered_map<std::string, int> last_use_;
};

#endif