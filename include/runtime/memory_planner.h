#ifndef MEMORY_PLANNER_H
#define MEMORY_PLANNER_H

#include "graph/graph.h"

#include <string>
#include <unordered_map>


struct TensorLifetime{
    int birth = -1;
    int last_use = -1;
};


class MemoryPlanner{
    
    public: 
        
        /*
        void analyze(const Graph& graph);

        int last_use(const std::string& tensor_name) const;

        const std::unordered_map <std::string, int>& last_uses() const;

        int birth(const std::string& tensor_name) const;

        const std::unordered_map <std::string, int>& births() const;

        bool is_intermediate(const std::string& tensor_name) const;
        */

        void analyze(const Graph& graph);
        const TensorLifetime& lifetime(const std::string& tensor_name) const;
        
        bool is_intermediate(const std::string& tensor_name) const;
        bool lifetimes_overlap(const std::string& tensor_a, const std::string& tensor_b) const;
        bool can_reuse(const std::string& tensor_a, const std::string& tensor_b) const;

        const std::unordered_map<std::string, TensorLifetime>& lifetimes() const;
    
    private:
        
        /*
        std::unordered_map<std::string, int> last_use_;
        std::unordered_map<std::string, int> birth_;
        */

       std::unordered_map<std::string, TensorLifetime> lifetimes_;


};

#endif