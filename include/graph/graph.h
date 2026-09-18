#ifndef GRAPH_H
#define GRAPH_H

#include "graph/node.h"

#include <string>
#include <unordered_map>
#include <vector>


class Graph{
    
    public:
        void add_node(const Node& node);

        void build_dependencies();


        // Getter at here
        const std::vector<Node>& nodes() const;
        const std::vector<std::vector<int>>& edges() const;
        const std::vector<int>& indegrees() const;



    private:
        std::vector<Node> nodes_;
        std::vector<std::vector<int>> edges_; // edges_[2] = {0,1,3} 代表着三条从2出发的单向边。
        std::vector<int> indegrees_;
};


#endif