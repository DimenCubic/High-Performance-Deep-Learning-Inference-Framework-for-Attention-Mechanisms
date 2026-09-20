#include "graph/parser.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json; // rename the official reference

Graph GraphParser(const std::string& filename){
    std::ifstream file(filename);

    if(!file)
        throw std::runtime_error("Failed to open the graph file: " + filename);


    json root;   // 创建JSON对象
    file >> root;


    if(!root.contains("nodes"))
        throw std::runtime_error("Grapg JSON doesn't contain 'Nodes'.");
    

    Graph graph;

    for(const auto& node_json : root.at("nodes")){
        std::string name = node_json.at("name").get<std::string>();
        std::string type = node_json.at("type").get<std::string>();
        std::vector<std::string> inputs = node_json.at("inputs").get<std::vector<std::string>>();
        std::vector<std::string> outputs = node_json.at("outputs").get<std::vector<std::string>>();



        graph.add_node(Node(name, type, inputs, outputs));
    }


    graph.build_dependencies();

    return graph;

}