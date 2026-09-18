#include "graph/node.h"



// Cosntructor only made a initialization.
Node::Node(const std::string& name,
            const std::string& type,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs)
            : name_(name), type_(type), inputs_(inputs), outputs_(outputs){}


const std::string& Node::name() const{
    return name_;
}

const std::string& Node::type() const{
    return type_;
}

const std::vector<std::string>& Node::inputs() const{
    return inputs_;
}

const std::vector<std::string>& Node::outputs() const{
    return outputs_;
}
