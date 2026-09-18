#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>

class Node{
    public:
        Node(
            const std::string& name,
            const std::string& type,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs
        );


        // Getter Function
        const std::string& name() const;
        const std::string& type() const;
        const std::vector<std::string>& inputs() const;
        const std::vector<std::string>& outputs() const;


    private:
        std::string name_;
        std::string type_;
        std::vector<std::string> inputs_;   // includes name of the tensor
        std::vector<std::string> outputs_;  // includes name of the tensor
};

#endif