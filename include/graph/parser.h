#ifndef PARSER_H
#define PARSER_H

#include "graph/graph.h"
#include <string>

class GraphParser{
    public:
        static Graph parse(const std::string& filename);  // Static 允许直接用类调用，因为这个函数没有需要用到对象存储的数据
};

#endif