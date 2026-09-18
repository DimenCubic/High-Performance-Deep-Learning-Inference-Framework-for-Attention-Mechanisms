#include "tensor/tensor.h"
#include <stdexcept>

Tensor::Tensor(const std::string& name, const std::vector<int>& shape) : name_(name), shape_(shape){   // 定义Tensor类里面的Tensor构造函数，后面指的是创建这个Tensor对象的时候直接把name赋值给name_
    std::size_t total_size = 1;

    for(int dim : shape_){
        if(dim <= 0)   // Every dimention must be positive. 
            throw std::invalid_argument("Tensor dimensions must be positive");
    
        total_size *= static_cast<std::size_t>(dim);

    
    }

    data_.resize(total_size, 0.0f); // adjuct size of the data_, since data_ is vector<float>


}


// Getter Function
const std::string& Tensor::name() const{
    return name_;
}

const std::vector<int>& Tensor::shape() const{
    return shape_;
}


std::size_t Tensor::size() const{
    return data_.size();
}


// data()
float* Tensor::data(){
    return data_.data();   // 自带的一个函数，返回的是vector第一个element的指针。
}

const float* Tensor::data() const{
    return data_.data();
}


// operator
float& Tensor::operator[](std::size_t index){
    return data_[index];
}


const float& Tensor::operator[](std::size_t index) const{
    return data_[index];
}