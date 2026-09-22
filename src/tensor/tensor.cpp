#include "tensor/tensor.h"
#include <stdexcept>


// Helper
static std::size_t compute_size(const std::vector<int>& shape){
    std::size_t total_size = 1;

    for(int dim : shape){
        if(dim <= 0)
            throw std::invalid_argument("Dimension must be positive");

        total_size *= static_cast<size_t>(dim);
    }

    return total_size;
}







Tensor::Tensor(const std::string& name, const std::vector<int>& shape) : name_(name), shape_(shape), size_(compute_size(shape)){   // 定义Tensor类里面的Tensor构造函数，后面指的是创建这个Tensor对象的时候直接把name赋值给name_
    /*
    std::size_t total_size = 1;

    for(int dim : shape_){
        if(dim <= 0)   // Every dimention must be positive. 
            throw std::invalid_argument("Tensor dimensions must be positive");
    
        total_size *= static_cast<std::size_t>(dim);

    
    }
    */

    ownered_data_.resize(size_, 0.0f); // adjuct size of the data_, since data_ is vector<float>
    data_ = ownered_data_.data();
}


Tensor::Tensor(const std::string& name, const std::vector<int>& shape, float* external_data) :
name_(name), shape_(shape), size_(compute_size(shape)), data_(external_data){

    if(external_data == nullptr)
        throw std::invalid_argument("Exeternal tensor data cannot be null.");
}





// Getter Function
const std::string& Tensor::name() const{
    return name_;
}

const std::vector<int>& Tensor::shape() const{
    return shape_;
}


std::size_t Tensor::size() const{
    return size_;
}


// data()
float* Tensor::data(){
    return data_;   // 自带的一个函数，返回的是vector第一个element的指针。
}

const float* Tensor::data() const{
    return data_;
}


// operator
float& Tensor::operator[](std::size_t index){
    return data_[index];
}


const float& Tensor::operator[](std::size_t index) const{
    return data_[index];
}



// Owns memory judge
bool Tensor::owns_memory()const{
    return !ownered_data_.empty();
}


