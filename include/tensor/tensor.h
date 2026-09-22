#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>
#include <cstddef>

class Tensor{
    public:

        // Constructor, got the read only reference from outside.
        Tensor(  
            const std::string& name,
            const std::vector<int>& shape
        );


        // New external Tensor
        Tensor(
            const std::string& name,
            const std::vector<int>& shape,
            float* external_data
        );



        // Copy onstructor
        Tensor(const Tensor& other);
        
        // Copy Assignment Operator
        Tensor& operator=(const Tensor& other);




        const std::string& name() const; // First const means 不允许通过这个函数提供的内部引用去修改内部的值，第二个const 指的是这个函数承诺不去修改内部值
        const std::vector<int>& shape() const; // 加上第一个const的最大原因也是
        std::size_t size() const;

        // const overload, will use the corresponding version based on the object data type.
        float* data();
        const float* data() const;

        // 运算符重载，定义符号[]在这里的运算意义，我们这边就是重新定义[]就是返回引用，所以Tensor A[1] 就会返回data_[1]，而不用写A.data()[1].
        float& operator[](std::size_t index);  
        const float& operator[](std::size_t index) const;


        bool owns_memory() const;


    private:

        std::string name_;
        std::vector<int> shape_;
        std::vector<float> ownered_data_;

        // 把之前所有的data全部指向allocate这一部分
        std::size_t size_ = 0;
        float* data_ = nullptr;

};


#endif