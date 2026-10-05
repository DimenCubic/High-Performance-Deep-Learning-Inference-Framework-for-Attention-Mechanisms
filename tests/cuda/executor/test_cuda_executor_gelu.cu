#include "cuda/cuda_ops.cuh"

#include "graph/graph.h"
#include "graph/node.h"

#include "ops/operator.h"

#include "tensor/tensor.h"

#include "runtime/cuda_memory_pool.h"
#include "runtime/device_transfer.h"
#include "runtime/executor.h"


#include <cmath>
#include <iostream>
#include <vector>


int main(){
    const int size = 513;
    const float tolerance = 1e-5f;

    const std::vector<int> shape = {size};


    // Create CPU input tensor.
    Tensor host_input("host_input", shape);

    for(int i = 0; i < size; i++)
        host_input[static_cast<std::size_t>(i)] = static_cast<float>((i % 17 - 8) * 0.25f);

    

    // CPU Reference
    Tensor cpu_reference("cpu_reference", shape);

    gelu(host_input.data(), cpu_reference.data(), size);



    // Create an input on the GPU
    CudaMemoryPool input_cuda_pool;
    float* input_device_buffer = input_cuda_pool.allocate(host_input.size());

    Tensor device_input("input_gpu", shape, input_device_buffer, DeviceType::CUDA);



    // H2D Transfer
    copy_host_to_device(host_input, device_input);




    // Create graph
    Graph graph;

    Node gelu_node(
        "gelu_node", 
        "GELU", 
        {"input_gpu"},
        {"output_gpu"}
    );

    graph.add_node(gelu_node);

    graph.build_dependencies();




    // Configure Executor.
    Executor executor;

    executor.add_tensor(device_input);
    executor.register_tensor("output_gpu", shape, DeviceType::CUDA);




    // Run Graph
    executor.run(graph);





    // Obtain CUDA output tensor
    Tensor& device_output = executor.get_tensor()

}