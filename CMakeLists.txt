cmake_minimum_required(VERSION 3.20)

project(
    HighPerformanceInference       # name
    VERSION 1.0
    LANGUAGES CXX
)

 

set(CMAKE_CXX_STANDARD 17)          # using C++ 17 standard  == g++ -std=c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Necessary C++ 17




# OpenMP on macOS
if(APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)




# Make a variable OPT_LEVEL for CMake
set(
    OPT_LEVEL
    "O0"
    CACHE STRING  # Set default
    "Compiler optimization level: O0, O1, O2, O3, Ofast" # Explanation for colleges
)

# Update complier based on the optimization level
if(OPT_LEVEL STREQUAL "O0")
    add_compile_options(-O0)
elseif(OPT_LEVEL STREQUAL "O1")
    add_compile_options(-O1)
elseif(OPT_LEVEL STREQUAL "O2")
    add_compile_options(-O2)
elseif(OPT_LEVEL STREQUAL "O3")
    add_compile_options(-O3)
elseif(OPT_LEVEL STREQUAL "Ofast")
    add_compile_options(-O3 -ffast-math)

else()
    message(
        FATAL_ERROR
        "Invalid OPT_LEVEL: ${OPT_LEVEL}"
    )
endif()


message(
    STATUS
    "Complier optimization level: ${OPT_LEVEL}"
)





include_directories(
    include
    /opt/homebrew/opt/libomp/include
)


# For correctness
add_executable(
    gemm_test

    tests/cpp/test_gemm.cpp
    src/utils/matrix.cpp

    src/gemm/naive_gemm.cpp
    src/gemm/reordered_gemm.cpp
    src/gemm/unrolled_gemm.cpp
    src/gemm/blocked_gemm.cpp
    src/gemm/blocked2_gemm.cpp
    src/gemm/simd_gemm.cpp
    src/gemm/simd_blocked_gemm.cpp
    src/gemm/optimized_gemm.cpp
)

target_include_directories(
    gemm_test
    PRIVATE
    /opt/homebrew/opt/libomp/include
)


target_link_libraries(
    gemm_test
    PRIVATE
    OpenMP::OpenMP_CXX
)


# For efficiency
add_executable(
    gemm_benchmark

    benchmark/benchmark_gemm.cpp
    src/gemm/naive_gemm.cpp
    src/gemm/reordered_gemm.cpp
    src/gemm/unrolled_gemm.cpp
    src/gemm/blocked_gemm.cpp
    src/gemm/blocked2_gemm.cpp
    src/gemm/simd_gemm.cpp
    src/gemm/simd_blocked_gemm.cpp
    src/gemm/optimized_gemm.cpp
    
    src/utils/matrix.cpp
    src/utils/timer.cpp
)

target_include_directories(
    gemm_benchmark
    PRIVATE
    /opt/homebrew/opt/libomp/include
)


target_link_libraries(
    gemm_benchmark
    PRIVATE
    OpenMP::OpenMP_CXX
)





# For Operator Tests
add_executable(
    ops_test

    tests/cpp/test_ops.cpp
    #tests/cpp/test_ops_temp.cpp

    src/ops/softmax.cpp
    src/ops/layernorm.cpp
    src/ops/gelu.cpp
    src/ops/matmul.cpp
    src/ops/qkv.cpp
    src/ops/attention_scores.cpp
    src/ops/attention_softmax.cpp
    src/ops/scaled_dot_product_attention.cpp
)




