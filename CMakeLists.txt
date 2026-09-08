cmake_minimum_required(VERSION 3.20)

project(
    HighPerformanceInference       # name
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)          # using C++ 17 standard  == g++ -std=c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Necessary C++ 17

include_directories(include)


# For correctness
add_executable(
    gemm_test

    src/main.cpp
    src/gemm/naive_gemm.cpp
)

# For efficiency
add_executable(
    gemm_benchmark

    benchmark/benchmark_gemm.cpp
    src/gemm/naive_gemm.cpp
    src/utils/matrix.cpp
    src/utils/timer.cpp
)
