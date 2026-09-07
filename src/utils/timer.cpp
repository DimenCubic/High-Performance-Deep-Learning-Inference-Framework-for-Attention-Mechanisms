#include "utils/timer.h"

void Timer::start(){
    start_time = std::chrono::high_resolution_clock::now();
}

double Timer::stop(){
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;  // elapsed is an object which contains time and unit.

    return elapsed.count();
}