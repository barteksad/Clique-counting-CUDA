#ifndef __TIMERS_H__
#define __TIMERS_H__

#include <chrono>
#include <string>
#include <iostream>

#include "helpers.h"

class CpuTimer
{
public:
  CpuTimer(std::string name) 
    : name(name), start_(clock::now()) {}
  void reset() { start_ = clock::now(); }
  double elapsed() const
  {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               clock::now() - start_)
        .count();
  }
  void tick(std::string desc) {
    std::cout << desc << " took " << elapsed() << " ms\n";
    reset();
  }
  ~CpuTimer()
  {
    std::cout << name << " took " << elapsed() << " ms\n";
  }

private:
  std::string name;
  using clock = std::chrono::high_resolution_clock;
  std::chrono::time_point<clock> start_;
};

class CudaTimer
{
public:
  CudaTimer(std::string name) 
    : name(name) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    reset();
    }
  void reset() { cudaEventRecord(start, 0); }
  double elapsed() const
  {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
  }
  void tick(std::string desc) {
    std::cout << desc << " took " << elapsed() << " ms\n";
    reset();
  }
  ~CudaTimer()
  {
    std::cout << name << " took " << elapsed() << " ms\n";
  }

private:
  std::string name;
  cudaEvent_t start, stop;
};

#endif  // __TIMERS_H__