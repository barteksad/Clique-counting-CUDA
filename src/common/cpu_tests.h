#include <vector>

#include "timers.h"

std::vector<uint32_t> histogram_cpu(const std::vector<uint32_t>&A, const std::vector<uint32_t>&B, uint32_t n_vertex) {
  CpuTimer timer("histogram_cpu");
  std::vector<uint32_t> hist(n_vertex, 0);
  for(auto x: A) {
    hist[x]++;
  }
  for(auto x: B) {
    hist[x]++;
  }
  return hist;
}
