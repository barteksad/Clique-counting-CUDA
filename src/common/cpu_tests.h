#include <vector>
#include <tuple>

#include "timers.h"

std::vector<uint32_t> histogram_cpu(const std::vector<uint32_t>&A, const std::vector<uint32_t>&B, uint32_t n_vertex) {
  CpuTimer timer("histogram cpu");
  std::vector<uint32_t> hist(n_vertex, 0);
  for(auto x: A) {
    hist[x]++;
  }
  for(auto x: B) {
    hist[x]++;
  }
  return hist;
}

void grap_orientation_cpu(std::vector<uint32_t>&A, std::vector<uint32_t>&B, std::vector<uint32_t> hist) {
  CpuTimer timer("grap orientation cpu");
  for(uint32_t i=0; i<A.size(); i++) {
      if(hist[A[i]] >= hist[B[i]]) {
        uint32_t tmp = A[i];
        A[i] = B[i];
        B[i] = tmp;
      }
  }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> csr_cpu(const std::vector<uint32_t>&A_src, const std::vector<uint32_t>&B_src, uint32_t n_vertex) {
  auto A(A_src);
  auto B(B_src);

  CpuTimer timer("csr cpu");
  std::vector<uint32_t> hist = histogram_cpu(A, B, n_vertex);

  grap_orientation_cpu(A, B, hist);

  hist = histogram_cpu(A, {}, n_vertex  + 1);
  std::vector<uint32_t> offsets(n_vertex+1, 0);
  thrust::exclusive_scan(thrust::host, hist.begin(), hist.end(), offsets.begin(), 0);

  thrust::sort_by_key(thrust::host, B.begin(), B.end(), A.begin());
  thrust::stable_sort_by_key(thrust::host, A.begin(), A.end(), B.begin());

  return std::make_tuple(offsets, B);
}
