#include "./common/input.h"
#include "./common/helpers.h"
#include "./common/timers.h"

#ifdef DEBUG
#include <cassert>
#include "./common/cpu_tests.h"
#endif // DEBUG
// #include <cooperative_groups/scan.h>

#define HIST_ELEM_PER_THREAD 10
#define HIST_THREADS 256
__global__ void histo_kernel(uint32_t *buffer, long size, uint32_t *histo) { 
  int i = threadIdx.x + (blockIdx.x * blockDim.x * HIST_ELEM_PER_THREAD);
  for(int j = 0; j < HIST_ELEM_PER_THREAD; j++) {
    if(i < size) {
      atomicAdd(&histo[buffer[i]], 1);
    }
    i += blockDim.x;
  }
}

// orients graph from vertex with lower degree to vertex with higher degree, edges_a becomes source and edges_b becomes destination of new edges
#define ORIENT_ELEM_PER_THREAD HIST_ELEM_PER_THREAD
#define ORIENT_THREADS HIST_THREADS
__global__ void graph_orientation(uint32_t *edges_a, uint32_t *edges_b, long size, uint32_t *histo) {
  int i = threadIdx.x + (blockIdx.x * blockDim.x * ORIENT_ELEM_PER_THREAD);
  for(int j = 0; j < ORIENT_ELEM_PER_THREAD; j++) {
    if(i < size) {
      if(!histo[edges_a[i]] < !histo[edges_b[i]]) {
        uint32_t tmp = edges_a[i];
        edges_a[i] = edges_b[i];
        edges_b[i] = tmp;
      }
    }
  }
}

size_t N_STREAMS = 2;

void sync_streams(cudaStream_t *streams, int n_streams) {
    for(int i = 0; i < n_streams; i++) {
      HANDLE_ERROR(cudaStreamSynchronize(streams[i]));
    }
}

int main(int argc, char *argv[]) {
  CpuTimer timer("main");

  const int K = std::stoi(argv[2]);

  const auto [A, B, n_vertex] = read_input(argv[1]);
  const int n_edges = A.size();

  std::cout << "n_vertex: " << n_vertex << " n_edges: " << n_edges << std::endl;

  cudaStream_t streams[N_STREAMS];
  for(int i = 0; i < N_STREAMS; i++) {
      cudaStreamCreate(&streams[i]);
  }
  
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount;

  // compute histogram to count nodes degrees
  CudaTimer hist_timer("histogram");
  uint32_t *host_hist;
  uint32_t *dev_hist, *dev_a, *dev_b;

  HANDLE_ERROR(cudaHostAlloc((void**)&host_hist, n_vertex * sizeof(uint32_t), cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void**)&dev_hist, n_vertex * sizeof(uint32_t)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, n_edges * sizeof(uint32_t)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, n_edges * sizeof(uint32_t)));
  
  HANDLE_ERROR(cudaMemset(dev_hist, 0, n_vertex * sizeof(uint32_t)));
  HANDLE_ERROR(cudaMemcpyAsync(dev_a, A.data(), n_edges * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[0]));
  HANDLE_ERROR(cudaMemcpyAsync(dev_b, B.data(), n_edges * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[1]));

  int hist_blocks = (n_edges + HIST_ELEM_PER_THREAD * HIST_THREADS) / (HIST_ELEM_PER_THREAD * HIST_THREADS);

  histo_kernel<<<hist_blocks, 256, 0, streams[0]>>>(dev_a, n_edges, dev_hist);
  histo_kernel<<<hist_blocks, 256, 0, streams[1]>>>(dev_b, n_edges, dev_hist);
  sync_streams(streams, N_STREAMS);

  HANDLE_ERROR(cudaMemcpyAsync(host_hist, dev_hist, n_vertex * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[0]));
  sync_streams(streams, 1);

  #ifdef DEBUG
  auto cpu_hist = histogram_cpu(A, B, n_vertex);
  for(auto i = 0; i < n_vertex; i++) {
    assert(cpu_hist[i] == host_hist[i]);
  }
  std::cout << "Histograms on CPU and GPU match!\n";
  #endif // NDEBUG

  cudaFreeHost(host_hist);
  cudaFree(dev_hist);	
  cudaFree(dev_a);	
  cudaFree(dev_b);

  for(int i = 0; i < N_STREAMS; i++) {
    HANDLE_ERROR(cudaStreamDestroy(streams[i]));
  }

  return 0;
}