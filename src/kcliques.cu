#include "./common/input.h"
#include "./common/helpers.h"
#include "./common/timers.h"

#include <thrust/sort.h>

#ifdef DEBUG
#include <cassert>
#include "./common/cpu_tests.h"
#endif // DEBUG
// #include <cooperative_groups/scan.h>

#define HIST_ELEM_PER_THREAD 100
#define HIST_THREADS 1024
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
      if(histo[edges_a[i]] >= histo[edges_b[i]]) {
        uint32_t tmp = edges_a[i];
        edges_a[i] = edges_b[i];
        edges_b[i] = tmp;
      }
    }
    i += blockDim.x;
  }
}

struct CSR {
  uint32_t *row_array;
  uint32_t *col_array;
  uint32_t n_vertex;
  uint32_t n_edges;
};

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
  // we can create histogram array with one more element to use it later as row array for CSR format
  const int CSR_row_size = n_vertex + 1;

  #ifdef DEBUG
  std::cout << "n_vertex: " << n_vertex << " n_edges: " << n_edges << std::endl;
  #endif // DEBUG

  cudaStream_t streams[N_STREAMS];
  for(int i = 0; i < N_STREAMS; i++) {
      cudaStreamCreate(&streams[i]);
  }
  
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount;

  // compute histogram to count nodes degrees
  uint32_t *host_hist;
  uint32_t *dev_hist, *dev_a, *dev_b;
  {
    CudaTimer hist_timer("histogram gpu");

    HANDLE_ERROR(cudaMalloc((void**)&dev_hist, CSR_row_size * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, n_edges * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, n_edges * sizeof(uint32_t)));
    
    HANDLE_ERROR(cudaMemset(dev_hist, 0, CSR_row_size * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMemcpyAsync(dev_a, A.data(), n_edges * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[0]));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b, B.data(), n_edges * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[1]));
    hist_timer.tick("memory management");

    int hist_blocks = (n_edges + HIST_ELEM_PER_THREAD * HIST_THREADS) / (HIST_ELEM_PER_THREAD * HIST_THREADS);

    histo_kernel<<<hist_blocks, HIST_THREADS, 0, streams[0]>>>(dev_a, n_edges, dev_hist);
    histo_kernel<<<hist_blocks, HIST_THREADS, 0, streams[1]>>>(dev_b, n_edges, dev_hist);
    sync_streams(streams, N_STREAMS);

  }

  #ifdef DEBUG
  HANDLE_ERROR(cudaHostAlloc((void**)&host_hist, CSR_row_size * sizeof(uint32_t), cudaHostAllocDefault));
  HANDLE_ERROR(cudaMemcpyAsync(host_hist, dev_hist, CSR_row_size * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[0]));
  sync_streams(streams, 1);

  auto cpu_hist = histogram_cpu(A, B, n_vertex);
  for(auto i = 0; i < n_vertex; i++) {
    assert(cpu_hist[i] == host_hist[i]);
  }
  std::cout << "Histograms on CPU and GPU match!\n";
  cudaFreeHost(host_hist);
  #endif // NDEBUG

  // -- end of histogram computation

  // graph orientation
  {
    CudaTimer orient_timer("graph_orientation");

    int orient_blocks = (n_edges + ORIENT_ELEM_PER_THREAD * ORIENT_THREADS) / (ORIENT_ELEM_PER_THREAD * ORIENT_THREADS);
    graph_orientation<<<orient_blocks, ORIENT_THREADS, 0, streams[0]>>>(dev_a, dev_b, n_edges, dev_hist);
    sync_streams(streams, 1);
  }

  // create CSR format
  // edges_b will become col_array and edges_a will be used to compute row_array which we can store in dev_hist memory since it is not needed anymore
  CSR csr;
  {
    CudaTimer csr_timer("csr");

    HANDLE_ERROR(cudaMemset(dev_hist, 0, CSR_row_size * sizeof(uint32_t)));
    int hist_blocks = (n_edges + HIST_ELEM_PER_THREAD * HIST_THREADS) / (HIST_ELEM_PER_THREAD * HIST_THREADS);
    histo_kernel<<<hist_blocks, HIST_THREADS, 0, streams[0]>>>(dev_a, n_edges, dev_hist);
    thrust::exclusive_scan(thrust::cuda::par.on(streams[0]), dev_hist, dev_hist + CSR_row_size, dev_hist, 0);

    thrust::sort_by_key(thrust::cuda::par.on(streams[0]), dev_b, dev_b + n_edges, dev_a);
    thrust::stable_sort_by_key(thrust::cuda::par.on(streams[0]), dev_a, dev_a + n_edges, dev_b);

    sync_streams(streams, 1);

    csr.col_array = dev_b;
    csr.row_array = dev_hist;
    cudaFree(dev_a);
    csr.n_vertex = n_vertex;
    csr.n_edges = n_edges;
  }

  #ifdef DEBUG
  auto [row_array, col_array] = csr_cpu(A, B, n_vertex);
  uint32_t *host_row_array, *host_col_array;
  HANDLE_ERROR(cudaHostAlloc((void**)&host_row_array, CSR_row_size * sizeof(uint32_t), cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void**)&host_col_array, n_edges * sizeof(uint32_t), cudaHostAllocDefault));
  HANDLE_ERROR(cudaMemcpyAsync(host_row_array, csr.row_array, CSR_row_size * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[0]));
  HANDLE_ERROR(cudaMemcpyAsync(host_col_array, csr.col_array, n_edges * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[1]));
  sync_streams(streams, 2);

  for(auto i = 0; i < n_vertex; i++) {
    assert(row_array[i] == host_row_array[i]);
  }
  for(auto i = 0; i < n_edges; i++) {
    assert(col_array[i] == host_col_array[i]);
  }
  std::cout << "CSR on CPU and GPU match!\n";

  cudaFreeHost(host_row_array);
  cudaFreeHost(host_col_array);
  #endif // DEBUG


  cudaFree(csr.col_array); // dev_b
  cudaFree(csr.row_array); // dev_hist

  for(int i = 0; i < N_STREAMS; i++) {
    HANDLE_ERROR(cudaStreamDestroy(streams[i]));
  }

  return 0;
}