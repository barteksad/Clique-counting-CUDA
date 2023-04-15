#include <stdlib.h>
#include <assert.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>

#include "./common/input.h"
#include "./common/helpers.h"
#include "./common/timers.h"
#include "./common/cpu_tests.h"

namespace cg = cooperative_groups;

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
// #define ORIENT_ELEM_PER_THREAD HIST_ELEM_PER_THREAD
#define ORIENT_ELEM_PER_THREAD 1
#define ORIENT_THREADS HIST_THREADS
__global__ void graph_orientation(uint32_t *edges_a, uint32_t *edges_b, long size, uint32_t *histo) {
  int i = threadIdx.x + (blockIdx.x * blockDim.x * ORIENT_ELEM_PER_THREAD);
  for(int j = 0; j < ORIENT_ELEM_PER_THREAD; j++) {
    if(i < size) {
      if(histo[edges_a[i]] > histo[edges_b[i]] || (histo[edges_a[i]] == histo[edges_b[i]] && edges_a[i] > edges_b[i])) {
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

#define D_OUT_MAX 1024U
#define SUB_WARP_SIZE 8U
// OA - orientation approach
#define OA_WARP_GROUPS 128U
#define OA_THREADS OA_WARP_GROUPS * SUB_WARP_SIZE
static_assert(SUB_WARP_SIZE && !(SUB_WARP_SIZE & (SUB_WARP_SIZE - 1)));
static_assert(D_OUT_MAX % 32 == 0);
#define K_MAX 11

__global__ void graph_orientation_approach(
  const CSR csr, 
  const uint32_t vertex_idx_offset, 
  const uint32_t K, 
  unsigned long long int K_global[K_MAX - 2], 
  uint32_t induced_sub_graph[],
  uint32_t vertex2idx[], // [D_OUT_MAX]
  uint32_t intersections[] // [OA_WARP_GROUPS][K_MAX - 2][D_OUT_MAX / 32]
  ) {
  __shared__ unsigned long long int K_local[K_MAX - 2]; 
  // create mapping between original vertex index and index in induced subgraph
  // eg. vertexes [11, 100, 123] -> at indexes [0, 1, 2]
  // then we can binsearch to find index of neighbor in induced subgraph

  cg::thread_group tile = cg::tiled_partition(cg::this_thread_block(), SUB_WARP_SIZE);

  const uint32_t vertex_idx = vertex_idx_offset + blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t sub_warp_id = tid / SUB_WARP_SIZE;
  const uint32_t sub_warp_tid = tile.thread_rank();
  const size_t isg_offset = blockIdx.x * D_OUT_MAX * (D_OUT_MAX / 32);
  const size_t v2i_offset = blockIdx.x * D_OUT_MAX;
  const size_t its_offset = blockIdx.x * OA_WARP_GROUPS * (K_MAX - 2) * (D_OUT_MAX / 32);
  assert(vertex_idx < csr.n_vertex);
  assert(K <= K_MAX);
  // initialize K_local
  for(int i = tid; i < K_MAX - 2; i += blockDim.x) {
    K_local[i] = 0;
  }
  
  for(int row = sub_warp_id; row < D_OUT_MAX; row += OA_WARP_GROUPS) {
    for(int col = sub_warp_tid; col < D_OUT_MAX / 32; col += SUB_WARP_SIZE) {
      induced_sub_graph[isg_offset + row * (D_OUT_MAX / 32) + col] = 0;
    }
  }

  for(int i = tid; i < D_OUT_MAX; i += blockDim.x) {
    vertex2idx[v2i_offset + i] = UINT32_MAX;
  }
  __syncthreads();

  for(uint32_t i = csr.row_array[vertex_idx] + tid; i < csr.row_array[vertex_idx + 1]; i += blockDim.x) {
    vertex2idx[v2i_offset + i - csr.row_array[vertex_idx]] = csr.col_array[i];
  }
  __syncthreads();

  // create induced subgraph indicating connection between nodes and store it in shared memory as bitmap, map vertex indexes to interval [0, vertex degree]
  // warp per neighbor
  const uint32_t n_neighbors = csr.row_array[vertex_idx + 1] - csr.row_array[vertex_idx];
  assert(n_neighbors <= D_OUT_MAX);
  if(n_neighbors == 0)
    return;
  for(uint32_t neighbor_i = sub_warp_id; neighbor_i < n_neighbors; neighbor_i += OA_WARP_GROUPS) {
    assert(csr.row_array[vertex_idx] + neighbor_i < csr.row_array[vertex_idx + 1]);
    const uint32_t neighbor = csr.col_array[csr.row_array[vertex_idx] + neighbor_i];
    for(uint32_t i = csr.row_array[neighbor] + sub_warp_tid; i < csr.row_array[neighbor + 1]; i += SUB_WARP_SIZE) {
      const uint32_t neighbor_neighbor_vertex = csr.col_array[i];
      const uint32_t neighbor_neighbor_idx = bsearch_dev(vertex2idx + v2i_offset, neighbor_neighbor_vertex, n_neighbors);
      assert(neighbor_neighbor_idx < n_neighbors || neighbor_neighbor_idx == UINT32_MAX);
      if(neighbor_neighbor_idx != UINT32_MAX) {
        assert(neighbor_neighbor_idx / 32 < D_OUT_MAX / 32);
        atomicOr(&induced_sub_graph[isg_offset + neighbor_i * (D_OUT_MAX / 32) + neighbor_neighbor_idx / 32], 1 << (neighbor_neighbor_idx % 32));
      }
    }
  }
  __syncthreads();

  __shared__ uint32_t curr_neighbor_to_check[OA_WARP_GROUPS][K_MAX - 2];
  __shared__ uint32_t next_possible_neighbor[OA_WARP_GROUPS];
  // __shared__ uint32_t intersections[OA_WARP_GROUPS][K_MAX - 2][D_OUT_MAX / 32];

  for(uint32_t warp_subtree = sub_warp_id; warp_subtree < n_neighbors; warp_subtree += OA_WARP_GROUPS) {
    for(uint32_t i = sub_warp_tid; i < K_MAX - 2; i += SUB_WARP_SIZE) {
      curr_neighbor_to_check[sub_warp_id][i] = 0;
    }

    for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
      // intersections[sub_warp_id][0][i] = induced_sub_graph[isg_offset + warp_subtree * (D_OUT_MAX / 32) + i];
      intersections[its_offset + sub_warp_id * (K_MAX - 2) * (D_OUT_MAX / 32) + i] = induced_sub_graph[isg_offset + warp_subtree * (D_OUT_MAX / 32) + i];
    }

    tile.sync();
    for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
      uint64_t n_neighbors_k = __popc(intersections[its_offset + sub_warp_id * (K_MAX - 2) * (D_OUT_MAX / 32) + i]);
      if(n_neighbors_k > 0) {
        atomicAdd(&K_local[0], n_neighbors_k);
      }
    }

    tile.sync();

    uint32_t curr_k = 3;
    while(curr_k >= 2) {
      if(curr_k == K) {
        curr_k--;
        if(curr_k == 2) {
          break;
        }
      }

      // find first non zero bit in to_check >= prev_check
      uint32_t prev = curr_neighbor_to_check[sub_warp_id][curr_k - 3];
      if(sub_warp_tid == 0) {
        next_possible_neighbor[sub_warp_id] = UINT32_MAX;
      }
      tile.sync();

      for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
        // uint32_t x = intersections[sub_warp_id][curr_k - 3][i];
        uint32_t x = intersections[its_offset + sub_warp_id * (K_MAX - 2) * (D_OUT_MAX / 32) + (curr_k - 3) * (D_OUT_MAX / 32) + i];
        if(x == 0) {
          continue;
        } else {
          uint32_t mask = UINT32_MAX;
          if(prev / 32 == i) {
            mask ^= (~(~0 << (prev % 32)));
          }
          int pos = __ffs(x & mask) - 1;
          if(pos == -1 || (i * 32 + pos) < prev) {
            continue;
          } else {
            atomicMin(&next_possible_neighbor[sub_warp_id], i * 32 + pos);
          }
        }
      }
      tile.sync();

      // if no next possible neighbor, backtrack
      uint32_t next = next_possible_neighbor[sub_warp_id];
      if(next == UINT32_MAX) {
        // backtrack
        if(sub_warp_tid == 0) {
          curr_neighbor_to_check[sub_warp_id][curr_k - 3] = 0;
        }
        curr_k--;
        if(curr_k == 2) {
          break;
        }
        continue;
      }

      tile.sync();

      if(sub_warp_tid == 0) {
        curr_neighbor_to_check[sub_warp_id][curr_k - 3] = next + 1;
      }

      for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
        intersections[its_offset + sub_warp_id * (K_MAX - 2) * (D_OUT_MAX / 32) + (curr_k - 2) * (D_OUT_MAX / 32) + i] = intersections[its_offset + sub_warp_id * (K_MAX - 2) * (D_OUT_MAX / 32) + (curr_k - 3) * (D_OUT_MAX / 32) + i] & induced_sub_graph[isg_offset + next * (D_OUT_MAX / 32) + i];
      }
      tile.sync();

      for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
        uint64_t n_neighbors_k = __popc(intersections[its_offset + sub_warp_id * (K_MAX - 2) * (D_OUT_MAX / 32) + (curr_k - 2) * (D_OUT_MAX / 32) + i]);
        if(n_neighbors_k > 0) {
          atomicAdd(&K_local[curr_k - 2], n_neighbors_k);
        }
      }
      curr_k++;
    }
  }
  __syncthreads();

  // reduce local kclique counters to global counters
  for(uint32_t i = tid; i < K_MAX - 2; i += blockDim.x) {
    atomicAdd(&K_global[i], K_local[i] % 1000000000);
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
  std::cout << "K: " << K << std::endl;
  const std::string out_file_path = argv[3];

  std::vector<uint32_t> A, B;
  uint32_t n_vertex;
  std::cout << "Reading input file: " << argv[1] << std::endl;
  std::tie(A, B, n_vertex) = read_input(argv[1]);
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
  std::cout << prop.concurrentKernels << " concurrentKernels\n";
  std::cout << prop.multiProcessorCount << " multiProcessorCount\n";
  int blocks = prop.multiProcessorCount * 16;
  int warps = prop.warpSize;
  #ifdef DEBUG
  printf("warps: %d\n", warps);
  printf("blocks: %d\n", blocks);
  #endif // DEBUG

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
  // -- end of graph orientation

  // create CSR format
  // edges_b will become col_array and edges_a will be used to compute row_array which we can store in dev_hist memory since it is not needed anymore
  CSR csr;
  {
    CudaTimer csr_timer("csr");

    HANDLE_ERROR(cudaMemset(dev_hist, 0, CSR_row_size * sizeof(uint32_t)));
    int hist_blocks = (n_edges + HIST_ELEM_PER_THREAD * HIST_THREADS) / (HIST_ELEM_PER_THREAD * HIST_THREADS);
    histo_kernel<<<hist_blocks, HIST_THREADS, 0, streams[0]>>>(dev_a, n_edges, dev_hist);
    sync_streams(streams, 1);

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

  // orientation approach
  unsigned long long int *K_global_dev;
  unsigned long long int *K_global_host;
  {
    CudaTimer oa_timer("orientation_approach");
    uint32_t *induced_sub_graph;
    uint32_t *vertex2idx;
    uint32_t *intersections;

    HANDLE_ERROR(cudaMalloc((void**)&K_global_dev, (K_MAX - 2) * sizeof(unsigned long long int)));
    HANDLE_ERROR(cudaMalloc((void**)&induced_sub_graph, blocks * D_OUT_MAX * (D_OUT_MAX / 32) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&vertex2idx, blocks * D_OUT_MAX * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&intersections, blocks * OA_WARP_GROUPS * (K_MAX - 2) * (D_OUT_MAX / 32) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaHostAlloc((void**)&K_global_host, (K_MAX - 2) * sizeof(unsigned long long int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaMemset(K_global_dev, 0, (K_MAX - 2) * sizeof(unsigned long long int)));

    for(uint32_t vertex_idx = 0; vertex_idx < n_vertex; vertex_idx += blocks) {
      int n_blocks = std::min(n_vertex - vertex_idx, (uint32_t)blocks);
      graph_orientation_approach<<<n_blocks, OA_THREADS, 0, streams[0]>>>(csr, vertex_idx, K, K_global_dev, induced_sub_graph, vertex2idx, intersections);
      sync_streams(streams, 1);
    }

    HANDLE_ERROR(cudaMemcpyAsync(K_global_host, K_global_dev, (K_MAX - 2) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost, streams[0]));
    sync_streams(streams, 1);
    cudaFree(induced_sub_graph);
    cudaFree(vertex2idx);
    cudaFree(intersections);
  }

  std::cout << "K:\n";
  std::cout << "1: " << n_vertex << "\n";
  std::cout << "2: " << n_edges << "\n";
  for(auto i = 0; i < K_MAX - 2; i++) {
    std::cout << i + 3 << ": " << K_global_host[i] % 1000000000 << "\n";
  }

  cudaFree(csr.col_array); // dev_b
  cudaFree(csr.row_array); // dev_hist
  cudaFree(K_global_dev); 
  cudaFreeHost(K_global_host);

  for(int i = 0; i < N_STREAMS; i++) {
    HANDLE_ERROR(cudaStreamDestroy(streams[i]));
  }

  return 0;
}