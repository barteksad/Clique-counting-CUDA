#include <stdlib.h>
#include <thrust/sort.h>

#include "./common/input.h"
#include "./common/helpers.h"
#include "./common/timers.h"
#include "./common/cpu_tests.h"

#ifdef DEBUG
#include <cassert>
#endif // DEBUG

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

#define D_OUT_MAX 1024
#define SUB_WARP_SIZE 32
// OA - orientation approach
#define OA_WARP_GROUPS 4
#define OA_THREADS OA_WARP_GROUPS * SUB_WARP_SIZE
static_assert(SUB_WARP_SIZE && !(SUB_WARP_SIZE & (SUB_WARP_SIZE - 1)));
static_assert(D_OUT_MAX % 32 == 0);
static_assert(D_OUT_MAX % (SUB_WARP_SIZE * OA_WARP_GROUPS) == 0);
#define K_MAX 11
__global__ void graph_orientation_approach(
  CSR csr, uint32_t K) {
  __shared__ uint32_t K_local[K_MAX - 2]; 
  __shared__ uint32_t induced_sub_graph[D_OUT_MAX / 32][D_OUT_MAX / 32];
  // create mapping between original vertex index and index in induced subgraph
  // eg. vertexes [11, 100, 123] -> at indexes [0, 1, 2]
  // then we can binsearch to find index of neighbor in induced subgraph
  __shared__ uint32_t vertex2idx[D_OUT_MAX];

  uint32_t vertex_idx = blockIdx.x;
  uint32_t tid = threadIdx.x;
  uint32_t sub_warp_id = tid / SUB_WARP_SIZE;
  uint32_t sub_warp_tid = tid % SUB_WARP_SIZE;

  // initialize K_local
  for(int i = tid; i < K_MAX - 2; i += blockDim.x) {
    K_local[i] = 0;
  }
  
  for(int row = sub_warp_id; row < D_OUT_MAX / 32; row += OA_WARP_GROUPS) {
    for(int col = sub_warp_tid; col < D_OUT_MAX / 32; col += SUB_WARP_SIZE) {
      induced_sub_graph[row][col] = 0;
    }
  }

  for(int i = tid; i < D_OUT_MAX; i += blockDim.x) {
    vertex2idx[i] = UINT32_MAX;
  }
  __syncthreads();

  for(uint32_t i = csr.row_array[vertex_idx] + tid; i < csr.row_array[vertex_idx + 1]; i += blockDim.x) {
    vertex2idx[i - csr.row_array[vertex_idx]] = csr.col_array[i];
  }
  __syncthreads();

  // create induced subgraph indicating connection between nodes and store it in shared memory as bitmap, map vertex indexes to interval [0, vertex degree]
  // warp per neighbor
  uint32_t n_neighbors = csr.row_array[vertex_idx + 1] - csr.row_array[vertex_idx];
  for(uint32_t neighbor_i = sub_warp_id; neighbor_i < n_neighbors; neighbor_i += OA_WARP_GROUPS) {
    uint32_t neighbor = csr.col_array[csr.row_array[vertex_idx] + neighbor_i];
    for(uint32_t i = csr.row_array[neighbor] + sub_warp_tid; i < csr.row_array[neighbor + 1]; i += SUB_WARP_SIZE) {
      uint32_t neighbor_neighbor_vertex = csr.col_array[i];
      uint32_t neighbor_neighbor_idx = bsearch_dev(vertex2idx, neighbor_neighbor_vertex, n_neighbors);
      if(neighbor_neighbor_idx != UINT32_MAX) {
        atomicOr(&induced_sub_graph[neighbor_i][neighbor_neighbor_idx / 32], 1 << (neighbor_neighbor_idx % 32));
      }
    }
  }
  __syncthreads();

  __shared__ uint32_t next_possible_neighbor[OA_WARP_GROUPS][K_MAX - 3];
  __shared__ uint32_t dfs_vertex[OA_WARP_GROUPS][K_MAX - 3];
  __shared__ uint32_t intersections[OA_WARP_GROUPS][K_MAX - 3][D_OUT_MAX / 32];
  for(uint32_t warp_subtree = sub_warp_id; warp_subtree < n_neighbors; warp_subtree += OA_WARP_GROUPS) {
    for(uint32_t i = sub_warp_tid; i < K_MAX - 3; i += SUB_WARP_SIZE) {
      next_possible_neighbor[sub_warp_id][i] = 0;
    }
    if(sub_warp_tid == 0) {
      dfs_vertex[sub_warp_id][0] = warp_subtree;
    }
    __syncwarp();

    uint32_t curr_k = 3;
    while(true) {
      // find first non zero bit in to_check >= prev_check
      uint32_t prev = next_possible_neighbor[sub_warp_id][curr_k - 3];
      __syncwarp();
      if(sub_warp_tid) {
        next_possible_neighbor[sub_warp_id][curr_k - 3] = UINT32_MAX;
      }
      __syncwarp();

      for(uint32_t i = sub_warp_tid; i < D_OUT_MAX / 32; i += SUB_WARP_SIZE) {
        uint32_t x = induced_sub_graph[dfs_vertex[sub_warp_id][curr_k - 3]][i];
        if(x == 0) {
          continue;
        } else {
          int pos = __ffs(x) - 1;
          if(pos == -1 || i * 32 + pos <= prev) {
            continue;
          } else {
            atomicMin(&next_possible_neighbor[sub_warp_id][curr_k - 3], i * 32 + pos);
          }
        }
      }
      __syncwarp();

      // if no next possible neighbor, backtrack
      if(next_possible_neighbor[sub_warp_id][curr_k - 3] == UINT32_MAX) {
        // backtrack
        curr_k--;
        if(curr_k == 2) {
          break;
        }
        continue;
      }
      
      // increment kclique counter
      atomicAdd(&K_local[curr_k - 3], 1);

    }
  }
  // __shared__ uint32_t to_check[OA_WARP_GROUPS][D_OUT_MAX / 32];
  // for(uint32_t warp_subtree = sub_warp_id; warp_subtree < n_neighbors; warp_subtree += OA_WARP_GROUPS) {
  //   // zero to_check
  //   for(uint32_t i = sub_warp_tid; i < D_OUT_MAX / 32; i += SUB_WARP_SIZE) {
  //     to_check[sub_warp_id][i] = 0;
  //   }
  //   __syncwarp();
  //   // set to check to next subtree
  //   if(sub_warp_tid = 0) {
  //     to_check[sub_warp_id][warp_subtree / 32] = 1 << (warp_subtree % 32);
  //   }
  //   __syncwarp();

  //   for(uint32_t curr_k = 3; curr_k <= K; curr_k++) {

  //     int 
  //     // count number of cliques of size curr_k
  //     for(uint32_t i = sub_warp_tid; i < n_neighbors; i += SUB_WARP_SIZE) {
  //       uint32_t n_cliques = __popc(induced_sub_graph[warp_subtree][warp_subtree / 32]);
  //     }
  //   }
  // }

  // print induced subgraph
  if(tid == 0) {
    printf("vertex %u: , induced[0][0]: %d induced[1][0]: %d, induced[2][0]: %d\n", vertex_idx, induced_sub_graph[0][0], induced_sub_graph[1][0], induced_sub_graph[2][0]);
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
  const std::string out_file_path = argv[3];

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
  // -- end of graph orientation

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

  std::cout << "CSR row array:\n";
  for(auto i = 0; i < CSR_row_size; i++) {
    std::cout << host_row_array[i] << " ";
  }
  std::cout << "\n";
  std::cout << "CSR col array:\n";
  for(auto i = 0; i < n_edges; i++) {
    std::cout << host_col_array[i] << " ";
  }
  std::cout << "\n";
  std::cout << "CSR n_vertex: " << n_vertex << "\n";
  std::cout << "CSR n_edges: " << n_edges << "\n";

  for(auto i = 0; i < CSR_row_size; i++) {
    assert(row_array[i] == host_row_array[i]);
  }
  for(auto i = 0; i < n_edges; i++) {
    assert(col_array[i] == host_col_array[i]);
  }
  std::cout << "CSR on CPU and GPU match!\n";

  cudaFreeHost(host_row_array);
  cudaFreeHost(host_col_array);
  #endif // DEBUG

  // orientation approach
  {
    CudaTimer oa_timer("orientation_approach");
    graph_orientation_approach<<<n_vertex, OA_THREADS, 0, streams[0]>>>(csr, K);
    sync_streams(streams, 1);
  }

  cudaFree(csr.col_array); // dev_b
  cudaFree(csr.row_array); // dev_hist

  for(int i = 0; i < N_STREAMS; i++) {
    HANDLE_ERROR(cudaStreamDestroy(streams[i]));
  }

  return 0;
}