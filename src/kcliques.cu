#include <stdlib.h>
#include <assert.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>

#include "./common/io.h"
#include "./common/helpers.h"
#include "./common/timers.h"

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

#define D_OUT_MAX 1024U // max degree of output graph
#define SUB_WARP_SIZE 8U // number of threads per sub warp, what is described in paper as threads group
// OA - orientation approach
#define OA_WARP_GROUPS 128U // number of thread groups per block
#define OA_THREADS OA_WARP_GROUPS * SUB_WARP_SIZE // number of threads per block
static_assert(SUB_WARP_SIZE && !(SUB_WARP_SIZE & (SUB_WARP_SIZE - 1))); // check if SUB_WARP_SIZE is power of 2
static_assert(D_OUT_MAX % 32 == 0); 
static_assert(OA_THREADS <= 1024);
#define K_MAX 11

// helper macros for indexing flat nd arrays
#define idx_intersections(its, its_offset, row, col, k) its[its_offset + row * (K_MAX - 2) * (D_OUT_MAX / 32) + col * (K_MAX - 2) + k]
#define idx_induced_sub_graph(isg, isg_offset, row, col) isg[isg_offset + row * (D_OUT_MAX / 32) + col]

__global__ void graph_orientation_approach(
  const CSR csr, // CSR representation of graph
  const uint32_t vertex_idx_offset, // number of kernel group currently processed
  const uint32_t K, 
  unsigned long long int K_global[K_MAX - 2], 
  uint32_t induced_sub_graph[],
  // for storing intersections per group at each recurrence level, shape: [OA_WARP_GROUPS][K_MAX - 2][D_OUT_MAX / 32]
  uint32_t intersections[] 
  ) {
  __shared__ unsigned long long int K_local[K_MAX - 2]; 

  // split warps into sub warps
  cg::thread_group tile = cg::tiled_partition(cg::this_thread_block(), SUB_WARP_SIZE);

  const uint32_t vertex_idx = vertex_idx_offset + blockIdx.x; // index of vertex in original graph
  const uint32_t tid = threadIdx.x;
  const uint32_t sub_warp_id = tid / SUB_WARP_SIZE; // id in threads group
  const uint32_t sub_warp_tid = tile.thread_rank(); // id of threads group
  const size_t isg_offset = blockIdx.x * D_OUT_MAX * (D_OUT_MAX / 32); // offset in induced_sub_graph array
  const size_t its_offset = blockIdx.x * OA_WARP_GROUPS * (K_MAX - 2) * (D_OUT_MAX / 32); // offset in intersections array
  assert(vertex_idx < csr.n_vertex);
  assert(K <= K_MAX);
  // initialize K_local
  for(int i = tid; i < K_MAX - 2; i += blockDim.x) {
    K_local[i] = 0;
  }

  // initialize induced_sub_graph
  for(int row = sub_warp_id; row < D_OUT_MAX; row += OA_WARP_GROUPS) {
    for(int col = sub_warp_tid; col < D_OUT_MAX / 32; col += SUB_WARP_SIZE) {
      idx_induced_sub_graph(induced_sub_graph, isg_offset, row, col) = 0;
    }
  }

  // create mapping between original vertex index and index in induced subgraph
  // eg. vertexes [11, 100, 123] -> at indexes [0, 1, 2]
  // then we can binsearch to find index of neighbor in induced subgraph
  __shared__ uint32_t vertex2idx[D_OUT_MAX];
  for(int i = tid; i < D_OUT_MAX; i += blockDim.x) {
    vertex2idx[i] = UINT32_MAX; // initialize with out of range value
  }
  __syncthreads();

  for(uint32_t i = csr.row_array[vertex_idx] + tid; i < csr.row_array[vertex_idx + 1]; i += blockDim.x) {
    vertex2idx[i - csr.row_array[vertex_idx]] = csr.col_array[i];
  }
  __syncthreads();

  // create induced subgraph indicating connection between nodes and store it in shared memory as bitmap, map vertex indexes to interval [0, vertex degree]
  // warp per neighbor
  const uint32_t n_neighbors = csr.row_array[vertex_idx + 1] - csr.row_array[vertex_idx];
  assert(n_neighbors <= D_OUT_MAX);
  if(n_neighbors == 0)
    return;
  // for each neighbor do
  for(uint32_t neighbor_i = sub_warp_id; neighbor_i < n_neighbors; neighbor_i += OA_WARP_GROUPS) {
    assert(csr.row_array[vertex_idx] + neighbor_i < csr.row_array[vertex_idx + 1]);
    const uint32_t neighbor = csr.col_array[csr.row_array[vertex_idx] + neighbor_i];
    // for each of neighbor neighbors do
    for(uint32_t i = csr.row_array[neighbor] + sub_warp_tid; i < csr.row_array[neighbor + 1]; i += SUB_WARP_SIZE) {
      const uint32_t neighbor_neighbor_vertex = csr.col_array[i];
      // find index of neighbor neighbor in vertex2idx array, it becomes its index in induced subgraph
      const uint32_t neighbor_neighbor_idx = bsearch_dev(vertex2idx, neighbor_neighbor_vertex, n_neighbors); 
      assert(neighbor_neighbor_idx < n_neighbors || neighbor_neighbor_idx == UINT32_MAX);
      // if neighbor neighbor is neighbor of vertex, set bit in induced subgraph
      if(neighbor_neighbor_idx != UINT32_MAX) {
        assert(neighbor_neighbor_idx / 32 < D_OUT_MAX / 32);
        atomicOr(&idx_induced_sub_graph(induced_sub_graph, isg_offset, neighbor_i, neighbor_neighbor_idx / 32), 1 << (neighbor_neighbor_idx % 32));
      }
    }
  }
  __syncthreads();

  // for each group and recurrence level, store the number of lowest not checked neighbor
  __shared__ uint32_t curr_neighbor_to_check[OA_WARP_GROUPS][K_MAX - 2];
  // used to find next possible neighbor at each level
  __shared__ uint32_t next_possible_neighbor[OA_WARP_GROUPS]; 

  for(uint32_t warp_subtree = sub_warp_id; warp_subtree < n_neighbors; warp_subtree += OA_WARP_GROUPS) {
    // initialize curr_neighbor_to_check with 0 since we haven't checked any neighbor yet
    for(uint32_t i = sub_warp_tid; i < K_MAX - 2; i += SUB_WARP_SIZE) {
      curr_neighbor_to_check[sub_warp_id][i] = 0;
    }

    // copy fresh induced subgraph to group memory
    for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
      idx_intersections(intersections, its_offset, sub_warp_id, 0, i) = idx_induced_sub_graph(induced_sub_graph, isg_offset, warp_subtree, i);
    }

    tile.sync();
    // count 3-cliques
    for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
      uint64_t n_neighbors_k = __popc(idx_intersections(intersections, its_offset, sub_warp_id, 0, i));
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
        next_possible_neighbor[sub_warp_id] = UINT32_MAX; // initialize with out of range value
      }
      tile.sync();

      for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
        // uint32_t x = intersections[sub_warp_id][curr_k - 3][i];
        uint32_t x = idx_intersections(intersections, its_offset, sub_warp_id, (curr_k - 3), i);
        if(x == 0) {
          continue;
        } else {
          // find first bit set >= prev
          // mask is used to set all bits < prev to 0 in array index i
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

      // tile.sync();

      // in next recursion level, start checking from next + 1
      if(sub_warp_tid == 0) {
        curr_neighbor_to_check[sub_warp_id][curr_k - 3] = next + 1;
      }

      // compute intersection of induced subgraph of next neighbor and intersection of previous level
      for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
        idx_intersections(intersections, its_offset, sub_warp_id, (curr_k - 2), i) = idx_intersections(intersections, its_offset, sub_warp_id, (curr_k - 3), i) & idx_induced_sub_graph(induced_sub_graph, isg_offset, next, i);
      }
      tile.sync();

      // count k-cliques
      for(uint32_t i = sub_warp_tid; i <= (n_neighbors - 1) / 32; i += SUB_WARP_SIZE) {
        uint64_t n_neighbors_k = __popc(idx_intersections(intersections, its_offset, sub_warp_id, (curr_k - 2), i));
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


size_t N_STREAMS = 32;

void sync_streams(cudaStream_t *streams, int n_streams) {
    for(int i = 0; i < n_streams; i++) {
      HANDLE_ERROR(cudaStreamSynchronize(streams[i]));
    }
}

int main(int argc, char *argv[]) {
  CpuTimer timer("main");

  const int K = std::stoi(argv[2]);
  const std::string out_file_path = argv[3];

  std::vector<uint32_t> A, B;
  uint32_t n_vertex;
  std::tie(A, B, n_vertex) = read_input(argv[1]);
  const int n_edges = A.size();
  // we can create histogram array with one more element to use it later as row array for CSR format
  const int CSR_row_size = n_vertex + 1;

  cudaStream_t streams[N_STREAMS];
  for(int i = 0; i < N_STREAMS; i++) {
      cudaStreamCreate(&streams[i]);
  }
  
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount * 2;

  // compute histogram to count nodes degrees
  uint32_t *dev_hist, *dev_a, *dev_b;
  {
    CudaTimer hist_timer("histogram gpu");

    HANDLE_ERROR(cudaMalloc((void**)&dev_hist, CSR_row_size * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, n_edges * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, n_edges * sizeof(uint32_t)));
    
    HANDLE_ERROR(cudaMemset(dev_hist, 0, CSR_row_size * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMemcpyAsync(dev_a, A.data(), n_edges * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[0]));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b, B.data(), n_edges * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[1]));

    int hist_blocks = (n_edges + HIST_ELEM_PER_THREAD * HIST_THREADS) / (HIST_ELEM_PER_THREAD * HIST_THREADS);

    histo_kernel<<<hist_blocks, HIST_THREADS, 0, streams[0]>>>(dev_a, n_edges, dev_hist);
    histo_kernel<<<hist_blocks, HIST_THREADS, 0, streams[1]>>>(dev_b, n_edges, dev_hist);
    sync_streams(streams, N_STREAMS);
  }
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
    CudaTimer csr_timer("convert to CSR");

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
    uint32_t *intersections;

    size_t isg_size = blocks * D_OUT_MAX * (D_OUT_MAX / 32); // isg = induced_sub_graph
    size_t inter_size = blocks * OA_WARP_GROUPS * (K_MAX - 2) * (D_OUT_MAX / 32); // inter = intersections

    HANDLE_ERROR(cudaMalloc((void**)&K_global_dev, (K_MAX - 2) * sizeof(unsigned long long int)));
    HANDLE_ERROR(cudaMalloc((void**)&induced_sub_graph, N_STREAMS * isg_size  * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&intersections, N_STREAMS * inter_size * sizeof(uint32_t)));
    HANDLE_ERROR(cudaHostAlloc((void**)&K_global_host, (K_MAX - 2) * sizeof(unsigned long long int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaMemset(K_global_dev, 0, (K_MAX - 2) * sizeof(unsigned long long int)));

    for(uint32_t vertex_idx = 0; vertex_idx < n_vertex; vertex_idx += blocks) {
      size_t s_idx = (vertex_idx / blocks) % N_STREAMS;
      int n_blocks = std::min(n_vertex - vertex_idx, (uint32_t)blocks);
      graph_orientation_approach<<<n_blocks, OA_THREADS, 0, streams[s_idx]>>>(
        csr, 
        vertex_idx, 
        K, 
        K_global_dev, 
        induced_sub_graph + s_idx * isg_size, 
        intersections + s_idx * inter_size);
    }
    sync_streams(streams, N_STREAMS);

    HANDLE_ERROR(cudaMemcpyAsync(K_global_host, K_global_dev, (K_MAX - 2) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost, streams[0]));
    sync_streams(streams, 1);
    cudaFree(induced_sub_graph);
    cudaFree(intersections);
  }

  write_output(out_file_path, K, n_vertex, n_edges, K_global_host);

  cudaFree(csr.col_array); // dev_b
  cudaFree(csr.row_array); // dev_hist
  cudaFree(K_global_dev); 
  cudaFreeHost(K_global_host);

  for(int i = 0; i < N_STREAMS; i++) {
    HANDLE_ERROR(cudaStreamDestroy(streams[i]));
  }

  return 0;
}