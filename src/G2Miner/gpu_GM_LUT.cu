#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
#include "codegen_LUT.cuh"
#include "codegen_utils.cuh"
#define FISSION
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

#include "GM_LUT.cuh"
#include "GM_build_LUT.cuh"
#include "GM_LUT_deep.cuh"
#include "BS_vertex.cuh"
#include "BS_edge.cuh"

// #define THREAD_PARALLEL

// K = 1: G2Miner + LUT
// K = 2: G2Miner

__global__ void clear(AccType *accumulators) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  accumulators[i] = 0;
}

void PatternSolver(Graph &g, int k, std::vector<uint64_t> &accum, int, int) {
  assert(k >= 1);
  size_t memsize = print_device_info(0);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize/1024/1024/1024
            << " GB, graph_mem = " << mem_graph/1024/1024 << " MB\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";
  // CUDA_SAFE_CALL(cudaSetDevice(CUDA_SELECT_DEVICE));
  GraphGPU gg(g);
  gg.init_edgelist(g);

  size_t npatterns = 3;
  AccType *h_counts = (AccType *)malloc(sizeof(AccType) * npatterns);
  for (int i = 0; i < npatterns; i++) h_counts[i] = 0;
  AccType *d_counts;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_counts, sizeof(AccType) * npatterns));
  clear<<<1, npatterns>>>(d_counts);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
 
  size_t nwarps = WARPS_PER_BLOCK;
  size_t n_lists;
  size_t n_bitmaps;
  if (k == 1){
    n_lists = __N_LISTS1;
    n_bitmaps = __N_BITMAPS1;
  }
  else if (k == 2){
    n_lists = __N_LISTS2;
    n_bitmaps = __N_BITMAPS2;
  }
  else if (k == 3) {
    n_lists = __N_LISTS3;
    n_bitmaps = __N_BITMAPS3;
  }
  else {
    n_lists = __N_LISTS1;
    n_bitmaps = __N_BITMAPS1;    
  }

  size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
  size_t per_block_bitmap_size = nwarps * n_bitmaps * ((size_t(md) + BITMAP_WIDTH-1)/BITMAP_WIDTH) * sizeof(vidType);

  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
  if (nb < nblocks) nblocks = nb;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM;
  if (k == 1){
    max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
    std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  }
  else {
    max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
    std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  } 
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;

  nblocks = std::min(6*max_blocks, nblocks);

  nblocks = 640;
  std::cout << "CUDA " << k << "-pattern listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
  vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

  size_t bitmap_size = nblocks * per_block_bitmap_size;
  std::cout << "lut rows size: " << bitmap_size/(1024*1024) << " MB\n";
  bitmapType *frontier_bitmap; // each thread has lut rows to store midresult of lut compute
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_bitmap, bitmap_size));

  // LUT声明
  LUTManager<> lut_manager(nblocks * nwarps, WARP_LIMIT, WARP_LIMIT, true); 

  // split vertex tasks
  std::vector<vidType> vid_warp, vid_block, vid_global;

  for (int vid = 0; vid < nv; ++vid) {
    auto degree = g.get_degree(vid);
    if (degree <= WARP_LIMIT) {
        vid_warp.push_back(vid);
    } else if (degree <= BLOCK_LIMIT) {
        vid_block.push_back(vid);
    } else {
        vid_global.push_back(vid);
    }
  }
  vidType vid_warp_size = vid_warp.size();
  vidType vid_block_size = vid_block.size();
  vidType vid_global_size = vid_global.size();
  std::cout << "warp_task: " << vid_warp_size << " block_task: " << vid_block_size << " global_task: " << vid_global_size << "\n";
  vidType *d_vid_warp;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_vid_warp, vid_warp_size * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_vid_warp, vid_warp.data(), vid_warp_size * sizeof(vidType), cudaMemcpyHostToDevice));

  vidType *d_vid_block;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_vid_block, vid_block_size * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_vid_block, vid_block.data(), vid_block_size * sizeof(vidType), cudaMemcpyHostToDevice));

  Timer t;
  t.Start();
  // G2Miner + LUT
  if (k == 1) {
    std::cout << "Run G2Miner + LUT\n";
    if (WARP_LIMIT != 0) GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    if (vid_block_size) {
      lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
      GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    if (vid_global_size){
      lut_manager.recreate(1, md, md, true);
      nblocks = BLOCK_GROUP;
      for (vidType i = 0; i < vid_global_size; i++) {
        vidType task_id = vid_global[i];
        lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
        GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
        GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
      }
    }
  }
  // G2Miner + LUT build deeper level
  else if (k == 2) {
    std::cout << "Run G2Miner + LUT\n";
    GM_LUT_warp_deep<<<nblocks, nthreads>>>(0, ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
    GM_LUT_block_deep<<<nblocks, nthreads>>>(0, ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    lut_manager.recreate(BLOCK_GROUP, md, md, true);
    nblocks = BLOCK_GROUP;
    GM_LUT_block_large_deep<<<nblocks, nthreads>>>(0, ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  // G2Miner
  else if (k == 3){
    std::cout << "Run G2Miner\n";
    BS_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else {
    std::cout << "Not supported right now\n";
  }
  CUDA_SAFE_CALL(cudaMemcpy(h_counts, d_counts, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < npatterns; i ++) accum[i] = h_counts[i];
  t.Stop();


  std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaFree(d_counts));
}

