#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
// #include "codegen_LUT.cuh"
// #include "codegen_utils.cuh"
#include "expand_LUT.cuh"
#define FISSION
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

#include "GF_kernels.cuh"
#include "P1_GF_LUT.cuh"
#include "P2_GF_LUT.cuh"
#include "P3_GF_LUT.cuh"
#include "P7_GF_LUT.cuh"

// #define THREAD_PARALLEL


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
            << " GB, graph_mem = " << mem_graph/1024/1024/1024 << " GB\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";
  // CUDA_SAFE_CALL(cudaSetDevice(CUDA_SELECT_DEVICE));
  GraphGPU gg(g);
  gg.init_edgelist(g);

  size_t npatterns = 2;
  AccType *h_counts = (AccType *)malloc(sizeof(AccType) * npatterns);
  for (int i = 0; i < npatterns; i++) h_counts[i] = 0;
  AccType *d_counts;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_counts, sizeof(AccType) * npatterns));
  clear<<<1, npatterns>>>(d_counts);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
 
  size_t nwarps = WARPS_PER_BLOCK;
  size_t n_lists = 8;
  size_t n_bitmaps = 2;

  size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
  size_t per_block_bitmap_size = nwarps * n_bitmaps * ((size_t(md) + BITMAP_WIDTH-1)/BITMAP_WIDTH) * sizeof(vidType);

  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
  if (nb < nblocks) nblocks = nb;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));

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

  Roaring_LUTManager<> lut_manager(nblocks * nwarps, WARP_LIMIT, WARP_LIMIT); 

  AccType *G_INDEX, *G_INDEX1, *G_INDEX2, *G_INDEX3;
  AccType nowindex = nblocks * nwarps;
  AccType nowindex1 = nblocks;
  AccType nowindex3 = 500;
  CUDA_SAFE_CALL(cudaMalloc((void**) &(G_INDEX), sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(G_INDEX, &nowindex, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(G_INDEX2), sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(G_INDEX2, &nowindex, sizeof(AccType), cudaMemcpyHostToDevice));  
  CUDA_SAFE_CALL(cudaMalloc((void**) &(G_INDEX1), sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(G_INDEX1, &nowindex1, sizeof(AccType), cudaMemcpyHostToDevice));  
  CUDA_SAFE_CALL(cudaMalloc((void**) &(G_INDEX3), sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(G_INDEX1, &nowindex3, sizeof(AccType), cudaMemcpyHostToDevice));  

  Timer t;
  t.Start();
  // LUT vertex
  if (k == 1) {
    std::cout << "P1 GF LUT\n";
    P1_GF_LUT_warp<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, G_INDEX, lut_manager);
    lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT);
    P1_GF_LUT_block<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, G_INDEX1, lut_manager);
    lut_manager.recreate(1, md, md);
    nblocks = BLOCK_GROUP;
    for (vidType i = 0; i < nv; i++) {
      if (g.get_degree(i) > BLOCK_LIMIT) {
        lut_manager.update_para(1, g.get_degree(i), g.get_degree(i));
        clear_counterlist<<<nblocks, nthreads>>>(gg, frontier_list, md, i);
        GF_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, i);
        P1_GF_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, i);
      }
    }
  }
  // BS Edge
  else if (k == 2){
    std::cout << "P1 GF\n";
    P1_frequency_count<<<nblocks, nthreads>>>(nv, gg, frontier_list, md, d_counts, G_INDEX);
    P1_count_correction<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_counts, G_INDEX2);
  }
  else if (k == 3){
    std::cout << "P1 GF\n";
    P1_fused_matching<<<nblocks, nthreads>>>(nv, ne, gg, frontier_list, md, d_counts, G_INDEX, G_INDEX2);
  }
  else if (k == 4) {
    std::cout << "P3 GF\n";
    P3_fused_matching<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_counts, G_INDEX);
  }
  else if (k == 5) {
    std::cout << "P3 GF LUT\n";
    P3_GF_LUT_warp<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, G_INDEX, lut_manager);
    lut_manager.recreate(500, md, md);
    P3_GF_LUT_block<<<500, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, G_INDEX3, lut_manager);
  }
  else if (k == 6) {
    std::cout << "P7 GF\n";
    P7_fused_matching<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_counts, G_INDEX);
  }
  else if (k == 7) {
    std::cout << "P7 GF LUT\n";
    P7_GF_LUT_warp<<<nblocks, nthreads>>>(0, ne, gg, frontier_list, frontier_bitmap, md, d_counts, G_INDEX, lut_manager);
    lut_manager.recreate(500, md, md);
    P7_GF_LUT_block<<<500, nthreads>>>(0, ne, gg, frontier_list, frontier_bitmap, md, d_counts, G_INDEX3, lut_manager);
  }
  else {
    std::cout << "Not supported right now\n";
  }
  CUDA_SAFE_CALL(cudaMemcpy(h_counts, d_counts, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
  // for (size_t i = 0; i < npatterns; i ++) accum[i] = h_counts[i];
  accum[0] = h_counts[0] - h_counts[1];
  // accum[0] = h_counts[1];
  t.Stop();

  std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaFree(d_counts));
}

