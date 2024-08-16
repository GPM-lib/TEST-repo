void LUT_5anti_tailed_diamond(Graph &g, StorageMeta &meta, uint64_t &total){

  uint64_t counter = 0;

  // BEGIN OF CODEGEN
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = 0; v0_idx < candidate_v0.size(); v0_idx += 1){
    auto v0 = candidate_v0[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    
    __build_LUT(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));;
    for (vidType v1_idx = 0; v1_idx < candidate_v1.size(); v1_idx += 1) {
      __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), /*slot_id=*/4);
      auto candidate_v2_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/4);
      for(vidType v2_idx_idx = 0; v2_idx_idx < candidate_v2_idx.size(); v2_idx_idx ++){
        auto v2_idx = candidate_v2_idx[v2_idx_idx];
        __intersect(meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/false, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/true, /*upper_bound=*/-1), /*upper_bound=*/v1_idx, /*output_slot=*/0);
        __build_index_from_vmap(g, meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/0, /*slot_id=*/-1), /*slot_id=*/5);
        __difference(meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/false, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/-1, /*output_slot=*/1);
        auto candidate_v3_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/5);
        for(vidType v3_idx_idx = 0; v3_idx_idx < candidate_v3_idx.size(); v3_idx_idx += 1){
          auto v3_idx = candidate_v3_idx[v3_idx_idx];
          counter += __difference_num(__get_vmap_from_heap(g, meta, /*bitmap_id=*/1, /*slot_id=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v3_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/-1);
        }
      }
    }
  }
  // END OF CODEGEN

  total = counter;
}
