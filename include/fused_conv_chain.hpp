#pragma once
#include "ir.hpp"
#include "schedule.hpp"
#include "weight_packing.hpp"
#include "mk_compiler.hpp"

#ifndef MK_OW_TILE
#define MK_OW_TILE 8   // try 8, 12, 16 later
#endif

#ifndef MK_OCB_STEP
#define MK_OCB_STEP 2  // process out-channel blocks in batches of 2 to manage registers
#endif

namespace mk {

// Returns true if region is exactly: Conv → [Act?] → Conv → [Act?],
// with both Convs being 3x3,s=1,p=1,dilation=1 and Act either None or ReLU.
bool region_is_two_conv_chain_3x3_s1_p1_relu_only(const FusedRegion& ir);

// Run fused two-conv chain using an AVX-512 ring buffer (NCHWc, vec=16).
// Preconditions: region_is_two_conv_chain_3x3_s1_p1_relu_only(ir) == true
// Input: N=1, NCHWc (vec=16). Output: NCHWc (vec=16).
void run_fused_two_conv_chain_avx512(
    const FusedRegion& ir,
    const PackedWeights& W,
    const float* in_nchwc, int inC, int H, int Wspatial,
    float* out_nchwc, int vec = 16,
    int num_threads = 1);


void run_fused_two_conv_chain_avx512_ringless(
    const FusedRegion& ir,
    const PackedWeights& Wp,
    const float* MK_RESTRICT in_nchwc, int inC, int H, int W,
    float* MK_RESTRICT out_nchwc, int vec,
    int num_threads);


} // namespace mk