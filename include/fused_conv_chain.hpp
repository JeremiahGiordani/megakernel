#pragma once
#include "ir.hpp"
#include "schedule.hpp"
#include "weight_packing.hpp"

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

} // namespace mk
