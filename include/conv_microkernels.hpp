#pragma once
#include "schedule.hpp"

namespace mk {

// AVX-512 FP32, NCHWc (vec=16) input/output, weights packed as OIhw16i16o.
// Supports kH=kW=3, stride=1, pad=1, dilation=1.
void conv3x3_s1_nchwc_oihw16i16o_avx512(
    const float* in, int inC, int H, int W,
    float* out, int outC,
    const ConvPackInfo& meta, const float* Wp,
    int padH, int padW, int vec);

} // namespace mk
