#include "conv_microkernels.hpp"
#include <immintrin.h>
#include <cassert>
#include <cstring>

// Reuse helpers if you have them elsewhere; keep local for compilation unit
static inline long long ceil_div_ll(long long a, long long b) { return (a + b - 1) / b; }
static inline int conv_out_dim_i(int in, int pad, int k, int stride) {
  const int eff_k = (k - 1) + 1; // dilation=1
  return ((in + 2*pad - eff_k) / stride) + 1;
}

namespace mk {

void conv3x3_s1_nchwc_oihw16i16o_avx512(
    const float* in, int inC, int H, int W,
    float* out, int outC,
    const ConvPackInfo& meta, const float* Wp,
    int padH, int padW, int vec)
{
    // Preconditions
    assert(vec == 16 && "This microkernel is AVX-512 FP32 (vec=16) only");
    assert(meta.vec == 16);
    assert(meta.kH == 3 && meta.kW == 3);
    (void)vec;

    const int inCb  = (int)ceil_div_ll(inC, 16);
    const int outCb = (int)ceil_div_ll(outC, 16);

    const int outH = conv_out_dim_i(H, padH, /*k=*/3, /*s=*/1);
    const int outW = conv_out_dim_i(W, padW, /*k=*/3, /*s=*/1);

    // We compute each output vector from zero; no need to memset whole buffer.
    // Tail handling: last oc block may be <16 lanes.
    const int tail = outC - (outCb - 1) * 16;
    const __mmask16 tail_mask = (outCb > 0 && tail > 0 && tail < 16)
                                    ? (__mmask16)((1u << tail) - 1u)
                                    : (__mmask16)0xFFFF;

    for (int ocb = 0; ocb < outCb; ++ocb) {
        const bool tail_block = (ocb == outCb - 1) && (tail < 16);
        const __mmask16 store_mask = tail_block ? tail_mask : (__mmask16)0xFFFF;

        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                // Output pixel base (vector of 16 oc lanes)
                float* out_pix = out + ((((long long)ocb * outH) * outW + (long long)oh * outW + ow) * 16);

                __m512 acc = _mm512_setzero_ps();

                // Sum over input channel blocks and 3x3 taps
                for (int icb = 0; icb < inCb; ++icb) {
                    for (int kh = 0; kh < 3; ++kh) {
                        const int ih = oh - padH + kh;
                        if ((unsigned)ih >= (unsigned)H) continue;
                        for (int kw = 0; kw < 3; ++kw) {
                            const int iw = ow - padW + kw;
                            if ((unsigned)iw >= (unsigned)W) continue;

                            // Input pixel for this (icb, ih, iw)
                            const float* in_pix =
                                in + ((((long long)icb * H) * W + (long long)ih * W + iw) * 16);

                            // Base into packed weights for (ocb, icb, kh, kw)
                            const long long Wbase =
                                (long long)ocb * meta.stride_ocb +
                                (long long)icb * meta.stride_icb +
                                (long long)kh  * meta.stride_kh  +
                                (long long)kw  * meta.stride_kw;

                        // ic_i inner (broadcast input lane; FMA with vector of 16 oc lanes)
                            for (int ic_i = 0; ic_i < 16; ++ic_i) {
                                const float x = in_pix[ic_i];                  // scalar
                                if (x == 0.0f) continue;                       // tiny speedup on zeros
                                const long long Wrow = Wbase + (long long)ic_i * meta.stride_ic_i;
                                const __m512 w = _mm512_loadu_ps(Wp + Wrow);    // 16 oc lanes
                                const __m512 b = _mm512_set1_ps(x);
                                acc = _mm512_fmadd_ps(b, w, acc);
                            }
                        }
                    }
                }

                // Store with mask for tail block
                if (tail_block) {
                    _mm512_mask_storeu_ps(out_pix, store_mask, acc);
                } else {
                    _mm512_storeu_ps(out_pix, acc);
                }
            } // ow
        }   // oh
    }     // ocb
}

} // namespace mk
