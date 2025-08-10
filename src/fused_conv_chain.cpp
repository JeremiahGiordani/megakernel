#include "fused_conv_chain.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <omp.h>

namespace mk {

static inline int64_t ceil_div_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }
static inline int conv_out_dim_i(int in, int pad, int k, int stride) {
  const int eff_k = (k - 1) + 1; // dilation=1
  return ((in + 2*pad - eff_k) / stride) + 1;
}

bool region_is_two_conv_chain_3x3_s1_p1_relu_only(const FusedRegion& ir) {
  // Expect exactly: Conv, [Act], Conv, [Act]  (total ops 2..4)
  int conv_idx[2] = {-1,-1};
  int found_conv = 0;
  for (int i = 0; i < (int)ir.ops.size(); ++i) {
    if (ir.ops[i].kind == OpKind::Conv2D) {
      if (found_conv >= 2) return false;
      conv_idx[found_conv++] = i;
    } else if (ir.ops[i].kind == OpKind::Activation) {
      if (ir.ops[i].act.kind != ActKind::ReLU) return false; // MVP: ReLU only
    } else {
      return false;
    }
  }
  if (found_conv != 2) return false;

  const auto& c1 = ir.ops[conv_idx[0]].conv;
  const auto& c2 = ir.ops[conv_idx[1]].conv;

  auto ok_conv = [](const Conv2D& c) {
    return c.kH==3 && c.kW==3 && c.strideH==1 && c.strideW==1 &&
           c.padH==1 && c.padW==1 && c.dilationH==1 && c.dilationW==1;
  };
  if (!ok_conv(c1) || !ok_conv(c2)) return false;

  // Spatial dims must match with s=1,p=1,k=3
  // (ir.{inH,inW} == after c1 == after c2)
  return true;
}

// Produce one row (oh1) of Conv1 into the ring buffer for all mid channel blocks.
// Layout: NCHWc (vec=16). Packed weights: OIhw16i16o.
// Ring layout: [icb_mid][row_slot (0..2)][W][vec] — where icb_mid is Conv1 oc block.
// Produce one row (oh1) of Conv1 into the ring buffer
static void produce_conv1_row_avx512(
    const float* in, int inC, int H, int W,              // input shape
    float* ring, int midC,                               // ring target channels
    const ConvPackInfo& m1, const float* w1p,            // packed meta + weights
    const float* bias1_padded, bool relu1,
    int oh1, int vec)
{
  const int inCb  = (int)ceil_div_i64(inC,  vec);
  const int midCb = (int)ceil_div_i64(midC, vec);
  const int outH1 = H;          // k=3,s=1,p=1
  const int outW1 = W;

  const int slot = ((oh1 % 3) + 3) % 3;

  for (int ocb = 0; ocb < midCb; ++ocb) {
    __m512 bvec = _mm512_setzero_ps();
    if (bias1_padded) bvec = _mm512_loadu_ps(bias1_padded + ocb * vec);

    for (int ow = 0; ow < outW1; ++ow) {
      __m512 acc = _mm512_setzero_ps();

      for (int icb = 0; icb < inCb; ++icb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = oh1 - 1 + kh;
          if ((unsigned)ih >= (unsigned)H) continue;
          for (int kw = 0; kw < 3; ++kw) {
            const int iw = ow - 1 + kw;
            if ((unsigned)iw >= (unsigned)W) continue;

            const float* in_pix =
              in + ((((int64_t)icb * H) * W + (int64_t)ih * W + iw) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m1.stride_ocb +
                (int64_t)icb * m1.stride_icb +
                (int64_t)kh  * m1.stride_kh  +
                (int64_t)kw  * m1.stride_kw;

            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const float x = in_pix[ic_i];
              if (x == 0.0f) continue;
              const int64_t Wrow = Wbase + (int64_t)ic_i * m1.stride_ic_i;
              const __m512 wv = _mm512_loadu_ps(w1p + Wrow);
              const __m512 xv = _mm512_set1_ps(x);
              acc = _mm512_fmadd_ps(xv, wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu1) {
        const __m512 z = _mm512_setzero_ps();
        acc = _mm512_max_ps(acc, z);
      }

      float* dst = ring + ((((int64_t)ocb * 3 + slot) * outW1 + ow) * vec);
      _mm512_storeu_ps(dst, acc);
    }
  }
}

// Consume from ring to produce one row (oh2) of Conv2
static void consume_conv2_row_avx512(
    const float* ring, int midC, int H1s, int W1s,
    float* out, int outC, int H2s, int W2s, int oh2,
    const ConvPackInfo& m2, const float* w2p,
    const float* bias2_padded, bool relu2, int vec)
{
  const int midCb = (int)ceil_div_i64(midC, vec);
  const int outCb = (int)ceil_div_i64(outC, vec);

  const int tail = outC - (outCb - 1) * vec;
  const __mmask16 tail_mask = (outCb > 0 && tail > 0 && tail < vec)
                                ? (__mmask16)((1u << tail) - 1u)
                                : (__mmask16)0xFFFF;

  for (int ocb = 0; ocb < outCb; ++ocb) {
    const bool tail_block = (ocb == outCb - 1) && (tail < vec);
    const __mmask16 store_mask = tail_block ? tail_mask : (__mmask16)0xFFFF;

    __m512 bvec = _mm512_setzero_ps();
    if (bias2_padded) bvec = _mm512_loadu_ps(bias2_padded + ocb * vec);

    for (int ow = 0; ow < W2s; ++ow) {
      __m512 acc = _mm512_setzero_ps();

      for (int icb = 0; icb < midCb; ++icb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ry = oh2 - 1 + kh;
          if ((unsigned)ry >= (unsigned)H1s) continue;
          const int slot = ((ry % 3) + 3) % 3;
          for (int kw = 0; kw < 3; ++kw) {
            const int rx = ow - 1 + kw;
            if ((unsigned)rx >= (unsigned)W1s) continue;

            const float* in_pix =
              ring + ((((int64_t)icb * 3 + slot) * W1s + rx) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m2.stride_ocb +
                (int64_t)icb * m2.stride_icb +
                (int64_t)kh  * m2.stride_kh  +
                (int64_t)kw  * m2.stride_kw;

            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const float x = in_pix[ic_i];
              if (x == 0.0f) continue;
              const int64_t Wrow = Wbase + (int64_t)ic_i * m2.stride_ic_i;
              const __m512 wv = _mm512_loadu_ps(w2p + Wrow);
              const __m512 xv = _mm512_set1_ps(x);
              acc = _mm512_fmadd_ps(xv, wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu2) {
        const __m512 z = _mm512_setzero_ps();
        acc = _mm512_max_ps(acc, z);
      }

      float* out_pix =
        out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + ow) * vec);
      if (tail_block) _mm512_mask_storeu_ps(out_pix, store_mask, acc);
      else            _mm512_storeu_ps(out_pix, acc);
    }
  }
}


void run_fused_two_conv_chain_avx512(
    const FusedRegion& ir,
    const PackedWeights& Wp,
    const float* in_nchwc, int inC, int H, int Ws,
    float* out_nchwc, int vec,
    int num_threads) 
{
    assert(vec == 16 && Wp.vec == 16);

    int cpos[2] = {-1,-1};
    int ci = 0;
    for (int i = 0; i < (int)ir.ops.size(); ++i)
        if (ir.ops[i].kind == OpKind::Conv2D) cpos[ci++] = i;
    assert(ci == 2);

    const auto& c1 = ir.ops[cpos[0]].conv;
    const auto& c2 = ir.ops[cpos[1]].conv;

    auto has_relu_after = [&](int conv_pos)->bool {
        if (conv_pos+1 < (int)ir.ops.size() && ir.ops[conv_pos+1].kind == OpKind::Activation)
        return ir.ops[conv_pos+1].act.kind == ActKind::ReLU;
        return false;
    };
    const bool relu1 = has_relu_after(cpos[0]);
    const bool relu2 = has_relu_after(cpos[1]);

    const int midC = (int)c1.outC;
    const int outC = (int)c2.outC;
    const int H1s = H, W1s = Ws;
    const int H2s = H1s, W2s = W1s;

    // conv indices inside PackedWeights (conv-only order)
    int idx0 = 0, idx1 = 1;
    const ConvPackInfo& m1 = Wp.meta[idx0];
    const ConvPackInfo& m2 = Wp.meta[idx1];
    const float* w1p = reinterpret_cast<const float*>(Wp.per_conv[idx0].ptr);
    const float* w2p = reinterpret_cast<const float*>(Wp.per_conv[idx1].ptr);
    const float* b1p = Wp.bias_ptrs[idx0] ? reinterpret_cast<const float*>(Wp.bias_ptrs[idx0]) : nullptr;
    const float* b2p = Wp.bias_ptrs[idx1] ? reinterpret_cast<const float*>(Wp.bias_ptrs[idx1]) : nullptr;

    const int midCb = (int)ceil_div_i64(midC, vec);

    #pragma omp parallel num_threads(num_threads)
    {
        const int nth = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int rows_per = (H2s + nth - 1) / nth;
        const int y0 = tid * rows_per;
        const int y1 = std::min(H2s, y0 + rows_per);

        // Thread-private ring buffer: midCb × 3 × W × vec
        const size_t ring_elems = (size_t)midCb * 3 * (size_t)W1s * (size_t)vec;
        std::unique_ptr<float[]> ring(new float[ring_elems]);

        int last_produced = -1;
        for (int oh = y0; oh < y1; ++oh) {
        const int need_hi = std::min(H1s - 1, oh + 1);
        for (int ry = std::max(last_produced + 1, 0); ry <= need_hi; ++ry) {
            produce_conv1_row_avx512(
                in_nchwc, inC, H1s, W1s,
                ring.get(), midC,
                m1, w1p, b1p, relu1, ry, vec);
            last_produced = ry;
        }

        consume_conv2_row_avx512(
            ring.get(), midC, H1s, W1s,
            out_nchwc, outC, H2s, W2s, oh,
            m2, w2p, b2p, relu2, vec);
        }
    } // omp parallel
}


} // namespace mk
