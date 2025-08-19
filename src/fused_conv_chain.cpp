#include "fused_conv_chain.hpp"
#include "aligned_alloc.hpp"
#include "mk_compiler.hpp"
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


// Ringless fused Conv3x3->ReLU->Conv3x3->ReLU, NCHWc (vec==16), stride=1, pad=1.
void run_fused_two_conv_chain_avx512_ringless(
    const FusedRegion& ir,
    const PackedWeights& Wp,
    const float* MK_RESTRICT in_nchwc, int inC, int H, int W,
    float* MK_RESTRICT out_nchwc, int vec,
    int num_threads)
{
  assert(vec == 16 && Wp.vec == 16);

  // locate the two convs
  int cpos[2] = {-1,-1}, ci=0;
  for (int i=0;i<(int)ir.ops.size();++i)
    if (ir.ops[i].kind==OpKind::Conv2D) cpos[ci++]=i;
  assert(ci==2);

  const auto& c1 = ir.ops[cpos[0]].conv;
  const auto& c2 = ir.ops[cpos[1]].conv;
  auto has_relu_after = [&](int p){ return (p+1<(int)ir.ops.size() && ir.ops[p+1].kind==OpKind::Activation && ir.ops[p+1].act.kind==ActKind::ReLU); };
  const bool relu1 = has_relu_after(cpos[0]);
  const bool relu2 = has_relu_after(cpos[1]);

  const int midC = (int)c1.outC;
  const int outC = (int)c2.outC;
  const int H1s = H, W1s = W; // same spatial (k=3,s=1,p=1)

  // weights & meta
  const int idx0=0, idx1=1;
  const ConvPackInfo& m1 = Wp.meta[idx0];
  const ConvPackInfo& m2 = Wp.meta[idx1];
  const float* MK_RESTRICT w1p = reinterpret_cast<const float*>(Wp.per_conv[idx0].ptr);
  const float* MK_RESTRICT w2p = reinterpret_cast<const float*>(Wp.per_conv[idx1].ptr);
  const float* MK_RESTRICT b1p = Wp.bias_ptrs[idx0] ? reinterpret_cast<const float*>(Wp.bias_ptrs[idx0]) : nullptr;
  const float* MK_RESTRICT b2p = Wp.bias_ptrs[idx1] ? reinterpret_cast<const float*>(Wp.bias_ptrs[idx1]) : nullptr;

  const int inCb  = (inC  + vec - 1) / vec;
  const int midCb = (midC + vec - 1) / vec;
  const int outCb = (outC + vec - 1) / vec;

  #pragma omp parallel num_threads(num_threads)
  {
    const int nth = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int rows_per = (H1s + nth - 1) / nth;
    const int y0 = tid * rows_per;
    const int y1 = std::min(H1s, y0 + rows_per);

    alignas(64) float ybuf[3][MK_OW_TILE+2][16]; // [k2h][rx_idx 0..T+1][lane 0..15]

    for (int oh = y0; oh < y1; ++oh) {
      for (int ow0 = 0; ow0 < W1s; ow0 += MK_OW_TILE) {
        const int T = std::min(MK_OW_TILE, W1s - ow0);

        // ---- PRECOMPUTE Conv1 outputs ybuf for the 3 rows and (T+2) columns ----
        // k2h maps to ry = oh + (k2h-1); rx_idx 0..T+1 maps to rx = ow0-1 + rx_idx
        for (int k2h = 0; k2h < 3; ++k2h) {
          const int ry = oh - 1 + k2h;
          const bool ry_valid = (0 <= ry && ry < H1s);

          for (int rx_idx = 0; rx_idx < T + 2; ++rx_idx) {
            const int rx = ow0 - 1 + rx_idx;
            const bool rx_valid = (0 <= rx && rx < W1s);

            __m512 yv = _mm512_setzero_ps(); // 16-lane (mid oc_i)

            if (ry_valid && rx_valid) {
              // Conv1 at (ry, rx) across all mid lanes (mcb loop outside)
              // We accumulate across midCb below, so init to zero here.
            }

            // We will accumulate yv over mcb below, but we need to rebuild per mcb.
            // Instead: compute full y over input channels for each midCb separately and add.
            // Loop midCb and fold into yv:
            for (int mcb = 0; mcb < midCb; ++mcb) {
              __m512 yv_mcb = _mm512_setzero_ps();

              if (ry_valid && rx_valid) {
                for (int icb = 0; icb < inCb; ++icb) {
                  for (int k1h = 0; k1h < 3; ++k1h) {
                    const int ih = ry - 1 + k1h;
                    if ((unsigned)ih >= (unsigned)H) continue;
                    for (int k1w = 0; k1w < 3; ++k1w) {
                      const int iw = rx - 1 + k1w;
                      if ((unsigned)iw >= (unsigned)W) continue;

                      const float* MK_RESTRICT in_pix =
                        in_nchwc + ((((int64_t)icb * H) * W + (int64_t)ih * W + iw) * vec);

                      const int64_t Wbase =
                          (int64_t)mcb * m1.stride_ocb +
                          (int64_t)icb * m1.stride_icb +
                          (int64_t)k1h * m1.stride_kh  +
                          (int64_t)k1w * m1.stride_kw;

                      // accumulate over input SIMD lanes
                      for (int ic_i = 0; ic_i < vec; ++ic_i) {
                        const float x = in_pix[ic_i];
                        if (x == 0.0f) continue;
                        const __m512 wv1 = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
                        yv_mcb = _mm512_fmadd_ps(_mm512_set1_ps(x), wv1, yv_mcb);
                      }
                    }
                  }
                }
                // bias1 + relu1 for this midCb slice
                if (b1p) {
                  const __m512 bvec = _mm512_load_ps(b1p + mcb * vec);
                  yv_mcb = _mm512_add_ps(yv_mcb, bvec);
                }
                if (relu1) {
                  const __m512 z = _mm512_setzero_ps();
                  yv_mcb = _mm512_max_ps(yv_mcb, z);
                }
              } // (ry,rx) valid

              yv = _mm512_add_ps(yv, yv_mcb);
            } // mcb

            // store final Conv1(mid) vector to stack buffer
            _mm512_store_ps(ybuf[k2h][rx_idx], yv);
          } // rx_idx
        } // k2h

        // ---- ACCUMULATE Conv2 into final outputs (in ocb batches) ----
        for (int ocb0 = 0; ocb0 < outCb; ocb0 += MK_OCB_STEP) {
          const int occ = std::min(MK_OCB_STEP, outCb - ocb0);

          __m512 acc[MK_OCB_STEP][MK_OW_TILE];
          for (int j=0;j<occ;++j)
            for (int t=0;t<T;++t) acc[j][t] = _mm512_setzero_ps();

          // taps over Conv2 kernel and mid lanes
          for (int k2h = 0; k2h < 3; ++k2h) {
            for (int k2w = 0; k2w < 3; ++k2w) {
              // rx_idx used: for local t (0..T-1), rx_idx = t + k2w
              for (int ic_i = 0; ic_i < vec; ++ic_i) {
                // weights for each ocb in batch
                __m512 w2v[MK_OCB_STEP];
                for (int j=0;j<occ;++j) {
                  const int ocb = ocb0 + j;
                  const int64_t Wbase2 =
                      (int64_t)ocb * m2.stride_ocb +
                      /*mcb folded already via ybuf*/ 0 + // handled by ybuf sum over midCb
                      (int64_t)k2h * m2.stride_kh +
                      (int64_t)k2w * m2.stride_kw +
                      (int64_t)ic_i * m2.stride_ic_i;
                  // NOTE: since we summed over midCb in ybuf, we should also sum over mcb in weights.
                  // But weights are per (ocb, mcb, k2h, k2w, ic_i). So we must accumulate across mcb here.
                  // Load-and-sum across mcb:
                  __m512 wsum = _mm512_setzero_ps();
                  for (int mcb = 0; mcb < midCb; ++mcb) {
                    const int64_t Wb = (int64_t)ocb * m2.stride_ocb +
                                       (int64_t)mcb * m2.stride_icb +
                                       (int64_t)k2h * m2.stride_kh +
                                       (int64_t)k2w * m2.stride_kw +
                                       (int64_t)ic_i * m2.stride_ic_i;
                    wsum = _mm512_add_ps(wsum, _mm512_load_ps(w2p + Wb));
                  }
                  w2v[j] = wsum;
                }

                // FMA across tile columns
                for (int t = 0; t < T; ++t) {
                  const int rx_idx = t + k2w; // 0..T+1
                  // Get scalar lane from ybuf
                  // ybuf already summed over mcb and has bias1+relu1
                  const float s0 = ybuf[k2h][rx_idx][ic_i];

                  const __m512 sv = _mm512_set1_ps(s0);
                  for (int j=0;j<occ;++j) {
                    acc[j][t] = _mm512_fmadd_ps(sv, w2v[j], acc[j][t]);
                  }
                } // t
              } // ic_i
            } // k2w
          } // k2h

          // Epilogue + store
          for (int j=0;j<occ;++j) {
            const int ocb = ocb0 + j;
            __m512 b2 = _mm512_setzero_ps();
            if (b2p) b2 = _mm512_load_ps(b2p + ocb * vec);

            for (int t=0;t<T;++t) {
              __m512 v = _mm512_add_ps(acc[j][t], b2);
              if (relu2) { const __m512 z = _mm512_setzero_ps(); v = _mm512_max_ps(v, z); }

              float* MK_RESTRICT out_pix =
                out_nchwc + ((((int64_t)ocb * H1s) * W1s + (int64_t)oh * W1s + (ow0 + t)) * vec);
              _mm512_store_ps(out_pix, v);
            }
          }
        } // ocb batch
      } // tiles
    } // rows
  } // omp
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

// Zero one ring column (all vec floats) for a given slot across all midCb
static inline void zero_ring_column(
    float* ring, int midCb, int curW2, int vec, int slot, int li)
{
  for (int mcb = 0; mcb < midCb; ++mcb) {
    float* dst = ring + ((((int64_t)mcb * 3 + slot) * curW2 + li) * vec);
    std::memset(dst, 0, (size_t)vec * sizeof(float));
  }
}

// Produce a SINGLE output column of Conv1 (branchy, with bounds) into ring at local index `li`
static inline void produce_conv1_one_col_branchy_to_ring(
    const float* in, int inC, int H, int W,
    float* ring, int curW2, int midC,
    const ConvPackInfo& m1, const float* w1p,
    const float* bias1, bool relu1,
    int oh1, int global_ow, int li, int vec)
{
  const int inCb  = (int)((inC + vec - 1) / vec);
  const int midCb = (int)((midC + vec - 1) / vec);
  const int slot = ((oh1 % 3) + 3) % 3;

  for (int mcb = 0; mcb < midCb; ++mcb) {
    __m512 acc = _mm512_setzero_ps();
    __m512 bvec = _mm512_setzero_ps();
    if (bias1) bvec = _mm512_load_ps(bias1 + mcb * vec);

    for (int icb = 0; icb < inCb; ++icb) {
      for (int kh = 0; kh < 3; ++kh) {
        const int ih = oh1 - 1 + kh;
        if ((unsigned)ih >= (unsigned)H) continue;
        for (int kw = 0; kw < 3; ++kw) {
          const int iw = global_ow - 1 + kw;
          if ((unsigned)iw >= (unsigned)W) continue;

          const float* in_pix =
              in + ((((int64_t)icb * H) * W + (int64_t)ih * W + iw) * vec);

          const int64_t Wbase =
              (int64_t)mcb * m1.stride_ocb +
              (int64_t)icb * m1.stride_icb +
              (int64_t)kh  * m1.stride_kh  +
              (int64_t)kw  * m1.stride_kw;

          for (int ic_i = 0; ic_i < vec; ++ic_i) {
            const float x = in_pix[ic_i];
            if (x == 0.0f) continue;
            const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
            acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
          }
        }
      }
    }

    acc = _mm512_add_ps(acc, bvec);
    if (relu1) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }

    float* dst = ring + ((((int64_t)mcb * 3 + slot) * curW2 + li) * vec);
    _mm512_store_ps(dst, acc);
  }
}

// Branchy tile producer for BODY columns only. Note: `ow0_shifted = ow0 - 1`.
// Local li range is [li_lo .. li_hi] with li==1 -> global ow0.
static inline void produce_conv1_row_tile_branchy_body(
    const float* in, int inC, int H, int W,
    float* ring, int curW2, int midC,
    const ConvPackInfo& m1, const float* w1p,
    const float* bias1, bool relu1,
    int oh1, int ow0_shifted, int li_lo, int li_hi, int vec)
{
  const int inCb  = (int)((inC + vec - 1) / vec);
  const int midCb = (int)((midC + vec - 1) / vec);
  const int slot = ((oh1 % 3) + 3) % 3;

  for (int mcb = 0; mcb < midCb; ++mcb) {
    __m512 bvec = _mm512_setzero_ps();
    if (bias1) bvec = _mm512_load_ps(bias1 + mcb * vec);

    for (int li = li_lo; li <= li_hi; ++li) {
      const int ow = (ow0_shifted + li); // li=1 -> global ow0
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
                (int64_t)mcb * m1.stride_ocb +
                (int64_t)icb * m1.stride_icb +
                (int64_t)kh  * m1.stride_kh  +
                (int64_t)kw  * m1.stride_kw;

            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const float x = in_pix[ic_i];
              if (x == 0.0f) continue;
              const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu1) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }

      float* dst = ring + ((((int64_t)mcb * 3 + slot) * curW2 + li) * vec);
      _mm512_store_ps(dst, acc);
    }
  }
}

// Interior, width-tiled producer that writes BODY columns with +1 halo offset.
// Note: pass `ow0_shifted = ow0 - 1`, and `curW2 = curW + 2`.
static inline void produce_conv1_row_tile8_interior_halo(
    const float* in, int inC, int H, int W,
    float* ring, int curW2, int midC,
    const ConvPackInfo& m1, const float* w1p,
    const float* bias1, bool relu1,
    int oh1, int ow0_shifted, int ow0, int curW, int vec)
{
  constexpr int OW_TILE = MK_OW_TILE;
  const int inCb  = (int)((inC + vec - 1) / vec);
  const int midCb = (int)((midC + vec - 1) / vec);
  const int slot = ((oh1 % 3) + 3) % 3;

  // BODY global interior columns intersected with this tile:
  const int ow_start = std::max(1, ow0);
  const int ow_end   = std::min(W - 2, ow0 + curW - 1);
  if (ow_start > ow_end) return;

  const int main_start = ow_start;
  const int main_end = main_start + ((ow_end - main_start + 1) / OW_TILE) * OW_TILE;

  for (int mcb = 0; mcb < midCb; ++mcb) {
    __m512 bvec = _mm512_setzero_ps();
    if (bias1) bvec = _mm512_load_ps(bias1 + mcb * vec);

    for (int ow = main_start; ow < main_end; ow += OW_TILE) {
      __m512 acc[OW_TILE];
      for (int t=0;t<OW_TILE;++t) acc[t] = _mm512_setzero_ps();

      for (int icb = 0; icb < inCb; ++icb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = oh1 - 1 + kh; // interior row
          for (int kw = 0; kw < 3; ++kw) {
            const int base_col = ow - 1 + kw;
            const float* in_base =
              in + ((((int64_t)icb * H) * W + (int64_t)ih * W + base_col) * vec);

            const int64_t Wbase =
                (int64_t)mcb * m1.stride_ocb +
                (int64_t)icb * m1.stride_icb +
                (int64_t)kh  * m1.stride_kh  +
                (int64_t)kw  * m1.stride_kw;

            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
              for (int t=0;t<OW_TILE;++t) {
                const float x = in_base[ic_i + t*vec];
                acc[t] = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc[t]);
              }
            }
          }
        }
      }

      // store to ring with +1 offset: li = (ow - ow0_shifted) + t
      for (int t=0;t<OW_TILE;++t) {
        __m512 v = _mm512_add_ps(acc[t], bvec);
        if (relu1) { const __m512 z = _mm512_setzero_ps(); v = _mm512_max_ps(v, z); }
        const int li = (ow - ow0_shifted) + t;
        float* dst = ring + ((((int64_t)mcb * 3 + slot) * curW2 + li) * vec);
        _mm512_store_ps(dst, v);
      }
    }

    // leftovers
    for (int ow = main_end; ow <= ow_end; ++ow) {
      __m512 acc = _mm512_setzero_ps();
      for (int icb = 0; icb < inCb; ++icb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = oh1 - 1 + kh;
          for (int kw = 0; kw < 3; ++kw) {
            const int base_col = ow - 1 + kw;
            const float* in_base =
              in + ((((int64_t)icb * H) * W + (int64_t)ih * W + base_col) * vec);
            const int64_t Wbase =
                (int64_t)mcb * m1.stride_ocb +
                (int64_t)icb * m1.stride_icb +
                (int64_t)kh  * m1.stride_kh  +
                (int64_t)kw  * m1.stride_kw;
            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
              const float x = in_base[ic_i];
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }
      acc = _mm512_add_ps(acc, bvec);
      if (relu1) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }
      const int li = (ow - ow0_shifted);
      float* dst = ring + ((((int64_t)mcb * 3 + slot) * curW2 + li) * vec);
      _mm512_store_ps(dst, acc);
    }
  }
}

// Branchy tile consumer for arbitrary li range (uses halo: li -> rx = ow-1+kw)
static inline void consume_conv2_row_tile_branchy_halo(
    const float* ring, int curW2, int midC, int H1s, int W1s,
    float* out, int outC, int H2s, int W2s, int oh2, int ow0,
    int li_lo, int li_hi,
    const ConvPackInfo& m2, const float* w2p,
    const float* bias2, bool relu2, int vec)
{
  const int midCb = (int)((midC + vec - 1) / vec);
  const int outCb = (int)((outC + vec - 1) / vec);

  const int tail = outC - (outCb - 1) * vec;
  const __mmask16 tail_mask = (outCb > 0 && tail > 0 && tail < vec)
                                ? (__mmask16)((1u << tail) - 1u)
                                : (__mmask16)0xFFFF;

  for (int ocb = 0; ocb < outCb; ++ocb) {
    const bool tail_block = (ocb == outCb - 1) && (tail < vec);
    const __mmask16 store_mask = tail_block ? tail_mask : (__mmask16)0xFFFF;
    __m512 bvec = _mm512_setzero_ps();
    if (bias2) bvec = _mm512_load_ps(bias2 + ocb * vec);

    for (int li = li_lo; li <= li_hi; ++li) {
      const int ow = ow0 + (li - 1); // li=1 -> global ow0
      __m512 acc = _mm512_setzero_ps();

      for (int mcb = 0; mcb < midCb; ++mcb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ry = oh2 - 1 + kh;
          if ((unsigned)ry >= (unsigned)H1s) continue;
          const int slot = ((ry % 3) + 3) % 3;
          for (int kw = 0; kw < 3; ++kw) {
            const int rx = ow - 1 + kw;
            if ((unsigned)rx >= (unsigned)W1s) continue;

            const int li_src = (rx - ow0) + 1; // halo shift
            if ((unsigned)li_src >= (unsigned)curW2) continue;

            const float* in_pix =
              ring + ((((int64_t)mcb * 3 + slot) * curW2 + li_src) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m2.stride_ocb +
                (int64_t)mcb * m2.stride_icb +
                (int64_t)kh  * m2.stride_kh  +
                (int64_t)kw  * m2.stride_kw;

            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const float x = in_pix[ic_i];
              if (x == 0.0f) continue;
              const __m512 wv = _mm512_load_ps(w2p + Wbase + (int64_t)ic_i * m2.stride_ic_i);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu2) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }

      float* out_pix =
        out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + ow) * vec);
      if (tail_block) _mm512_mask_storeu_ps(out_pix, store_mask, acc);
      else            _mm512_store_ps(out_pix, acc);
    }
  }
}

// Interior tile consumer using only the tile ring (with halo)
static inline void consume_conv2_row_tile8_interior_halo(
    const float* ring, int curW2, int midC,
    float* out, int outC, int H2s, int W2s, int oh2, int ow0,
    int ow_start, int ow_end,        // global interior range in this tile
    const ConvPackInfo& m2, const float* w2p,
    const float* bias2, bool relu2, int vec)
{
  constexpr int OW_TILE = MK_OW_TILE;
  const int midCb = (int)((midC + vec - 1) / vec);
  const int outCb = (int)((outC + vec - 1) / vec);

  const int tail = outC - (outCb - 1) * vec;
  const __mmask16 tail_mask = (outCb > 0 && tail > 0 && tail < vec)
                                ? (__mmask16)((1u << tail) - 1u)
                                : (__mmask16)0xFFFF;

  const int main_start = ow_start;
  const int main_end = main_start + ((ow_end - main_start + 1) / OW_TILE) * OW_TILE;

  for (int ocb = 0; ocb < outCb; ++ocb) {
    const bool tail_block = (ocb == outCb - 1) && (tail < vec);
    const __mmask16 store_mask = tail_block ? tail_mask : (__mmask16)0xFFFF;
    __m512 bvec = _mm512_setzero_ps();
    if (bias2) bvec = _mm512_load_ps(bias2 + ocb * vec);

    for (int ow = main_start; ow < main_end; ow += OW_TILE) {
      __m512 acc[OW_TILE];
      for (int t=0;t<OW_TILE;++t) acc[t] = _mm512_setzero_ps();

      for (int mcb = 0; mcb < midCb; ++mcb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ry = oh2 - 1 + kh;   // interior row -> valid
          const int slot = ((ry % 3) + 3) % 3;
          for (int kw = 0; kw < 3; ++kw) {
            // halo shift: base_col_local = (ow - ow0) + kw
            const int base_col_local = (ow - ow0) + kw;
            const float* in_base =
              ring + ((((int64_t)mcb * 3 + slot) * curW2 + base_col_local) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m2.stride_ocb +
                (int64_t)mcb * m2.stride_icb +
                (int64_t)kh  * m2.stride_kh  +
                (int64_t)kw  * m2.stride_kw;

            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w2p + Wbase + (int64_t)ic_i * m2.stride_ic_i);
              for (int t=0;t<OW_TILE;++t) {
                const float x = in_base[ic_i + t*vec];
                acc[t] = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc[t]);
              }
            }
          }
        }
      }

      for (int t=0;t<OW_TILE;++t) {
        __m512 v = _mm512_add_ps(acc[t], bvec);
        if (relu2) { const __m512 z = _mm512_setzero_ps(); v = _mm512_max_ps(v, z); }
        float* out_pix =
          out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + (ow + t)) * vec);
        if (tail_block) _mm512_mask_storeu_ps(out_pix, store_mask, v);
        else            _mm512_store_ps(out_pix, v);
      }
    }

    for (int ow = main_end; ow <= ow_end; ++ow) {
      __m512 acc = _mm512_setzero_ps();

      for (int mcb = 0; mcb < midCb; ++mcb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ry = oh2 - 1 + kh;
          const int slot = ((ry % 3) + 3) % 3;
          for (int kw = 0; kw < 3; ++kw) {
            const int li = (ow - ow0) + kw; // halo shift
            const float* in_pix =
              ring + ((((int64_t)mcb * 3 + slot) * curW2 + li) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m2.stride_ocb +
                (int64_t)mcb * m2.stride_icb +
                (int64_t)kh  * m2.stride_kh  +
                (int64_t)kw  * m2.stride_kw;

            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const float x = in_pix[ic_i];
              const __m512 wv = _mm512_load_ps(w2p + Wbase + (int64_t)ic_i * m2.stride_ic_i);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu2) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }
      float* out_pix =
        out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + ow) * vec);
      if (tail_block) _mm512_mask_storeu_ps(out_pix, store_mask, acc);
      else            _mm512_store_ps(out_pix, acc);
    }
  }
}



// ---- Tile-local helpers (ring is midCb x 3 x curW x vec) ----

static inline void produce_conv1_row_tile_branchy(
    const float* MK_RESTRICT in, int inC, int H, int W,
    float* MK_RESTRICT ring, int curW, int midC,
    const ConvPackInfo& m1, const float* MK_RESTRICT w1p,
    const float* MK_RESTRICT bias1, bool relu1,
    int oh1, int ow0, int ow_lo_local, int ow_hi_local, // local inclusive range [0..curW-1]
    int vec)
{
  const int inCb  = (int)((inC + vec - 1) / vec);
  const int midCb = (int)((midC + vec - 1) / vec);
  const int slot = ((oh1 % 3) + 3) % 3;

  for (int mcb = 0; mcb < midCb; ++mcb) {
    __m512 bvec = _mm512_setzero_ps();
    if (bias1) bvec = _mm512_load_ps(bias1 + mcb * vec);

    for (int li = ow_lo_local; li <= ow_hi_local; ++li) {
      const int ow = ow0 + li;
      __m512 acc = _mm512_setzero_ps();

      for (int icb = 0; icb < inCb; ++icb) {
        MK_UNROLL_3
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = oh1 - 1 + kh;
          if ((unsigned)ih >= (unsigned)H) continue;
          MK_UNROLL_3
          for (int kw = 0; kw < 3; ++kw) {
            const int iw = ow - 1 + kw;
            if ((unsigned)iw >= (unsigned)W) continue;

            const float* in_pix =
              in + ((((int64_t)icb * H) * W + (int64_t)ih * W + iw) * vec);

            const int64_t Wbase =
                (int64_t)mcb * m1.stride_ocb +
                (int64_t)icb * m1.stride_icb +
                (int64_t)kh  * m1.stride_kh  +
                (int64_t)kw  * m1.stride_kw;

            MK_UNROLL_16
            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const float x = in_pix[ic_i];
              if (x == 0.0f) continue;
              const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu1) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }

      float* dst = ring + ((((int64_t)mcb * 3 + slot) * curW + li) * vec);
      _mm512_store_ps(dst, acc);
    }
  }
}

// Interior, width-tiled compute for the tile range [ow0 .. ow0+curW-1],
// computing only interior columns ow ∈ [1..W-2]. Edges are handled separately.
static inline void produce_conv1_row_tile8_interior(
    const float* MK_RESTRICT in, int inC, int H, int W,
    float* MK_RESTRICT ring, int curW, int midC,
    const ConvPackInfo& m1, const float* MK_RESTRICT w1p,
    const float* MK_RESTRICT bias1, bool relu1,
    int oh1, int ow0, int vec)
{
  constexpr int OW_TILE = MK_OW_TILE;
  const int inCb  = (int)((inC + vec - 1) / vec);
  const int midCb = (int)((midC + vec - 1) / vec);
  const int slot = ((oh1 % 3) + 3) % 3;

  // local interior range within this tile
  const int ow_start = std::max(1, ow0);
  const int ow_end_inclusive = std::min(W - 2, ow0 + curW - 1);
  if (ow_start > ow_end_inclusive) return;

  const int main_start = ow_start;
  const int main_end = main_start + ((ow_end_inclusive - main_start + 1) / OW_TILE) * OW_TILE;

  for (int mcb = 0; mcb < midCb; ++mcb) {
    __m512 bvec = _mm512_setzero_ps();
    if (bias1) bvec = _mm512_load_ps(bias1 + mcb * vec);

    // process blocks of OW_TILE
    for (int ow = main_start; ow < main_end; ow += OW_TILE) {
      __m512 acc[OW_TILE];
      for (int t=0;t<OW_TILE;++t) acc[t] = _mm512_setzero_ps();

      for (int icb = 0; icb < inCb; ++icb) {
        MK_UNROLL_3
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = oh1 - 1 + kh; // interior row -> 0..H-1
          MK_UNROLL_3
          for (int kw = 0; kw < 3; ++kw) {
            const int base_col = ow - 1 + kw;
            const float* in_base =
              in + ((((int64_t)icb * H) * W + (int64_t)ih * W + base_col) * vec);

            const int64_t Wbase =
                (int64_t)mcb * m1.stride_ocb +
                (int64_t)icb * m1.stride_icb +
                (int64_t)kh  * m1.stride_kh  +
                (int64_t)kw  * m1.stride_kw;

            MK_UNROLL_16
            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);

              // 8 adjacent scalars across columns
              MK_UNROLL_16
              for (int t=0;t<OW_TILE;++t) {
                const float x = in_base[ic_i + t*vec];
                acc[t] = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc[t]);
              }
            }
          }
        }
      }

      // epilogue & store into ring (local index li = ow - ow0 + t)
      MK_UNROLL_16
      for (int t=0;t<OW_TILE;++t) {
        __m512 v = _mm512_add_ps(acc[t], bvec);
        if (relu1) { const __m512 z = _mm512_setzero_ps(); v = _mm512_max_ps(v, z); }
        const int li = (ow - (ow0 - 1)) + t;
        float* dst = ring + ((((int64_t)mcb * 3 + slot) * curW + li) * vec);
        _mm512_store_ps(dst, v);
      }
    }

    // leftover columns inside tile (scalar-per-pixel loop)
    for (int ow = main_end; ow <= ow_end_inclusive; ++ow) {
      __m512 acc = _mm512_setzero_ps();

      for (int icb = 0; icb < inCb; ++icb) {
        MK_UNROLL_3
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = oh1 - 1 + kh;
          MK_UNROLL_3
          for (int kw = 0; kw < 3; ++kw) {
            const int base_col = ow - 1 + kw;
            const float* in_base =
              in + ((((int64_t)icb * H) * W + (int64_t)ih * W + base_col) * vec);
            const int64_t Wbase =
                (int64_t)mcb * m1.stride_ocb +
                (int64_t)icb * m1.stride_icb +
                (int64_t)kh  * m1.stride_kh  +
                (int64_t)kw  * m1.stride_kw;

            MK_UNROLL_16
            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
              const float x = in_base[ic_i];
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu1) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }
      const int li = ow - ow0;
      float* dst = ring + ((((int64_t)mcb * 3 + slot) * curW + li) * vec);
      _mm512_store_ps(dst, acc);
    }
  }
}

static inline void consume_conv2_row_tile_branchy(
    const float* MK_RESTRICT ring, int curW, int midC, int H1s, int W1s,
    float* MK_RESTRICT out, int outC, int H2s, int W2s, int oh2, int ow0,
    int ow_lo_local, int ow_hi_local,
    const ConvPackInfo& m2, const float* MK_RESTRICT w2p,
    const float* MK_RESTRICT bias2, bool relu2, int vec)
{
  const int midCb = (int)((midC + vec - 1) / vec);
  const int outCb = (int)((outC + vec - 1) / vec);

  const int tail = outC - (outCb - 1) * vec;
  const __mmask16 tail_mask = (outCb > 0 && tail > 0 && tail < vec)
                                ? (__mmask16)((1u << tail) - 1u)
                                : (__mmask16)0xFFFF;

  for (int ocb = 0; ocb < outCb; ++ocb) {
    const bool tail_block = (ocb == outCb - 1) && (tail < vec);
    const __mmask16 store_mask = tail_block ? tail_mask : (__mmask16)0xFFFF;
    __m512 bvec = _mm512_setzero_ps();
    if (bias2) bvec = _mm512_load_ps(bias2 + ocb * vec);

    for (int li = ow_lo_local; li <= ow_hi_local; ++li) {
      const int ow = ow0 + li;
      __m512 acc = _mm512_setzero_ps();

      for (int mcb = 0; mcb < midCb; ++mcb) {
        MK_UNROLL_3
        for (int kh = 0; kh < 3; ++kh) {
          const int ry = oh2 - 1 + kh;
          if ((unsigned)ry >= (unsigned)H1s) continue;
          const int slot = ((ry % 3) + 3) % 3;

          MK_UNROLL_3
          for (int kw = 0; kw < 3; ++kw) {
            const int rx = ow - 1 + kw;
            if ((unsigned)rx >= (unsigned)W1s) continue;

            const float* in_pix =
              ring + ((((int64_t)mcb * 3 + slot) * curW + (rx - ow0)) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m2.stride_ocb +
                (int64_t)mcb * m2.stride_icb +
                (int64_t)kh  * m2.stride_kh  +
                (int64_t)kw  * m2.stride_kw;

            MK_UNROLL_16
            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const float x = in_pix[ic_i];
              if (x == 0.0f) continue;
              const __m512 wv = _mm512_load_ps(w2p + Wbase + (int64_t)ic_i * m2.stride_ic_i);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu2) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }

      float* out_pix =
        out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + ow) * vec);
      if (tail_block) _mm512_mask_storeu_ps(out_pix, store_mask, acc);
      else            _mm512_store_ps(out_pix, acc);
    }
  }
}

static inline void consume_conv2_row_tile8_interior(
    const float* MK_RESTRICT ring, int curW, int midC,
    float* MK_RESTRICT out, int outC, int H2s, int W2s, int oh2, int ow0,
    int ow_start, int ow_end_inclusive, // interior range in global coords
    const ConvPackInfo& m2, const float* MK_RESTRICT w2p,
    const float* MK_RESTRICT bias2, bool relu2, int vec)
{
  constexpr int OW_TILE = MK_OW_TILE;
  const int midCb = (int)((midC + vec - 1) / vec);
  const int outCb = (int)((outC + vec - 1) / vec);

  const int tail = outC - (outCb - 1) * vec;
  const __mmask16 tail_mask = (outCb > 0 && tail > 0 && tail < vec)
                                ? (__mmask16)((1u << tail) - 1u)
                                : (__mmask16)0xFFFF;

  const int main_start = ow_start;
  const int main_end = main_start + ((ow_end_inclusive - main_start + 1) / OW_TILE) * OW_TILE;

  for (int ocb = 0; ocb < outCb; ++ocb) {
    const bool tail_block = (ocb == outCb - 1) && (tail < vec);
    const __mmask16 store_mask = tail_block ? tail_mask : (__mmask16)0xFFFF;
    __m512 bvec = _mm512_setzero_ps();
    if (bias2) bvec = _mm512_load_ps(bias2 + ocb * vec);

    // main blocks
    for (int ow = main_start; ow < main_end; ow += OW_TILE) {
      __m512 acc[OW_TILE];
      for (int t=0;t<OW_TILE;++t) acc[t] = _mm512_setzero_ps();

      for (int mcb = 0; mcb < (int)((midC + vec - 1)/vec); ++mcb) {
        // taps over the three rows above/at/below (interior -> valid)
        MK_UNROLL_3
        for (int kh = 0; kh < 3; ++kh) {
          const int ry = oh2 - 1 + kh;
          const int slot = ((ry % 3) + 3) % 3;

          MK_UNROLL_3
          for (int kw = 0; kw < 3; ++kw) {
            const int base_col_local = (ow - ow0) + kw;
            const float* in_base =
                ring + ((((int64_t)mcb * 3 + slot) *  base_col_local) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m2.stride_ocb +
                (int64_t)mcb * m2.stride_icb +
                (int64_t)kh  * m2.stride_kh  +
                (int64_t)kw  * m2.stride_kw;

            MK_UNROLL_16
            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w2p + Wbase + (int64_t)ic_i * m2.stride_ic_i);
              MK_UNROLL_16
              for (int t=0;t<OW_TILE;++t) {
                const float x = in_base[ic_i + t*vec];
                acc[t] = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc[t]);
              }
            }
          }
        }
      }

      // epilogue & store (global positions ow..ow+OW_TILE-1)
      MK_UNROLL_16
      for (int t=0;t<OW_TILE;++t) {
        __m512 v = _mm512_add_ps(acc[t], bvec);
        if (relu2) { const __m512 z = _mm512_setzero_ps(); v = _mm512_max_ps(v, z); }
        float* out_pix =
          out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + (ow + t)) * vec);
        if (tail_block) _mm512_mask_storeu_ps(out_pix, store_mask, v);
        else            _mm512_store_ps(out_pix, v);
      }
    }

    // leftover columns in interior range
    for (int ow = main_end; ow <= ow_end_inclusive; ++ow) {
      __m512 acc = _mm512_setzero_ps();

      for (int mcb = 0; mcb < (int)((midC + vec - 1)/vec); ++mcb) {
        MK_UNROLL_3
        for (int kh = 0; kh < 3; ++kh) {
          const int ry = oh2 - 1 + kh;
          const int slot = ((ry % 3) + 3) % 3;
          MK_UNROLL_3
          for (int kw = 0; kw < 3; ++kw) {
            const int li = (ow - ow0) - 1 + kw;
            const float* in_pix =
              ring + ((((int64_t)mcb * 3 + slot) * curW + li) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m2.stride_ocb +
                (int64_t)mcb * m2.stride_icb +
                (int64_t)kh  * m2.stride_kh  +
                (int64_t)kw  * m2.stride_kw;

            MK_UNROLL_16
            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const float x = in_pix[ic_i];
              const __m512 wv = _mm512_load_ps(w2p + Wbase + (int64_t)ic_i * m2.stride_ic_i);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu2) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }
      float* out_pix =
        out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + ow) * vec);
      if (tail_block) _mm512_mask_storeu_ps(out_pix, store_mask, acc);
      else            _mm512_store_ps(out_pix, acc);
    }
  }
}

// Produce one row (oh1) of Conv1 into the ring buffer for all mid channel blocks.
// Layout: NCHWc (vec=16). Packed weights: OIhw16i16o.
// Ring layout: [icb_mid][row_slot (0..2)][W][vec] — where icb_mid is Conv1 oc block.
// Produce one row (oh1) of Conv1 into the ring buffer
static void produce_conv1_row_avx512(
    const float* MK_RESTRICT in, int inC, int H, int W,              // input shape
    float* MK_RESTRICT ring, int midC,                               // ring target channels
    const ConvPackInfo& m1, const float* MK_RESTRICT w1p,            // packed meta + weights
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
              const __m512 wv = _mm512_load_ps(w1p + Wrow);
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
      _mm512_store_ps(dst, acc);
    }
  }
}

// Consume from ring to produce one row (oh2) of Conv2
static void consume_conv2_row_avx512(
    const float* MK_RESTRICT ring, int midC, int H1s, int W1s,
    float* MK_RESTRICT out, int outC, int H2s, int W2s, int oh2,
    const ConvPackInfo& m2, const float* MK_RESTRICT w2p,
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
              const __m512 wv = _mm512_load_ps(w2p + Wrow);
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
        else            _mm512_store_ps(out_pix, acc);
    }
  }
}

static inline void consume_conv2_row_edges_avx512(
    const float* MK_RESTRICT ring, int midC, int H1s, int W1s,
    float* MK_RESTRICT out, int outC, int H2s, int W2s, int oh2,
    const ConvPackInfo& m2, const float* MK_RESTRICT w2p,
    const float* bias2_padded, bool relu2, int vec)
{
  const int midCb = (int)((midC + vec - 1) / vec);
  const int outCb = (int)((outC + vec - 1) / vec);
  const int tail = outC - (outCb - 1) * vec;
  const __mmask16 tail_mask = (outCb > 0 && tail > 0 && tail < vec)
                                ? (__mmask16)((1u << tail) - 1u)
                                : (__mmask16)0xFFFF;

  // handle up to two edges (if W2s==1, they collapse to the same column)
  const int cols[2] = {0, std::max(0, W2s - 1)};
  const int ncols = (W2s > 1) ? 2 : 1;

  for (int ocb = 0; ocb < outCb; ++ocb) {
    const bool tail_block = (ocb == outCb - 1) && (tail < vec);
    const __mmask16 store_mask = tail_block ? tail_mask : (__mmask16)0xFFFF;
    __m512 bvec = _mm512_setzero_ps();
    if (bias2_padded) bvec = _mm512_load_ps(bias2_padded + ocb * vec);

    for (int c = 0; c < ncols; ++c) {
      const int ow = cols[c];
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
              const __m512 wv = _mm512_load_ps(w2p + Wbase + (int64_t)ic_i * m2.stride_ic_i);
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }

      acc = _mm512_add_ps(acc, bvec);
      if (relu2) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }

      float* out_pix = out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + ow) * vec);
      if (tail_block) _mm512_mask_storeu_ps(out_pix, store_mask, acc);
      else            _mm512_store_ps(out_pix, acc);
    }
  }
}



// ---- INTERIOR (no bounds checks), width-tiled (OW_TILE=8) ----
static inline void produce_conv1_row_avx512_tile8_interior(
    const float* MK_RESTRICT in, int inC, int H, int W,        // input shape
    float* MK_RESTRICT ring, int midC,                          // ring target channels
    const ConvPackInfo& m1, const float* MK_RESTRICT w1p,
    const float* bias1_padded, bool relu1,
    int oh1, int vec)
{
  constexpr int OW_TILE = 8;
  const int inCb  = (int)((inC + vec - 1) / vec);
  const int midCb = (int)((midC + vec - 1) / vec);

  // interior columns are [1 .. W-2]
  const int ow_start = 1;
  const int ow_end   = W - 1;          // exclusive upper bound for scalar fallback
  const int ow_main_end = ow_start + ((W - 2 - ow_start + 1) / OW_TILE) * OW_TILE;

  const int slot = ((oh1 % 3) + 3) % 3;

  for (int ocb = 0; ocb < midCb; ++ocb) {
    __m512 bvec = _mm512_setzero_ps();
    if (bias1_padded) bvec = _mm512_load_ps(bias1_padded + ocb * vec);

    {
        __m512 acc = _mm512_setzero_ps();
        for (int icb = 0; icb < inCb; ++icb) {
            for (int kh = 0; kh < 3; ++kh) {
            const int ih = oh1 - 1 + kh;
            if ((unsigned)ih >= (unsigned)H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    const int iw = 0 - 1 + kw; // {-1,0,1}
                    if ((unsigned)iw >= (unsigned)W) continue;
                    const float* in_pix =
                    in + ((((int64_t)icb * H) * W + (int64_t)ih * W + iw) * vec);
                    const int64_t Wbase =
                        (int64_t)ocb * m1.stride_ocb +
                        (int64_t)icb * m1.stride_icb +
                        (int64_t)kh  * m1.stride_kh  +
                        (int64_t)kw  * m1.stride_kw;
                    for (int ic_i = 0; ic_i < vec; ++ic_i) {
                        const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(in_pix[ic_i]), wv, acc);
                    }
                }
            }
        }
        acc = _mm512_add_ps(acc, bvec);
        if (relu1) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }
        float* dst = ring + ((((int64_t)ocb * 3 + slot) * W + 0) * vec);
        _mm512_store_ps(dst, acc);
    }

    // right edge (ow = W-1)
    {
        __m512 acc = _mm512_setzero_ps();
        for (int icb = 0; icb < inCb; ++icb) {
            for (int kh = 0; kh < 3; ++kh) {
            const int ih = oh1 - 1 + kh;
            if ((unsigned)ih >= (unsigned)H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    const int iw = (W - 1) - 1 + kw; // {W-2, W-1, W}
                    if ((unsigned)iw >= (unsigned)W) continue;
                    const float* in_pix =
                    in + ((((int64_t)icb * H) * W + (int64_t)ih * W + iw) * vec);
                    const int64_t Wbase =
                        (int64_t)ocb * m1.stride_ocb +
                        (int64_t)icb * m1.stride_icb +
                        (int64_t)kh  * m1.stride_kh  +
                        (int64_t)kw  * m1.stride_kw;
                    for (int ic_i = 0; ic_i < vec; ++ic_i) {
                        const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(in_pix[ic_i]), wv, acc);
                    }
                }
            }
        }
        acc = _mm512_add_ps(acc, bvec);
        if (relu1) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }
        float* dst = ring + ((((int64_t)ocb * 3 + slot) * W + (W - 1)) * vec);
        _mm512_store_ps(dst, acc);
    }

    // Main tiles of 8 columns
    for (int ow = ow_start; ow < ow_main_end; ow += OW_TILE) {
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();
      __m512 acc4 = _mm512_setzero_ps();
      __m512 acc5 = _mm512_setzero_ps();
      __m512 acc6 = _mm512_setzero_ps();
      __m512 acc7 = _mm512_setzero_ps();

      for (int icb = 0; icb < inCb; ++icb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = oh1 - 1 + kh;  // interior -> guaranteed in [0,H-1]
          // base col for this tap (ow-1 .. ow+6) plus kw (0..2)
          for (int kw = 0; kw < 3; ++kw) {
            const int base_col = ow - 1 + kw; // first of 8 inputs for this tap
            const float* in_base =
              in + ((((int64_t)icb * H) * W + (int64_t)ih * W + base_col) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m1.stride_ocb +
                (int64_t)icb * m1.stride_icb +
                (int64_t)kh  * m1.stride_kh  +
                (int64_t)kw  * m1.stride_kw;

            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);

              // 8 adjacent inputs, scalar each, stride vec between columns
              const float x0 = in_base[ic_i + 0*vec];
              const float x1 = in_base[ic_i + 1*vec];
              const float x2 = in_base[ic_i + 2*vec];
              const float x3 = in_base[ic_i + 3*vec];
              const float x4 = in_base[ic_i + 4*vec];
              const float x5 = in_base[ic_i + 5*vec];
              const float x6 = in_base[ic_i + 6*vec];
              const float x7 = in_base[ic_i + 7*vec];

              acc0 = _mm512_fmadd_ps(_mm512_set1_ps(x0), wv, acc0);
              acc1 = _mm512_fmadd_ps(_mm512_set1_ps(x1), wv, acc1);
              acc2 = _mm512_fmadd_ps(_mm512_set1_ps(x2), wv, acc2);
              acc3 = _mm512_fmadd_ps(_mm512_set1_ps(x3), wv, acc3);
              acc4 = _mm512_fmadd_ps(_mm512_set1_ps(x4), wv, acc4);
              acc5 = _mm512_fmadd_ps(_mm512_set1_ps(x5), wv, acc5);
              acc6 = _mm512_fmadd_ps(_mm512_set1_ps(x6), wv, acc6);
              acc7 = _mm512_fmadd_ps(_mm512_set1_ps(x7), wv, acc7);
            }
          }
        }
      }

      // Epilogue (bias + relu) and store 8 columns
      acc0 = _mm512_add_ps(acc0, bvec);
      acc1 = _mm512_add_ps(acc1, bvec);
      acc2 = _mm512_add_ps(acc2, bvec);
      acc3 = _mm512_add_ps(acc3, bvec);
      acc4 = _mm512_add_ps(acc4, bvec);
      acc5 = _mm512_add_ps(acc5, bvec);
      acc6 = _mm512_add_ps(acc6, bvec);
      acc7 = _mm512_add_ps(acc7, bvec);
      if (relu1) {
        const __m512 z = _mm512_setzero_ps();
        acc0 = _mm512_max_ps(acc0, z); acc1 = _mm512_max_ps(acc1, z);
        acc2 = _mm512_max_ps(acc2, z); acc3 = _mm512_max_ps(acc3, z);
        acc4 = _mm512_max_ps(acc4, z); acc5 = _mm512_max_ps(acc5, z);
        acc6 = _mm512_max_ps(acc6, z); acc7 = _mm512_max_ps(acc7, z);
      }
      float* dst0 = ring + ((((int64_t)ocb * 3 + slot) * W + (int64_t)ow + 0) * vec);
      float* dst1 = dst0 + vec;
      float* dst2 = dst1 + vec;
      float* dst3 = dst2 + vec;
      float* dst4 = dst3 + vec;
      float* dst5 = dst4 + vec;
      float* dst6 = dst5 + vec;
      float* dst7 = dst6 + vec;
      _mm512_store_ps(dst0, acc0); _mm512_store_ps(dst1, acc1);
      _mm512_store_ps(dst2, acc2); _mm512_store_ps(dst3, acc3);
      _mm512_store_ps(dst4, acc4); _mm512_store_ps(dst5, acc5);
      _mm512_store_ps(dst6, acc6); _mm512_store_ps(dst7, acc7);
    }

    // Leftover interior columns (if any): fall back to scalar-per-pixel store
    for (int ow = ow_main_end; ow < ow_end; ++ow) {
      __m512 acc = _mm512_setzero_ps();
      for (int icb = 0; icb < inCb; ++icb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ih = oh1 - 1 + kh;
          for (int kw = 0; kw < 3; ++kw) {
            const int base_col = ow - 1 + kw;
            const float* in_base =
              in + ((((int64_t)icb * H) * W + (int64_t)ih * W + base_col) * vec);
            const int64_t Wbase =
                (int64_t)ocb * m1.stride_ocb +
                (int64_t)icb * m1.stride_icb +
                (int64_t)kh  * m1.stride_kh  +
                (int64_t)kw  * m1.stride_kw;
            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w1p + Wbase + (int64_t)ic_i * m1.stride_ic_i);
              const float x = in_base[ic_i];
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }
      acc = _mm512_add_ps(acc, bvec);
      if (relu1) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }
      float* dst = ring + ((((int64_t)ocb * 3 + slot) * W + ow) * vec);
      _mm512_store_ps(dst, acc);
    }
  }
}

static inline void consume_conv2_row_avx512_tile8_interior(
    const float* MK_RESTRICT ring, int midC, int H1s, int W1s,
    float* MK_RESTRICT out, int outC, int H2s, int W2s, int oh2,
    const ConvPackInfo& m2, const float* MK_RESTRICT w2p,
    const float* bias2_padded, bool relu2, int vec)
{
  constexpr int OW_TILE = 8;
  const int midCb = (int)((midC + vec - 1) / vec);
  const int outCb = (int)((outC + vec - 1) / vec);

  const int tail = outC - (outCb - 1) * vec;
  const __mmask16 tail_mask = (outCb > 0 && tail > 0 && tail < vec)
                                ? (__mmask16)((1u << tail) - 1u)
                                : (__mmask16)0xFFFF;

  const int ow_start = 1;
  const int ow_end   = W2s - 1;
  const int ow_main_end = ow_start + ((W2s - 2 - ow_start + 1) / OW_TILE) * OW_TILE;

  for (int ocb = 0; ocb < outCb; ++ocb) {
    const bool tail_block = (ocb == outCb - 1) && (tail < vec);
    const __mmask16 store_mask = tail_block ? tail_mask : (__mmask16)0xFFFF;

    __m512 bvec = _mm512_setzero_ps();
    if (bias2_padded) bvec = _mm512_load_ps(bias2_padded + ocb * vec);

    for (int ow = ow_start; ow < ow_main_end; ow += OW_TILE) {
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();
      __m512 acc4 = _mm512_setzero_ps();
      __m512 acc5 = _mm512_setzero_ps();
      __m512 acc6 = _mm512_setzero_ps();
      __m512 acc7 = _mm512_setzero_ps();

      for (int icb = 0; icb < midCb; ++icb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ry = oh2 - 1 + kh;       // interior -> 0..H1s-1
          const int slot = ((ry % 3) + 3) % 3;
          for (int kw = 0; kw < 3; ++kw) {
            const int base_col = ow - 1 + kw;
            const float* in_base =
              ring + ((((int64_t)icb * 3 + slot) * W1s + base_col) * vec);

            const int64_t Wbase =
                (int64_t)ocb * m2.stride_ocb +
                (int64_t)icb * m2.stride_icb +
                (int64_t)kh  * m2.stride_kh  +
                (int64_t)kw  * m2.stride_kw;

            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w2p + Wbase + (int64_t)ic_i * m2.stride_ic_i);
              const float x0 = in_base[ic_i + 0*vec];
              const float x1 = in_base[ic_i + 1*vec];
              const float x2 = in_base[ic_i + 2*vec];
              const float x3 = in_base[ic_i + 3*vec];
              const float x4 = in_base[ic_i + 4*vec];
              const float x5 = in_base[ic_i + 5*vec];
              const float x6 = in_base[ic_i + 6*vec];
              const float x7 = in_base[ic_i + 7*vec];
              acc0 = _mm512_fmadd_ps(_mm512_set1_ps(x0), wv, acc0);
              acc1 = _mm512_fmadd_ps(_mm512_set1_ps(x1), wv, acc1);
              acc2 = _mm512_fmadd_ps(_mm512_set1_ps(x2), wv, acc2);
              acc3 = _mm512_fmadd_ps(_mm512_set1_ps(x3), wv, acc3);
              acc4 = _mm512_fmadd_ps(_mm512_set1_ps(x4), wv, acc4);
              acc5 = _mm512_fmadd_ps(_mm512_set1_ps(x5), wv, acc5);
              acc6 = _mm512_fmadd_ps(_mm512_set1_ps(x6), wv, acc6);
              acc7 = _mm512_fmadd_ps(_mm512_set1_ps(x7), wv, acc7);
            }
          }
        }
      }

      // Epilogue + store 8 columns
      acc0 = _mm512_add_ps(acc0, bvec);
      acc1 = _mm512_add_ps(acc1, bvec);
      acc2 = _mm512_add_ps(acc2, bvec);
      acc3 = _mm512_add_ps(acc3, bvec);
      acc4 = _mm512_add_ps(acc4, bvec);
      acc5 = _mm512_add_ps(acc5, bvec);
      acc6 = _mm512_add_ps(acc6, bvec);
      acc7 = _mm512_add_ps(acc7, bvec);
      if (relu2) {
        const __m512 z = _mm512_setzero_ps();
        acc0 = _mm512_max_ps(acc0, z); acc1 = _mm512_max_ps(acc1, z);
        acc2 = _mm512_max_ps(acc2, z); acc3 = _mm512_max_ps(acc3, z);
        acc4 = _mm512_max_ps(acc4, z); acc5 = _mm512_max_ps(acc5, z);
        acc6 = _mm512_max_ps(acc6, z); acc7 = _mm512_max_ps(acc7, z);
      }

      float* out0 = out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + ow + 0) * vec);
      float* out1 = out0 + vec; float* out2 = out1 + vec; float* out3 = out2 + vec;
      float* out4 = out3 + vec; float* out5 = out4 + vec; float* out6 = out5 + vec; float* out7 = out6 + vec;

      if (tail_block) {
        _mm512_mask_storeu_ps(out0, store_mask, acc0);
        _mm512_mask_storeu_ps(out1, store_mask, acc1);
        _mm512_mask_storeu_ps(out2, store_mask, acc2);
        _mm512_mask_storeu_ps(out3, store_mask, acc3);
        _mm512_mask_storeu_ps(out4, store_mask, acc4);
        _mm512_mask_storeu_ps(out5, store_mask, acc5);
        _mm512_mask_storeu_ps(out6, store_mask, acc6);
        _mm512_mask_storeu_ps(out7, store_mask, acc7);
      } else {
        _mm512_store_ps(out0, acc0); _mm512_store_ps(out1, acc1);
        _mm512_store_ps(out2, acc2); _mm512_store_ps(out3, acc3);
        _mm512_store_ps(out4, acc4); _mm512_store_ps(out5, acc5);
        _mm512_store_ps(out6, acc6); _mm512_store_ps(out7, acc7);
      }
    }

    // leftover interior columns
    for (int ow = ow_main_end; ow < ow_end; ++ow) {
      __m512 acc = _mm512_setzero_ps();
      for (int icb = 0; icb < midCb; ++icb) {
        for (int kh = 0; kh < 3; ++kh) {
          const int ry = oh2 - 1 + kh;
          const int slot = ((ry % 3) + 3) % 3;
          for (int kw = 0; kw < 3; ++kw) {
            const int base_col = ow - 1 + kw;
            const float* in_base =
              ring + ((((int64_t)icb * 3 + slot) * W1s + base_col) * vec);
            const int64_t Wbase =
                (int64_t)ocb * m2.stride_ocb +
                (int64_t)icb * m2.stride_icb +
                (int64_t)kh  * m2.stride_kh  +
                (int64_t)kw  * m2.stride_kw;
            for (int ic_i = 0; ic_i < vec; ++ic_i) {
              const __m512 wv = _mm512_load_ps(w2p + Wbase + (int64_t)ic_i * m2.stride_ic_i);
              const float x = in_base[ic_i];
              acc = _mm512_fmadd_ps(_mm512_set1_ps(x), wv, acc);
            }
          }
        }
      }
      acc = _mm512_add_ps(acc, bvec);
      if (relu2) { const __m512 z = _mm512_setzero_ps(); acc = _mm512_max_ps(acc, z); }
      float* out_pix = out + ((((int64_t)ocb * H2s) * W2s + (int64_t)oh2 * W2s + ow) * vec);
      if (tail_block) _mm512_mask_storeu_ps(out_pix, store_mask, acc);
      else            _mm512_store_ps(out_pix, acc);
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

  const int tileW = MK_OW_TILE;
  const int midCb = (int)((midC + vec - 1) / vec);

  // TILE ring with halo: (curW + 2)
  const size_t ring_capacity = (size_t)midCb * 3 * (size_t)(tileW + 2) * (size_t)vec;
  float* ring = (float*)mk::aligned_alloc64(ring_capacity * sizeof(float));
  auto ring_free = std::unique_ptr<float, void(*)(float*)>(ring, [](float* p){ mk::aligned_free64(p); });

  for (int oh = y0; oh < y1; ++oh) {
    const int need_lo = std::max(0, oh - 1);
    const int need_hi = std::min(H1s - 1, oh + 1);
    const int slot_lo = ((need_lo % 3) + 3) % 3; // for zeroing halos quickly if needed

    for (int ow0 = 0; ow0 < W1s; ow0 += tileW) {
      const int curW = std::min(tileW, W1s - ow0);
      const int curW2 = curW + 2;
      const int ow0_shifted = ow0 - 1;  // so li=1 maps to global ow0

      // Produce rows needed for THIS TILE (with halo columns filled)
      for (int ry = need_lo; ry <= need_hi; ++ry) {
        const int slot = ((ry % 3) + 3) % 3;

        // Left halo (li=0) is global ow0-1
        if (ow0 > 0) {
          produce_conv1_one_col_branchy_to_ring(
              in_nchwc, inC, H1s, W1s, ring, curW2, midC,
              m1, w1p, b1p, relu1, ry, ow0 - 1, /*li=*/0, vec);
        } else {
          zero_ring_column(ring, midCb, curW2, vec, slot, /*li=*/0);
        }

        // Right halo (li=curW+1) is global ow0+curW
        if (ow0 + curW < W1s) {
          produce_conv1_one_col_branchy_to_ring(
              in_nchwc, inC, H1s, W1s, ring, curW2, midC,
              m1, w1p, b1p, relu1, ry, ow0 + curW, /*li=*/curW + 1, vec);
        } else {
          zero_ring_column(ring, midCb, curW2, vec, slot, /*li=*/curW + 1);
        }

        const bool edge_row = (ry == 0) || (ry == H1s - 1);
        if (edge_row) {
          // Produce BODY columns (li=1..curW) using branchy path
          produce_conv1_row_tile_branchy_body(
              in_nchwc, inC, H1s, W1s, ring, curW2, midC,
              m1, w1p, b1p, relu1, ry, ow0_shifted,
              /*li_lo=*/1, /*li_hi=*/curW, vec);
        } else {
          // Fast interior BODY columns
          produce_conv1_row_tile8_interior_halo(
              in_nchwc, inC, H1s, W1s, ring, curW2, midC,
              m1, w1p, b1p, relu1, ry, ow0_shifted, ow0, curW, vec);
        }
      } // produce rows

      // Consume THIS TILE for row 'oh'
      const bool edge_consume = (oh == 0) || (oh == H2s - 1);
      if (edge_consume) {
        // Consume whole BODY (li=1..curW) with branchy halo consumer
        consume_conv2_row_tile_branchy_halo(
            ring, curW2, midC, H1s, W1s,
            out_nchwc, outC, H2s, W2s, oh, ow0,
            /*li_lo=*/1, /*li_hi=*/curW,
            m2, w2p, b2p, relu2, vec);
      } else {
        // Interior range in this tile (global coords), fully covered by halo
        const int ow_int_start = std::max(1, ow0);
        const int ow_int_end   = std::min(W2s - 2, ow0 + curW - 1);
        if (ow_int_start <= ow_int_end) {
          consume_conv2_row_tile8_interior_halo(
              ring, curW2, midC,
              out_nchwc, outC, H2s, W2s, oh, ow0,
              ow_int_start, ow_int_end,
              m2, w2p, b2p, relu2, vec);
        }
        // If the tile touches actual image edges (ow0==0 or ow0+curW==W), the two
        // edge columns 0 and W-1 belong to those tiles; branchy path above handles them
        // when edge_consume==true. For interior rows, no edge columns are in range.
      }

    } // tiles
  }   // rows
}     // omp parallel
}


} // namespace mk
