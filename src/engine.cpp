#include "engine.hpp"
#include "onnx_loader.hpp"
#include "regionizer.hpp"
#include "weight_packing.hpp"
#include "epilogue.hpp"
#include "conv_microkernels.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mk {

// ---------- helpers ----------
static inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

static inline int64_t conv_out_dim(int64_t in, int64_t pad, int64_t dilation,
                                   int64_t k, int64_t stride) {
  const int64_t eff_k = dilation * (k - 1) + 1;
  return ((in + 2*pad - eff_k) / stride) + 1;
}

struct MaxShape { int C, H, W; };


// NCHW -> NCHWc (vec=16), N=1
static void nchw_to_nchwc(const float* in, int C, int H, int W,
                          int vec, float* out) {
  const int Cb = (int)ceil_div(C, vec);
  const int64_t HW = (int64_t)H * W;
  std::memset(out, 0, (size_t)Cb * H * W * vec * sizeof(float));
  for (int c = 0; c < C; ++c) {
    const int cb = c / vec, ci = c % vec;
    const float* pin = in + (int64_t)c * HW;
    float* pout = out + (((int64_t)cb * H) * W * vec);
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        pout[( (int64_t)h * W + w) * vec + ci] = pin[(int64_t)h * W + w];
      }
    }
  }
}

// NCHWc -> NCHW (vec=16), N=1
static void nchwc_to_nchw(const float* in, int C, int H, int W,
                          int vec, float* out) {
    const int Cb = (int)ceil_div(C, vec);
    const int64_t HW = (int64_t)H * W;
    for (int c = 0; c < C; ++c) {
        const int cb = c / vec, ci = c % vec;
        const float* pin = in + (((int64_t)cb * H) * W * vec);
        float* pout = out + (int64_t)c * HW;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                pout[(int64_t)h * W + w] = pin[( (int64_t)h * W + w) * vec + ci];
            }
        }
    }
}

// Add bias (packed to vec, tail zeroed) to NCHWc tensor
static void add_bias_nchwc(float* out_nchwc, int outC, int H, int W,
                           int vec, const float* bias_padded) {
    const int Cb = (int)ceil_div(outC, vec);
    for (int cb = 0; cb < Cb; ++cb) {
        const float* b = bias_padded + cb * vec;
        float* base = out_nchwc + ((int64_t)cb * H) * W * vec;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float* px = base + ((int64_t)h * W + w) * vec;
                for (int ci = 0; ci < vec; ++ci) px[ci] += b[ci];
            }
        }
    }
}

// In-place activation over NCHWc
static void activation_nchwc(float* buf, int C, int H, int W, int vec, ActKind k) {
    const int64_t total = (int64_t)ceil_div(C, vec) * H * W * vec;
    apply_activation(buf, (int)total, k);
}

// Reference conv: NCHWc (vec) + packed weights [ocb][icb][kH][kW][ic_i][oc_i]
// Produces NCHWc output (materialized). Stride/pad supported; dilation=1 for MVP.
static void conv_ref_packed_nchwc(const float* in, int inC, int H, int W,
                                  float* out, int outC,
                                  const ConvPackInfo& meta, const float* Wp,
                                  int strideH, int strideW, int padH, int padW,
                                  int vec)
{
    assert(vec == meta.vec);
    const int inCb  = (int)ceil_div(inC,  vec);
    const int outCb = (int)ceil_div(outC, vec);
    const int kH = (int)meta.kH, kW = (int)meta.kW;

    const int outH = (int)conv_out_dim(H, padH, /*dilation*/1, kH, strideH);
    const int outW = (int)conv_out_dim(W, padW, /*dilation*/1, kW, strideW);

    // zero output
    std::memset(out, 0, (size_t)outCb * outH * outW * vec * sizeof(float));

    for (int ocb = 0; ocb < outCb; ++ocb) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                float* out_pix = out + ((((int64_t)ocb * outH) * outW + (int64_t)oh * outW + ow) * vec);
                
                for (int icb = 0; icb < inCb; ++icb) {
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = oh * strideH - padH + kh;
                        if ((unsigned)ih >= (unsigned)H) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int iw = ow * strideW - padW + kw;
                            if ((unsigned)iw >= (unsigned)W) continue;
                            const float* in_pix =
                                in + ((((int64_t)icb * H) * W + (int64_t)ih * W + iw) * vec);
                            const int64_t Wbase =
                            (int64_t)ocb * meta.stride_ocb +
                            (int64_t)icb * meta.stride_icb +
                            (int64_t)kh  * meta.stride_kh  +
                            (int64_t)kw  * meta.stride_kw;
                            for (int ic_i = 0; ic_i < vec; ++ic_i) {
                                const float x = in_pix[ic_i];
                                if (x == 0.0f) continue;
                                const int64_t Wrow = Wbase + (int64_t)ic_i * meta.stride_ic_i;
                                for (int oc_i = 0; oc_i < vec; ++oc_i) {
                                out_pix[oc_i] += Wp[Wrow + oc_i] * x;
                                }
                            }
                        }
                    }
                }

            } // ow
        }   // oh
    }     // ocb
}

static MaxShape region_max_shape(const FusedRegion& ir, int C0, int H0, int W0) {
    int C = C0, H = H0, W = W0;
    int maxC = C, maxH = H, maxW = W;

    for (const auto& op : ir.ops) {
        if (op.kind == OpKind::Conv2D) {
            const auto& c = op.conv;
            const int outH = (int)conv_out_dim(H, c.padH, /*dilation*/1, c.kH, c.strideH);
            const int outW = (int)conv_out_dim(W, c.padW, /*dilation*/1, c.kW, c.strideW);
            const int outC = (int)c.outC;
            C = outC; H = outH; W = outW;
            maxC = std::max(maxC, C);
            maxH = std::max(maxH, H);
            maxW = std::max(maxW, W);
        } else {
        // Activations don’t change shape
        }
    }
    return {maxC, maxH, maxW};
}

// ---------- MegakernelModel methods ----------

MegakernelModel::MegakernelModel(const std::string& onnx_path, TargetISA isa)
: isa_(isa) {
    compile_from_onnx(onnx_path);
    plan_layouts_and_tiles();
    pack_all_weights();
    build_executors();
}

MegakernelModel::~MegakernelModel() {
    for (auto& r : regions_) {
        free_packed_weights(r.weights);
    }
}

void MegakernelModel::compile_from_onnx(const std::string& path) {
    auto lg = load_onnx_conv_act_only(path);
    onnx_inits_ = lg.inits; // keep initializers around

    auto regs = make_conv_act_regions(lg);
    if (regs.empty())
        throw std::runtime_error("No fusible Conv/Activation regions found.");

    regions_.clear();
    regions_.reserve(regs.size());
    for (auto& ir : regs) {
        CompiledRegion cr;
        cr.ir = std::move(ir);
        regions_.push_back(std::move(cr));
    }

    model_input_  = TensorDesc{1, lg.inC,  lg.inH,  lg.inW};
    // Output is last region's out*
    const auto& last = regions_.back().ir;
    model_output_ = TensorDesc{1, last.outC, last.outH, last.outW};
}

void MegakernelModel::plan_layouts_and_tiles() {
    for (auto& r : regions_) {
        r.schedule.internal = Layout::NCHWc;
        r.schedule.vec = 16;
        r.schedule.tile = Tile{32, 32, 8, 64}; // placeholder; unused in ref path
    }
}

void MegakernelModel::pack_all_weights() {
    for (auto& r : regions_) {
        r.weights = pack_weights_for_region(r.ir, onnx_inits_, /*vec=*/16);
    }
}

void MegakernelModel::build_executors() {
    // Materializing MVP uses the reference conv directly in run(); nothing to do here.
}

void MegakernelModel::run(const float* input_nchw, int N, int C, int H, int W,
                          float* output_nchw, int outC, int outH, int outW,
                          int /*num_threads*/, Affinity /*affinity*/) const
{
    if (N != 1) throw std::runtime_error("Phase 1 supports N=1 only");
    if ((model_input_.C != C) || (model_input_.H != H) || (model_input_.W != W))
        throw std::runtime_error("Input shape mismatch vs compiled model.");
    if ((model_output_.C != outC) || (model_output_.H != outH) || (model_output_.W != outW))
        throw std::runtime_error("Output shape mismatch vs compiled model.");

    const int vec = 16;

    // Allocate two NCHWc work buffers sized to the **max** intermediate
    int maxC = C, maxH = H, maxW = W;
    int curC = C, curH = H, curW = W;
    for (const auto& r : regions_) {
        MaxShape m = region_max_shape(r.ir, curC, curH, curW);
        maxC = std::max(maxC, m.C);
        maxH = std::max(maxH, m.H);
        maxW = std::max(maxW, m.W);
        // After the region, shapes become the region's outputs
        curC = (int)r.ir.outC; curH = (int)r.ir.outH; curW = (int)r.ir.outW;
    }

    // Allocate NCHWc scratch using the maximums
    const int Cb_max = (int)ceil_div((int64_t)maxC, (int64_t)vec);
    const size_t scratch_elems = (size_t)Cb_max * (size_t)maxH * (size_t)maxW * (size_t)vec;
    std::unique_ptr<float[]> bufA(new float[scratch_elems]);
    std::unique_ptr<float[]> bufB(new float[scratch_elems]);

    float* cur  = bufA.get();
    float* next = bufB.get();

    // 1) Convert NCHW -> NCHWc
    nchw_to_nchwc(input_nchw, C, H, W, vec, cur);
    curC = C; curH = H; curW = W;

    // 2) Execute regions (Phase 1 likely just one)
    for (const auto& r : regions_) {
        // Walk ops in order; materialize after each Conv; apply Act in-place
        size_t conv_idx = 0; // index into packed weights/meta arrays

        for (const auto& op : r.ir.ops) {
        if (op.kind == OpKind::Conv2D) {
            const auto& c = op.conv;
            const auto& meta = r.weights.meta[conv_idx];
            const float* Wp  = reinterpret_cast<const float*>(r.weights.per_conv[conv_idx].ptr);

            const int outC_now = (int)meta.outC;
            const int outH_now = (int)conv_out_dim(curH, c.padH, /*dilation*/1, c.kH, c.strideH);
            const int outW_now = (int)conv_out_dim(curW, c.padW, /*dilation*/1, c.kW, c.strideW);

            // Run reference conv
            const bool can_use_avx512_3x3_s1 =
                (vec == 16) &&
                (c.kH == 3 && c.kW == 3) &&
                (c.strideH == 1 && c.strideW == 1) &&
                (c.dilationH == 1 && c.dilationW == 1) &&
                (c.padH == 1 && c.padW == 1);

            if (can_use_avx512_3x3_s1) {
            conv3x3_s1_nchwc_oihw16i16o_avx512(
                cur, curC, curH, curW,
                next, outC_now,
                meta, Wp,
                c.padH, c.padW, vec);
            } else {
            // Fallback reference (slow but correct)
            conv_ref_packed_nchwc(cur, curC, curH, curW,
                                    next, outC_now,
                                    meta, Wp,
                                    c.strideH, c.strideW, c.padH, c.padW,
                                    vec);
            }

            // Bias (if any)
            if (r.weights.bias_ptrs[conv_idx]) {
            const float* b = reinterpret_cast<const float*>(r.weights.bias_ptrs[conv_idx]);
            add_bias_nchwc(next, outC_now, outH_now, outW_now, vec, b);
            }

            // Swap buffers; update shape trackers
            std::swap(cur, next);
            curC = outC_now; curH = outH_now; curW = outW_now;
            conv_idx++;
        } else if (op.kind == OpKind::Activation) {
            activation_nchwc(cur, curC, curH, curW, vec, op.act.kind);
        }
        }
    }

    // 3) Convert NCHWc -> NCHW
    nchwc_to_nchw(cur, model_output_.C, model_output_.H, model_output_.W, vec, output_nchw);
}

} // namespace mk
