#include "weight_packing.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace mk {

static inline size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

static void* aligned_alloc64(size_t bytes) {
  void* p = nullptr;
#if defined(_MSC_VER)
  p = _aligned_malloc(bytes, 64);
  if (!p) throw std::bad_alloc();
#else
  if (posix_memalign(&p, 64, bytes) != 0) throw std::bad_alloc();
#endif
  return p;
}

static void aligned_free64(void* p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

// Layout helper to compute strides for [ocb][icb][kH][kW][ic_i][oc_i]
static ConvPackInfo make_pack_info(int64_t O, int64_t I, int64_t kH, int64_t kW, int vec) {
  ConvPackInfo info;
  info.outC = O; info.inC = I; info.kH = kH; info.kW = kW; info.vec = vec;
  info.oc_blocks = static_cast<int>((O + vec - 1) / vec);
  info.ic_blocks = static_cast<int>((I + vec - 1) / vec);

  // Strides in element counts (floats)
  // order: ocb, icb, kh, kw, ic_i, oc_i
  const int64_t S_oc_i = 1;
  const int64_t S_ic_i = vec * S_oc_i;
  const int64_t S_kw   = vec * S_ic_i;
  const int64_t S_kh   = kW * S_kw;
  const int64_t S_icb  = kH * S_kh;
  const int64_t S_ocb  = info.ic_blocks * S_icb;

  info.stride_oc_i = S_oc_i;
  info.stride_ic_i = S_ic_i;
  info.stride_kw   = S_kw;
  info.stride_kh   = S_kh;
  info.stride_icb  = S_icb;
  info.stride_ocb  = S_ocb;

  return info;
}

// Pack one OIHW tensor into [ocb][icb][kH][kW][ic_i][oc_i] with zero-padding tails.
static void pack_OIHW_16i16o(const float* OIHW, const int64_t O, const int64_t I,
                             const int64_t kH, const int64_t kW, int vec,
                             const ConvPackInfo& meta, float* dst)
{
  const int oc_blocks = meta.oc_blocks;
  const int ic_blocks = meta.ic_blocks;

  for (int ocb = 0; ocb < oc_blocks; ++ocb) {
    for (int icb = 0; icb < ic_blocks; ++icb) {
      for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
          for (int ic_i = 0; ic_i < vec; ++ic_i) {
            for (int oc_i = 0; oc_i < vec; ++oc_i) {
              const int64_t o = ocb * vec + oc_i;
              const int64_t i = icb * vec + ic_i;

              float v = 0.0f;
              if (o < O && i < I) {
                // OIHW index: (((o*I + i)*kH + kh)*kW + kw)
                const int64_t idx = (((o * I + i) * kH + kh) * kW + kw);
                v = OIHW[idx];
              }

              const int64_t out_idx =
                ocb * meta.stride_ocb +
                icb * meta.stride_icb +
                kh  * meta.stride_kh  +
                kw  * meta.stride_kw  +
                ic_i* meta.stride_ic_i+
                oc_i* meta.stride_oc_i;

              dst[out_idx] = v;
            }
          }
        }
      }
    }
  }
}

static void* pack_bias_padded(const float* bias, int64_t outC, int vec, size_t& bytes_out) {
  const size_t padded = round_up(static_cast<size_t>(outC), static_cast<size_t>(vec));
  const size_t bytes  = padded * sizeof(float);
  float* buf = reinterpret_cast<float*>(aligned_alloc64(bytes));
  // zero init for tail
  std::memset(buf, 0, bytes);
  std::memcpy(buf, bias, outC * sizeof(float));
  bytes_out = bytes;
  return buf;
}

PackedWeights pack_weights_for_region(const FusedRegion& region,
    const std::unordered_map<std::string, Initializer>& inits,
    int vec)
{
  PackedWeights pw;
  pw.vec = vec;

  pw.per_conv.reserve(region.ops.size());
  pw.meta.reserve(region.ops.size());

  for (size_t idx = 0; idx < region.ops.size(); ++idx) {
    const auto& op = region.ops[idx];
    if (op.kind != OpKind::Conv2D) continue;

    const auto& c = op.conv;
    // Look up weight initializer (OIHW)
    auto itW = inits.find(c.weight_name);
    if (itW == inits.end())
      throw std::runtime_error("Weight not found: " + c.weight_name);
    const auto& W = itW->second;
    if (W.dims.size() != 4)
      throw std::runtime_error("Weight must be 4D OIHW: " + c.weight_name);
    const int64_t O = W.dims[0], I = W.dims[1], kH = W.dims[2], kW = W.dims[3];
    if ((int)kH != c.kH || (int)kW != c.kW) {
      // Keep permissive but warn later if you add logging
    }

    // Metadata & buffer size
    ConvPackInfo meta = make_pack_info(O, I, kH, kW, vec);
    const size_t elems = static_cast<size_t>(meta.stride_ocb) * meta.oc_blocks;
    const size_t bytes = elems * sizeof(float);

    // Allocate and pack
    float* dst = reinterpret_cast<float*>(aligned_alloc64(bytes));
    std::memset(dst, 0, bytes);
    pack_OIHW_16i16o(W.data.data(), O, I, kH, kW, vec, meta, dst);

    // Bias (optional)
    void* bias_ptr = nullptr; size_t bias_bytes = 0;
    if (!c.bias_name.empty()) {
      auto itB = inits.find(c.bias_name);
      if (itB == inits.end())
        throw std::runtime_error("Bias not found: " + c.bias_name);
      const auto& B = itB->second;
      if (B.dims.size() != 1 || B.dims[0] != O)
        throw std::runtime_error("Bias must be [O]: " + c.bias_name);
      bias_ptr = pack_bias_padded(B.data.data(), O, vec, bias_bytes);
      pw.owned.push_back(bias_ptr);
    }

    // Record
    pw.per_conv.push_back(PackedWeights::PerConv{
      static_cast<int64_t>(idx), bytes, dst
    });
    pw.bias_ptrs.push_back(bias_ptr);
    pw.bias_bytes.push_back(bias_bytes);
    pw.meta.push_back(meta);
    pw.owned.push_back(dst);
  }

  return pw;
}

void free_packed_weights(PackedWeights& pw) {
  for (void* p : pw.owned) {
    if (p) aligned_free64(p);
  }
  pw.owned.clear();
  pw.per_conv.clear();
  pw.bias_ptrs.clear();
  pw.bias_bytes.clear();
  pw.meta.clear();
}

} // namespace mk
