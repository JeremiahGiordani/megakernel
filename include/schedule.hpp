// include/schedule.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>

namespace mk {

enum class Layout { NCHW, NCHWc, NHWC, NHWCc };
enum class TargetISA { Auto, AVX2, AVX512 };
enum class Affinity { None, Compact, Scatter };

struct Tile { int oc_block, ic_block, y_strip, x_strip; };

struct Schedule {
  Layout internal = Layout::NCHWc;
  int vec = 16; // AVX-512 FP32 lanes
  Tile tile{32, 32, 8, 64};
};

struct TensorDesc { int64_t N,C,H,W; };

// --- NEW: metadata describing one packed conv buffer ---
struct ConvPackInfo {
  int64_t outC=0, inC=0, kH=0, kW=0;
  int vec=16;
  int oc_blocks=0, ic_blocks=0;       // ceil(outC/vec), ceil(inC/vec)
  // Logical strides (in floats) for address calc in the packed layout:
  // index = ((((ocb*ic_blocks + icb)*kH + kh)*kW + kw)*vec + ic_i)*vec + oc_i
  int64_t stride_ocb=0, stride_icb=0, stride_kh=0, stride_kw=0, stride_ic_i=0, stride_oc_i=1;
};

// Packed weights for a region, aligned with each Conv op in region.ops
struct PackedWeights {
  struct PerConv { int64_t op_index; size_t bytes; void* ptr; };
  std::vector<PerConv> per_conv;

  // Bias (padded to vec), same order as per_conv (nullptr if none)
  std::vector<void*> bias_ptrs;
  std::vector<size_t> bias_bytes;

  // Metadata per conv
  std::vector<ConvPackInfo> meta;

  // Ownership for buffers we allocate (weights + biases)
  std::vector<void*> owned;

  int vec = 16;
};

} // namespace mk
