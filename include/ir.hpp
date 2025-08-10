#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mk {

enum class OpKind { Conv2D, Activation };
enum class ActKind { None, ReLU, SiLU, GELU_Tanh };

struct Conv2D {
  int64_t inC, outC, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW;
  std::string weight_name;          // <-- added
  std::string bias_name;            // <-- added (may be empty)
};

struct Activation {
  ActKind kind = ActKind::ReLU;
};

struct Op {
  OpKind kind;
  Conv2D conv;         // valid if kind==Conv2D
  Activation act;      // valid if kind==Activation
};

struct FusedRegion {
  // Phase 1: linear chain of Conv/Act ops with single input/output
  std::vector<Op> ops;
  // Boundaries:
  int64_t inC=0, inH=0, inW=0;
  int64_t outC=0, outH=0, outW=0;
  // Chosen during scheduling:
  // - layout (NCHWc / NHWCc), tile sizes, vec width…
};

} // namespace mk
