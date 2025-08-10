// include/onnx_loader.hpp
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "ir.hpp"

namespace mk {

struct Initializer {
  std::string name;
  std::vector<int64_t> dims;
  std::vector<float> data;  // FP32 only for Phase 1
};

struct LoadedGraph {
  std::vector<Op> ops;  // linear Conv/Act chain in model order
  int64_t inC=0, inH=0, inW=0;
  int64_t outC=0, outH=0, outW=0;
  std::unordered_map<std::string, Initializer> inits; // by name
};

// Parses an ONNX file and returns only Conv/Activation ops and initializers.
// Assumes N=1, FP32. Ignores BN/pool/branches (we’ll split earlier in the pipeline).
LoadedGraph load_onnx_conv_act_only(const std::string& onnx_path);

} // namespace mk
