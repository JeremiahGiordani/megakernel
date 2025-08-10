#pragma once
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "ir.hpp"
#include "schedule.hpp"
#include "onnx_loader.hpp"   // <-- add

namespace mk {

struct CompiledRegion {
  FusedRegion ir;
  Schedule schedule;
  PackedWeights weights;
};

class MegakernelModel {
public:
  explicit MegakernelModel(const std::string& onnx_path,
                           TargetISA isa = TargetISA::AVX512);
  ~MegakernelModel();

  // Phase 1: N=1, FP32, NCHW in/out
  void run(const float* input_nchw, int N, int C, int H, int W,
           float* output_nchw, int outC, int outH, int outW,
           int num_threads = 0, Affinity affinity = Affinity::None) const;

  // Handy for Python binding to allocate an output array of the right shape
  TensorDesc output_desc() const { return model_output_; }

private:
  void compile_from_onnx(const std::string& path);
  void plan_layouts_and_tiles();
  void pack_all_weights();
  void build_executors(); // (no-op for materializing MVP)

  TargetISA isa_;
  std::vector<CompiledRegion> regions_;
  TensorDesc model_input_{1,0,0,0}, model_output_{1,0,0,0};

  // Keep initializers for packing
  std::unordered_map<std::string, Initializer> onnx_inits_;
};

} // namespace mk
