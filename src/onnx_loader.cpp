// src/onnx_loader.cpp
#include "onnx_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <onnx.pb.h>

namespace mk {

static std::vector<int64_t> read_dims(const onnx::TypeProto_Tensor& t) {
  std::vector<int64_t> d;
  d.reserve(t.shape().dim_size());
  for (const auto& dim : t.shape().dim()) {
    if (dim.has_dim_value()) d.push_back(static_cast<int64_t>(dim.dim_value()));
    else d.push_back(-1); // unknown; Phase 1 expects known dims
  }
  return d;
}

static std::vector<int64_t> value_info_dims(const onnx::ValueInfoProto& v) {
  if (!v.has_type() || !v.type().has_tensor_type())
    throw std::runtime_error("Expected tensor type");
  return read_dims(v.type().tensor_type());
}

static void extract_initializer_fp32(const onnx::TensorProto& t, Initializer& out) {
  out.name = t.name();
  out.dims.assign(t.dims().begin(), t.dims().end());
  out.data.clear(); out.data.reserve( (size_t) t.raw_data().size()/sizeof(float) );
  if (t.has_raw_data()) {
    const auto& raw = t.raw_data();
    const float* p = reinterpret_cast<const float*>(raw.data());
    size_t n = raw.size() / sizeof(float);
    out.data.assign(p, p + n);
  } else {
    // Fallback to field storage
    out.data.assign(t.float_data().begin(), t.float_data().end());
  }
  if (out.data.empty())
    throw std::runtime_error("Initializer has no FP32 data: " + out.name);
}

LoadedGraph load_onnx_conv_act_only(const std::string& onnx_path) {
  // Read model
  std::ifstream ifs(onnx_path, std::ios::binary);
  if (!ifs) throw std::runtime_error("Cannot open ONNX: " + onnx_path);

  onnx::ModelProto model;
  if (!model.ParseFromIstream(&ifs))
    throw std::runtime_error("Failed to parse ONNX protobuf");

  const auto& graph = model.graph();

  LoadedGraph lg;

  // Inputs/outputs (assume single input/output, NCHW)
  if (graph.input_size() < 1 || graph.output_size() < 1)
    throw std::runtime_error("Model must have >=1 input and >=1 output");
  auto in_dims  = value_info_dims(graph.input(0));   // [N,C,H,W]
  auto out_dims = value_info_dims(graph.output(0));  // [N,C,H,W]
  if (in_dims.size() != 4 || out_dims.size() != 4)
    throw std::runtime_error("Phase 1 expects 4D NCHW tensors");

  lg.inC  = in_dims[1]; lg.inH  = in_dims[2]; lg.inW  = in_dims[3];
  lg.outC = out_dims[1]; lg.outH = out_dims[2]; lg.outW = out_dims[3];

  // Initializers (weights/biases)
  for (const auto& init : graph.initializer()) {
    Initializer I;
    extract_initializer_fp32(init, I);
    lg.inits[I.name] = std::move(I);
  }

  // Node parsing: Conv + Activation (Relu/SiLU/GELU as we add)
  for (const auto& node : graph.node()) {
    const std::string& op = node.op_type();

    if (op == "Conv") {
      // Inputs: X, W, (B?)
      if (node.input_size() < 2)
        throw std::runtime_error("Conv missing weights");
      std::string X = node.input(0);
      std::string W = node.input(1);
      std::string B = node.input_size() >= 3 ? node.input(2) : "";

      Conv2D c{};
      // defaults
      c.strideH=1; c.strideW=1; c.padH=0; c.padW=0; c.dilationH=1; c.dilationW=1;
      c.kH=0; c.kW=0;

      // Attributes
      for (const auto& a : node.attribute()) {
        if (a.name() == "strides" && a.ints_size() == 2) {
          c.strideH = (int)a.ints(0); c.strideW = (int)a.ints(1);
        } else if (a.name() == "pads" && a.ints_size() >= 2) {
          c.padH = (int)a.ints(0); c.padW = (int)a.ints(1);
        } else if (a.name() == "dilations" && a.ints_size() == 2) {
          c.dilationH = (int)a.ints(0); c.dilationW = (int)a.ints(1);
        } else if (a.name() == "kernel_shape" && a.ints_size() == 2) {
          c.kH = (int)a.ints(0); c.kW = (int)a.ints(1);
        }
      }

      // Infer in/out channels from weight dims (OIHW)
      auto itW = lg.inits.find(W);
      if (itW == lg.inits.end())
        throw std::runtime_error("Conv weight initializer not found: " + W);
      const auto& w_dims = itW->second.dims; // [O,I,kH,kW]
      if (w_dims.size() != 4)
        throw std::runtime_error("Conv weights must be 4D OIHW");
      c.outC = w_dims[0];
      c.inC  = w_dims[1];
      if (c.kH == 0 && c.kW == 0) {
        c.kH = (int)w_dims[2];
        c.kW = (int)w_dims[3];
      }
      c.weight_name = W;
      c.bias_name   = B; // may be empty

      Op convOp;
      convOp.kind = OpKind::Conv2D;
      convOp.conv = std::move(c);
      lg.ops.push_back(std::move(convOp));
    }
    else if (op == "Relu") {
      Op actOp;
      actOp.kind = OpKind::Activation;
      actOp.act.kind = ActKind::ReLU;
      lg.ops.push_back(std::move(actOp));
    }
    // (Optional) add others later:
    // else if (op == "SiLU") { ... }  else if (op == "Gelu") { ... }
    else {
      // Phase 1: ignore unsupported ops; higher-level pipeline should split before them.
      // You can also throw if you prefer strict behavior now.
    }
  }

  return lg;
}

} // namespace mk
