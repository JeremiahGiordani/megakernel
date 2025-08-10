#include "regionizer.hpp"
#include <stdexcept>
#include <cmath>

namespace mk {

// ONNX Conv2D output shape (floor division semantics)
static inline int64_t conv_out_dim(int64_t in, int64_t pad, int64_t dilation,
                                   int64_t k, int64_t stride) {
  // out = floor((in + 2*pad - dilation*(k-1) - 1)/stride + 1)
  const int64_t eff_k = dilation * (k - 1) + 1;
  return ( (in + 2*pad - eff_k) / stride ) + 1;
}

std::vector<FusedRegion> make_conv_act_regions(const LoadedGraph& g) {
  std::vector<FusedRegion> regions;
  if (g.ops.empty()) return regions;

  // Current running shape through the graph (N=1 always in Phase 1)
  int64_t curC = g.inC, curH = g.inH, curW = g.inW;

  FusedRegion cur{};
  cur.inC = curC; cur.inH = curH; cur.inW = curW;

  auto flush_region = [&]() {
    if (!cur.ops.empty()) {
      // cur.out* already updated as we appended ops
      regions.push_back(cur);
      cur = FusedRegion{};
    }
  };

  for (const auto& op : g.ops) {
    switch (op.kind) {
      case OpKind::Conv2D: {
        const auto& c = op.conv;

        // Basic sanity: inC should match running channels if provided.
        if (c.inC > 0 && c.inC != curC) {
          // Split region here to keep invariants simple.
          flush_region();
          // Start a new region from the current point.
          cur.inC = curC; cur.inH = curH; cur.inW = curW;
        }

        // Infer output spatial dims & channels
        const int64_t outH = conv_out_dim(curH, c.padH, c.dilationH, c.kH, c.strideH);
        const int64_t outW = conv_out_dim(curW, c.padW, c.dilationW, c.kW, c.strideW);
        const int64_t outC = c.outC;

        // Append op to current region
        cur.ops.push_back(op);

        // Update running shape
        curC = outC; curH = outH; curW = outW;

        // Keep region boundaries current
        cur.outC = curC; cur.outH = curH; cur.outW = curW;
        if (cur.inC == 0) { cur.inC = g.inC; cur.inH = g.inH; cur.inW = g.inW; }

        break;
      }

      case OpKind::Activation: {
        // Activations don’t change shape; just append.
        cur.ops.push_back(op);
        // Boundaries unchanged (but keep them set if empty)
        if (cur.outC == 0) { cur.outC = curC; cur.outH = curH; cur.outW = curW; }
        if (cur.inC  == 0) { cur.inC  = g.inC; cur.inH  = g.inH; cur.inW  = g.inW; }
        break;
      }

      default: {
        // Unsupported op: close current region and start fresh after it.
        flush_region();
        // Skip/ignore the op for Phase 1 (loader shouldn't emit others anyway).
        break;
      }
    }
  }

  // Push the last open region
  flush_region();

  // If we created nothing but we had ops, make one region as a fallback.
  if (regions.empty() && !g.ops.empty()) {
    FusedRegion one{};
    one.ops = g.ops;
    one.inC = g.inC; one.inH = g.inH; one.inW = g.inW;
    one.outC = curC; one.outH = curH; one.outW = curW;
    regions.push_back(std::move(one));
  }

  return regions;
}

} // namespace mk
