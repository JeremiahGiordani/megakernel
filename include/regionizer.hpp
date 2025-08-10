#pragma once
#include <vector>
#include "ir.hpp"
#include "onnx_loader.hpp"

namespace mk {

// Turn a flat Conv/Activation op list into 1+ FusedRegion(s).
// For Phase 1, your loader only emits Conv/Activation, so this will usually
// return exactly one region. Still, this code is robust if we add other ops later.
std::vector<FusedRegion> make_conv_act_regions(const LoadedGraph& g);

} // namespace mk
