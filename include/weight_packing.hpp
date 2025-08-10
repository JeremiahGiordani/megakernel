#pragma once
#include <string>
#include <unordered_map>
#include "ir.hpp"
#include "onnx_loader.hpp"
#include "schedule.hpp"

namespace mk {

// Pack all Conv2D weights (and biases) for a fused region into OIhw16i16o layout.
// - inits: ONNX initializers by name (weights/biases).
// - vec:   16 for AVX-512 (Phase 1).
// Returns a PackedWeights with owned aligned buffers + metadata.
PackedWeights pack_weights_for_region(const FusedRegion& region,
    const std::unordered_map<std::string, Initializer>& inits,
    int vec = 16);

// Free all owned buffers in a PackedWeights (idempotent).
void free_packed_weights(PackedWeights& pw);

} // namespace mk
