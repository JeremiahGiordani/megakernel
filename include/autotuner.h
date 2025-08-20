#pragma once
#include <cstddef>

namespace gemm {

struct TileParams {
  int MC;
  int KC;
  int NC;
};

/// Pick (MC,KC,NC) for AVX-512 FP32 microkernel with MR=8, NR=48.
/// - Respects SGEMM_MC/KC/NC env overrides if set (same behavior you had).
/// - Otherwise, computes tiles from cache sizes + shape.
/// - Optional env knobs (bytes unless suffixed with K/M/G):
///     SGEMM_L1D, SGEMM_L2, SGEMM_L3
///   Heuristic multipliers (floats): SGEMM_ALPHA (L1), SGEMM_BETA (L2),
///     SGEMM_GAMMA (LLC → NC), SGEMM_LLC_EFF (reserve factor)
/// - dtype_bytes should be 4 for float.
TileParams pick_tiles_avx512(int M, int N, int K, int dtype_bytes = 4);

} // namespace gemm
