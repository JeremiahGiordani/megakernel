#pragma once
#include <cstddef>

namespace gemm {

// Row-major, float32, contiguous.
// A: [M x K], B: [K x N], C: [M x N]
// Computes: C = alpha * (A @ B) + beta * C
// Tunables default to reasonable AVX-512 FP32 values; safe on non-AVX too (scalar loops inside).
// , int Mb, int Nb, int Kb, int mr, int nr, int ku
void sgemm_blocked(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha = 1.0f, float beta = 0.0f);


void gemm_blocked_jit(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha = 1.0f, float beta = 0.0f);

} // namespace gemm
