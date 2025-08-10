#pragma once
#include <cmath>
#include "ir.hpp"

namespace mk {
inline void apply_activation(float* __restrict v, int len, ActKind k) {
  if (k == ActKind::ReLU) {
    for (int i=0;i<len;++i) v[i] = v[i] > 0.f ? v[i] : 0.f;
  } else if (k == ActKind::SiLU) {
    for (int i=0;i<len;++i) { float x=v[i]; v[i] = x/(1.f+std::exp(-x)); }
  }
  // GELU approx later
}
} // namespace mk
