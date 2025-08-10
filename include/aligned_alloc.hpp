#pragma once
#include <cstdlib>
#include <stdexcept>

namespace mk {
inline void* aligned_alloc64(size_t bytes) {
  void* p = nullptr;
#if defined(_MSC_VER)
  p = _aligned_malloc(bytes, 64);
  if (!p) throw std::bad_alloc();
#else
  if (posix_memalign(&p, 64, bytes) != 0) throw std::bad_alloc();
#endif
  return p;
}
inline void aligned_free64(void* p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  std::free(p);
#endif
}
} // namespace mk
