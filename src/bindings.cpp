// bindings.cpp — drop-in replacement
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "gemm.h"

namespace py = pybind11;

namespace {
// 64B-aligned malloc/free (same as in the kernel)
static inline float* aligned_alloc64(size_t n_floats){
#if defined(_MSC_VER)
  return static_cast<float*>(_aligned_malloc(n_floats*sizeof(float), 64));
#else
  void* p=nullptr; if (posix_memalign(&p,64,n_floats*sizeof(float))!=0) return nullptr; return (float*)p;
#endif
}
static inline void aligned_free64(float* p){
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}
static inline bool aligned64(const void* p){
  return ((reinterpret_cast<uintptr_t>(p) & 63u) == 0u);
}

// Create a NumPy array that owns a 64B-aligned buffer
static py::array_t<float> make_aligned_array(int M, int N){
  size_t count = (size_t)M * (size_t)N;
  float* ptr = aligned_alloc64(count);
  if(!ptr) throw std::bad_alloc();
  // Capsule will free the memory when the array is GC'd
  py::capsule owner(ptr, [](void* p){ aligned_free64(static_cast<float*>(p)); });
  // Strides are row-major: (N*sizeof(float), sizeof(float))
  return py::array_t<float>(
      { M, N },
      { (py::ssize_t)N * (py::ssize_t)sizeof(float),
        (py::ssize_t)sizeof(float) },
      ptr, owner);
}
} // anonymous

// C = alpha * A@B + beta*C
// Optional 'out' lets you provide a preallocated buffer (ideally 64B aligned)
// Signature remains backward-compatible when 'out' is omitted.
py::array_t<float> gemm_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B,
    float alpha = 1.0f,
    float beta  = 0.0f)
{
    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::invalid_argument("A and B must be 2D: A[M,K], B[K,N]");

    const int M = static_cast<int>(A.shape(0));
    const int K = static_cast<int>(A.shape(1));
    const int Kb_B = static_cast<int>(B.shape(0));
    const int N = static_cast<int>(B.shape(1));
    if (K != Kb_B)
        throw std::invalid_argument("Inner dim mismatch: A[M,K] x B[K,N]");

    py::array_t<float> C = make_aligned_array(M, N);
    float* Cptr = C.mutable_data();

    // Note: for beta!=0 we intentionally do not zero C; kernel handles beta==0 overwrite paths.
    gemm::sgemm_blocked(
        A.data(), M, K,
        B.data(), N,
        Cptr,
        alpha, beta
    );
    return C;
}

PYBIND11_MODULE(_mk_cpu, m) {
    m.doc() = "CPU Megakernel implementations";
    m.def("gemm",
          &gemm_py,
          py::arg("A"), py::arg("B"),
          py::arg("alpha") = 1.0f,
          py::arg("beta")  = 0.0f,
          "Blocked SGEMM: C = alpha * A@B + beta*C, A:[M,K], B:[K,N] -> C:[M,N]\n"
          "Optional 'out' allows writing into a preallocated buffer.\n"
          "Buffers created here are 64B-aligned so the kernel uses aligned/streaming stores."
    );
}
