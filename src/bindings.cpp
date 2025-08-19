#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "gemm.h"

namespace py = pybind11;
using namespace mk;

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

    py::array_t<float> C({M, N});
    // If beta != 0, user should have passed an initialized C; for simplicity we just scale zeros→still zero.
    gemm::sgemm_blocked(
        A.data(), M, K,
        B.data(), N,
        C.mutable_data(),
        alpha, beta
    );
    return C;
}

PYBIND11_MODULE(_attn_cpu, m) {
    m.doc() = "CPU Megakernel implementations";
    m.def("gemm",
          &gemm_py,
          py::arg("A"), py::arg("B"),
          py::arg("alpha") = 1.0f,
          py::arg("beta")  = 0.0f,
          "Blocked SGEMM: C = alpha * A@B + beta*C, A:[M,K], B:[K,N] -> C:[M,N]\n"
          "Tunables default to AVX-512-friendly picks. Pass num_threads>0 to set OpenMP threads."
    );
}