#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "engine.hpp"
#include "schedule.hpp"

namespace py = pybind11;
using namespace mk;

PYBIND11_MODULE(mkcpu, m) {
  // 1) Register enums BEFORE using them in defaults
  py::enum_<TargetISA>(m, "TargetISA")
      .value("Auto",   TargetISA::Auto)
      .value("AVX2",   TargetISA::AVX2)
      .value("AVX512", TargetISA::AVX512)
      .export_values();

  py::enum_<Affinity>(m, "Affinity")
      .value("None",    Affinity::None)
      .value("Compact", Affinity::Compact)
      .value("Scatter", Affinity::Scatter)
      .export_values();

  // 2) Class binding
  py::class_<MegakernelModel>(m, "MegakernelModel")
    // two-arg ctor with enum default
    .def(py::init<const std::string&, TargetISA>(),
         py::arg("onnx_path"), py::arg("isa") = TargetISA::AVX512)
    // one-arg convenience ctor that picks AVX-512 in C++
    .def(py::init([](const std::string& path) {
        return new MegakernelModel(path, TargetISA::AVX512);
      }),
      py::arg("onnx_path"))

    .def("run",
      [](MegakernelModel &self,
         py::array_t<float, py::array::c_style | py::array::forcecast> x) {
        if (x.ndim() != 4) throw std::runtime_error("Expected NCHW input");
        int N = (int)x.shape(0), C=(int)x.shape(1), H=(int)x.shape(2), W=(int)x.shape(3);
        if (N != 1) throw std::runtime_error("Phase 1 supports N=1 only");
        auto od = self.output_desc();
        py::array_t<float> out({od.N, od.C, od.H, od.W});
        self.run(x.data(), N, C, H, W,
                 out.mutable_data(), (int)od.C, (int)od.H, (int)od.W);
        return out;
      },
      "Run inference on input X (NCHW, FP32; N=1)");
}
