// DepthWizard — pybind11 bindings for the native raster kernels.
//
// Builds the importable module `dw_native` (dw_native.pyd / .so):
//     import dw_native
//     filled, iters = dw_native.fill_invalid(np_array_2d_float32)
//     out           = dw_native.stretch_to_01(src, lo, hi)  # new float32 array
//     u8            = dw_native.quantize_u8(src)            # new uint8 array
//
// All functions return NEW arrays — inputs are never mutated — matching
// the pure-Python fallback semantics exactly (see
// backend/app/services/native_accel.py). Arrays are accepted as any dtype
// / layout (forcecast produces a contiguous float32 temp when needed).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>

#include "dw_raster.hpp"

namespace py = pybind11;

using f32_arr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using u8_arr = py::array_t<std::uint8_t>;

PYBIND11_MODULE(dw_native, m) {
  m.doc() = "DepthWizard native SIMD raster kernels (fill_invalid, "
            "normalize, quantize). Pure-Python fallbacks live in "
            "backend.app.services.native_accel.";

  m.def(
      "fill_invalid",
      [](f32_arr src) {
        auto out = py::array_t<float>({src.shape(0), src.shape(1)});
        std::memcpy(out.request().ptr, src.request().ptr,
                    sizeof(float) * static_cast<size_t>(src.size()));
        const long iterations = dw::fill_invalid(
            static_cast<float *>(out.request().ptr),
            static_cast<long>(src.shape(0)), static_cast<long>(src.shape(1)));
        return py::make_tuple(out, iterations);
      },
      py::arg("src"),
      "Fill NaN/Inf samples with iterative finite-neighbour means. "
      "Returns (filled_array, iterations); the input is not modified.");

  m.def(
      "stretch_to_01",
      [](f32_arr src, double lo, double hi) {
        auto out = py::array_t<float>({src.shape(0), src.shape(1)});
        dw::stretch_to_01(static_cast<const float *>(src.request().ptr),
                          static_cast<float *>(out.request().ptr),
                          static_cast<long>(src.size()), static_cast<float>(lo),
                          static_cast<float>(hi));
        return out;
      },
      py::arg("src"), py::arg("lo"), py::arg("hi"),
      "Affine stretch (src - lo) / (hi - lo) clamped to [0, 1].");

  m.def(
      "quantize_u8",
      [](f32_arr src) {
        auto out = u8_arr({src.shape(0), src.shape(1)});
        dw::quantize_u8(static_cast<const float *>(src.request().ptr),
                        static_cast<std::uint8_t *>(out.request().ptr),
                        static_cast<long>(src.size()));
        return out;
      },
      py::arg("src"),
      "Quantize [0, 1] floats to uint8 display samples (x255, rounded).");
}
