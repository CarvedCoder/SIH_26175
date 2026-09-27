// DepthWizard — native raster kernels (header)
//
// SIMD-friendly implementations of the CPU hotspots measured in the
// serving stack (see native/benchmark.py):
//   1. fill_invalid  — iterative neighbour-mean fill of NaN/Inf holes in a
//                      height raster. Was the #1 backend CPU hotspot
//                      (0.6 s per call on a 2540x2180 DSM in pure NumPy;
//                      runs on EVERY height-tile request and layer-PNG
//                      generation).
//   2. stretch_to_01 — affine normalization (lo/hi -> [0,1], clamped).
//                      Runs for every viewer layer PNG (depth/dsm/slope).
//   3. quantize_u8   — [0,1] float -> uint8 PNG sample buffer (x255 round).
//
// All functions operate on contiguous row-major float32 buffers. The
// implementations use plain scalar loops over tightly-packed data that
// clang/gcc auto-vectorize (build with -O3; -march=native enables AVX2
// where available). No external dependencies.

#pragma once

#include <cstdint>

namespace dw {

// Replace non-finite samples with the mean of their finite 8-neighbours,
// iterating until filled (same algorithm as
// TerrainService._fill_invalid). Surviving all-invalid regions get the
// global mean of finite samples (0.0 if none). Returns the number of
// iterations performed.
long fill_invalid(float *data, long rows, long cols);

// Clamp-and-stretch: out = clamp((in - lo) / (hi - lo), 0, 1).
// Non-finite inputs map to 0.
void stretch_to_01(const float *in, float *out, long count, float lo, float hi);

// out = (uint8) lround(clamp(in, 0, 1) * 255.0f)
void quantize_u8(const float *in, std::uint8_t *out, long count);

} // namespace dw
