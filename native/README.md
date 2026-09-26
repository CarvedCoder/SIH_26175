# native — C++ SIMD raster kernels

Native acceleration for the DepthWizard serving stack's measured CPU
hotspots, exposed to Python through **pybind11** with a pure-Python
fallback (the application MUST work without this module — see
`backend/app/services/native_accel.py`).

## Why these kernels

Profiling the backend on a reference-sized DSM (2540x2180, the recorded
demo scene) showed:

| kernel | role | NumPy cost |
|---|---|---|
| `fill_invalid` | NaN/Inf hole fill — runs on every height-tile request and every viewer-layer PNG | **620 ms** |
| `stretch_to_01` + `quantize_u8` | normalization + PNG encode prep — every layer PNG (depth/dsm/slope) | 29.5 ms |

After the invalid-filled raster, a sidecar cache (`dsm_filled.npy`,
mtime-guarded against `dsm.npy`) removes the fill from the per-tile
serving path entirely — the native kernel matters for first generation
and reprocessing.

## Benchmarks (Windows, clang-derived toolchain, -O3)

```
fill_invalid  [python]    620.1 ms
fill_invalid  [native ]    17.5 ms   speedup x35.4
stretch+quant [python]     29.5 ms
stretch+quant [native ]     7.7 ms   speedup  x3.8
```

Numerics: the kernels are **bit-exact** against the pure-Python fallback
on every tested hole pattern (single holes, craters, strips; the
64-iteration-cap leftover case included). Do not change the update order
without re-verifying with `python native/benchmark.py` +
`backend_tests/test_native_accel.py`.

## Build

```bash
pip install pybind11           # build-time dependency only
python native/build.py         # -> native/build/dw_native.<abi>.so|pyd
python native/build.py --test  # + compile & run the kernel unit tests
python native/benchmark.py     # Python vs native timings
```

Toolchain resolution (first that works wins — Windows, macOS, Linux):

1. CMake + system compiler (`CMakeLists.txt`)
2. Direct `clang++` / `g++`
3. `zig cc` from `pip install ziglang` — self-contained cross toolchain
   (bundles libc + kernel import libs), so a system C++ SDK is not
   required.

The built module is found automatically by
`backend.app.services.native_accel` (it searches `native/build/`).
Without it the pure-Python path runs — identical numerics, slower.

## Layout

```
native/
  CMakeLists.txt          canonical build definition
  build.py                cross-platform build script (cmake -> g++/clang++ -> zig cc)
  include/dw_raster.hpp   kernel declarations
  src/raster.cpp          SIMD-friendly kernel implementations
  src/bindings.cpp        pybind11 module (`dw_native`)
  tests/test_dw_raster.cpp  C++ unit tests (assert-style, no framework)
  benchmark.py            Python vs C++ benchmark
```

## Scope

Deliberately ONLY the measured hotspots. PyTorch inference (Depth
Anything V2 + CalibrationNet) already runs on optimized kernels
(CUDA/CPU BLAS); a browser-side renderer is not affected by backend C++
at all. No CUDA is added: the kernels run inside the serving process
alongside torch GPU inference, and profiling showed the cost is CPU-side
raster prep, not GPU.
