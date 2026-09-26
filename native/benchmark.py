#!/usr/bin/env python
"""Benchmark: pure-Python (NumPy) vs native C++ kernels.

Run after building:  python native/build.py && python native/benchmark.py

Uses a raster sized like the repository's recorded reference scene
(2540x2180 — a Cartosat-class DSM) with ~0.5% invalid samples.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "build"))

from backend.app.services import native_accel  # noqa: E402


def bench(label: str, fn, repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"{label:<44s} {best * 1000:9.1f} ms")
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    rows, cols = 2180, 2540  # the reference scene raster size
    dsm = rng.uniform(0, 18.8, (rows, cols)).astype(np.float32)
    dsm[rng.random(dsm.shape) < 0.005] = np.nan
    print(f"raster {cols}x{rows} ({dsm.size / 1e6:.1f} Mpx, "
          f"{int((~np.isfinite(dsm)).sum())} invalid)\n")

    native = native_accel.native_available()
    print(f"native kernels: {'AVAILABLE' if native else 'NOT BUILT — pure python only'}\n")

    # fill_invalid: NumPy fallback vs native. The fallback runs the same
    # dispatch function with the native module forced off.
    def python_fill():
        saved, saved_state = native_accel._native, native_accel._native_state
        native_accel._native, native_accel._native_state = None, "unavailable"
        try:
            return native_accel.fill_invalid(dsm.copy())
        finally:
            native_accel._native, native_accel._native_state = saved, saved_state

    t_py = bench("fill_invalid  [python]", python_fill)
    if native:
        t_nat = bench("fill_invalid  [native ]", lambda: native_accel.fill_invalid(dsm.copy()))
        print(f"{'':44s} speedup  x{t_py / t_nat:5.1f}\n")
    else:
        print()

    # stretch + quantize path (every viewer layer PNG)
    lo, hi = 2.0, 16.0
    t_py2 = bench(
        "stretch+quant [python]",
        lambda: (np.clip((dsm - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8),
    )
    if native:
        def native_stretch():
            norm = native_accel.stretch_to_01(dsm, lo, hi)
            return native_accel.quantize_u8(norm)

        t_nat2 = bench("stretch+quant [native ]", native_stretch)
        print(f"{'':44s} speedup  x{t_py2 / t_nat2:5.1f}")


if __name__ == "__main__":
    main()
