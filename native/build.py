#!/usr/bin/env python
"""Build the dw_native pybind11 extension — Windows, macOS and Linux.

Toolchain resolution (first that works wins):
    1. CMake + system compiler (the canonical path; see CMakeLists.txt)
    2. Direct invocation of clang++/g++ (no cmake needed)
    3. `zig cc` from the pip package `ziglang` — a fully self-contained
       cross-compiling toolchain (bundles its own libc/SDK), so the native
       module builds even on machines without a system C++ SDK. Install it
       with `pip install ziglang` (no admin rights required).

Usage:
    python native/build.py            # build into native/build/
    python native/build.py --test     # also compile+run the kernel tests
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

NATIVE_DIR = Path(__file__).resolve().parent
BUILD_DIR = NATIVE_DIR / "build"


def _pybind11_include() -> str:
    try:
        import pybind11

        return pybind11.get_include()
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pybind11 is required to build the native extension: "
            "pip install pybind11"
        ) from exc


def _python_include() -> str:
    return sysconfig.get_paths()["include"]


def _python_libs() -> tuple[str | None, str]:
    """(libs_dir, library_name) for linking the extension."""
    if sys.platform == "win32":
        lib_name = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
        # The venv's base_prefix may be a junction without libs/; search the
        # base interpreter directory tree for pythonXY.lib.
        roots = [Path(sys.base_prefix), Path(sys.base_prefix).parent]
        for root in roots:
            cand = root / "libs" / lib_name
            if cand.is_file():
                return str(cand.parent), lib_name.removesuffix(".lib")
        for cand in Path(sys.base_prefix).parent.glob(f"*/libs/{lib_name}"):
            return str(cand.parent), lib_name.removesuffix(".lib")
        return None, ""
    # Unix: extension modules resolve libpython symbols via the interpreter
    # itself (or leave them undefined at build time) — no -lpython needed.
    return None, ""


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _find_compiler() -> str | None:
    for cand in ("clang++", "g++", "c++"):
        if shutil.which(cand):
            return cand
    return None


def _zig_cmd() -> list[str] | None:
    """pip-provided zig (self-contained toolchain, no system SDK needed)."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "ziglang", "version"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            return [sys.executable, "-m", "ziglang"]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def build_with_cmake(test: bool) -> bool:
    if not shutil.which("cmake"):
        return False
    BUILD_DIR.mkdir(exist_ok=True)
    subprocess.run(
        [
            "cmake", "-S", str(NATIVE_DIR), "-B", str(BUILD_DIR),
            # FindPython3 honors Python3_EXECUTABLE; PYTHON_EXECUTABLE is the
            # legacy FindPythonInterp spelling cmake ignores in this project.
            "-DPython3_EXECUTABLE=" + sys.executable,
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "--config", "Release"], check=True)
    if test:
        subprocess.run(["ctest", "--test-dir", str(BUILD_DIR), "--output-on-failure"], check=True)
    return True


def _compile_direct(compiler_cmd: list[str], test: bool, win_abi: bool) -> None:
    """One direct compiler invocation set. `compiler_cmd` is e.g.
    ["clang++"], ["g++"], or ["<python>", "-m", "ziglang", "c++"].
    `win_abi` selects the Windows import-library linking convention."""
    BUILD_DIR.mkdir(exist_ok=True)
    inc = [_python_include(), _pybind11_include(), str(NATIVE_DIR / "include")]
    common = list(compiler_cmd) + ["-std=c++17", "-O3"]
    if not win_abi:
        common.append("-fPIC")
    for i in inc:
        common += ["-I", i]

    # Unit tests (plain C++ executable — never pass -shared)
    if test:
        test_bin = BUILD_DIR / ("dw_raster_tests.exe" if win_abi else "dw_raster_tests")
        subprocess.run(
            common + [
                str(NATIVE_DIR / "tests" / "test_dw_raster.cpp"),
                str(NATIVE_DIR / "src" / "raster.cpp"),
                "-o", str(test_bin),
            ],
            check=True,
        )
        subprocess.run([str(test_bin)], check=True)

    # Module. Windows needs the python import library (pythonXY.lib).
    mod = BUILD_DIR / f"dw_native{_ext_suffix()}"
    cmd = list(common) + ["-shared"] + [
        str(NATIVE_DIR / "src" / "bindings.cpp"),
        str(NATIVE_DIR / "src" / "raster.cpp"),
        "-o", str(mod),
    ]
    libs_dir, lib_name = _python_libs()
    if win_abi and libs_dir:
        cmd += ["-L", libs_dir, "-l", lib_name]
    subprocess.run(cmd, check=True)
    print(f"built {mod}")


def build_direct(test: bool) -> None:
    import platform

    # (label, command, windows_abi?) — first strategy that links wins.
    strategies: list[tuple[str, list[str] | None, bool]] = []
    compiler = _find_compiler()
    if compiler:
        strategies.append((compiler, [compiler], sys.platform == "win32"))
    zig = _zig_cmd()
    if zig is not None:
        if sys.platform == "win32":
            target = ["-target", "x86_64-windows-gnu"]
        elif sys.platform == "darwin":
            arch = "aarch64" if platform.machine() == "arm64" else "x86_64"
            target = ["-target", f"{arch}-macos-gnu"]
        else:
            target = ["-target", "x86_64-linux-gnu"]
        strategies.append(("zig cc", zig + ["c++"] + target, True))

    errors: list[str] = []
    for label, cmd, win_abi in strategies:
        try:
            _compile_direct(cmd, test, win_abi)
            return
        except (subprocess.CalledProcessError, OSError) as exc:
            errors.append(f"{label}: {exc}")
    raise SystemExit(
        "No usable C++ toolchain produced the module. Tried:\n  " +
        "\n  ".join(errors) +
        "\nThe pure-Python fallback keeps the app working without the "
        "native module."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="compile & run kernel tests")
    args = ap.parse_args()
    if build_with_cmake(args.test):
        print("built with cmake")
        return
    build_direct(args.test)


if __name__ == "__main__":
    main()
