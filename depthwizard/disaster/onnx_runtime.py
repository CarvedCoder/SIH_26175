"""Reusable ONNX Runtime inference session wrapper.

Handles:
    * Provider selection (CUDA > CPU with automatic fallback)
    * Model I/O introspection (shapes, dtypes, names)
    * Health validation without expensive inference
    * Session lifecycle (load once, infer many)
    * Typed errors for missing/invalid models
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class OnnxModelError(Exception):
    """Raised when an ONNX model cannot be loaded or validated."""


# Set process-wide when a warmup probe proves the CUDA EP produces NaN
# outputs, so later sessions in the same process skip straight to CPU.
_CUDA_UNUSABLE = False


@dataclass(frozen=True)
class TensorSpec:
    """Describes a single model input or output tensor."""
    name: str
    shape: list[int | str]   # may contain symbolic dims like 'batch'
    dtype: str               # e.g. 'tensor(float)'

    @property
    def numpy_dtype(self) -> np.dtype:
        mapping = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(uint8)": np.uint8,
        }
        return np.dtype(mapping.get(self.dtype, np.float32))


@dataclass(frozen=True)
class ModelSignature:
    """Complete I/O signature of an ONNX model."""
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    provider: str
    metadata: dict[str, str]

    def input_by_name(self, name: str) -> TensorSpec:
        for spec in self.inputs:
            if spec.name == name:
                return spec
        raise KeyError(f"No input named '{name}'; available: {[s.name for s in self.inputs]}")

    def output_by_name(self, name: str) -> TensorSpec:
        for spec in self.outputs:
            if spec.name == name:
                return spec
        raise KeyError(f"No output named '{name}'; available: {[s.name for s in self.outputs]}")


def _preload_nvidia_libraries() -> None:
    """Preload NVIDIA CUDA/cuDNN libraries installed in Python site-packages.

    When CUDA packages (nvidia-cublas, nvidia-cudnn, etc.) are installed via pip/uv,
    their shared libraries live in site-packages/nvidia/*/lib. Because this directory
    is not in default dynamic linker search paths or LD_LIBRARY_PATH, ONNX Runtime's
    CUDAExecutionProvider fails to find them unless preloaded into the global symbol table.
    """
    import sys
    import ctypes
    from pathlib import Path

    for sp in sys.path:
        if "site-packages" not in sp:
            continue
        nv_dir = Path(sp) / "nvidia"
        if not nv_dir.is_dir():
            continue
        for lib_dir in sorted(nv_dir.glob("*/lib")):
            if lib_dir.is_dir():
                for so in sorted(lib_dir.glob("*.so*")):
                    try:
                        ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass


class OnnxSession:
    """Lightweight wrapper around onnxruntime.InferenceSession.

    Usage:
        session = OnnxSession(path, device="auto")
        signature = session.signature
        outputs = session.run({"image": np.zeros(...)})
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        restore_fp32: bool = True,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise OnnxModelError(
                f"ONNX model not found: {self.model_path}"
            )
        if not self.model_path.is_file():
            raise OnnxModelError(
                f"ONNX model path is not a file: {self.model_path}"
            )

        # Internally-fp16 graphs overflow to inf/NaN in the attention
        # MatMul on the CUDA EP (see graph_fp32.py docstring). Restoring
        # the graph to fp32 makes GPU inference correct; on CPU it is
        # harmless (and more accurate than the fp16 export).
        load_path = self.model_path
        self.fp32_restored = False
        if restore_fp32:
            try:
                from .graph_fp32 import restore_fp32_model

                restored = restore_fp32_model(self.model_path)
                self.fp32_restored = restored != self.model_path
                load_path = restored
            except Exception as exc:  # restore is best-effort
                print(f"[onnx] fp32 restore unavailable ({exc}); "
                      f"loading original model")

        # Provider selection — skip the GPU entirely once a warmup probe
        # has proven it broken in this process.
        if device != "cpu" and _CUDA_UNUSABLE:
            providers = ["CPUExecutionProvider"]
        else:
            providers = self._select_providers(device)
        self._session = self._create_session(providers, load_path)

        # Determine which provider is actually active
        active_providers = self._session.get_providers()
        if "CUDAExecutionProvider" in active_providers:
            self._provider = "CUDA"
        else:
            self._provider = "CPU"

        # Some EP/driver stacks silently produce all-NaN outputs (observed
        # with the CUDA EP of onnxruntime-gpu 1.30 on CUDA 13 + RTX 4050,
        # for every input). Validate the GPU path with one cheap forward
        # pass and fall back to CPU when it is broken. The result is
        # remembered process-wide so later sessions skip the broken GPU.
        if self._provider == "CUDA" and not _CUDA_UNUSABLE:
            self._validate_gpu_or_fallback()

        # Build signature once
        self._signature = self._build_signature()

    def _create_session(self, providers: list, load_path: Path | None = None):
        import onnxruntime as ort

        path = Path(load_path) if load_path else self.model_path
        try:
            return ort.InferenceSession(
                str(path),
                providers=providers,
            )
        except Exception as exc:
            raise OnnxModelError(
                f"Failed to load ONNX model '{path.name}': {exc}"
            ) from exc

    def _validate_gpu_or_fallback(self) -> None:
        """Run one cheap zeros forward pass on the GPU; fall back to CPU
        if the output contains NaN. Skipped when input dims are dynamic
        (a safe warmup input cannot be constructed) — the per-run NaN
        guard in run() still catches broken outputs."""
        global _CUDA_UNUSABLE
        try:
            feeds = {}
            for spec in self._session.get_inputs():
                dims = list(spec.shape)
                if any(not isinstance(d, int) or d <= 0 for d in dims):
                    return  # dynamic dims — warmup inconclusive
                feeds[spec.name] = np.zeros(
                    dims, dtype=TensorSpec(spec.name, dims, spec.type).numpy_dtype
                )
            first_output = self._session.get_outputs()[0].name
            out = self._session.run([first_output], feeds)[0]
            if np.issubdtype(out.dtype, np.floating) and np.isnan(out).any():
                print(
                    "[onnx] CUDA provider produced NaN output for "
                    f"'{self.model_path.name}' — falling back to CPU"
                )
                _CUDA_UNUSABLE = True
                self._session = self._create_session(["CPUExecutionProvider"])
                self._provider = "CPU"
        except Exception:
            # Warmup itself failed — leave the session as-is; run() will
            # surface any real inference error as a typed error.
            return

    @staticmethod
    def _select_providers(device: str) -> list[str]:
        """Select ONNX Runtime execution providers based on device hint."""
        if device in ("cuda", "gpu", "auto"):
            _preload_nvidia_libraries()

        import onnxruntime as ort

        available = ort.get_available_providers()

        if device in ("cuda", "gpu"):
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            raise OnnxModelError(
                "CUDA requested but CUDAExecutionProvider is not available in ONNX Runtime. "
                f"Available providers: {available}. "
                "Ensure 'onnxruntime-gpu' is installed instead of 'onnxruntime' "
                "(run: uv pip install onnxruntime-gpu) and compatible CUDA/cuDNN libraries are available."
            )

        if device == "cpu":
            return ["CPUExecutionProvider"]

        # auto: prefer CUDA, fall back to CPU
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _build_signature(self) -> ModelSignature:
        inputs = [
            TensorSpec(
                name=inp.name,
                shape=list(inp.shape),
                dtype=inp.type,
            )
            for inp in self._session.get_inputs()
        ]
        outputs = [
            TensorSpec(
                name=out.name,
                shape=list(out.shape),
                dtype=out.type,
            )
            for out in self._session.get_outputs()
        ]
        meta = self._session.get_modelmeta()
        return ModelSignature(
            inputs=inputs,
            outputs=outputs,
            provider=self._provider,
            metadata=dict(meta.custom_metadata_map),
        )

    @property
    def signature(self) -> ModelSignature:
        return self._signature

    def run(
        self,
        inputs: dict[str, np.ndarray],
        output_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run inference and return named outputs as a dict."""
        if output_names is None:
            output_names = [o.name for o in self._signature.outputs]

        results = self._session.run(output_names, inputs)
        for name, arr in zip(output_names, results):
            if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
                raise OnnxModelError(
                    f"Model '{self.model_path.name}' produced NaN in output "
                    f"'{name}'. This indicates a broken execution-provider "
                    "configuration — try device='cpu'."
                )
        return dict(zip(output_names, results))

    def validate_health(self) -> dict[str, Any]:
        """Non-inference health check — verifies the model loads and
        reports its configuration. Does NOT run a forward pass."""
        return {
            "model_path": str(self.model_path),
            "model_exists": True,
            "loadable": True,
            "provider": self._provider,
            "inputs": [
                {"name": s.name, "shape": s.shape, "dtype": s.dtype}
                for s in self._signature.inputs
            ],
            "outputs": [
                {"name": s.name, "shape": s.shape, "dtype": s.dtype}
                for s in self._signature.outputs
            ],
        }
