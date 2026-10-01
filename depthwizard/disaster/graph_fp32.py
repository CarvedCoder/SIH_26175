"""Restore internally-fp16 ONNX graphs to full fp32.

WHY THIS EXISTS
---------------
The HOTOSM building/damage ONNX models were exported with internal fp16
compute (Cast-to-float16 wrappers around most ops, fp16 initializers).
On this project's GPU stack (RTX 4050, onnxruntime-gpu CUDA 12/13 builds
1.24–1.30), the fp16 attention MatMul for the DINOv3 token layout
(261 tokens = 256 patches + 5 registers) produces +inf/-inf attention
scores where the CPU path — which accumulates in fp32 — produces finite
values. Softmax(inf - inf) then yields NaN for every input, so GPU
inference silently returned garbage (all-NaN logits) while CPU was fine.

Diagnosis trail (2026-10-02): trivial conv models run fine on the CUDA
EP; the corruption survived provider options, graph-optimization levels,
memory-pattern/arena settings, three ORT versions, and an fp16
conversion of the whole graph. Exposing every intermediate tensor as a
graph output localised the first NaN to the attention Softmax, and the
bisection to its fp16 MatMul input containing infinities. Restoring the
graph to fp32 eliminates the overflow and GPU output then matches the
CPU reference (max |diff| ≈ 0.016 on the model's own fp16-CPU scale).

This module rewrites such graphs to fp32 end-to-end:
    * fp16 initializers  -> fp32
    * Cast(to=FLOAT16)   -> Cast(to=FLOAT)
    * fp16 value types   -> fp32
The result is mathematically the model the authors trained (the fp16
wrapping was an export-time compression), with strictly better numeric
range. Output is cached beside the source model, keyed by a hash of the
source file's size/mtime so a replaced model invalidates its cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _cache_key(model_path: Path) -> str:
    st = model_path.stat()
    digest = hashlib.sha1(
        f"{model_path.name}:{st.st_size}:{st.st_mtime_ns}".encode()
    ).hexdigest()[:10]
    return digest


def _has_internal_fp16(proto) -> bool:
    """True if the graph computes internally in fp16."""
    from onnx import TensorProto

    for init in proto.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            return True
    for node in proto.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    return True
    return False


def restore_fp32_model(model_path: str | Path, *, use_cache: bool = True) -> Path:
    """Return a path to an fp32-restored copy of ``model_path``.

    If the model has no internal fp16 compute, the original path is
    returned unchanged. Otherwise the rewritten graph is saved next to
    the source as ``<name>.fp32-<hash>.onnx`` (once) and reused.

    Raises nothing on ONNX parse failure is NOT guaranteed — callers
    should treat exceptions as "restore unavailable" and fall back to
    the original file.
    """
    import onnx
    import numpy as np
    from onnx import numpy_helper, TensorProto

    src = Path(model_path)
    cache = src.with_name(f"{src.stem}.fp32-{_cache_key(src)}{src.suffix}")
    if use_cache and cache.exists():
        return cache

    proto = onnx.load(str(src))
    if not _has_internal_fp16(proto):
        return src

    # fp16 initializers -> fp32
    for init in proto.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))

    # Constant nodes carrying fp16 tensors
    for node in proto.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))

    # Cast(to=FLOAT16) -> Cast(to=FLOAT)
    for node in proto.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT

    # Retype fp16 value infos (graph outputs and intermediate value_info)
    def _retype(value_info) -> None:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT

    for vi in proto.graph.value_info:
        _retype(vi)
    for vi in proto.graph.output:
        _retype(vi)
    for vi in proto.graph.input:
        _retype(vi)

    # IR version cap: some onnx writers stamp versions ORT rejects
    if proto.ir_version > 13:
        proto.ir_version = 13

    onnx.save(proto, str(cache))
    del proto
    return cache
