"""CLI diagnostic for ONNX model I/O signatures.

Usage:
    python -m depthwizard.disaster.inspect_models

Prints the complete input/output signature of both the building
detection and damage assessment ONNX models. Does NOT run inference.
"""

from __future__ import annotations

import os
import sys


def inspect_model(path: str, label: str) -> None:
    """Inspect a single ONNX model and print its signature."""
    from pathlib import Path

    model_path = Path(path)
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Path: {model_path}")

    if not model_path.exists():
        print(f"  STATUS: NOT FOUND")
        return

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

    try:
        import onnxruntime as ort

        print(f"  ONNX Runtime version: {ort.__version__}")
        print(f"  Available providers: {ort.get_available_providers()}")

        sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        print(f"\n  Inputs:")
        for inp in sess.get_inputs():
            print(f"    {inp.name}: {inp.type} {inp.shape}")

        print(f"\n  Outputs:")
        for out in sess.get_outputs():
            print(f"    {out.name}: {out.type} {out.shape}")

        meta = sess.get_modelmeta()
        if meta.custom_metadata_map:
            print(f"\n  Metadata:")
            for k, v in meta.custom_metadata_map.items():
                print(f"    {k}: {v}")

        print(f"\n  STATUS: OK")
        del sess

    except Exception as exc:
        print(f"  STATUS: ERROR — {exc}")


def main() -> None:
    """Inspect both disaster assessment ONNX models."""
    building_model = os.environ.get("DW_BUILDING_MODEL_ONNX", "")
    damage_model = os.environ.get("DW_DAMAGE_MODEL_ONNX", "")

    # Try to find models if env vars not set
    if not building_model:
        from pathlib import Path
        candidates = [
            Path("local_model.onnx"),
            Path("models/building/model.onnx"),
        ]
        for c in candidates:
            if c.exists():
                building_model = str(c)
                break

    if not damage_model:
        from pathlib import Path
        candidates = [
            Path("model.onnx"),
            Path("models/damage/model.onnx"),
        ]
        for c in candidates:
            if c.exists():
                damage_model = str(c)
                break

    print("DepthWizard — Disaster Assessment Model Inspector")
    print("=" * 60)

    if building_model:
        inspect_model(building_model, "BUILDING MODEL (HOTOSM DINOv3)")
    else:
        print("\n  BUILDING MODEL: Not configured")
        print("  Set DW_BUILDING_MODEL_ONNX environment variable")

    if damage_model:
        inspect_model(damage_model, "DAMAGE MODEL (HOTOSM Earthquake)")
    else:
        print("\n  DAMAGE MODEL: Not configured")
        print("  Set DW_DAMAGE_MODEL_ONNX environment variable")


if __name__ == "__main__":
    main()
