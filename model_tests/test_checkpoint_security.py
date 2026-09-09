"""Checkpoint deserialization security regression tests (audit finding C2).

torch.load(..., weights_only=False) is arbitrary-code-execution-by-pickle.
The training payload is a state dict + primitives, so the restricted
``weights_only=True`` loader must be used everywhere and must still load a
real, train-command-shaped checkpoint successfully.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _train_style_checkpoint(tmp_path, a0=2.0, b0=4.0):
    """A checkpoint EXACTLY as `python model.py train` writes it (primitives
    + state dict + scaler state) — must load under weights_only=True."""
    torch = pytest.importorskip("torch")
    from depthwizard.calibration_net import CalibrationNet

    net = CalibrationNet(in_ch=4, widths=(8, 16, 32), a0=a0, b0=b0)
    ckpt = {
        "model_state": net.state_dict(),
        "use_rgb": True,
        "use_dem": False,
        "use_sem": False,
        "sem_classes": 0,
        "sem_aux_head": False,
        "in_ch": 4,
        "widths": [8, 16, 32],
        "clamp_min": 0.0,
        "affine_init": {"a": a0, "b": b0},
        "loss": "l1",
        "loss_weights": {"w_grad": 0.0, "w_smooth": 0.0, "w_sem": 0.0},
        "epoch": 3,
        "amp_enabled": False,
        "scaler_state": {},
        "val_subset_mae": 1.5,
        "splits_json": "/tmp/splits.json",
        "dataset": "dfc2019",
        "dem_source": None,
        "created": "2026-09-09T00:00:00+00:00",
    }
    p = tmp_path / "best.pt"
    torch.save(ckpt, p)
    return p


def test_train_style_checkpoint_loads_securely(tmp_path):
    """Regression: the real checkpoint contract loads with weights_only=True."""
    torch = pytest.importorskip("torch")
    from depthwizard.tifops import load_calib_net

    p = _train_style_checkpoint(tmp_path)
    model = load_calib_net(p, device="cpu")
    assert model.use_rgb is True
    assert model.epoch == 3
    assert model.affine_init == {"a": 2.0, "b": 4.0}
    assert model.widths == (8, 16, 32)


def test_pickle_payload_checkpoint_is_refused(tmp_path):
    """A planted checkpoint carrying arbitrary objects must raise, never
    execute (the RCE vector the audit flagged)."""
    torch = pytest.importorskip("torch")

    class ExplodingPayload:
        def __reduce__(self):  # would execute os/system calls if unpickled
            return eval, ("1+1",)

    p = tmp_path / "evil.pt"
    torch.save({"model_state": {"w": ExplodingPayload()}}, p)

    from depthwizard.tifops import load_calib_net

    with pytest.raises(Exception):
        load_calib_net(p, device="cpu")


def test_checkpoint_payload_validation_rejects_malformed():
    from depthwizard.tifops import validate_checkpoint_payload

    with pytest.raises(ValueError, match="must be a dict"):
        validate_checkpoint_payload(["not", "a", "dict"])

    with pytest.raises(ValueError, match="missing required keys"):
        validate_checkpoint_payload({"model_state": {}})

    with pytest.raises(ValueError, match="model_state"):
        validate_checkpoint_payload(
            {
                "model_state": {"layer.weight": np.zeros((2, 2))},
                "use_rgb": True,
                "widths": [8, 16, 32],
                "affine_init": {"a": 0.0, "b": 0.0},
                "epoch": 1,
            }
        )

    with pytest.raises(ValueError, match="affine_init"):
        validate_checkpoint_payload(
            {
                "model_state": {},
                "use_rgb": True,
                "widths": [8, 16, 32],
                "affine_init": {"a": 0.0},
                "epoch": 1,
            }
        )


def test_sha256_file_helper():
    from depthwizard.tifops import sha256_file

    p = Path(tmpdir := __import__("tempfile").mkdtemp()) / "f.bin"
    p.write_bytes(b"depthwizard")
    import hashlib

    assert sha256_file(p) == hashlib.sha256(b"depthwizard").hexdigest()
