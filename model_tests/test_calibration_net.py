"""Tests for depthwizard.calibration_net — Phase 2 model contract.

Pins the properties that make the 'baseline-as-special-case' story true:
exact affine init, output clamping, shape safety on odd sizes, gradient
flow, and loss semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.calibration_net import (CalibrationNet, masked_huber_loss,
                                         masked_l1_loss)


def _dn(n=1, h=64, w=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, h, w, generator=g)


def test_output_shapes_and_clamp():
    net = CalibrationNet(in_ch=1, a0=2.0, b0=4.0, clamp_min=0.0)
    dn = _dn()
    out = net(dn)
    assert out["pred"].shape == dn.shape
    assert out["a"].shape == dn.shape and out["b"].shape == dn.shape
    assert float(out["pred"].min()) >= 0.0          # clamp is active


def test_init_reproduces_global_affine_exactly():
    """Zero-weight head + biases (a0, b0) => pred == clamp(a0*Dn + b0).
    THE property that makes the Phase-1 baseline a special case."""
    a0, b0 = 2.076463, 4.110271
    net = CalibrationNet(in_ch=1, a0=a0, b0=b0, clamp_min=0.0)
    dn = _dn(seed=3)
    out = net(dn)
    expected = torch.clamp(a0 * dn + b0, min=0.0)
    assert torch.allclose(out["pred"], expected, atol=1e-5), \
        "init must reproduce the frozen global affine bit-for-bit"


def test_odd_input_size_padded_correctly():
    net = CalibrationNet(in_ch=1, a0=1.0, b0=2.0)
    dn = _dn(h=62, w=70, seed=1)                    # not divisible by 4
    out = net(dn)
    assert out["pred"].shape == dn.shape
    expected = torch.clamp(1.0 * dn + 2.0, min=0.0)
    assert torch.allclose(out["pred"], expected, atol=1e-5)


def test_rgb_mode_forward():
    net = CalibrationNet(in_ch=4, a0=1.5, b0=3.0)
    dn, rgb = _dn(seed=2), torch.randn(1, 3, 64, 64)
    out = net(dn, rgb)
    assert out["pred"].shape == dn.shape
    assert torch.allclose(out["pred"], torch.clamp(1.5 * dn + 3.0, min=0.0),
                          atol=1e-5)                # init ignores RGB weights (zero)


def test_dem_mode_forward_in_ch_5_starts_at_affine_baseline():
    """Method-D challenger variant: in_ch=5 (Dn+RGB+DEM). The zero-weight
    head + a0,b0 bias init means it starts at EXACTLY the flagship's
    numbers — the same property that makes the Phase-1 baseline a special
    case (see worklog Section 1, "Affine init"). Ablation discipline: any
    val improvement is attributable to the DEM conditioning channel, not
    to architecture luck."""
    a0, b0 = 2.076463, 4.110271     # flagship's exact affine init
    net = CalibrationNet(in_ch=5, a0=a0, b0=b0, clamp_min=0.0)
    dn = _dn(seed=11)
    rgb = torch.randn(1, 3, 64, 64)
    dem = torch.randn(1, 1, 64, 64)
    out = net(dn, rgb, dem)
    assert out["pred"].shape == dn.shape
    expected = torch.clamp(a0 * dn + b0, min=0.0)
    assert torch.allclose(out["pred"], expected, atol=1e-5), \
        "in_ch=5 (Dn+RGB+DEM) must start exactly at the affine baseline"

    # Unbatched 3-D DEM input must auto-batch identically to the RGB path
    # (test_unbatched_3d_input_auto_batched pins the same property for RGB).
    dn3 = _dn(h=64, w=64, seed=12)[0]            # [1,64,64]
    rgb3 = torch.randn(3, 64, 64)
    dem3 = torch.randn(1, 64, 64)
    out3 = net(dn3, rgb3, dem3)
    assert out3["pred"].shape == (1, 1, 64, 64)
    expected3 = torch.clamp(a0 * dn3[None] + b0, min=0.0)
    assert torch.allclose(out3["pred"], expected3, atol=1e-5)


def test_gradient_flows_to_head_and_body():
    net = CalibrationNet(in_ch=1, a0=2.0, b0=4.0)
    dn, target = _dn(seed=4), _dn(seed=5) * 10.0
    loss = masked_l1_loss(net(dn)["pred"], target)
    loss.backward()
    assert net.head.weight.grad is not None and net.head.weight.grad.abs().sum() > 0
    assert net.head.bias.grad is not None and net.head.bias.grad.abs().sum() > 0
    # deeper layers get zero grad while head.weight == 0 — expected at step 0
    assert net.enc1[0].weight.grad is not None


def test_overfits_tiny_batch():
    """Adam steps on one fixed batch must reduce the loss decisively.
    NOTE: zero-weight head init means only the head moves at first, and the
    bias must physically travel from b0 to the truth — hence lr 5e-2 and 120
    steps here. (Real training: thousands of steps at lr 1e-3, fine.)"""
    torch.manual_seed(0)
    net = CalibrationNet(in_ch=1, a0=2.0, b0=4.0)
    opt = torch.optim.Adam(net.parameters(), lr=5e-2)
    dn = _dn(seed=6)
    net.clamp_min = None                       # truth is reachable exactly
    target = 5.0 * dn
    first = None
    for i in range(150):
        opt.zero_grad()
        loss = masked_l1_loss(net(dn)["pred"], target)
        if first is None:
            first = float(loss)
        loss.backward()
        opt.step()
    # 70%+ reduction proves useful motion away from the affine init
    assert float(loss) < first * 0.3, f"loss {float(loss):.4f} vs init {first:.4f}"


def test_masked_l1_hand_value():
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[1.0, float("nan")], [5.0, 8.0]]]])
    # valid pixels: (1-1)=0, (3-5)=-2, (4-8)=-4 -> (0+2+4)/3 = 2.0
    assert abs(float(masked_l1_loss(pred, target)) - 2.0) < 1e-6


def test_masked_huber_quadratic_below_delta():
    """Huber is 0.5*e^2 below delta (L2-ish), delta*(|e|-delta/2) above."""
    pred = torch.tensor([[[[0.0, 0.0, 0.0]]]])
    target = torch.tensor([[[[1.0, -1.0, 7.0]]]])
    hb = float(masked_huber_loss(pred, target, delta=5.0))
    expected = (0.5 * 1.0 + 0.5 * 1.0 + 5.0 * (7.0 - 2.5)) / 3.0
    assert abs(hb - expected) < 1e-6


def test_all_invalid_target_gives_zero_loss():
    pred = torch.ones(1, 1, 4, 4)
    target = torch.full((1, 1, 4, 4), float("nan"))
    assert float(masked_l1_loss(pred, target)) == 0.0


def test_unbatched_3d_input_auto_batched():
    """REGRESSION: dataset samples are [C,H,W] (3D). dim=1 cat on 3D used to
    concat HEIGHT instead of channels (32,256,128) -> conv crash. The model
    must auto-batch 3D inputs and always return [N,1,H,W]."""
    net = CalibrationNet(in_ch=1, a0=2.0, b0=4.0)
    dn3 = _dn(h=64, w=64, seed=9)[0]            # [1,64,64]
    out = net(dn3)
    assert out["pred"].shape == (1, 1, 64, 64)
    expected = torch.clamp(2.0 * dn3[None] + 4.0, min=0.0)
    assert torch.allclose(out["pred"], expected, atol=1e-5)
    net4 = CalibrationNet(in_ch=4, a0=2.0, b0=4.0)
    out4 = net4(dn3, torch.randn(3, 64, 64))     # unbatched RGB too
    assert out4["pred"].shape == (1, 1, 64, 64)
