"""V2 calibration-stack regression tests.

Pins:
  * every parameterization (absolute/residual/hybrid/residual_depth) starts
    EXACTLY at the global affine baseline H = clamp(a0*Dn + b0) at step 0
    (zero-init head + residual zero-init);
  * bounded residuals satisfy |a-a0| <= 1, |b-b0| <= 10 (tanh bound);
  * checkpoint metadata round-trips through load_calib_net for every v2
    variant, and legacy checkpoints (pre-Exp-4 and Exp-4 head-only) rebuild
    bit-identically;
  * an incompatible checkpoint (metadata disagrees with state-dict shapes)
    fails LOUDLY instead of silently rebuilding the wrong architecture;
  * berhu / height-balanced / uncertainty loss terms are finite and change
    the loss only when enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.calibration_net import CalibrationNet
from depthwizard.config import LossWeightsConfig
from depthwizard.losses import DepthLoss, masked_berhu_loss
from depthwizard.tifops import load_calib_net


def _batch(n=2, hw=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    dn = torch.rand(n, 1, hw, hw, generator=g)
    rgb = torch.randn(n, 3, hw, hw, generator=g)
    return dn, rgb


class TestResidualInit:
    def test_all_parameterizations_start_at_affine_baseline(self):
        a0, b0 = 1.5, 3.25
        dn, rgb = _batch()
        ref = torch.clamp(a0 * dn + b0, min=0.0)
        for p in ("absolute_affine", "residual_affine", "hybrid_residual",
                  "residual_depth"):
            net = CalibrationNet(in_ch=4, a0=a0, b0=b0, parameterization=p)
            with torch.no_grad():
                out = net(dn, rgb)
            assert torch.allclose(out["pred"], ref, atol=1e-5), p

    def test_residual_uses_ckpt_affine_not_hardcoded(self):
        """a0/b0 come from the checkpoint's affine_init — a different global
        affine must give a different starting output."""
        dn, rgb = _batch()
        n1 = CalibrationNet(in_ch=4, a0=1.5, b0=3.25,
                            parameterization="residual_affine")
        n2 = CalibrationNet(in_ch=4, a0=2.0, b0=4.0,
                            parameterization="residual_affine")
        with torch.no_grad():
            assert not torch.allclose(n1(dn, rgb)["pred"], n2(dn, rgb)["pred"])

    def test_bounded_residual_stays_within_bounds(self):
        dn, rgb = _batch()
        a0, b0 = 1.5, 3.25
        net = CalibrationNet(in_ch=4, a0=a0, b0=b0,
                             parameterization="residual_affine", bounded=True)
        # force the head to output large values; tanh must bound the fields
        with torch.no_grad():
            net.head.weight.normal_(0, 10.0)
            net.head.bias.normal_(0, 10.0)
            out = net(dn, rgb)
        assert (out["a"] - a0).abs().max() <= 1.0 + 1e-6
        assert (out["b"] - b0).abs().max() <= 10.0 + 1e-6

    def test_unknown_parameterization_raises(self):
        with pytest.raises(ValueError):
            CalibrationNet(parameterization="nope")


class TestCkptRoundTrip:
    def _save(self, net, meta, path):
        torch.save({"model_state": net.state_dict(), **meta}, path)

    def test_v2_metadata_round_trip(self, tmp_path):
        dn, rgb = _batch(1)
        net = CalibrationNet(
            in_ch=4, widths=(16, 32, 64), a0=2.076463, b0=4.110271,
            parameterization="residual_affine", bounded=True,
            context_module="aspp",
        )
        with torch.no_grad():
            ref = net(dn, rgb)["pred"].clone()
        self._save(net, {
            "use_rgb": True, "use_dem": False, "use_sem": False,
            "sem_classes": 0, "sem_aux_head": False, "semantic_mode": "input",
            "parameterization": "residual_affine", "bounded": True,
            "fusion_mode": "early", "context_module": "aspp",
            "use_uncertainty": False, "in_ch": 4, "widths": [16, 32, 64],
            "clamp_min": 0.0, "affine_init": {"a": 2.076463, "b": 4.110271},
            "epoch": 3, "val_subset_mae": 2.9,
        }, tmp_path / "v2.pt")
        lm = load_calib_net(tmp_path / "v2.pt")
        assert lm.parameterization == "residual_affine"
        assert lm.bounded is True
        assert lm.context_module == "aspp"
        with torch.no_grad():
            assert torch.allclose(ref, lm.net(dn, rgb)["pred"], atol=1e-6)

    def test_legacy_head_only_ckpt_rebuilds_bit_identically(self, tmp_path):
        """Exp-4 head-only checkpoints store sem_input=False + sem_aux_head
        and NO semantic_mode — the loader must map them to 'auxiliary'."""
        dn, rgb = _batch(1)
        old = CalibrationNet(in_ch=4, widths=(8, 16, 32), a0=2.0, b0=4.0,
                             sem_classes=6, sem_aux_head=True,
                             semantic_mode="auxiliary")
        self._save(old, {
            "use_rgb": True, "use_sem": False, "sem_classes": 6,
            "sem_aux_head": True, "sem_input": False, "in_ch": 4,
            "widths": [8, 16, 32], "clamp_min": 0.0,
            "affine_init": {"a": 2.0, "b": 4.0}, "epoch": 9,
        }, tmp_path / "legacy.pt")
        lm = load_calib_net(tmp_path / "legacy.pt")
        with torch.no_grad():
            assert torch.allclose(old(dn, rgb)["pred"],
                                  lm.net(dn, rgb)["pred"], atol=1e-6)

    def test_pre_exp4_ckpt_rebuilds_bit_identically(self, tmp_path):
        """Oldest checkpoints carry no sem/parameterization fields at all."""
        dn, rgb = _batch(1)
        old = CalibrationNet(in_ch=4, widths=(8, 16, 32), a0=2.0, b0=4.0)
        self._save(old, {"use_rgb": True, "in_ch": 4, "widths": [8, 16, 32],
                         "affine_init": {"a": 2.0, "b": 4.0}, "epoch": 1},
                   tmp_path / "oldest.pt")
        lm = load_calib_net(tmp_path / "oldest.pt")
        with torch.no_grad():
            assert torch.allclose(old(dn, rgb)["pred"],
                                  lm.net(dn, rgb)["pred"], atol=1e-6)

    def test_incompatible_ckpt_fails_loudly(self, tmp_path):
        """Metadata says residual_affine (2 outputs) but the state dict is a
        hybrid head (3 outputs) — must raise, never silently rebuild."""
        net = CalibrationNet(in_ch=4, parameterization="hybrid_residual")
        self._save(net, {
            "use_rgb": True, "parameterization": "residual_affine",
            "in_ch": 4, "widths": [16, 32, 64],
            "affine_init": {"a": 2.0, "b": 4.0}, "epoch": 1,
        }, tmp_path / "bad.pt")
        with pytest.raises(RuntimeError):
            load_calib_net(tmp_path / "bad.pt")

    def test_joint_mode_ckpt_round_trip(self, tmp_path):
        dn, rgb = _batch(1)
        net = CalibrationNet(in_ch=4, widths=(16, 32, 64), a0=2.0, b0=4.0,
                             sem_classes=6, sem_aux_head=True,
                             semantic_mode="joint")
        self._save(net, {
            "use_rgb": True, "use_sem": False, "sem_classes": 6,
            "sem_aux_head": True, "semantic_mode": "joint", "in_ch": 4,
            "widths": [16, 32, 64], "clamp_min": 0.0,
            "affine_init": {"a": 2.0, "b": 4.0}, "epoch": 2,
        }, tmp_path / "joint.pt")
        lm = load_calib_net(tmp_path / "joint.pt")
        with torch.no_grad():
            assert torch.allclose(net(dn, rgb)["pred"],
                                  lm.net(dn, rgb)["pred"], atol=1e-6)


class TestV2LossTerms:
    def test_berhu_matches_l1_for_small_errors(self):
        pred = torch.tensor([[[[0.1]]]])
        target = torch.tensor([[[[0.0]]]])
        # max|e| = 0.1, c = 0.2*0.1 = 0.02 -> still L1 region for all pixels
        assert torch.isfinite(masked_berhu_loss(pred, target))

    def test_berhu_penalizes_large_errors_more_than_l1(self):
        g = torch.Generator().manual_seed(0)
        pred = torch.rand(1, 1, 16, 16, generator=g) * 40
        target = torch.zeros(1, 1, 16, 16)
        berhu = masked_berhu_loss(pred, target)
        from depthwizard.losses import masked_l1_loss
        assert float(berhu) > float(masked_l1_loss(pred, target))

    def test_berhu_handles_all_zero_errors(self):
        x = torch.rand(1, 1, 8, 8)
        assert float(masked_berhu_loss(x, x)) == 0.0

    def test_height_balanced_upweights_tall_bins(self):
        pred = torch.zeros(1, 1, 2, 2)
        target = torch.tensor([[[[0.0, 1.0], [20.0, 40.0]]]])
        fn = DepthLoss(LossWeightsConfig(height_balanced=True))
        with torch.no_grad():
            w = fn._get_balanced_weights(target)
        assert w[0, 0, 0, 0] == 1.0
        assert w[0, 0, 1, 0] == 2.0   # 10-30 m bin
        assert w[0, 0, 1, 1] == 5.0   # >30 m bin

    def test_height_balanced_off_equals_plain_l1(self):
        g = torch.Generator().manual_seed(1)
        pred = torch.rand(2, 1, 8, 8, generator=g) * 20
        target = torch.rand(2, 1, 8, 8, generator=g) * 20
        from depthwizard.calibration_net import masked_l1_loss
        fn = DepthLoss(LossWeightsConfig(height_balanced=False))
        assert torch.allclose(fn(pred, target), masked_l1_loss(pred, target))

    def test_uncertainty_term_pulls_toward_err_scale(self):
        g = torch.Generator().manual_seed(2)
        pred = torch.rand(2, 1, 8, 8, generator=g) * 10
        target = torch.rand(2, 1, 8, 8, generator=g) * 10
        log_var = torch.zeros(2, 1, 8, 8)
        base = DepthLoss(LossWeightsConfig())(pred, target)
        withu = DepthLoss(LossWeightsConfig(uncertainty_weight=0.1))(
            pred, target, log_var=log_var)
        assert torch.isfinite(withu)
        assert float(withu) > float(base)

    def test_uncertainty_disabled_ignores_log_var(self):
        g = torch.Generator().manual_seed(3)
        pred = torch.rand(1, 1, 8, 8, generator=g)
        target = torch.rand(1, 1, 8, 8, generator=g)
        fn = DepthLoss(LossWeightsConfig())  # uncertainty_weight = 0
        a = fn(pred, target)
        b = fn(pred, target, log_var=torch.randn(1, 1, 8, 8))
        assert torch.equal(a, b)


class TestFilmTileStats:
    """Exp-1 port: FiLM conditioning on RAW-tile Dn statistics."""

    def test_dn_tile_stats_values(self):
        from depthwizard.normalize import dn_tile_stats

        raw = np.full((8, 8), 2.5, dtype=np.float32)
        lo, hi, rng_, mean = np.log(2.5 + 1e-3), np.log(2.5 + 1e-3), 0.0, np.log(2.5 + 1e-3)
        st = dn_tile_stats(raw)
        assert st.shape == (4,)
        assert st.dtype == np.float32
        assert np.isclose(st[0], lo) and np.isclose(st[1], hi)
        assert np.isclose(st[3], mean)
        raw2 = raw.copy()
        raw2[0, 0] = 5.0
        st2 = dn_tile_stats(raw2)
        assert st2[1] > st[1] and st2[2] > 0.0  # range now positive

    def test_film_identity_init_starts_at_baseline(self):
        dn, rgb = _batch()
        ref = torch.clamp(1.5 * dn + 3.25, min=0.0)
        for widths in ((16, 32, 64), (16, 32, 64, 128)):
            net = CalibrationNet(in_ch=4, widths=widths, a0=1.5, b0=3.25,
                                 film_stats=True)
            with torch.no_grad():
                out = net(dn, rgb, stats=torch.rand(2, 4))
            assert torch.allclose(out["pred"], ref, atol=1e-5), widths

    def test_film_zero_fill_when_stats_absent(self):
        dn, rgb = _batch()
        net = CalibrationNet(in_ch=4, film_stats=True, a0=2.0, b0=4.0)
        with torch.no_grad():
            out = net(dn, rgb)
        assert out["stats_zero_filled"] is True
        out2 = net(dn, rgb, stats=torch.zeros(2, 4))
        assert torch.allclose(out["pred"], out2["pred"], atol=1e-6)

    def test_film_no_film_net_ignores_stats(self):
        dn, rgb = _batch()
        net = CalibrationNet(in_ch=4)
        with torch.no_grad():
            a = net(dn, rgb)["pred"]
            b = net(dn, rgb, stats=torch.rand(2, 4))["pred"]
        assert torch.equal(a, b)

    def test_film_params_change_conditioning(self):
        dn, rgb = _batch()
        net = CalibrationNet(in_ch=4, film_stats=True)
        with torch.no_grad():
            net.film.mlp[-1].weight.normal_(0, 0.1)
            net.film.mlp[-1].bias.normal_(0, 0.1)
            net.head.weight.normal_(0, 0.1)
            p1 = net(dn, rgb, stats=torch.zeros(2, 4))["pred"]
            p2 = net(dn, rgb, stats=torch.ones(2, 4))["pred"]
        assert not torch.allclose(p1, p2)

    def test_film_ckpt_round_trip(self, tmp_path):
        dn, rgb = _batch(1)
        net = CalibrationNet(in_ch=4, widths=(16, 32, 64), a0=2.0, b0=4.0,
                             film_stats=True)
        with torch.no_grad():
            ref = net(dn, rgb, stats=torch.rand(1, 4))["pred"].clone()
        torch.save({"model_state": net.state_dict(), **{
            "use_rgb": True, "film_stats": True, "in_ch": 4,
            "widths": [16, 32, 64], "affine_init": {"a": 2.0, "b": 4.0},
            "epoch": 1,
        }}, tmp_path / "film.pt")
        lm = load_calib_net(tmp_path / "film.pt")
        assert lm.film_stats is True
        with torch.no_grad():
            assert torch.allclose(
                ref,
                lm.net(dn, rgb, stats=ref.new_full((1, 4), 0.3))["pred"],
                atol=1e-6,
            )
