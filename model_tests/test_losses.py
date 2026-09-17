"""Tests for depthwizard.losses (Phase 4) + CalibrationNet semantics.

Pins:
  * DEFAULT WEIGHTS = EXACT PRE-PHASE-4 BEHAVIOR (the minimal-first rule):
    DepthLoss with w_grad=w_smooth=w_sem=0 must equal masked_l1_loss /
    masked_huber_loss bit-for-bit;
  * each extra term is finite, non-negative-ish, and actually different from
    the main loss;
  * masked_semantic_ce ignores the ignore mask;
  * CalibrationNet: in_ch derivation, channel order, ZERO-FILL when sem is
    absent (risk R8), aux-head state-dict compatibility with old ckpts;
  * collate_dict_none_safe: None pass-through + nested meta dicts (the
    torch>=2.13 fix).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.calibration_net import (CalibrationNet, derive_in_ch,
                                         masked_huber_loss, masked_l1_loss)
from depthwizard.losses import (DepthLoss, LossConfig, build_loss,
                                edge_aware_smoothness, gradient_loss,
                                masked_semantic_ce)


def _toy_batch(n=2, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(n, 1, h, w, generator=g) * 20
    target = torch.rand(n, 1, h, w, generator=g) * 20
    rgb = torch.randn(n, 3, h, w, generator=g)
    return pred, target, rgb


class TestDefaultEquivalence:
    def test_default_weights_equal_frozen_l1(self):
        pred, target, _ = _toy_batch()
        got = DepthLoss(LossConfig(main="l1"))(pred, target)
        ref = masked_l1_loss(pred, target)
        assert torch.equal(got, ref)

    def test_default_weights_equal_frozen_huber(self):
        pred, target, _ = _toy_batch()
        got = DepthLoss(LossConfig(main="huber", huber_delta=5.0))(pred, target)
        ref = masked_huber_loss(pred, target, 5.0)
        assert torch.equal(got, ref)

    def test_build_loss_factory(self):
        fn = build_loss(LossConfig())
        assert isinstance(fn, DepthLoss)


class TestExtraTerms:
    def test_gradient_loss_zero_for_identical(self):
        pred, _, _ = _toy_batch()
        assert float(gradient_loss(pred, pred)) == 0.0

    def test_gradient_loss_positive_for_different(self):
        pred, target, _ = _toy_batch()
        assert float(gradient_loss(pred, target)) > 0.0

    def test_w_grad_adds_term(self):
        pred, target, _ = _toy_batch()
        base = DepthLoss(LossConfig())(pred, target)
        withg = DepthLoss(LossConfig(w_grad=0.5))(pred, target)
        assert float(withg) > float(base)

    def test_w_smooth_requires_rgb_but_is_finite(self):
        pred, target, rgb = _toy_batch()
        s = edge_aware_smoothness(pred, rgb)
        assert torch.isfinite(s)
        withs = DepthLoss(LossConfig(w_smooth=0.3))(pred, target, rgb=rgb)
        assert float(withs) > float(DepthLoss(LossConfig())(pred, target))

    def test_sem_ce_ignores_masked_pixels(self):
        n, k, h, w = 2, 6, 8, 8
        logits = torch.randn(n, k, h, w)
        onehot = torch.zeros(n, k, h, w)
        onehot[:, 0] = 1.0
        ignore = torch.zeros(n, 1, h, w, dtype=torch.bool)
        ignore[:, 0, :4] = True                 # ignored rows
        # perfect logits at non-ignored pixels -> tiny CE
        logits[:, 0] += 10.0
        ce = masked_semantic_ce(logits, onehot, ignore)
        assert float(ce) < 1.0

    def test_sem_ce_all_ignored_returns_zero(self):
        logits = torch.randn(1, 6, 4, 4)
        onehot = torch.zeros(1, 6, 4, 4)
        ignore = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        assert float(masked_semantic_ce(logits, onehot, ignore)) == 0.0

    def test_last_parts_breakdown(self):
        pred, target, rgb = _toy_batch()
        fn = DepthLoss(LossConfig(w_grad=0.5, w_smooth=0.1))
        _ = fn(pred, target, rgb=rgb)
        assert set(fn.last_parts) == {"main", "grad", "smooth"}


class TestCalibrationNetSemantics:
    def test_derive_in_ch(self):
        assert derive_in_ch() == 1
        assert derive_in_ch(use_rgb=True) == 4
        assert derive_in_ch(use_dem=True) == 2
        assert derive_in_ch(use_rgb=True, use_dem=True) == 5
        assert derive_in_ch(use_sem=True) == 7
        assert derive_in_ch(use_rgb=True, use_sem=True) == 10
        assert derive_in_ch(use_rgb=True, use_sem=True, use_dem=True) == 11

    def test_sem_channels_change_output_but_start_at_baseline(self):
        """Exp 4 variant: sem input flows through; the zero-weight head +
        a0,b0 init means ANY variant starts at the EXACT affine baseline."""
        dn = torch.rand(2, 1, 16, 16)
        sem = torch.zeros(2, 6, 16, 16)
        sem[:, 3] = 1.0
        net = CalibrationNet(in_ch=derive_in_ch(use_sem=True),
                             sem_classes=6, a0=2.0, b0=4.0)
        out = net(dn, None, None, sem)
        # affine exactness: H = clamp(2*Dn + 4)
        ref = torch.clamp(2.0 * dn + 4.0, min=0.0)
        assert torch.allclose(out["pred"], ref, atol=1e-4)
        assert out["sem_zero_filled"] is False

    def test_sem_zero_fill_when_absent(self):
        """R8 contract: a sem-expecting net stays evaluable without GT
        semantics — channels ZERO-FILLED, flag set."""
        dn = torch.rand(1, 1, 16, 16)
        net = CalibrationNet(in_ch=derive_in_ch(use_sem=True),
                             sem_classes=6, a0=2.0, b0=4.0)
        out = net(dn)
        assert out["sem_zero_filled"] is True
        # equals the same net fed explicit zeros
        out2 = net(dn, None, None, torch.zeros(1, 6, 16, 16))
        assert torch.allclose(out["pred"], out2["pred"], atol=1e-6)

    def test_unbatched_sem_input(self):
        dn = torch.rand(1, 16, 16)
        sem = torch.zeros(6, 16, 16)
        sem[0] = 1.0
        net = CalibrationNet(in_ch=7, sem_classes=6, a0=2.0, b0=4.0)
        out = net(dn, None, None, sem)
        assert out["pred"].shape == (1, 1, 16, 16)

    def test_aux_sem_head_shapes(self):
        net = CalibrationNet(in_ch=7, sem_classes=6, sem_aux_head=True,
                             a0=1.0, b0=1.0)
        dn = torch.rand(2, 1, 16, 16)
        sem = torch.zeros(2, 6, 16, 16)
        out = net(dn, None, None, sem)
        assert out["sem_logits"].shape == (2, 6, 16, 16)

    def test_state_dict_compatible_with_old_checkpoints(self):
        """A checkpoint saved by the PRE-Exp-4 class (no sem head) must load
        into the new class bit-identically (sem_aux_head=False default)."""
        old_net = CalibrationNet(in_ch=4, widths=(8, 16, 32),
                                 a0=2.0, b0=4.0)
        sd = old_net.state_dict()
        new_net = CalibrationNet(in_ch=4, widths=(8, 16, 32),
                                 a0=2.0, b0=4.0)
        new_net.load_state_dict(sd)          # must not raise
        dn = torch.rand(1, 1, 16, 16)
        rgb = torch.randn(1, 3, 16, 16)
        assert torch.allclose(old_net(dn, rgb)["pred"],
                              new_net(dn, rgb)["pred"])


class TestCollateNoneSafe:
    def test_none_layers_pass_through(self):
        from depthwizard.datasets.base import collate_dict_none_safe
        s1 = {"dn": torch.ones(1, 4, 4), "dem": None,
              "meta": {"stem": "a", "dem_tag": None, "dataset": "gamus"}}
        s2 = {"dn": torch.zeros(1, 4, 4), "dem": None,
              "meta": {"stem": "b", "dem_tag": None, "dataset": "gamus"}}
        b = collate_dict_none_safe([s1, s2])
        assert b["dem"] is None
        assert b["dn"].shape == (2, 1, 4, 4)
        # meta stays a per-sample LIST (bookkeeping, never tensor-collated)
        assert isinstance(b["meta"], list) and len(b["meta"]) == 2
        assert b["meta"][0]["stem"] == "a"
        assert b["meta"][1]["dem_tag"] is None

    def test_heterogeneous_meta_keys(self):
        """Mixed datasets: GAMUS meta (split/height_semantics/units) and
        DFC meta (dem_tag) have DIFFERENT key sets — must not crash."""
        from depthwizard.datasets.base import collate_dict_none_safe
        s1 = {"dn": torch.ones(1, 4, 4),
              "meta": {"sample_id": "DC_01_25", "split": "train",
                       "height_semantics": "nDSM", "units": "ASSUMED m"}}
        s2 = {"dn": torch.zeros(1, 4, 4),
              "meta": {"sample_id": "JAX_004_006", "dem_tag": None}}
        b = collate_dict_none_safe([s1, s2])
        assert b["dn"].shape == (2, 1, 4, 4)
        assert b["meta"][0]["split"] == "train"
        assert "split" not in b["meta"][1]

    def test_partial_key_presence_all_none(self):
        """A key present in only SOME samples, None-compatible -> None."""
        from depthwizard.datasets.base import collate_dict_none_safe
        s1 = {"dn": torch.ones(1, 4, 4), "sem_onehot": torch.zeros(6, 4, 4)}
        s2 = {"dn": torch.zeros(1, 4, 4), "sem_onehot": torch.ones(6, 4, 4)}
        b = collate_dict_none_safe([s1, s2])
        assert b["sem_onehot"].shape == (2, 6, 4, 4)

    def test_mixed_none_and_tensor_raises(self):
        from depthwizard.datasets.base import collate_dict_none_safe
        s1 = {"dn": torch.ones(1, 4, 4)}
        s2 = {"dn": None}
        with pytest.raises(TypeError):
            collate_dict_none_safe([s1, s2])

    def test_sem_layers_collate(self):
        from depthwizard.datasets.base import collate_dict_none_safe
        s1 = {"sem_onehot": torch.zeros(6, 4, 4), "sem_ignore": torch.zeros(1, 4, 4, dtype=torch.bool)}
        s2 = {"sem_onehot": torch.ones(6, 4, 4), "sem_ignore": torch.ones(1, 4, 4, dtype=torch.bool)}
        b = collate_dict_none_safe([s1, s2])
        assert b["sem_onehot"].shape == (2, 6, 4, 4)
        assert b["sem_ignore"].dtype == torch.bool
