"""Edge-aware weighted least-squares refinement ( Farbman et al. 2008
style energy, RGB + semantic gated, solved with preconditioned CG).

Energy:

    E(z) = SUM_i  c_i (z_i - d_i)^2                       (data fidelity)
         + lambda SUM_(i,j) w_ij (z_i - z_j)^2            (edge-aware smoothness)

    d   = AGL_raw (metric metres — the calibrated prediction; NEVER rescaled)
    c_i = data weight: confidence-aware (default 1.0 everywhere)
    w_ij= smoothness weight between 4-neighbours i,j:

              w_ij = exp(-(dRGB_ij / sigma_rgb)^2) * sem_sim_ij^gamma

          dRGB_ij  = L1 RGB distance between the two pixels (in [0,1] units)
          sem_sim  = Bhattacharyya similarity of predicted class probs
                     (1.0 when no semantic_probs are supplied)

First-order optimality gives the sparse SPD system

    (C + lambda * L) z = C d        C = diag(c),  L = graph Laplacian( w )

solved with conjugate gradients + Jacobi preconditioning. The solution
INTERPOLATES where confidence is low but is pinned to d where confidence is
high — it cannot shift the global height scale beyond what lambda*c permits
(the data term has unit total weight per pixel; pinned by unit tests:
constant input -> constant output bit-exactly).

Warm start: the caller may pass ``x0`` (e.g. the guided-filter result) —
a good approximation halves the CG iterations at zero quality cost.

All heavy operations are scipy-sparse / NumPy vectorized; no pixel loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WLSResult:
    z: np.ndarray  # refined AGL, float32 [H,W], NaN where input invalid
    iterations: int
    residual: float  # final relative residual
    converged: bool


def _neighbour_rgb_distance(rgb_u8: np.ndarray) -> dict:
    """L1 RGB distance (normalized [0,1]) to right and down neighbours.

    Returns {'right': [H,W], 'down': [H,W]} — entry (y,x) couples (y,x)
    with its right/down neighbour. Borders are +inf (disconnected edge).
    """
    rgb = np.asarray(rgb_u8)[..., :3].astype(np.float32) / 255.0
    inf = np.inf
    right = np.full(rgb.shape[:2], inf, dtype=np.float32)
    down = np.full(rgb.shape[:2], inf, dtype=np.float32)
    if rgb.shape[1] > 1:
        right[:, :-1] = np.abs(rgb[:, :-1] - rgb[:, 1:]).sum(axis=2)
    if rgb.shape[0] > 1:
        down[:-1, :] = np.abs(rgb[:-1] - rgb[1:]).sum(axis=2)
    return {"right": right, "down": down}


def _neighbour_weights(
    rgb_u8: np.ndarray,
    sigma_rgb: float,
    sem_probs: np.ndarray | None,
    semantic_edge_weight: float,
    semantic_hard_boundary: bool,
    valid: np.ndarray,
) -> dict:
    """Final smoothness weights for right/down neighbour pairs."""
    from .semantic import pairwise_semantic_similarity

    d_rgb = _neighbour_rgb_distance(rgb_u8)
    inv_s2 = 1.0 / max(sigma_rgb, 1e-6) ** 2

    w_right = np.exp(-(d_rgb["right"].astype(np.float64) ** 2) * inv_s2)
    w_down = np.exp(-(d_rgb["down"].astype(np.float64) ** 2) * inv_s2)

    if sem_probs is not None and semantic_edge_weight > 0:
        sim = pairwise_semantic_similarity(
            sem_probs, power=semantic_edge_weight, hard=semantic_hard_boundary
        )
        w_right *= sim["right"].astype(np.float64)
        w_down *= sim["down"].astype(np.float64)

    # pairs touching an invalid pixel carry no smoothness (they would couple
    # the solve to undefined data); border pairs are +inf distance -> w=0
    ok_r = np.zeros(valid.shape, dtype=bool)
    ok_d = np.zeros(valid.shape, dtype=bool)
    ok_r[:, :-1] = valid[:, :-1] & valid[:, 1:]
    ok_d[:-1, :] = valid[:-1] & valid[1:]
    w_right = np.where(ok_r, w_right, 0.0)
    w_down = np.where(ok_d, w_down, 0.0)
    return {"right": w_right, "down": w_down}


def wls_refine(
    signal: np.ndarray,
    rgb_u8: np.ndarray,
    lambda_: float = 1.0,
    sigma_rgb: float = 0.1,
    confidence: np.ndarray | None = None,
    sem_probs: np.ndarray | None = None,
    semantic_edge_weight: float = 0.0,
    semantic_hard_boundary: bool = False,
    tol: float = 1e-6,
    max_iter: int = 120,
    x0: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> WLSResult:
    """Solve the edge-aware WLS energy. See module docstring for the math.

    signal [H,W] metric AGL; rgb_u8 [H,W,3]; confidence [H,W] in [0,1]
    (None -> uniform); sem_probs [K,H,W] predicted probabilities or None.
    """
    signal = np.asarray(signal, dtype=np.float64)
    h, w = signal.shape
    if valid is None:
        valid = np.isfinite(signal)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(signal)
    n = h * w

    d = np.where(valid, signal, 0.0)
    if confidence is None:
        c = np.ones(n)
    else:
        c = np.clip(np.asarray(confidence, dtype=np.float64).ravel(), 0.0, 1.0)
    c = np.where(valid.ravel(), c, 0.0)  # invalid: no data term

    wts = _neighbour_weights(
        rgb_u8, sigma_rgb, sem_probs, semantic_edge_weight,
        semantic_hard_boundary, valid,
    )
    # pair arrays: right pairs couple (y, x)-(y, x+1) for x in [0, w-2];
    # down pairs couple (y, x)-(y+1, x) for y in [0, h-2]
    wr = (wts["right"][:, :-1] * lambda_).ravel()  # h*(w-1)
    wd = (wts["down"][:-1, :] * lambda_).ravel()  # (h-1)*w
    wr_2d = wr.reshape(h, w - 1)
    wd_2d = wd.reshape(h - 1, w)

    idx = np.arange(n).reshape(h, w)
    diag = c.copy()
    np.add.at(diag, idx[:, :-1].ravel(), wr)
    np.add.at(diag, idx[:, 1:].ravel(), wr)
    np.add.at(diag, idx[:-1].ravel(), wd)
    np.add.at(diag, idx[1:].ravel(), wd)

    def A_dot(v: np.ndarray) -> np.ndarray:
        # (C + lam*L) v = c*v + sum_j w_ij (v_i - v_j)
        #               = diag*v  -  sum_j w_ij v_j
        # (diag already includes c + all incident weights)
        out = diag * v
        v2 = v.reshape(h, w)
        acc = np.zeros((h, w))
        acc[:, :-1] += wr_2d * v2[:, 1:]  # right neighbour value
        acc[:, 1:] += wr_2d * v2[:, :-1]  # left neighbour value
        acc[:-1] += wd_2d * v2[1:]  # down neighbour value
        acc[1:] += wd_2d * v2[:-1]  # up neighbour value
        return out - acc.ravel()

    b = c * d.ravel()
    if x0 is not None:
        x = np.where(valid, x0, 0.0).astype(np.float64).ravel()
    else:
        x = d.ravel().copy()

    r = b - A_dot(x)
    b_norm = max(float(np.linalg.norm(b)), 1e-12)
    if float(np.linalg.norm(r)) / b_norm < tol:
        # exact solve already (e.g. constant signal or x0 == solution)
        z_out = np.where(valid, x.reshape(h, w), np.nan).astype(np.float32)
        return WLSResult(z=z_out, iterations=0, residual=0.0, converged=True)

    # Jacobi preconditioner
    inv_diag = np.where(diag > 1e-12, 1.0 / np.maximum(diag, 1e-12), 0.0)
    z = inv_diag * r
    p = z.copy()
    rz = float(r @ z)
    iters, converged = 0, False
    for iters in range(1, max_iter + 1):
        Ap = A_dot(p)
        denom = float(p @ Ap)
        if denom <= 1e-30:
            converged = float(np.linalg.norm(r)) / b_norm < tol
            break
        alpha = rz / denom
        x += alpha * p
        r -= alpha * Ap
        resid = float(np.linalg.norm(r)) / b_norm
        if resid < tol:
            converged = True
            break
        z = inv_diag * r
        rz_new = float(r @ z)
        beta = rz_new / max(rz, 1e-30)
        p = z + beta * p
        rz = rz_new

    final_resid = float(np.linalg.norm(b - A_dot(x))) / b_norm
    z_out = np.where(valid, x.reshape(h, w), np.nan).astype(np.float32)
    return WLSResult(z=z_out, iterations=iters, residual=final_resid,
                     converged=converged)
