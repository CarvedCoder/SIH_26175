"""Semantic-aware boundary protection for the smoothness term.

Inputs are PREDICTED semantic probabilities [K,H,W] in [0,1] (softmax over
the model's auxiliary semantic head — NEVER the GT mask at inference). The
project class order is the frozen ``datasets.semantics.PROJECT_CLASSES``:

    0 building, 1 vegetation, 2 road, 3 water, 4 ground, 5 other

Pairwise similarity between neighbouring pixels i, j uses the
Bhattacharyya coefficient of their class distributions:

    sim(i,j) = sum_k sqrt(p_ik * p_jk)

which is 1 for identical distributions and ~0 for confident-but-different
classes. This is strictly softer and more informative than comparing
argmax labels: an uncertain pixel (flat distribution) stays similar to
everything (permits smoothing), while a confident building pixel next to a
confident ground pixel gets sim ~ 0 (blocks smoothing — exactly the
roof->ground bleeding the pipeline must prevent).

Building boundaries get no special-cased code path: because class 0 vs any
other confident class drives sim -> 0, the generic mechanism already
protects building/ground, building/road and building/vegetation
transitions. The ``semantic_edge_weight`` exponent controls how sharply:
    w_sem = sim^gamma     (gamma = 1 -> linear; >1 -> sharper boundaries)
plus an optional hard cut (argmax differs -> weight exactly 0) for
experiments.
"""

from __future__ import annotations

import numpy as np

BUILDING_CLASS = 0  # project class index (datasets.semantics.PROJECT_CLASSES)


def _check_probs(sem_probs: np.ndarray) -> np.ndarray:
    p = np.asarray(sem_probs, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"semantic_probs must be [K,H,W], got {p.shape}")
    if p.min() < -1e-6 or p.max() > 1 + 1e-6:
        raise ValueError("semantic_probs must lie in [0,1] (softmax output)")
    return p


def pairwise_semantic_similarity(
    sem_probs: np.ndarray, power: float = 1.0, hard: bool = False
) -> dict:
    """Right-neighbour and down-neighbour semantic similarities.

    Returns dict with 'right', 'down': float32 [H,W] where element (y,x) is
    the similarity between pixel (y,x) and its right/down neighbour. Border
    entries (last column / last row) are 1.0 (no neighbour -> no penalty
    contribution; the caller's Laplacian simply ignores them).
    """
    p = _check_probs(sem_probs)
    k = p.shape[0]
    s = np.sqrt(np.clip(p, 0.0, 1.0))

    right = np.ones(p.shape[1:], dtype=np.float32)
    down = np.ones(p.shape[1:], dtype=np.float32)
    if p.shape[1] > 1:
        right[:, :-1] = np.clip((s[:, :, :-1] * s[:, :, 1:]).sum(axis=0), 0.0, 1.0)
    if p.shape[2] > 1:
        down[:-1, :] = np.clip((s[:, :-1, :] * s[:, 1:, :]).sum(axis=0), 0.0, 1.0)

    if hard:
        arg = p.argmax(axis=0)
        if p.shape[1] > 1:
            diff_r = arg[:, :-1] != arg[:, 1:]
            right[:, :-1] = np.where(diff_r, 0.0, right[:, :-1])
        if p.shape[2] > 1:
            diff_d = arg[:-1, :] != arg[1:, :]
            down[:-1, :] = np.where(diff_d, 0.0, down[:-1, :])

    if power != 1.0:
        right = np.clip(right, 0.0, 1.0) ** float(power)
        down = np.clip(down, 0.0, 1.0) ** float(power)
    return {"right": right.astype(np.float32), "down": down.astype(np.float32)}


def semantic_boundary_strength(sem_probs: np.ndarray) -> np.ndarray:
    """[H,W] in [0,1]: 1 - max neighbour similarity (diagnostic map).

    High values = confident class change between adjacent pixels = a
    semantic boundary. Used for visual diagnostics and (in evaluation) to
    locate boundary bands — never as deployed GT.
    """
    sim = pairwise_semantic_similarity(sem_probs)
    both = np.minimum(sim["right"], sim["down"])
    return (1.0 - both).astype(np.float32)


def building_mask_from_probs(sem_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Soft building mask argmax-free where possible: prob >= threshold."""
    p = _check_probs(sem_probs)
    return p[BUILDING_CLASS] >= threshold
