"""Synthetic terrain + image rendering for tools/generate_demo_bundles.py."""
import numpy as np

SIZE = 256


def _grad(y, x):
    gy, gx = np.gradient(y, x)
    return np.hypot(gy, gx)


def _blur(a, k=5):
    """Simple box blur (separable) to keep per-pixel slopes realistic."""
    def blur1d(x, axis):
        pad = k - 1
        p = np.pad(x, [(pad, pad) if ax == axis else (0, 0) for ax in range(x.ndim)], mode="edge")
        c = np.cumsum(p, axis=axis)
        out = (c[tuple(slice(k, None) if ax == axis else slice(None) for ax in range(x.ndim))]
               - c[tuple(slice(None, -k) if ax == axis else slice(None) for ax in range(x.ndim))]) / k
        sl = [slice(0, x.shape[ax]) for ax in range(x.ndim)]
        return out[tuple(sl)]
    return blur1d(blur1d(a, 0), 1)


def build_terrain(scene):
    """Return (dsm metres float32, normalised heights 0..1 float32)."""
    rng = np.random.default_rng(42)
    v, u = np.mgrid[0:SIZE, 0:SIZE]
    u = u / SIZE
    v = v / SIZE
    shape = scene["shape"]

    if shape == "ridge":
        h = (
            0.55 * np.exp(-(((u - 0.5) * 1.9 + (v - 0.5) * 0.9) ** 2) / 0.06)
            + 0.22 * np.exp(-(((u - 0.72) ** 2 + (v - 0.3) ** 2) / 0.02))
            + 0.10 * np.sin(u * 11) * np.sin(v * 9)
            + 0.06 * rng.normal(0, 1, (SIZE, SIZE))
        )
    elif shape == "valley":
        channel = np.abs(u - 0.55 + 0.25 * np.sin(v * 5.5))
        h = (
            0.62 * np.exp(-(channel ** 2) / 0.055)
            + 0.16 * np.exp(-(((u - 0.85) ** 2 + (v - 0.75) ** 2) / 0.03))
            + 0.07 * np.sin(u * 7) * np.sin(v * 13)
            + 0.05 * rng.normal(0, 1, (SIZE, SIZE))
        )
        h = 1.05 - h * 0.9  # invert: valley floor low, plateaus high
    else:  # quarry: benched terraces + pit
        terraces = np.floor(v * 5) / 5
        pit = 0.35 * np.exp(-(((u - 0.45) ** 2 + (v - 0.5) ** 2) / 0.035))
        h = 0.25 + 0.5 * terraces - pit + 0.05 * rng.normal(0, 1, (SIZE, SIZE))

    h = (h - h.min()) / (h.max() - h.min())
    h = np.clip(_blur(h, 7), 0, 1)
    h = (h - h.min()) / (h.max() - h.min())
    lo, hi = scene["min_elev"], scene["max_elev"]
    dsm = (lo + h * (hi - lo)).astype(np.float32)
    return dsm, h.astype(np.float32)


# ---- colour ramps -----------------------------------------------------------

def _ramp(t, stops):
    """t: HxW in [0,1]; stops: list of (pos, (r,g,b))."""
    t = np.clip(t, 0, 1)
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=float)
    out = np.zeros(t.shape + (3,), dtype=float)
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, cols[:, ch])
    return out


TERRAIN_STOPS = [
    (0.00, (38, 70, 60)),
    (0.25, (72, 122, 66)),
    (0.50, (150, 158, 82)),
    (0.72, (176, 128, 74)),
    (0.88, (150, 110, 100)),
    (1.00, (245, 245, 245)),
]
DEPTH_STOPS = [
    (0.00, (20, 24, 82)),
    (0.35, (24, 110, 160)),
    (0.65, (40, 180, 130)),
    (0.85, (190, 200, 50)),
    (1.00, (240, 230, 90)),
]
ERROR_STOPS = [
    (0.00, (30, 160, 70)),
    (0.35, (190, 200, 60)),
    (0.65, (230, 140, 40)),
    (1.00, (200, 40, 40)),
]


def render_images(scene, dsm, norm):
    """Return dict name -> HxWx3 uint8."""
    rng = np.random.default_rng(7)
    out = {}

    # fake RGB ortho: greenish-brown mottled texture correlated with terrain
    base = 90 + 70 * norm[..., None]
    mottle = rng.normal(0, 14, (SIZE, SIZE, 3))
    rgb = np.dstack([
        base[..., 0] * 0.75 + 40,
        base[..., 0] * 0.95,
        base[..., 0] * 0.55 + 25,
    ]) + mottle
    out["preview"] = np.clip(rgb, 0, 255).astype(np.uint8)
    out["rgb"] = out["preview"]

    # shaded relief texture
    gy, gx = np.gradient(norm)
    shade = np.clip(0.55 + 3.2 * gx - 3.2 * gy, 0.2, 1.25)
    relief = _ramp(norm, TERRAIN_STOPS) * shade[..., None] * 1.6
    out["texture"] = np.clip(relief, 0, 255).astype(np.uint8)
    out["dsm_texture"] = out["texture"]

    out["depth"] = np.clip(_ramp(norm, DEPTH_STOPS), 0, 255).astype(np.uint8)
    out["dsm"] = np.clip(_ramp(norm, TERRAIN_STOPS), 0, 255).astype(np.uint8)

    slope = np.degrees(np.arctan(_grad(dsm, scene["pixel_size"])))
    snorm = np.clip(slope / 40.0, 0, 1)
    out["slope"] = np.clip(_ramp(snorm, [(0, (20, 20, 20)), (1, (255, 255, 255))]), 0, 255).astype(np.uint8)

    # error map: low errors on gentle slopes, bigger on steep + random clusters
    err = scene["error_bias"] * (0.4 + 1.6 * snorm) * np.abs(rng.normal(1, 0.35, (SIZE, SIZE)))
    enorm = np.clip(err / (scene["error_bias"] * 4.0), 0, 1)
    out["error_map"] = np.clip(_ramp(enorm, ERROR_STOPS), 0, 255).astype(np.uint8)

    mini = out["dsm"].copy()
    mini[:3] = mini[-3:] = mini[:, :3] = mini[:, -3:] = 240
    out["minimap"] = mini

    ref = out["dsm"].astype(float)
    ref[..., 0] *= 0.55
    ref[..., 1] *= 0.85
    out["reference"] = np.clip(ref, 0, 255).astype(np.uint8)
    out["reference_preview"] = out["reference"]

    # route heatmap: red where too steep for a fire truck
    blocked = snorm > 0.55
    caution = (snorm > 0.35) & ~blocked
    heat = out["dsm"].astype(float) * 0.45
    heat[caution] = heat[caution] * 0.4 + np.array([230, 180, 40]) * 0.6
    heat[blocked] = np.array([200, 45, 45])
    out["heatmap"] = np.clip(heat, 0, 255).astype(np.uint8)

    passab = out["dsm"].astype(float) * 0.5
    passab[blocked] = passab[blocked] * 0.3 + np.array([180, 60, 60]) * 0.7
    out["passability"] = np.clip(passab, 0, 255).astype(np.uint8)

    # 16-bit grayscale heightmap PNG (color type 0) — what TerrainCanvas parses
    out["heightmap"] = np.clip(norm, 0, 1) * np.float64(65535)

    return out
