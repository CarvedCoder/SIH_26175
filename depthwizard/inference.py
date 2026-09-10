"""Image -> AGL/DSM inference pipeline (the ONE code path for CLI + service).

Who uses this module:
    * the ``infer`` CLI command (``python model.py infer ...``)
    * the FastAPI service (``/service/api.py``) that backs the webapp
    * the scene payload builder that feeds the Three.js viewer

Because every consumer imports the same functions, the webapp can never
drift from the certified CLI forward pass.

Flow:
    read image (rasterio; PNG/JPG/TIF, georef state reported honestly)
      -> resolve RAW Dn (explicit .npy | depth cache | LIVE Depth Anything
         V2, one 1024 tile per forward — the training granularity)
      -> flagship CalibrationNet  H = clamp(a(x,y)*Dn + b(x,y), 0)
         Dn min-max normalized at the granularity the net consumes
         (per 1024 tile in tiles mode — the training contract)
         modes: crop (center 1024) | resize (letterboxed 1024, aspect
         preserved) | tiles (any size, edge-pad, per-tile Dn)
      -> optional Track-2 anchoring  DSM = AGL + ground  [ANCHORED (not learned)]
      -> outputs: dsm.npy (+ dsm.tif when georeferenced) (+ _anchored), preview PNG
      -> scene payload (downsampled grid + RGB PNG + stats) for the webapp

This is a DEMONSTRATION path, never a citable evaluation (numbers come only
from the ``evaluate`` command).
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio import CRS, Affine

from .anchoring import ANCHORED_LABEL, AnchorResult, anchor
from .normalize import minmax_normalize

TILE = 1024  # training tile size; crop/resize/tiles all target it
MAX_GRID_SIDE = 512  # webapp mesh grid cap (stride-downsampled)


# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------


def read_image(path: Path | str) -> tuple[np.ndarray, dict]:
    """Read any raster rasterio can open -> (rgb_u8 [H,W,3], profile).

    Single-band inputs are replicated to 3 channels (the net's contract).
    Non-uint8 dtypes produce a loud warning — the training contract is
    uint8/255 and silently rescaling would be a domain shift.
    """
    import rasterio

    with rasterio.open(path) as ds:
        bands = [1, 2, 3] if ds.count >= 3 else [1]
        raw = ds.read(bands)
        arr = np.stack([raw[0]] * 3) if ds.count < 3 else raw
        profile = ds.profile.copy()
        # keep crs/transform as first-class fields for the anchoring path
        profile["_crs_obj"] = ds.crs
        profile["_transform_obj"] = ds.transform
        if ds.dtypes[0] != "uint8":
            print(
                f"[!] input dtype={ds.dtypes[0]} — training contract is "
                f"uint8/255. Verify scaling!"
            )
    rgb = np.ascontiguousarray(arr.transpose(1, 2, 0), dtype=np.uint8)
    return rgb, profile


def georef_state(profile: dict) -> tuple[bool, CRS | None, Affine | None]:
    """(is_georeferenced, crs, transform) — CRS None means pixel-space only."""
    crs = profile.get("_crs_obj")
    tf = profile.get("_transform_obj")
    return (crs is not None, crs, tf)


# ---------------------------------------------------------------------------
# Dn resolution: explicit file -> cache -> live backbone
# ---------------------------------------------------------------------------


def load_raw_dn_file(path: Path | str) -> np.ndarray:
    """Raw relative-depth .npy -> [H,W] float32 (NOT normalized).

    Normalization is deliberately NOT done here: predict() normalizes at the
    granularity the network actually consumes (per 1024 tile in tiles mode,
    whole input otherwise) — mirroring the training cache recipe.
    """
    raw = np.load(path)
    if raw.ndim != 2:
        raise ValueError(f"Dn array must be 2-D (H,W), got shape {raw.shape}")
    return raw.astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy/PIL — unit-testable without torch)
# ---------------------------------------------------------------------------


def resize_to_tile(arr: np.ndarray, size: int = TILE) -> np.ndarray:
    """Bilinear resize of [H,W] or [H,W,C] to (size, size). PIL-based,
    deterministic, works for float32 single-band and uint8 RGB alike.

    NOTE: this SQUEEZES the aspect ratio. Inference uses letterbox_to_tile
    instead; this helper remains for utilities/tests that genuinely want a
    square output."""
    from PIL import Image

    h, w = arr.shape[:2]
    if (h, w) == (size, size):
        return arr
    if arr.ndim == 2:
        im = (
            Image.fromarray(arr.astype(np.float32))
            if arr.dtype == np.float32
            else Image.fromarray(arr)
        )
        out = im.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(out, dtype=np.float32)
    im = Image.fromarray(arr)
    return np.asarray(
        im.resize((size, size), Image.Resampling.BILINEAR), dtype=arr.dtype
    )


def _resize_bilinear(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize [H,W] or [H,W,C] to (height, width), dtype-preserving
    for uint8, float32 for single-band float arrays."""
    from PIL import Image

    if (arr.shape[0], arr.shape[1]) == (height, width):
        return arr
    if arr.ndim == 2:
        im = (
            Image.fromarray(arr.astype(np.float32))
            if arr.dtype == np.float32
            else Image.fromarray(arr)
        )
        out = im.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(out, dtype=np.float32)
    im = Image.fromarray(arr)
    return np.asarray(
        im.resize((width, height), Image.Resampling.BILINEAR), dtype=arr.dtype
    )


def letterbox_placement(h: int, w: int, size: int = TILE) -> tuple[int, int, int, int]:
    """Aspect-preserving placement of a (h, w) image inside a (size, size)
    canvas: (y0, x0, h2, w2) — top-left offset and scaled size of the real
    content. The content is scaled so its LARGEST side equals `size`
    (matching the old resize-mode scale, without the distortion)."""
    scale = size / max(h, w)
    h2 = max(1, round(h * scale))
    w2 = max(1, round(w * scale))
    y0 = (size - h2) // 2
    x0 = (size - w2) // 2
    return y0, x0, h2, w2


def letterbox_to_tile(
    arr: np.ndarray, size: int = TILE
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Fit [H,W] or [H,W,C] into a (size, size) canvas WITHOUT distortion.

    The content is bilinear-scaled to fit, centered, and the remaining
    margin is filled by edge replication (never invents new extrema, so
    normalizing over the canvas == normalizing over the real content).

    Returns (canvas, (y0, x0, h2, w2)) so the forward output can be cropped
    back to exactly the content region before it is resized to the source
    grid.
    """
    h, w = arr.shape[:2]
    y0, x0, h2, w2 = letterbox_placement(h, w, size)
    if arr.ndim == 2:
        content = _resize_bilinear(arr, h2, w2)
        pad = ((y0, size - h2 - y0), (x0, size - w2 - x0))
    else:
        content = _resize_bilinear(arr, h2, w2)
        pad = ((y0, size - h2 - y0), (x0, size - w2 - x0), (0, 0))
    canvas = np.pad(content, pad, mode="edge")
    return np.ascontiguousarray(canvas), (y0, x0, h2, w2)


def normalize_dn_per_tile(raw: np.ndarray, tile: int = TILE) -> np.ndarray:
    """Per-tile min-max normalization of RAW relative depth — the training
    contract (dataset.py normalizes each cached 1024 tile independently).

    The raw map is edge-padded to the tile grid FIRST, then each padded
    tile is normalized in isolation and the padding is cropped away, so
    inference tiles see exactly the Dn distribution training tiles saw:
    full [0,1] range per tile, zero cross-tile magnitude coupling.
    """
    h, w = raw.shape
    ny, nx, hp, wp = tile_bounds(h, w, tile)
    padded = np.pad(
        raw.astype(np.float32, copy=False), ((0, hp - h), (0, wp - w)), mode="edge"
    )
    out = np.empty((hp, wp), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            y, x = i * tile, j * tile
            out[y : y + tile, x : x + tile] = minmax_normalize(
                padded[y : y + tile, x : x + tile]
            )
    return out[:h, :w]


def center_crop_to_tile(arr: np.ndarray, size: int = TILE) -> np.ndarray:
    """Center crop [H,W] or [H,W,C] to (size, size); pads by reflection if
    the image is smaller (keeps a valid net input without inventing content
    beyond the borders)."""
    h, w = arr.shape[:2]
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        pad = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)) + (
            ((0, 0),) if arr.ndim == 3 else ()
        )
        arr = np.pad(arr, pad, mode="reflect")
        h, w = arr.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return arr[y0 : y0 + size, x0 : x0 + size]


def tile_bounds(h: int, w: int, tile: int = TILE) -> tuple[int, int, int, int]:
    """(ny, nx, padded_h, padded_w) — edge-padded dims so tiles divide exactly."""
    ny, nx = -(-h // tile), -(-w // tile)
    return ny, nx, ny * tile, nx * tile


def downsample_stride(h: int, w: int, max_side: int = MAX_GRID_SIDE) -> int:
    """Integer stride keeping both dims <= max_side (>=1)."""
    return max(1, int(np.ceil(max(h, w) / max_side)))


def downsample_grid(grid: np.ndarray, stride: int) -> np.ndarray:
    """Stride-subsample [H,W] -> [ceil(H/s), ceil(W/s)] (mesh-friendly, cheap)."""
    if stride <= 1:
        return grid
    return grid[::stride, ::stride]


def compute_stats(dsm: np.ndarray) -> dict[str, float]:
    """The [stats] line of the infer path — descriptive, NOT citable metrics."""
    d = np.asarray(dsm, dtype=np.float64)
    return {
        "n": int(d.size),
        "min": float(d.min()),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "max": float(d.max()),
        "neg": int((d < 0).sum()),
    }


def rgb_png_data_url(rgb_u8: np.ndarray, max_side: int = MAX_GRID_SIDE) -> str:
    """uint8 [H,W,3] -> 'data:image/png;base64,...' (downsampled for the mesh)."""
    from PIL import Image

    h, w = rgb_u8.shape[:2]
    s = downsample_stride(h, w, max_side)
    if s > 1:
        rgb_u8 = rgb_u8[::s, ::s]
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


@dataclass
class DnResolution:
    raw: np.ndarray  # [H,W] float32 RAW relative depth (NOT normalized)
    source: str  # "explicit" | "cache" | "live"


class DepthWizardPredictor:
    """Flagship predictor: Dn (+RGB) -> AGL metres.

    Dn resolution order (honest about which path fired):
        1. explicit raw .npy path      -> source="explicit"
        2. depth cache  <stem>.npy     -> source="cache"
        3. LIVE Depth Anything V2      -> source="live"   (downloads weights!)
       (4. none of the above -> error; we never fabricate a depth substitute)

    Dn here is RAW relative depth. Per-tile min-max normalization happens in
    predict(), exactly where the training cache recipe applies it (per 1024
    tile) — see normalize_dn_per_tile.
    """

    def __init__(
        self,
        ckpt_path: Path | str,
        device: str = "cpu",
        cache_dir: Path | str | None = None,
        live_backbone: bool = True,
        backbone_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
    ):
        from .tifops import load_calib_net, make_predict_fn, resolve_torch_device

        self.device = resolve_torch_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.live_backbone = live_backbone
        self.backbone_id = backbone_id
        self.model = load_calib_net(ckpt_path, self.device)
        self._predict_fn = make_predict_fn(self.model, self.device)
        self._backbone = None

    @property
    def model_tag(self) -> str:
        return self.model.tag

    # ------------------------------------------------------------------
    def _backbone_raw_per_tile(self, rgb_u8: np.ndarray) -> np.ndarray:
        """RAW relative depth for multi-tile images, computed per 1024 tile.

        Mirrors the cache-builder recipe (precompute_depth.py): the backbone
        sees exactly ONE 1024x1024 tile per forward (same per-object
        resolution as training), never a squeezed whole scene.
        """
        h, w = rgb_u8.shape[:2]
        ny, nx, hp, wp = tile_bounds(h, w)
        rgb_pad = np.pad(
            rgb_u8, ((0, hp - h), (0, wp - w), (0, 0)), mode="edge"
        )
        raw = np.empty((hp, wp), dtype=np.float32)
        for i in range(ny):
            for j in range(nx):
                y, x = i * TILE, j * TILE
                raw[y : y + TILE, x : x + TILE] = self._backbone.raw_depth(
                    rgb_pad[y : y + TILE, x : x + TILE]
                )
        return raw[:h, :w]

    def resolve_dn(
        self, rgb_u8: np.ndarray, stem: str = "", dn_path: Path | str | None = None
    ) -> DnResolution:
        h, w = rgb_u8.shape[:2]

        if dn_path is not None:
            raw = load_raw_dn_file(dn_path)
            if raw.shape != (h, w):
                raise ValueError(
                    f"Dn shape {raw.shape} does not match RGB shape {(h, w)}"
                )
            return DnResolution(raw=raw, source="explicit")

        if self.cache_dir is not None and stem:
            cached = self.cache_dir / f"{stem}.npy"
            if cached.exists():
                raw = load_raw_dn_file(cached)
                if raw.shape == (h, w):
                    return DnResolution(raw=raw, source="cache")
                print(
                    f"[!] cache shape mismatch for {stem}: {raw.shape} vs "
                    f"{(h, w)} — falling through to live backbone"
                )

        if self.live_backbone:
            if self._backbone is None:
                from .backbone import get_backbone

                print(
                    f"[i] no cached Dn — running LIVE {self.backbone_id} "
                    f"(first call may download weights)"
                )
                self._backbone = get_backbone(self.backbone_id, self.device)
            # Training fed the backbone one 1024 tile at a time. Match that
            # granularity: multi-tile scenes get per-tile backbone passes at
            # the SAME effective resolution the net was calibrated on.
            if h > TILE or w > TILE:
                raw = self._backbone_raw_per_tile(rgb_u8)
            else:
                raw = self._backbone.raw_depth(rgb_u8)
            return DnResolution(raw=raw, source="live")

        raise FileNotFoundError(
            f"no Dn for '{stem or '<array>'}': not found in cache "
            f"({self.cache_dir}), no --dn given, and live backbone is "
            f"disabled. Enable it or precompute the cache "
            f"(`python model.py depth`)."
        )

    # ------------------------------------------------------------------
    def predict(
        self, rgb_u8: np.ndarray, raw_dn: np.ndarray, mode: str = "auto"
    ) -> np.ndarray:
        """AGL metres [H,W] float32. Modes: auto|crop|resize|tiles.

        ``raw_dn`` is RAW relative depth (NOT normalized); normalization is
        applied here at exactly the granularity the net consumes, mirroring
        the training contract (dataset.py / precompute_depth.py):

        crop   : center 1024x1024 crop, min-max normalized over the crop
                 (output 1024x1024). Images smaller than the tile fall back
                 to letterbox semantics.
        resize : letterboxed 1024x1024 canvas (aspect preserved — NO
                 squeeze), min-max normalized over the canvas, prediction
                 cropped back to the content region and mapped to the
                 source grid.
        tiles  : any size, edge-padded 1024 tiles, Dn normalized PER TILE
                 (the training recipe), full-coverage output.
        """
        h, w = rgb_u8.shape[:2]
        if mode == "auto":
            mode = "tiles" if (h >= TILE and w >= TILE) else "resize"
        if raw_dn.shape != (h, w):
            raise ValueError(f"dn {raw_dn.shape} != rgb {(h, w)} grid")

        if mode in ("crop", "resize"):
            small = (h < TILE) or (w < TILE)
            if mode == "resize" or small:
                rgb_canvas, (y0, x0, h2, w2) = letterbox_to_tile(rgb_u8)
                dn_canvas, _ = letterbox_to_tile(raw_dn)
                pred = self._predict_fn(
                    minmax_normalize(dn_canvas),
                    rgb_canvas if self.model.use_rgb else None,
                )
                # keep only the real content, then map back to the source grid
                core = pred[y0 : y0 + h2, x0 : x0 + w2]
                return _resize_bilinear(core, h, w)
            y0, x0 = (h - TILE) // 2, (w - TILE) // 2
            return self._predict_fn(
                minmax_normalize(raw_dn[y0 : y0 + TILE, x0 : x0 + TILE]),
                rgb_u8[y0 : y0 + TILE, x0 : x0 + TILE] if self.model.use_rgb else None,
            )

        if mode == "tiles":
            ny, nx, hp, wp = tile_bounds(h, w)
            dn_pad = np.pad(
                raw_dn.astype(np.float32, copy=False),
                ((0, hp - h), (0, wp - w)),
                mode="edge",
            )
            rgb_pad = np.pad(rgb_u8, ((0, hp - h), (0, wp - w), (0, 0)), mode="edge")
            out = np.zeros((hp, wp), dtype=np.float32)
            print(f"[i] tiles: {ny}x{nx} = {ny * nx} tiles of {TILE}")
            for i in range(ny):
                for j in range(nx):
                    y, x = i * TILE, j * TILE
                    # Per-tile normalization — the EXACT training contract
                    # (each training tile was min-max normalized alone).
                    out[y : y + TILE, x : x + TILE] = self._predict_fn(
                        minmax_normalize(dn_pad[y : y + TILE, x : x + TILE]),
                        rgb_pad[y : y + TILE, x : x + TILE]
                        if self.model.use_rgb
                        else None,
                    )
            return out[:h, :w]

        raise ValueError(f"unknown mode '{mode}' (auto|crop|resize|tiles)")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def save_preview_png(dsm: np.ndarray, path: Path, title: str) -> None:
    """Terrain-colormapped preview (matplotlib, Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(dsm, cmap="terrain")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="elevation (m)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(
    out_dir: Path,
    dsm: np.ndarray,
    profile: dict,
    anchored: AnchorResult | None,
    preview_title: str,
) -> dict[str, str | None]:
    """Write dsm.npy (+ dsm.tif when georeferenced) (+ anchored DSM).

    Returns relative path strings for the payload's ``outputs`` block.
    """
    import rasterio

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str | None] = {}

    np.save(out_dir / "dsm.npy", dsm.astype(np.float32))
    outputs["dsm_npy"] = str(out_dir / "dsm.npy")

    georef, crs, tf = georef_state(profile)
    if georef:
        with rasterio.open(
            out_dir / "dsm.tif",
            "w",
            driver="GTiff",
            height=dsm.shape[0],
            width=dsm.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=tf,
            compress="deflate",
        ) as dst:
            dst.write(dsm.astype(np.float32), 1)
        outputs["dsm_tif"] = str(out_dir / "dsm.tif")
    else:
        outputs["dsm_tif"] = None

    if anchored is not None:
        suffix = out_dir.name
        p = out_dir / "dsm_anchored.tif" if georef else out_dir / "dsm_anchored.npy"
        if georef:
            with rasterio.open(
                p,
                "w",
                driver="GTiff",
                height=anchored.dsm.shape[0],
                width=anchored.dsm.shape[1],
                count=1,
                dtype="float32",
                crs=crs,
                transform=tf,
                compress="deflate",
            ) as dst:
                dst.write(anchored.dsm.astype(np.float32), 1)
        else:
            np.save(p, anchored.dsm.astype(np.float32))
        outputs["dsm_anchored"] = str(p)
    else:
        outputs["dsm_anchored"] = None

    save_preview_png(
        anchored.dsm if anchored is not None else dsm,
        out_dir / "dsm_preview.png",
        preview_title,
    )
    outputs["preview_png"] = str(out_dir / "dsm_preview.png")
    return outputs


# ---------------------------------------------------------------------------
# Scene payload — the backend/frontend contract (webapp consumes this)
# ---------------------------------------------------------------------------


def build_scene_payload(
    dsm: np.ndarray,
    rgb_u8: np.ndarray,
    *,
    stem: str,
    mode: str,
    dn_source: str,
    model_tag: str,
    device: str,
    profile: dict,
    anchored: AnchorResult | None,
    outputs: dict[str, str | None],
    elapsed_sec: float,
) -> dict:
    """JSON-serializable scene description for the Three.js viewer.

    Contract (webapp/src/lib/dw.ts mirrors these types):
        grid.data       row-major flattened [height*width] floats (metres)
        rgb_png         data-URL PNG, downsampled to the SAME grid footprint
        stats           descriptive only — never citable metrics
        anchored        false | {label: 'ANCHORED (not learned)', source: ...}
        georef.crs      string; "UNKNOWN" when the input carried no CRS
        meta.pixel_size_m   [float, float] metres-per-source-pixel [x, y]
                            or null when CRS is absent (Track-1 honest null;
                            viewers MUST branch on null and refuse metric
                            claims — see worklog Section 4)
    """
    georef, crs, tf = georef_state(profile)
    stride = downsample_stride(dsm.shape[0], dsm.shape[1])
    grid = downsample_grid(dsm, stride)
    stats = compute_stats(dsm)
    transform_repr = list(tf)[:6] if tf is not None else "UNKNOWN"

    # GSD honesty (worklog Section 4: "GSD honesty regression"): pixel_size_m is
    # the real ground-sample distance per source pixel, in metres, derived from
    # the raster's CRS+transform. Honest ``None`` for non-georeferenced Track-1
    # inputs — never a fallback guess. Consumers (Viewer3D, demprior) MUST
    # branch on null and refuse metric claims when it is absent.
    from .geo import pixel_size_metres

    pixel_size_m = pixel_size_metres(crs, tf)
    pixel_size_m_json = (
        [round(float(v), 6) for v in pixel_size_m] if pixel_size_m is not None else None
    )

    payload = {
        "ok": True,
        "stem": stem,
        "grid": {
            "height": int(grid.shape[0]),
            "width": int(grid.shape[1]),
            "stride": int(stride),
            "data": [round(float(v), 3) for v in grid.ravel()],
        },
        "rgb_png": rgb_png_data_url(rgb_u8),
        "stats": stats,
        "anchored": (
            {"label": ANCHORED_LABEL, "source": anchored.source}
            if anchored is not None
            else False
        ),
        "georef": {
            "crs": str(crs) if crs is not None else "UNKNOWN",
            "transform": transform_repr,
        },
        "meta": {
            "model_tag": model_tag,
            "device": device,
            "dn_source": dn_source,
            "mode": mode,
            "source_shape": [int(dsm.shape[0]), int(dsm.shape[1])],
            "pixel_size_m": pixel_size_m_json,
            "elapsed_sec": round(elapsed_sec, 2),
        },
        "outputs": outputs,
    }
    return payload


# ---------------------------------------------------------------------------
# Orchestrator — one entry used by CLI --json-out AND the FastAPI service
# ---------------------------------------------------------------------------


def run_inference(
    input_path: Path | str,
    ckpt_path: Path | str,
    *,
    out_dir: Path | str,
    device: str = "cpu",
    mode: str = "auto",
    dn_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    live_backbone: bool = True,
    backbone_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
    anchor_dem: Path | str | None = None,
    ground_elev: float | None = None,
    write_files: bool = True,
) -> dict:
    """Full inference run -> scene payload dict (see build_scene_payload)."""
    t0 = time.perf_counter()
    input_path = Path(input_path)

    predictor = DepthWizardPredictor(
        ckpt_path=ckpt_path,
        device=device,
        cache_dir=cache_dir,
        live_backbone=live_backbone,
        backbone_id=backbone_id,
    )

    rgb_u8, profile = read_image(input_path)
    georef, crs, tf = georef_state(profile)
    print(
        f"[i] input {input_path.name}  {rgb_u8.shape[1]}x{rgb_u8.shape[0]}  "
        f"georeferenced={georef} ({crs})"
    )

    resolution = predictor.resolve_dn(rgb_u8, stem=input_path.stem, dn_path=dn_path)
    print(f"[i] Dn source: {resolution.source}")

    dsm = predictor.predict(rgb_u8, resolution.raw, mode=mode)
    stats = compute_stats(dsm)
    print(
        f"[stats] DSM (m): min {stats['min']:.2f}  mean {stats['mean']:.2f}  "
        f"median {stats['median']:.2f}  max {stats['max']:.2f}  "
        f"neg {stats['neg']}"
    )
    if stats["neg"]:
        print(
            "[!] negatives exist — the model's clamp makes this impossible. "
            "The forward pass diverged from the certified path. STOP and diff."
        )

    anchored = None
    if anchor_dem is not None or ground_elev is not None:
        tile_profile = {
            "crs": crs,
            "transform": tf,
            "height": dsm.shape[0],
            "width": dsm.shape[1],
        }
        anchored = anchor(dsm, anchor_dem, ground_elev, tile_profile)
        assert anchored is not None
        a_stats = compute_stats(anchored.dsm)
        print(
            f"[stats] ANCHORED DSM (m): min {a_stats['min']:.2f}  "
            f"mean {a_stats['mean']:.2f}  max {a_stats['max']:.2f}  "
            f"[{ANCHORED_LABEL} | source={anchored.source}]"
        )
    elif not georef:
        print(
            "[i] AGL is RELATIVE to ground (Track-1 rDSM) — not an absolute "
            "DSM. Anchor with --anchor-dem/--ground-elev for Track 2."
        )

    outputs: dict[str, str | None] = {}
    if write_files:
        outputs = write_outputs(
            Path(out_dir),
            dsm,
            profile,
            anchored,
            preview_title=f"{predictor.model_tag} — {input_path.name}",
        )
        for k, v in outputs.items():
            if v:
                print(f"[out] {v}")

    return build_scene_payload(
        dsm,
        rgb_u8,
        stem=input_path.stem,
        mode=mode,
        dn_source=resolution.source,
        model_tag=predictor.model_tag,
        device=device,
        profile=profile,
        anchored=anchored,
        outputs=outputs,
        elapsed_sec=time.perf_counter() - t0,
    )
