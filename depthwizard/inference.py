"""Image -> AGL/DSM inference pipeline (the ONE code path for CLI + service).

Who uses this module:
    * the ``infer`` CLI command (``python model.py infer ...``)
    * the FastAPI service (``/service/api.py``) that backs the webapp
    * the scene payload builder that feeds the Three.js viewer

Because every consumer imports the same functions, the webapp can never
drift from the certified CLI forward pass.

Flow (height-model backend is pluggable — architecture registry in
``depthwizard/tifops.py``):
    read image (Pillow for PNG/JPG/JPEG, rasterio for GeoTIFF; georef
      state reported honestly, never invented)
      -> resolve RAW Dn (explicit .npy | depth cache | LIVE Depth Anything
         V2, one 1024 tile per forward — the training granularity)
      -> height backend, selected by checkpoint/config:
           "rdah"  (default after the RDAH integration): official
                    RDAH-Net — depth input = RAW Dn x 40 (0-255 range,
                    NOT min-max), RGB ImageNet-normalized; output = nDSM
                    metres, direct regression (NO a*Dn+b affine logic);
                    see depthwizard/rdah.py for the verified recipe.
           "calibration_net" (legacy/frozen): H = clamp(a(x,y)*Dn + b(x,y), 0)
                    with Dn min-max normalized per 1024 tile (training
                    contract).
         modes: crop (center 1024) | resize (letterboxed 1024, aspect
         preserved) | tiles (any size, overlapping windows + weighted
         stitching — depthwizard.tiling, per-window Dn)
      -> optional Track-2 anchoring  DSM = height + ground  [ANCHORED (not learned)]
      -> outputs: dsm.npy (+ dsm.tif when georeferenced) (+ _anchored), preview PNG
      -> scene payload (downsampled grid + RGB PNG + stats) for the webapp

Both backends keep the SAME downstream contracts: the predictor emits
metre heights on the source grid; anchoring, writing and geospatial
metadata handling are shared and untouched. The payload reports which
backend produced the heights (``meta.model_architecture``) and their
semantic role (``meta.height_type``: "AGL" for CalibrationNet, "nDSM"
for RDAH — never mislabelled as absolute terrain elevation).

This is a DEMONSTRATION path, never a citable evaluation (numbers come only
from the ``evaluate`` command).
"""

from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio import CRS, Affine

from .anchoring import ANCHORED_LABEL
from .normalize import minmax_normalize, minmax_normalize_with_stats
from .tiling import OverlapStitcher, TilingConfig, iter_tile_windows

TILE = 1024  # training tile size; crop/resize/tiles all target it
MAX_GRID_SIDE = 512  # webapp mesh grid cap (stride-downsampled)


class InferenceCancelled(RuntimeError):
    """Raised between tiles/stages when a cooperative cancel hook fires.

    The service catches this and marks the job cancelled; the CLI never
    passes a hook, so CLI behavior is unchanged.
    """



# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------


def read_image(path: Path | str) -> tuple[np.ndarray, dict]:
    """Read any supported input -> (rgb_u8 [H,W,3] uint8, profile).

    Delegates to :func:`depthwizard.imgio.preprocess_input_image`:

      .png/.jpg/.jpeg -> Pillow decode (palette expanded, alpha dropped,
                         16-bit range-checked — never a blind cast);
                         CRS stays None, no invented georeferencing.
      everything else -> rasterio (GeoTIFF + GDAL rasters) with CRS/
                         transform preserved.

    Both paths end at the SAME contract: uint8 [H,W,3] in [0,255] — the
    input the DAv2 backbone and CalibrationNet expect (each applies its
    own /255 + ImageNet normalization exactly once downstream). Decoding
    failures raise imgio.InvalidImageError instead of returning garbage.
    """
    from .imgio import preprocess_input_image

    rgb, meta = preprocess_input_image(path)
    profile: dict = dict(meta["rasterio_profile"]) if meta["rasterio_profile"] else {}
    # keep crs/transform as first-class fields for the anchoring path
    profile["_crs_obj"] = meta["crs"]
    profile["_transform_obj"] = meta["transform"]
    return rgb, profile



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


def _window_tile(arr: np.ndarray, window, tile: int = TILE) -> np.ndarray:
    """Extract one TileWindow's region from [H,W] or [H,W,C], edge-padded
    up to (tile, tile) when the window extends past the source (only
    possible when an axis is smaller than tile — see iter_tile_windows)."""
    vh, vw = window.valid_height, window.valid_width
    region = arr[
        window.row_off : window.row_off + vh, window.col_off : window.col_off + vw
    ]
    if vh == tile and vw == tile:
        return region
    pad2 = ((0, tile - vh), (0, tile - vw))
    pad = pad2 + ((0, 0),) if arr.ndim == 3 else pad2
    return np.pad(region, pad, mode="edge")


def normalize_dn_per_tile(raw: np.ndarray, tile: int = TILE) -> np.ndarray:
    """Per-window min-max normalization of RAW relative depth — the training
    contract (dataset.py normalizes each cached 1024 tile independently).

    Each OVERLAPPING inference window (depthwizard.tiling) is normalized in
    isolation, exactly as each training tile was, and the results are
    weighted-stitched (OverlapStitcher) so windows that disagree at a shared
    border blend smoothly instead of leaving a hard seam.
    """
    h, w = raw.shape
    cfg = TilingConfig(tile_size=tile)
    stitcher = OverlapStitcher(h, w, cfg)
    for window in iter_tile_windows(h, w, cfg):
        tile_raw = _window_tile(raw.astype(np.float32, copy=False), window, tile)
        stitcher.add_tile(window, minmax_normalize(tile_raw))
    return stitcher.finalize()


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




def _resize_sem_probs(
    sem: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Bilinear-resize per-channel semantic probabilities [K,h,w] -> [K,H,W].

    Bilinear interpolation of probabilities does not re-normalize to a
    probability simplex — acceptable for smoothness GATING (relative
    similarities matter, not absolute values); callers must not present
    the result as calibrated probabilities.
    """
    if sem.shape[1:] == (height, width):
        return sem
    out = np.stack(
        [_resize_bilinear(sem[k], height, width) for k in range(sem.shape[0])],
        axis=0,
    )
    return out.astype(np.float32)




# Scene-output + payload functions moved to depthwizard.pipeline.scene_outputs
# (tranche 3c) — re-exported here so EVERY existing import path keeps working
# (backend.app, depthwizard.cli, model_tests, golden regression).
from .pipeline.scene_outputs import (  # noqa: F401 — re-export
    MAX_GRID_SIDE,
    build_scene_payload,
    compute_stats,
    downsample_grid,
    downsample_stride,
    georef_state,
    rgb_png_data_url,
    save_preview_png,
    write_outputs,
    write_semantic_outputs,
)

# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


@dataclass
class DnResolution:
    raw: np.ndarray  # [H,W] float32 RAW relative depth (NOT normalized)
    source: str  # "explicit" | "cache" | "live"


class DepthWizardPredictor:
    """Flagship predictor: Dn (+RGB) -> height metres.

    Height backend (RDAH integration): ``architecture`` selects the model
    family — "rdah" (official RDAH-Net, default for new configs) or
    "calibration_net" (the frozen Phase-2 net). None = auto-detect from
    the checkpoint payload (see tifops.detect_architecture). Both share
    the exact same downstream path: RAW Dn in, metre heights out.

    Dn resolution order (honest about which path fired):
        1. explicit raw .npy path      -> source="explicit"
        2. depth cache  <stem>.npy     -> source="cache"
        3. LIVE Depth Anything V2      -> source="live"   (downloads weights!)
       (4. none of the above -> error; we never fabricate a depth substitute)

    Dn here is RAW relative depth. Normalization happens inside
    predict() at exactly the granularity the net consumes — per 1024
    tile min-max for CalibrationNet (the training contract, see
    normalize_dn_per_tile); RAW x depth_scale for the RDAH backend
    (reconstructed from the same per-tile stats — depthwizard/rdah.py).
    """

    def __init__(
        self,
        ckpt_path: Path | str,
        device: str = "cpu",
        cache_dir: Path | str | None = None,
        live_backbone: bool = True,
        backbone_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
        architecture: str | None = None,
        depth_scale: float = 40.0,
    ):
        from .tifops import (
            load_height_model,
            make_full_predict_fn,
            make_predict_fn,
            resolve_torch_device,
        )

        self.device = resolve_torch_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.live_backbone = live_backbone
        self.backbone_id = backbone_id
        # registry: calibration_net | rdah | None (auto-detect from ckpt)
        self.model = load_height_model(
            ckpt_path, self.device, architecture=architecture,
            depth_scale=depth_scale,
        )
        self._predict_fn = make_predict_fn(self.model, self.device)
        self._predict_full_fn = make_full_predict_fn(self.model, self.device)
        self._backbone = None

    @property
    def model_tag(self) -> str:
        return self.model.tag

    @property
    def architecture(self) -> str:
        """Active height backend: "calibration_net" | "rdah"."""
        return self.model.architecture

    # ------------------------------------------------------------------
    def _backbone_raw_per_tile(
        self, rgb_u8: np.ndarray, progress_cb: Callable[[int, int], None] | None = None
    ) -> np.ndarray:
        """RAW relative depth for multi-tile images, computed per 1024 window.

        Mirrors the cache-builder recipe (precompute_depth.py): the backbone
        sees exactly ONE 1024x1024 window per forward (same per-object
        resolution as training), never a squeezed whole scene. Windows
        OVERLAP and the per-window raw depths are weighted-stitched
        (depthwizard.tiling.OverlapStitcher) so no seam is introduced where
        two independent backbone passes disagree.
        """
        h, w = rgb_u8.shape[:2]
        cfg = TilingConfig(tile_size=TILE)
        windows = list(iter_tile_windows(h, w, cfg))
        stitcher = OverlapStitcher(h, w, cfg)
        for done, window in enumerate(windows, start=1):
            tile_rgb = _window_tile(rgb_u8, window, TILE)
            raw_tile = self._backbone.raw_depth(tile_rgb)
            stitcher.add_tile(window, raw_tile)
            if progress_cb is not None:
                progress_cb(done, len(windows))
        out = stitcher.finalize()
        if np.isnan(out).any():
            raise RuntimeError("backbone returned NaN raw depth in a covered window")
        return out

    def resolve_dn(
        self,
        rgb_u8: np.ndarray,
        stem: str = "",
        dn_path: Path | str | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
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
                raw = self._backbone_raw_per_tile(rgb_u8, progress_cb=progress_cb)
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
        self, rgb_u8: np.ndarray, raw_dn: np.ndarray, mode: str = "auto",
        should_cancel: Callable[[], bool] | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
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
        tiles  : any size, overlapping edge-padded 1024 windows, Dn
                 normalized PER WINDOW (the training recipe), weighted-
                 stitched (depthwizard.tiling) — seam-free, full coverage.
        """
        return self._predict_impl(
            rgb_u8, raw_dn, mode, want_semantics=False,
            should_cancel=should_cancel, progress_cb=progress_cb,
        )

    def predict_with_semantics(
        self, rgb_u8: np.ndarray, raw_dn: np.ndarray, mode: str = "auto",
        should_cancel: Callable[[], bool] | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Like predict(), but ALSO returns predicted semantic probabilities.

        Returns {"pred": [H,W] float32, "sem_probs": [K,H,W] float32 | None}.
        ``sem_probs`` is the softmax of the PREDICTED auxiliary semantic head
        (never GT) and is None for checkpoints without sem_aux_head — the
        post-processing stage degrades honestly in that case. Composition:
        {"pred", "sem_probs"} -> PostProcessConfig -> refine_agl.

        ``should_cancel`` is a cooperative hook polled between tiles; when it
        returns True, InferenceCancelled is raised (never silently ignored).
        """
        return self._predict_impl(
            rgb_u8, raw_dn, mode, want_semantics=True,
            should_cancel=should_cancel, progress_cb=progress_cb,
        )

    def _predict_impl(
        self,
        rgb_u8: np.ndarray,
        raw_dn: np.ndarray,
        mode: str = "auto",
        want_semantics: bool = False,
        should_cancel: Callable[[], bool] | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ):
        from .imgio import validate_rgb_u8

        validate_rgb_u8(rgb_u8, source="predict(rgb_u8)")
        h, w = rgb_u8.shape[:2]
        if mode == "auto":
            mode = "tiles" if (h >= TILE and w >= TILE) else "resize"
        if raw_dn.shape != (h, w):
            raise ValueError(f"dn {raw_dn.shape} != rgb {(h, w)} grid")

        def _forward(dn_n: np.ndarray, rgb_c: np.ndarray, stats_c: np.ndarray = None):
            """Single-tile forward -> (pred, sem_probs|None)."""
            if want_semantics:
                ex = self._predict_full_fn(
                    dn_n, rgb_c if self.model.use_rgb else None, stats_c
                )
                return ex["pred"], ex.get("sem_probs")
            return (
                self._predict_fn(dn_n, rgb_c if self.model.use_rgb else None, stats_c),
                None,
            )

        if mode in ("crop", "resize"):
            small = (h < TILE) or (w < TILE)
            if mode == "resize" or small:
                rgb_canvas, (y0, x0, h2, w2) = letterbox_to_tile(rgb_u8)
                dn_canvas, _ = letterbox_to_tile(raw_dn)
                dn_n, stats = minmax_normalize_with_stats(dn_canvas)
                pred, sem = _forward(
                    dn_n, rgb_canvas, stats
                )
                # keep only the real content, then map back to the source grid
                core = pred[y0 : y0 + h2, x0 : x0 + w2]
                out = _resize_bilinear(core, h, w)
                if sem is not None:
                    sem = _resize_sem_probs(
                        sem[y0 : y0 + h2, x0 : x0 + w2], h, w
                    )
                return self._pack(out, sem, want_semantics)
            y0, x0 = (h - TILE) // 2, (w - TILE) // 2
            dn_n, stats = minmax_normalize_with_stats(
                raw_dn[y0 : y0 + TILE, x0 : x0 + TILE]
            )
            pred, sem = _forward(
                dn_n,
                rgb_u8[y0 : y0 + TILE, x0 : x0 + TILE],
                stats,
            )
            return self._pack(pred, sem, want_semantics)

        if mode == "tiles":
            cfg = TilingConfig(tile_size=TILE)
            windows = list(iter_tile_windows(h, w, cfg))
            stitcher = OverlapStitcher(h, w, cfg)
            sem_stitchers: list[OverlapStitcher] | None = []
            print(f"[i] tiles: {len(windows)} windows of {TILE} (overlap {cfg.overlap})")
            for done, window in enumerate(windows, start=1):
                if should_cancel is not None and should_cancel():
                    raise InferenceCancelled(
                        "cancelled between tiles"
                    )
                tile_dn = _window_tile(raw_dn, window, TILE)
                tile_rgb = _window_tile(rgb_u8, window, TILE)
                # Per-window normalization — the EXACT training contract
                # (each training tile was min-max normalized alone); the
                # FiLM stats come from the SAME window so train/infer
                # conditioning always matches.
                dn_n, stats = minmax_normalize_with_stats(tile_dn)
                pred, sem = _forward(dn_n, tile_rgb, stats)
                stitcher.add_tile(window, pred)
                if progress_cb is not None:
                    progress_cb(done, len(windows))
                if want_semantics and sem is not None:
                    if not sem_stitchers:
                        sem_stitchers = [
                            OverlapStitcher(h, w, cfg) for _ in range(sem.shape[0])
                        ]
                    for k, st in enumerate(sem_stitchers):
                        st.add_tile(window, sem[k])
            out = stitcher.finalize()
            if np.isnan(out).any():
                raise RuntimeError(
                    "tiled forward left uncovered NaN pixels — stitcher bug"
                )
            sem_out: np.ndarray | None = None
            if want_semantics:
                if sem_stitchers:
                    sem_out = np.stack(
                        [st.finalize() for st in sem_stitchers], axis=0
                    )
                    if np.isnan(sem_out).any():
                        raise RuntimeError(
                            "semantic stitching left uncovered NaN pixels"
                        )
                # else: checkpoint has no aux head -> sem_probs stays None
            return self._pack(out, sem_out, want_semantics)

        raise ValueError(f"unknown mode '{mode}' (auto|crop|resize|tiles)")

    @staticmethod
    def _pack(
        pred: np.ndarray, sem_probs: np.ndarray | None, want_semantics: bool
    ) -> dict | np.ndarray:
        # predict() keeps the LEGACY contract: a bare [H,W] array.
        # predict_with_semantics() ALWAYS returns a dict (sem_probs=None
        # when the checkpoint has no auxiliary head) — its callers index
        # out["pred"], so a bare array would crash them (bug found by
        # model_tests/test_postprocess.py, headless-checkpoint path).
        if not want_semantics:
            return pred
        return {"pred": np.asarray(pred, dtype=np.float32), "sem_probs": sem_probs}

    def make_tta_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Full-path TTA closure: transformed RGB -> LIVE backbone Dn ->
        per-tile normalized forward -> height on the transformed grid.

        The identity orientation of a cached-Dn deployment must NOT be
        reused for flipped augmentations (a flipped image needs a flipped
        backbone pass — reusing the cached orientation would silently
        predict the unflipped world). This closure therefore always runs
        the live backbone and raises if it is disabled.

        Backend note: per-window (dn, stats) are computed together — the
        RDAH backend reconstructs the RAW depth scale from the stats, so
        both backends get exactly the inputs their training contract
        defined (min-max [0,1] + stats flows to both; each applies its own
        conversion exactly once).
        """
        if not self.live_backbone:
            raise ValueError(
                "TTA requires the live backbone (--no-live disables it); "
                "a cached Dn exists only for the identity orientation and "
                "reusing it for flipped augmentations would be wrong."
            )

        def predict_aug(rgb_aug_u8: np.ndarray) -> np.ndarray:
            from .tiling import OverlapStitcher, TilingConfig, iter_tile_windows

            h, w = rgb_aug_u8.shape[:2]
            cfg = TilingConfig(tile_size=TILE)
            stitcher = OverlapStitcher(h, w, cfg)
            for window in iter_tile_windows(h, w, cfg):
                tile_rgb = _window_tile(rgb_aug_u8, window, TILE)
                tile_raw = self._backbone.raw_depth(tile_rgb)
                # per-window normalization + stats (the SAME pair every
                # backend's training contract consumed)
                dn_n, stats = minmax_normalize_with_stats(tile_raw)
                pred = self._predict_fn(
                    dn_n, tile_rgb if self.model.use_rgb else None, stats
                )
                stitcher.add_tile(window, pred)
            out = stitcher.finalize()
            if np.isnan(out).any():
                raise RuntimeError(
                    "TTA tiled forward left uncovered NaN pixels"
                )
            return out

        return predict_aug


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Scene payload — the backend/frontend contract (webapp consumes this)
# ---------------------------------------------------------------------------



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
    dem_provider: str | None = None,
    dem_cache_dir: Path | str | None = None,
    write_files: bool = True,
    postprocess: str = "none",
    postprocess_params: dict | None = None,
    tta: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    architecture: str | None = None,
    depth_scale: float = 40.0,
    terraheight_tile_size: int | None = None,
    terraheight_overlap: int | None = None,
    terraheight_fp16: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Full inference run -> scene payload dict (see build_scene_payload).

    ``architecture``: None (auto-detect from the checkpoint) | "rdah" |
    "calibration_net" | "terraheight_s" — the height-model backend registry.

    TerraHeight-S dispatch: when the resolved backend is ``terraheight_s``
    this orchestrator delegates to
    :func:`depthwizard.terraheight.run_terraheight_inference` — RGB ->
    TerraHeight-S -> AGL metres, with NO depth-cache/Dn stage (the
    ``dn_path``/``cache_dir``/``mode``/backbone knobs and the RDAH
    post-processing/TTA path do not apply; requesting them prints an
    honest note and they are ignored). The scene payload and artifact
    contracts (dsm.npy/dsm.tif/previews + terraheight_* provenance) are
    identical, so downstream consumers are unchanged.

    Post-processing (task Sec. 19 API):
        postprocess="none"    raw height passthrough — byte-identical legacy path
        postprocess=<preset>  median|guided|bilateral|wls|conf|semantic|full
        postprocess_params    optional {field: value} overrides for
                              PostProcessConfig (e.g. wls_lambda=2.0)
        tta=True              flip/rotate ensemble inside the refinement
        should_cancel         cooperative cancel hook — polled between tiles
                              and at every stage boundary; True raises
                              InferenceCancelled (service-backed cancellation)
        progress_cb           optional (done, total) hook called after each
                              tiled forward (backbone Dn windows, prediction
                              tiles, TerraHeight AGL tiles) — lets callers
                              surface REAL tile progress; never fabricated

    Order is FROZEN: height_raw -> refine -> (anchor) -> write. The raw
    heights are always preserved (agl_raw.npy) and the anchored DSM is
    built from the REFINED heights, so DSM == DEM + height_refined holds
    exactly (nDSM + ground elevation = absolute DSM — never mislabel the
    relative prediction as absolute terrain elevation).
    """
    t0 = time.perf_counter()
    input_path = Path(input_path)

    # ---- backend dispatch (registry: tifops.load_height_model) ----------
    from .tifops import detect_architecture

    resolved_arch = architecture or detect_architecture(ckpt_path)
    if resolved_arch == "terraheight_s":
        ignored = [
            name
            for name, val in (
                ("--dn/dn_path", dn_path),
                ("cache_dir", cache_dir),
                ("postprocess", postprocess if postprocess not in (None, "", "none") else None),
                ("tta", tta if tta else None),
            )
            if val
        ]
        if ignored:
            print(
                f"[i] terraheight_s ignores {', '.join(ignored)} — TerraHeight "
                "computes AGL directly from RGB (no depth cache, no RDAH "
                "post-processing)."
            )
        from .terraheight import (
            TERRAHEIGHT_DEFAULT_OVERLAP,
            TERRAHEIGHT_TILE_SIZE,
            run_terraheight_inference,
        )

        return run_terraheight_inference(
            input_path,
            ckpt_path,
            out_dir=out_dir,
            device=device,
            tile_size=(terraheight_tile_size or TERRAHEIGHT_TILE_SIZE),
            overlap=(terraheight_overlap or TERRAHEIGHT_DEFAULT_OVERLAP),
            fp16=terraheight_fp16,
            anchor_dem=anchor_dem,
            ground_elev=ground_elev,
            dem_provider=dem_provider,
            dem_cache_dir=dem_cache_dir,
            write_files=write_files,
            should_cancel=should_cancel,
            progress_cb=progress_cb,
        )

    predictor = DepthWizardPredictor(
        ckpt_path=ckpt_path,
        device=device,
        cache_dir=cache_dir,
        live_backbone=live_backbone,
        backbone_id=backbone_id,
        architecture=architecture,
        depth_scale=depth_scale,
    )

    rgb_u8, profile = read_image(input_path)
    georef, crs, tf = georef_state(profile)
    print(
        f"[i] input {input_path.name}  {rgb_u8.shape[1]}x{rgb_u8.shape[0]}  "
        f"georeferenced={georef} ({crs})"
    )

    def _cancelled() -> bool:
        return bool(should_cancel is not None and should_cancel())

    if _cancelled():
        raise InferenceCancelled("cancelled before inference started")

    resolution = predictor.resolve_dn(
        rgb_u8, stem=input_path.stem, dn_path=dn_path, progress_cb=progress_cb
    )
    print(f"[i] Dn source: {resolution.source}")

    pp_active = postprocess not in (None, "", "none")
    pp_report_dict: dict | None = None
    agl_raw: np.ndarray | None = None
    sem_probs: np.ndarray | None = None

    # Always attempt predict_with_semantics so semantic artifacts are
    # generated for every scene when the checkpoint supports it,
    # regardless of the postprocess setting.
    has_sem_head = bool(getattr(predictor.model, 'sem_aux_head', False))

    if pp_active or has_sem_head:
        out = predictor.predict_with_semantics(
            rgb_u8, resolution.raw, mode=mode, should_cancel=should_cancel,
            progress_cb=progress_cb,
        )
        if pp_active:
            from .postprocess import config_from_preset, refine_agl

            pcfg = config_from_preset(postprocess, postprocess_params or {})
            pcfg = pcfg.with_updates(tta=tta or pcfg.tta)
            agl_raw = out["pred"]
            sem_probs = out.get("sem_probs")
            if sem_probs is None and pcfg.method in ("semantic_wls", "planar"):
                print(
                    "[i] checkpoint has no semantic auxiliary head — semantic "
                    "gating unavailable, refining RGB-only (no semantics "
                    "fabricated)"
                )
            tta_fn = None
            if pcfg.tta:
                tta_fn = predictor.make_tta_predict_fn()

            confidence = None
            if pcfg.method == "semantic_wls" and pcfg.confidence_weight > 0:
                from .postprocess.confidence import estimate_confidence

                confidence = estimate_confidence(agl_raw, rgb_u8)

            # Physical pixel size from the input raster's georeference — the
            # spike slope gate must use real metres/pixel, never assume 1 px
            # = 1 m (falls back to (1, 1) for non-georeferenced input).
            from .geo import pixel_size_metres

            gsd = pixel_size_metres(crs, tf) if georef else None

            dsm, pp_report = refine_agl(
                agl_raw,
                rgb_u8,
                pcfg,
                sem_probs=sem_probs,
                confidence=confidence,
                tta_predict_fn=tta_fn,
                gsd=gsd,
            )
            pp_report_dict = pp_report.to_dict()
            print(
                f"[postprocess] method={pp_report.method}  "
                f"changed={pp_report.n_changed}/{pp_report.n_valid}  "
                f"{pp_report.elapsed_sec:.2f}s"
            )
            if pp_report.calibration:
                c = pp_report.calibration
                print(
                    f"[postprocess] calibration: mean {c['raw_mean']:.3f}->"
                    f"{c['refined_mean']:.3f} (shift {c['mean_shift']:+.4f} m), "
                    f"std ratio {c['std_ratio']:.4f}"
                )
            for note in pp_report.notes:
                print(f"[postprocess] {note}")
        else:
            # No postprocess, but we ran predict_with_semantics for the
            # semantic head — use the bare AGL prediction.
            dsm = out["pred"]
            sem_probs = out.get("sem_probs")
    else:
        dsm = predictor.predict(
            rgb_u8, resolution.raw, mode=mode, should_cancel=should_cancel,
            progress_cb=progress_cb,
        )

    if _cancelled():
        raise InferenceCancelled("cancelled after depth inference")

    stats = compute_stats(dsm)
    print(
        f"[stats] DSM (m): min {stats['min']:.2f}  mean {stats['mean']:.2f}  "
        f"median {stats['median']:.2f}  max {stats['max']:.2f}  "
        f"neg {stats['neg']}"
    )
    if stats["neg"]:
        if predictor.model.architecture == "rdah":
            # RDAH is UNCLAMPED (official semantics): slightly negative
            # ground predictions are expected, not a divergence.
            print(
                f"[i] {stats['neg']} negative pixels — expected for the "
                "unclamped RDAH backend (official nDSM regression; ground "
                "can dip below 0). CalibrationNet clamps at 0, RDAH does not."
            )
        else:
            print(
                "[!] negatives exist — the model's clamp makes this impossible. "
                "The forward pass diverged from the certified path. STOP and diff."
            )

    anchored = None
    abs_result = None
    if _cancelled():
        raise InferenceCancelled("cancelled before anchoring")
    anchor_requested = (
        anchor_dem is not None or ground_elev is not None
        or dem_provider not in (None, "", "none")
    )
    if anchor_requested:
        from .absolute_dsm import maybe_acquire_reference_dem, write_provenance

        dem_result, _ = maybe_acquire_reference_dem(
            anchor_dem, dem_provider, dem_cache_dir, profile, dsm.shape
        )
    else:
        dem_result = None

    # The output-type classification ALWAYS runs (Part D): every scene
    # payload records output_type + absolute_reference_available, even
    # for the plain relative product.
    from .absolute_dsm import build_absolute_dsm

    abs_result = build_absolute_dsm(
        dsm,
        profile,
        dem_result,
        ground_elev if dem_result is None else None,
        height_model=predictor.model.architecture,
        height_model_version=str(
            getattr(predictor.model, "checkpoint", "unknown")
        ),
        calibration_enabled=(predictor.model.architecture == "calibration_net"),
        calibration_model=(
            predictor.model_tag
            if predictor.model.architecture == "calibration_net"
            else None
        ),
    )
    anchored = abs_result.anchored
    if anchored is not None:
        a_stats = compute_stats(anchored.dsm)
        print(
            f"[stats] ANCHORED DSM (m): min {a_stats['min']:.2f}  "
            f"mean {a_stats['mean']:.2f}  max {a_stats['max']:.2f}  "
            f"[{ANCHORED_LABEL} | source={anchored.source} | "
            f"output_type={abs_result.output_type}]"
        )
        if write_files and dem_result is not None:
            from .absolute_dsm import write_provenance as _write_prov

            prov_path = _write_prov(abs_result.provenance, Path(out_dir))
            print(f"[out] {prov_path}")
    elif not georef:
        print(
            "[i] Input raster is not georeferenced — the prediction is a "
            "RELATIVE height map ("
            f"{predictor.model.height_type}"
            "). Absolute georeferenced DSM cannot be emitted without "
            "external geospatial reference; anchor with --anchor-dem/"
            "--ground-elev for Track 2. No CRS was invented."
        )

    if _cancelled():
        raise InferenceCancelled("cancelled before writing outputs")
    outputs: dict[str, str | None] = {}
    if write_files:
        outputs = write_outputs(
            Path(out_dir),
            dsm,
            profile,
            anchored,
            preview_title=f"{predictor.model_tag} — {input_path.name}",
            agl_raw=agl_raw,
            postprocess_meta=pp_report_dict,
        )
        for k, v in outputs.items():
            if v:
                print(f"[out] {v}")

        # ---- Semantic artifact generation --------------------------------
        if sem_probs is not None:
            ckpt_name = Path(str(getattr(predictor.model, 'checkpoint', 'unknown'))).parent.name
            sem_outputs = write_semantic_outputs(
                Path(out_dir),
                sem_probs,
                checkpoint_name=ckpt_name,
                model_name=predictor.model_tag,
            )
            outputs.update(sem_outputs)
            from .semantic_segmenter import prediction_from_probs
            sem_pred = prediction_from_probs(sem_probs)
            print(
                f"[semantic] {sem_pred.num_classes} classes  "
                f"mean_conf={sem_pred.mean_confidence():.3f}  "
                f"low_conf(<0.7)={sem_pred.low_confidence_fraction(0.7):.1%}"
            )
            for k2, v2 in sem_outputs.items():
                if v2:
                    print(f"[out] {v2}")
        elif has_sem_head:
            print(
                "[semantic] checkpoint has sem_aux_head but predict_with_semantics "
                "returned None — semantic artifacts NOT generated"
            )

    payload = build_scene_payload(
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
    # Height semantics (RDAH integration): which backend produced the
    # heights and what they MEAN. "nDSM"/"AGL" are both above-ground
    # height in metres — neither is absolute terrain elevation. When
    # anchoring ran, DSM = heights + ground datum (ANCHORED, not learned).
    payload["meta"]["model_architecture"] = predictor.model.architecture
    payload["meta"]["height_type"] = predictor.model.height_type
    if abs_result is not None:
        from .absolute_dsm import payload_output_fields

        payload["meta"].update(payload_output_fields(abs_result))
        payload["meta"]["height_model_label"] = predictor.model_tag
    if anchored is not None:
        if abs_result is not None and abs_result.output_type == "absolute_dsm":
            payload["meta"]["height_semantics"] = (
                f"{predictor.model.height_type} + reference DEM = absolute "
                f"DSM ({ANCHORED_LABEL})"
            )
        else:
            payload["meta"]["height_semantics"] = (
                f"{predictor.model.height_type} + user-supplied constant "
                f"datum ({ANCHORED_LABEL}; datum asserted by the user, not "
                "an external DEM)"
            )
    else:
        payload["meta"]["height_semantics"] = (
            f"{predictor.model.height_type} (above-ground height in metres, "
            "relative — NOT absolute terrain elevation)"
        )
    if pp_report_dict is not None:
        payload["postprocess"] = {
            "method": pp_report_dict["method"],
            "config": pp_report_dict["config"],
            "calibration": pp_report_dict["calibration"],
            "notes": pp_report_dict["notes"],
        }
    # Semantic availability indicator for the frontend
    payload["semantic"] = {
        "available": sem_probs is not None,
        "reason": (
            None if sem_probs is not None
            else "Checkpoint has no semantic auxiliary head"
        ),
    }
    return payload
