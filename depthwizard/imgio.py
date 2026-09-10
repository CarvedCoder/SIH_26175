"""Input-image decoding + the shared uint8 RGB contract.

Two loaders, ONE contract (do not violate this elsewhere):

  * .png/.jpg/.jpeg -> Pillow decode (:func:`decode_rgb_u8`): robust for
    palette / alpha / grayscale / 16-bit / CMYK modes. NO geospatial
    metadata is invented — CRS stays None.
  * everything else (GeoTIFF + any GDAL-readable raster) -> rasterio,
    CRS/transform preserved exactly (the anchoring + dsm.tif output path
    depends on them).

Both end at the SAME training contract:

    rgb uint8 [H, W, 3], values in [0, 255], C-contiguous

which is what ``backbone.DepthAnythingBackbone.preprocess`` (Depth Anything
V2) and ``tifops.make_predict_fn`` (CalibrationNet) consume. Each of those
applies its own /255 + ImageNet normalization EXACTLY ONCE internally —
callers must never pre-normalize, and nothing here normalizes either.

Wider dtypes are range-checked, never blind-cast: a naive
``astype(np.uint8)`` wraps modulo 256 (uint16 300 -> 44) and silently
destroys the image (probe-confirmed on the old read_image).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Plain photos are decoded by Pillow; every other extension falls through
# to the rasterio path (GeoTIFF + any GDAL driver) so geospatial metadata
# is never lost.
_PILLOW_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


class InvalidImageError(ValueError):
    """Input cannot be decoded / does not satisfy the uint8 RGB contract.

    Subclasses ValueError so the serving layers' existing handler maps it
    to INVALID_INPUT / HTTP 400 automatically (service/api.py,
    backend processing_service.record_failure).
    """


# ---------------------------------------------------------------------------
# dtype policy — the ONLY sanctioned conversion to uint8
# ---------------------------------------------------------------------------


def to_uint8(arr: np.ndarray, source: str = "<input>") -> np.ndarray:
    """Range-checked conversion of an image array to uint8.

    Policy (deterministic, monotonic, never wraps, never stretches
    genuine 8-bit content):

      * uint8              -> returned unchanged
      * bool               -> 0/255
      * integers, all in [0, 255]
                           -> direct cast (8-bit content in a wider
                              container — bit-exact, no rescale)
      * integers, max > 255
                           -> linear rescale by the OBSERVED max so the
                              full 8-bit range is used (handles full-range
                              16-bit AND 11/12-bit-in-16-bit aerial data).
                              Negative integers are rejected — not image
                              data.
      * float with any NaN/Inf -> rejected (corrupted input, named loudly)
      * float in [0, 1]    -> x255 (normalized-float convention)
      * float in (1, 255]  -> rounded (already 8-bit range)
      * float outside [0, 255] -> rejected with the observed range — we
                              never guess a stretch for arbitrary floats

    Every applied rescale prints a one-line notice; nothing converts
    silently.
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.size == 0:
        raise InvalidImageError(f"{source}: image has no pixels")
    if arr.dtype == np.bool_:
        return (arr.astype(np.uint8)) * np.uint8(255)

    if np.issubdtype(arr.dtype, np.integer):
        lo = int(arr.min())
        if lo < 0:
            raise InvalidImageError(
                f"{source}: {arr.dtype} contains negative values (min {lo}) "
                "— not interpretable as image data. Convert to uint8 or "
                "float yourself with a documented mapping."
            )
        mx = int(arr.max())
        if mx <= 255:
            return arr.astype(np.uint8)
        scale = 255.0 / mx
        print(
            f"[i] {source}: {arr.dtype} range [0, {mx}] -> uint8 "
            f"(linear rescale x{scale:.6g})"
        )
        return np.rint(arr.astype(np.float64) * scale).astype(np.uint8)

    # floating point
    f = arr.astype(np.float64)
    n_bad = int((~np.isfinite(f)).sum())
    if n_bad:
        raise InvalidImageError(
            f"{source}: {n_bad} non-finite (NaN/Inf) pixels in a float "
            f"raster — corrupted or nodata-carrying input. Clean it or "
            "export uint8; refusing to cast garbage into the model input."
        )
    lo, hi = float(f.min()), float(f.max())
    if 0.0 <= lo and hi <= 1.0:
        print(f"[i] {source}: float {arr.dtype} in [0, 1] -> uint8 (x255)")
        return np.rint(f * 255.0).astype(np.uint8)
    if 0.0 <= lo and hi <= 255.0:
        print(f"[i] {source}: float {arr.dtype} in [0, 255] -> uint8 (round)")
        return np.rint(f).astype(np.uint8)
    raise InvalidImageError(
        f"{source}: float {arr.dtype} range [{lo:.4g}, {hi:.4g}] is not an "
        "image range — expected [0, 1] or [0, 255]. Rescale the data "
        "yourself with a documented mapping; refusing to guess."
    )


# ---------------------------------------------------------------------------
# Validation — the model-input contract
# ---------------------------------------------------------------------------


def validate_rgb_u8(rgb: np.ndarray, *, source: str = "<input>") -> None:
    """Fail loudly unless `rgb` satisfies the model-input contract:

        np.ndarray, dtype uint8, shape [H, W, 3], H >= 1, W >= 1,
        all values within [0, 255] (and therefore finite — uint8 cannot
        hold NaN/Inf).

    Raises :class:`InvalidImageError` with an actionable message.
    """
    if not isinstance(rgb, np.ndarray):
        raise InvalidImageError(
            f"{source}: expected a numpy array, got {type(rgb).__name__}"
        )
    if rgb.dtype != np.uint8:
        raise InvalidImageError(
            f"{source}: expected uint8 RGB (the /255 + ImageNet "
            f"normalization happens later, inside the models), got dtype "
            f"{rgb.dtype}"
        )
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise InvalidImageError(
            f"{source}: expected shape [H, W, 3] RGB, got {rgb.shape} — "
            "alpha/grayscale must be resolved before the model sees it"
        )
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h < 1 or w < 1:
        raise InvalidImageError(f"{source}: image has no pixels ({w}x{h})")
    if rgb.min() < 0 or rgb.max() > 255:
        raise InvalidImageError(
            f"{source}: pixel values outside [0, 255] "
            f"(min {rgb.min()}, max {rgb.max()})"
        )


# ---------------------------------------------------------------------------
# Pillow decode path (.png / .jpg / .jpeg)
# ---------------------------------------------------------------------------


def _pil_to_rgb_u8(im, source: str) -> np.ndarray:
    """PIL Image (any mode) -> uint8 [H, W, 3] array.

    Mode policy:
      * RGB                     -> as-is
      * high-bit-depth / float single channel (I, I;16*, F)
                                -> numeric range-checked conversion
                                (PIL's own convert() SILENTLY CLIPS
                                values > 255 — never use it here)
      * everything else (L, 1, LA, P, PA, RGBA, CMYK, ...)
                                -> PIL convert("RGB"): palette indices
                                are expanded to their palette colors,
                                alpha is DISCARDED (transparent pixels
                                keep their underlying RGB values), gray
                                is replicated to 3 channels.
    """
    mode = im.mode
    if mode in ("I", "F") or mode.startswith("I;16"):
        u8 = to_uint8(np.asarray(im), source=source)
        return np.repeat(u8[:, :, None], 3, axis=2)
    if mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def decode_rgb_u8(path: Path | str) -> tuple[np.ndarray, dict]:
    """Decode a PNG/JPG/JPEG with Pillow -> (rgb_u8 [H,W,3], info dict).

    info keys: ``format`` (PIL format string, e.g. "PNG"/"JPEG"),
    ``pil_mode`` (original mode BEFORE conversion — the audit trail for
    alpha/palette/16-bit handling), ``original_size`` [w, h].

    Decoding failures (corrupted, truncated, wrong extension, or a
    decompression-bomb guard hit) raise :class:`InvalidImageError` — the
    input is never silently skipped or garbage-cast.
    """
    path = Path(path)
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.load()  # force full decode; truncated files raise HERE
            info = {
                "format": im.format,
                "pil_mode": im.mode,
                "original_size": [int(im.width), int(im.height)],
            }
            if im.mode != "RGB":
                print(f"[i] {path.name}: mode {im.mode} -> RGB uint8")
            rgb = _pil_to_rgb_u8(im, source=path.name)
    except InvalidImageError:
        raise
    except Exception as e:  # PIL raises many unrelated types on bad data
        raise InvalidImageError(
            f"cannot decode '{path.name}' as an image "
            f"({type(e).__name__}: {e}). The file is corrupted, truncated, "
            "or not a supported image format."
        ) from e
    validate_rgb_u8(rgb, source=str(path))
    return np.ascontiguousarray(rgb), info


# ---------------------------------------------------------------------------
# Geospatial decode path (GeoTIFF + GDAL rasters) — rasterio preserved
# ---------------------------------------------------------------------------


def _read_georeferenced(path: Path) -> tuple[np.ndarray, dict]:
    """Rasterio read -> validated uint8 RGB + geospatial metadata.

    CRS/transform are carried through untouched (write_outputs writes
    dsm.tif from them; demprior anchors from them). Single-band rasters
    are replicated to 3 channels (the net's contract); bands beyond RGB
    are ignored (bands 1-3). Non-uint8 dtypes go through the shared
    range-checked :func:`to_uint8` — never a blind cast.
    """
    import rasterio

    try:
        with rasterio.open(path) as ds:
            bands = [1, 2, 3] if ds.count >= 3 else [1]
            raw = ds.read(bands)
            arr = np.stack([raw[0]] * 3, axis=0) if ds.count < 3 else raw
            profile = ds.profile.copy()
            crs = ds.crs
            transform = ds.transform
            original_dtype = ds.dtypes[0]
    except rasterio.RasterioIOError as e:
        raise InvalidImageError(
            f"cannot open '{path.name}' as a raster ({e}). If this is a "
            "plain PNG/JPG, keep the correct extension — decoding is "
            "dispatched by extension."
        ) from e

    rgb = np.ascontiguousarray(to_uint8(np.moveaxis(arr, 0, -1), source=path.name))
    validate_rgb_u8(rgb, source=str(path))
    meta = {
        "format": profile.get("driver"),
        "pil_mode": None,
        "original_dtype": original_dtype,
        "georeferenced": crs is not None,
        "crs": crs,
        "transform": transform,
        "rasterio_profile": profile,
    }
    return rgb, meta


# ---------------------------------------------------------------------------
# THE one preprocessing entry (CLI + service + backend all funnel here)
# ---------------------------------------------------------------------------


def preprocess_input_image(path: Path | str) -> tuple[np.ndarray, dict]:
    """Any supported input file -> (rgb_u8 [H,W,3] uint8, metadata dict).

    Dispatch (by extension, matching the georeferencing contract):

        .png/.jpg/.jpeg -> Pillow visual decode; georeferenced=False,
                           CRS/transform None (NEVER invented);
        anything else   -> rasterio (GeoTIFF + any GDAL raster),
                           CRS/transform preserved.

    No resize and no normalization happen here — load-time output is the
    full-resolution uint8 RGB the model paths consume.

    Metadata keys: ``source``, ``format``, ``pil_mode``,
    ``original_size`` [w, h], ``processed_size`` [w, h], ``channels``,
    ``dtype``, ``original_dtype``, ``georeferenced``, ``crs``,
    ``transform``, ``rasterio_profile`` (raw rasterio profile, or None
    for the Pillow path).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input image not found: {path}")
    if not path.is_file():
        raise ValueError(f"input path is not a file: {path}")

    if path.suffix.lower() in _PILLOW_SUFFIXES:
        rgb, info = decode_rgb_u8(path)
        meta = {
            "format": info["format"],
            "pil_mode": info["pil_mode"],
            "original_dtype": None,
            "georeferenced": False,
            "crs": None,
            "transform": None,
            "rasterio_profile": None,
        }
    else:
        rgb, meta = _read_georeferenced(path)

    meta.update(
        {
            "source": str(path),
            "original_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "processed_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "channels": 3,
            "dtype": str(rgb.dtype),
        }
    )
    return rgb, meta
