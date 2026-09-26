"""Regression tests: the Depth/DSM viewer layer contract.

Guards the fixes for:
  * /depth serving the Matplotlib dsm_preview.png as a WebGL texture
    (must serve the Pillow-generated greyscale depth_layer.png);
  * DSM layer 404ing for non-georeferenced scenes (dsm.npy only);
  * uint16 source imagery wrapping into coloured static in rgb_preview.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from backend.app.services.terrain_service import TerrainService


@pytest.fixture()
def gradient_scene(client, uploaded_scene, mock_inference, processed_scene):
    """A processed scene whose dsm.npy is a known gradient (not zeros)."""
    from backend.app.services.result_service import result_service

    # Upgrade the mocked inference output to a known gradient DSM so the
    # texture content can be verified end-to-end.
    scene_id = uploaded_scene["scene_id"]
    out_dir = result_service.get_output_dir(scene_id)
    dsm = np.zeros((256, 256), dtype=np.float32)
    for x in range(256):
        dsm[:, x] = x / 255.0 * 10.0
    np.save(out_dir / "dsm.npy", dsm)
    return scene_id


def test_depth_meta_url_is_not_the_matplotlib_preview(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/depth").json()
    assert body["available"] is True
    url = body["url"]
    assert url is not None
    # The Matplotlib diagnostic preview lives at /results/preview; the
    # interactive layer texture must be the /results/depth-texture route
    # (or a presigned URL that the file route materializes).
    assert "/results/preview" not in url
    assert "/results/depth-texture" in url or url.startswith("http")


def test_depth_texture_is_browser_renderable_greyscale(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/depth").json()
    response = client.get(body["url"])
    assert response.status_code == 200

    from PIL import Image

    image = Image.open(io.BytesIO(response.content))
    # A clean single-channel data texture: exactly the DSM raster shape.
    # A Matplotlib figure would be RGBA at dpi-scaled figure size.
    assert image.mode == "L"
    assert image.size == (256, 256)


def test_depth_texture_matches_dsm_values(client, gradient_scene):
    scene_id = gradient_scene
    body = client.get(f"/api/v1/scenes/{scene_id}/depth").json()
    response = client.get(body["url"])
    assert response.status_code == 200

    from PIL import Image

    image = np.asarray(Image.open(io.BytesIO(response.content)), dtype=np.float32)
    # The DSM is an east-west gradient 0..10 m stretched to [0, 255]:
    # column 0 -> 0, column 255 -> 255. A terrain-colormapped Matplotlib
    # preview could never produce this monotonic single-channel ramp.
    assert image[0, 0] <= 2
    assert abs(image[0, 255] - 255) <= 2
    assert np.all(np.diff(image[0]) >= -1)


def test_dsm_meta_works_for_non_georeferenced_scenes(client, processed_scene):
    """Non-georeferenced scenes have no dsm.tif — the DSM layer must fall
    back to the predicted surface array instead of 404ing."""
    import numpy as np

    from backend.app.services.result_service import result_service

    scene_id = processed_scene["scene"]["scene_id"]
    # The processed upload is a GeoTIFF WITH a CRS; simulate a JPG/PNG
    # upload by removing the GeoTIFF twin.
    out_dir = result_service.get_output_dir(scene_id)
    (out_dir / "dsm.tif").unlink(missing_ok=True)
    dsm = np.zeros((256, 256), dtype=np.float32)
    np.save(out_dir / "dsm.npy", dsm)

    body = client.get(f"/api/v1/scenes/{scene_id}/dsm").json()
    assert body["available"] is True
    assert body["format"] == "npy"
    assert body["url"] is not None
    texture = client.get(body["url"])
    assert texture.status_code == 200

    from PIL import Image

    image = Image.open(io.BytesIO(texture.content))
    assert image.mode == "L"
    assert image.size == (256, 256)


def test_rgb_preview_stretches_uint16_instead_of_wrapping():
    """A 16-bit raster cast straight to uint8 wraps (value mod 256) and
    renders genuine imagery as uniform coloured static — the observed
    'RGB noise' failure. The preview must stretch instead."""
    rng = np.random.default_rng(0)
    # uint16 reflectance range, e.g. Cartosat DN 400..16000
    arr = rng.uniform(400, 16000, (64, 64, 3)).astype(np.float32)
    out = TerrainService._stretch_to_uint8(arr)
    assert out.dtype == np.uint8
    # Stretched output has real contrast (not a 256-value wrap of the high
    # bytes, which would still fill the full range but lose monotonicity —
    # the key property is ORDER preservation for a monotonic input).
    mono = np.linspace(400, 16000, 256, dtype=np.float32)
    mono_out = TerrainService._stretch_to_uint8(
        np.stack([mono, mono, mono], axis=-1)
    )
    assert np.all(np.diff(mono_out[:, 0].astype(np.int32)) >= 0)
    assert mono_out[0, 0] == 0 and mono_out[-1, 0] == 255

    # True 8-bit data passes through unchanged.
    eight_bit = rng.integers(0, 256, (32, 32, 3)).astype(np.float32)
    out8 = TerrainService._stretch_to_uint8(eight_bit)
    assert np.array_equal(out8, np.clip(eight_bit, 0, 255).astype(np.uint8))


def test_depth_layer_uses_pillow_not_matplotlib(client, gradient_scene, monkeypatch):
    """The visible depth texture must be generatable with Matplotlib
    uninstalled/forbidden: importing matplotlib must never happen on the
    depth-texture path."""
    import sys as _sys

    scene_id = gradient_scene
    body = client.get(f"/api/v1/scenes/{scene_id}/depth").json()

    class _Blocker:
        def find_module(self, name, path=None):  # legacy hook
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("matplotlib forbidden on texture paths")
            return None

        def find_spec(self, name, path=None, target=None):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("matplotlib forbidden on texture paths")
            return None

    # Remove any cached matplotlib modules so the blocker would trigger.
    saved = {
        k: v for k, v in _sys.modules.items()
        if k == "matplotlib" or k.startswith("matplotlib.")
    }
    for k in saved:
        del _sys.modules[k]
    _sys.meta_path.insert(0, _Blocker())
    try:
        response = client.get(f"/api/v1/scenes/{scene_id}/results/depth-texture")
        assert response.status_code == 200
    finally:
        _sys.meta_path.pop(0)
        _sys.modules.update(saved)
