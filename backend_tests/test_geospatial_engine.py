"""Tests for Geospatial Engine: metric scale invariants, GSD derivation,
CRS handling, origin rebasing, and chunk tile streaming.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_geospatial_metadata_fields(client, processed_scene):
    """Verify that terrain metadata includes all required geospatial fields."""
    scene_id = processed_scene["scene"]["scene_id"]
    res = client.get(f"/api/v1/scenes/{scene_id}/terrain")
    assert res.status_code == 200
    body = res.json()

    assert "crs" in body
    assert "projected_crs" in body
    assert "affine_transform" in body
    assert "gsd_x" in body
    assert "gsd_y" in body
    assert "local_origin" in body
    assert "tile_config" in body

    tc = body["tile_config"]
    assert tc is not None
    assert tc["tile_size"] == 256
    assert "height_tile_url" in tc
    assert "texture_tile_url" in tc


def test_numerical_scale_invariants(client, tmp_path):
    """Verify exact scale relationships:
    GSD = 1.0m -> 1024px = 1024m
    GSD = 2.0m -> 1024px = 2048m
    GSD = 0.5m -> 1024px = 512m
    """
    from depthwizard.geo import pixel_size_metres
    import rasterio
    from rasterio.transform import from_origin

    cases = [
        (1.0, 1024, 1024.0),
        (2.0, 1024, 2048.0),
        (0.5, 1024, 512.0),
        (3.0, 512, 1536.0),
    ]

    crs = rasterio.crs.CRS.from_epsg(32617)  # UTM Zone 17N (metric)

    for gsd, px_dim, expected_m in cases:
        transform = from_origin(500000.0, 4000000.0, gsd, gsd)
        calc_gsd = pixel_size_metres(crs, transform)
        assert calc_gsd is not None
        assert abs(calc_gsd[0] - gsd) < 1e-6
        assert abs(calc_gsd[1] - gsd) < 1e-6

        world_size_m = px_dim * calc_gsd[0]
        assert abs(world_size_m - expected_m) < 1e-6


def test_chunk_tile_endpoints(client, processed_scene):
    """Verify that chunk tile endpoints stream valid 16-bit PNG height and RGB texture tiles."""
    scene_id = processed_scene["scene"]["scene_id"]

    # Height tile endpoint
    h_res = client.get(f"/api/v1/scenes/{scene_id}/terrain/tile/height?x=0&y=0&z=0&size=128")
    assert h_res.status_code == 200
    assert h_res.headers["content-type"] == "image/png"

    img_h = Image.open(io.BytesIO(h_res.content))
    assert img_h.size == (128, 128)
    assert img_h.mode in ("I;16", "I")

    # Texture tile endpoint
    t_res = client.get(f"/api/v1/scenes/{scene_id}/terrain/tile/texture?x=0&y=0&z=0&size=256")
    assert t_res.status_code == 200
    assert t_res.headers["content-type"] == "image/png"

    img_t = Image.open(io.BytesIO(t_res.content))
    assert img_t.size == (256, 256)


def test_chunk_tile_quadtree_levels(client, processed_scene):
    """Fractional-quadtree convention: z is the level, x/y grid indices < 2**z."""
    scene_id = processed_scene["scene"]["scene_id"]

    # Level 1 splits the raster into 2x2 tiles; all four must resolve.
    for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        res = client.get(
            f"/api/v1/scenes/{scene_id}/terrain/tile/height?x={x}&y={y}&z=1&size=64"
        )
        assert res.status_code == 200
        img = Image.open(io.BytesIO(res.content))
        assert img.size == (64, 64)

    tex_res = client.get(
        f"/api/v1/scenes/{scene_id}/terrain/tile/texture?x=1&y=0&z=1&size=64"
    )
    assert tex_res.status_code == 200
    assert Image.open(io.BytesIO(tex_res.content)).size == (64, 64)

    # Out-of-range indices are rejected with a client error, not a 500.
    bad = client.get(f"/api/v1/scenes/{scene_id}/terrain/tile/height?x=2&y=0&z=1&size=64")
    assert bad.status_code == 400
    bad_tex = client.get(f"/api/v1/scenes/{scene_id}/terrain/tile/texture?x=0&y=3&z=1&size=64")
    assert bad_tex.status_code == 400


def test_chunk_tile_shared_edge_pixels(client, processed_scene):
    """Adjacent tiles at the same level must sample the same source pixels on
    their shared edge: tile (0,0) right boundary == tile (1,0) left boundary."""
    from backend.app.services.terrain_service import TerrainService

    win_a = TerrainService._quadtree_window(1024, 0, 0, 1)
    win_b = TerrainService._quadtree_window(1024, 1, 0, 1)
    # (x0, x1, y0, y1)
    assert win_a[1] == win_b[0], "adjacent tiles must share the boundary pixel"

    # Level windows tile the full axis without gaps or overlaps.
    full = 1000  # deliberately not a power of two
    covered = 0
    for i in range(2):
        x0, x1, _, _ = TerrainService._quadtree_window(full, i, 0, 1)
        assert x0 == covered
        covered = x1
    assert covered == full
