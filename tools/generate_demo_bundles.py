"""
Generate precomputed demo scene bundles for tools/mock_backend_for_camera_test.py.

Each bundle under tools/demo_bundles/<scene_id>/ holds a complete set of
pipeline-style outputs (no live inference needed):
  scene.json      - every API payload the frontend consumes (results, terrain,
                    validation, error map, route defaults, heatmaps)
  heightmap.png   - 256x256 16-bit grayscale PNG of normalised heights (what
                    TerrainCanvas parses; heightmap.f32 kept as raw reference)
  dsm.npy         - 256x256 float32 DSM in metres
  preview.png     - fake RGB ortho input
  texture.png     - shaded-relief colour texture
  depth.png       - depth preview (viridis-ish)
  dsm.png         - DSM preview (terrain colour ramp)
  slope.png       - slope magnitude (grayscale)
  error_map.png   - validation error map (green-yellow-red)
  minimap.png     - bordered mini DSM preview
  reference.png   - reference DEM preview
  heatmap.png     - route passability heatmap
  passability.png - vehicle passability layer texture

Deterministic (seeded). Regenerate after tweaking terrain or metrics:
  python tools/generate_demo_bundles.py

The three bundles are the demo beats in docs/JURY_DEMO.md:
  demo_ridge  - georeferenced, best validation metrics (opening beat)
  demo_valley - georeferenced, high-relief (walkthrough/route beat)
  demo_quarry - relative mode, validation against a legacy survey DEM
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _demo_terrain import build_terrain, render_images

OUT_DIR = Path(__file__).resolve().parent / "demo_bundles"
SIZE = 256

SCENES = [
    {
        "scene_id": "demo_ridge",
        "filename": "ridge_traverse.tif",
        "beat": "Opening beat: best-in-class DSM accuracy + validation metrics.",
        "georeferenced": True,
        "crs": "EPSG:32643",
        "pixel_size": 12.0,
        "min_elev": 254.0,
        "max_elev": 578.0,
        "reference_source": "SRTM GL1 30m",
        "metrics": {"rmse": 3.12, "mae": 2.45, "correlation": 0.93, "sample_count": 41233},
        "error_bias": 1.4,   # mean error magnitude scale (metres) for the error map
        "shape": "ridge",
        "route": {"start": {"x": 38, "y": 200}, "end": {"x": 214, "y": 52}},
    },
    {
        "scene_id": "demo_valley",
        "filename": "valley_approach.tif",
        "beat": "High-relief walkthrough + Route Assist beat.",
        "georeferenced": True,
        "crs": "EPSG:32643",
        "pixel_size": 16.0,
        "min_elev": 402.0,
        "max_elev": 918.0,
        "reference_source": "SRTM GL1 30m",
        "metrics": {"rmse": 4.87, "mae": 3.61, "correlation": 0.89, "sample_count": 38874},
        "error_bias": 2.1,
        "shape": "valley",
        "route": {"start": {"x": 30, "y": 60}, "end": {"x": 226, "y": 210}},
    },
    {
        "scene_id": "demo_quarry",
        "filename": "quarry_overview.jpg",
        "beat": "Relative-mode scene (plain JPG) validated against a legacy survey DEM.",
        "georeferenced": False,
        "crs": None,
        "pixel_size": 3.0,
        "min_elev": 12.0,
        "max_elev": 132.0,
        "reference_source": "quarry_survey_2019_dem (uploaded reference)",
        "metrics": {"rmse": 2.04, "mae": 1.58, "correlation": 0.96, "sample_count": 51902},
        "error_bias": 0.9,
        "shape": "quarry",
        "route": {"start": {"x": 46, "y": 210}, "end": {"x": 208, "y": 66}},
    },
]

VEHICLES = [
    ("fire_truck", "Fire Truck", 15.0, 11.0),
    ("ambulance", "Ambulance", 12.0, 9.0),
    ("rescue_atv", "Rescue ATV", 30.0, 22.0),
    ("suv_4x4", "SUV 4x4", 20.0, 15.0),
    ("rescue_chopper", "Rescue Chopper", 45.0, 35.0),
]


def terrain_payload(scene, dsm, norm):
    span = scene["max_elev"] - scene["min_elev"]
    return {
        "available": True,
        "heightmap_url": "/api/v1/scenes/{sid}/results/heightmap",
        "texture_url": "/api/v1/scenes/{sid}/results/dsm-texture",
        "height_scale": round(span * 0.35, 2),
        "min_elevation": scene["min_elev"],
        "max_elevation": scene["max_elev"],
        "world_width_m": round(SIZE * scene["pixel_size"], 1),
        "world_depth_m": round(SIZE * scene["pixel_size"], 1),
        "is_georeferenced_scale": scene["georeferenced"],
        "terrain": {
            "scene_id": "{sid}",
            "dimensions": {"width": SIZE, "height": SIZE},
            "bounds": {"min_x": 0.0, "min_y": 0.0, "max_x": SIZE * scene["pixel_size"], "max_y": SIZE * scene["pixel_size"]},
            "coordinate_system": {"crs": scene["crs"], "units": "meters"},
            "elevation_mode": "absolute" if scene["georeferenced"] else "relative",
            "units": "m",
            "heightmap": {"name": "heightmap", "url": "/api/v1/scenes/{sid}/results/heightmap", "format": "f32", "width": SIZE, "height": SIZE},
            "texture": {"name": "dsm-texture", "url": "/api/v1/scenes/{sid}/results/dsm-texture", "format": "png", "width": SIZE, "height": SIZE},
            "minimap": {"name": "minimap", "url": "/api/v1/scenes/{sid}/results/minimap", "format": "png", "width": SIZE, "height": SIZE},
            "reference": {
                "available": True,
                "name": scene["reference_source"],
                "url": "/api/v1/scenes/{sid}/results/reference",
                "crs": scene["crs"],
                "units": "m",
                "width": SIZE,
                "height": SIZE,
            },
            "layers": [
                {"name": "rgb", "type": "image", "visible": False, "url": "/api/v1/scenes/{sid}/results/rgb"},
                {"name": "depth", "type": "image", "visible": False, "url": "/api/v1/scenes/{sid}/results/depth"},
                {"name": "dsm", "type": "image", "visible": True, "url": "/api/v1/scenes/{sid}/results/dsm"},
                {"name": "error", "type": "image", "visible": False, "url": "/api/v1/scenes/{sid}/results/error-map"},
                {"name": "slope", "type": "image", "visible": False, "url": "/api/v1/scenes/{sid}/results/slope"},
                {"name": "passability", "type": "image", "visible": False, "url": "/api/v1/scenes/{sid}/results/passability?vehicle=fire_truck"},
            ],
            "capabilities": {"walkthrough": True, "measurements": True, "layers": True, "route_assist": True},
            "metadata": {},
        },
    }


def results_payload(scene, dsm):
    stats = {
        "minimum": round(float(dsm.min()), 2),
        "maximum": round(float(dsm.max()), 2),
        "mean": round(float(dsm.mean()), 2),
        "median": round(float(np.median(dsm)), 2),
        "relief": round(float(dsm.max() - dsm.min()), 2),
        "units": "meters",
    }
    return {
        "scene_id": "{sid}",
        # ResultDashboard renders the RGB card from results.preview_url
        "preview_url": "/api/v1/scenes/{sid}/results/preview",
        "elevation_mode": "absolute" if scene["georeferenced"] else "relative",
        "units": "m",
        "available_layers": ["rgb", "depth", "dsm", "reference_dem", "error", "slope"],
        "reference_source": scene["reference_source"],
        "reference_dem_available": True,
        "min_elevation": scene["min_elev"],
        "max_elevation": scene["max_elev"],
        "resolution": {"width": SIZE, "height": SIZE},
        "depth": {
            "available": True,
            "preview": {"name": "depth", "url": "/api/v1/scenes/{sid}/results/depth", "format": "png", "size_bytes": None},
            "raw": {"name": "depth-raw", "url": "/api/v1/scenes/{sid}/export/depth", "format": "npy", "size_bytes": None},
            "width": SIZE,
            "height": SIZE,
            "statistics": stats,
        },
        "dsm": {
            "available": True,
            "preview": {"name": "dsm", "url": "/api/v1/scenes/{sid}/results/dsm", "format": "png", "size_bytes": None},
            "raster": {"name": "dsm-raster", "url": "/api/v1/scenes/{sid}/export/dsm", "format": "tif", "size_bytes": None},
            "width": SIZE,
            "height": SIZE,
            "statistics": stats,
            "crs": scene["crs"],
            "bounds": [0.0, 0.0, SIZE * scene["pixel_size"], SIZE * scene["pixel_size"]],
        },
        "capabilities": {"depth": True, "dsm": True, "terrain": True, "validation": True, "reference_comparison": True, "export": True},
        "assets": [
            {"name": "rgb", "url": "/api/v1/scenes/{sid}/results/rgb", "format": "png", "size_bytes": None},
            {"name": "depth", "url": "/api/v1/scenes/{sid}/results/depth", "format": "png", "size_bytes": None},
            {"name": "dsm", "url": "/api/v1/scenes/{sid}/results/dsm", "format": "png", "size_bytes": None},
            {"name": "error", "url": "/api/v1/scenes/{sid}/results/error-map", "format": "png", "size_bytes": None},
            {"name": "terrain", "url": "/api/v1/scenes/{sid}/terrain", "format": "json", "size_bytes": None},
        ],
        "metadata": {"beat": scene["beat"]},
    }


def validation_payload(scene, dsm):
    m = scene["metrics"]
    return {
        "scene_id": "{sid}",
        "available": True,
        "metrics": {
            "rmse": m["rmse"],
            "mae": m["mae"],
            "correlation": m["correlation"],
            "sample_count": m["sample_count"],
            "units": "meters",
        },
        "reference": {
            "available": True,
            "name": scene["reference_source"],
            "crs": scene["crs"],
            "units": "m",
            "width": SIZE,
            "height": SIZE,
        },
        "error_map": {"name": "error_map", "url": "/api/v1/scenes/{sid}/results/error-map", "format": "png"},
        "metadata": {"method": "co-registered reference DEM comparison", "beat": scene["beat"]},
    }


def build_scene(scene):
    out = OUT_DIR / scene["scene_id"]
    out.mkdir(parents=True, exist_ok=True)
    dsm, norm = build_terrain(scene)
    images = render_images(scene, dsm, norm)
    dsm.astype(np.float32).tofile(out / "heightmap.f32")
    np.save(out / "dsm.npy", dsm.astype(np.float32))
    for name, img in images.items():
        if name == "heightmap":
            # 16-bit grayscale PNG — the format TerrainCanvas parses natively
            Image.fromarray(img.astype(np.uint16)).save(out / "heightmap.png")
        else:
            Image.fromarray(img).save(out / f"{name}.png")

    payload = {
        "beat": scene["beat"],
        "filename": scene["filename"],
        "georeferenced": scene["georeferenced"],
        "create": {
            "scene_id": "{sid}",
            "filename": scene["filename"],
            "status": "completed",
            "format": "GeoTIFF" if scene["georeferenced"] else "JPEG",
            "dimensions": {"width": SIZE, "height": SIZE, "channels": 3},
            "georeference": {
                "available": scene["georeferenced"],
                "crs": scene["crs"],
                "min_x": 0.0 if scene["georeferenced"] else None,
                "min_y": 0.0 if scene["georeferenced"] else None,
                "max_x": SIZE * scene["pixel_size"] if scene["georeferenced"] else None,
                "max_y": SIZE * scene["pixel_size"] if scene["georeferenced"] else None,
                "pixel_width": scene["pixel_size"] if scene["georeferenced"] else None,
                "pixel_height": scene["pixel_size"] if scene["georeferenced"] else None,
            },
            "processing_path": "absolute" if scene["georeferenced"] else "relative",
            "capabilities": {
                "absolute_elevation": scene["georeferenced"],
                "relative_elevation": True,
                "reference_comparison": True,
                "slope": True,
                "height_measurement": True,
                "error_map": True,
                "local_refinement": True,
            },
        },
        "results": results_payload(scene, dsm),
        "terrain": terrain_payload(scene, dsm, norm),
        "validation": validation_payload(scene, dsm),
        "error_map": {
            "scene_id": "{sid}",
            "url": "/api/v1/scenes/{sid}/results/error-map",
            "units": "m",
            "min_error": 0.0,
            "max_error": round(scene["error_bias"] * 4.0, 2),
        },
        "depth_meta": {
            "scene_id": "{sid}",
            "available": True,
            "url": "/api/v1/scenes/{sid}/results/depth",
            "download_url": "/api/v1/scenes/{sid}/export/depth",
            "format": "png",
            "width": SIZE,
            "height": SIZE,
            "statistics": {
                "minimum": scene["min_elev"],
                "maximum": scene["max_elev"],
                "mean": round((scene["min_elev"] + scene["max_elev"]) / 2, 2),
                "median": round((scene["min_elev"] + scene["max_elev"]) / 2, 2),
                "relief": scene["max_elev"] - scene["min_elev"],
                "units": "meters",
            },
        },
        "dsm_meta": {
            "scene_id": "{sid}",
            "available": True,
            "url": "/api/v1/scenes/{sid}/results/dsm",
            "download_url": "/api/v1/scenes/{sid}/export/dsm",
            "format": "png",
            "width": SIZE,
            "height": SIZE,
            "crs": scene["crs"],
        },
        "reference_meta": {
            "scene_id": "{sid}",
            "available": True,
            "name": scene["reference_source"],
            "download_url": "/api/v1/scenes/{sid}/export/dsm",
            "visualization_url": "/api/v1/scenes/{sid}/results/reference",
            "crs": scene["crs"] or "scene-local",
            "units": "m",
        },
        "minimap": {
            "scene_id": "{sid}",
            "url": "/api/v1/scenes/{sid}/results/minimap",
            "image_url": "/api/v1/scenes/{sid}/results/minimap",
            "format": "png",
            "width": SIZE,
            "height": SIZE,
            "extent": {"min_x": 0.0, "min_z": 0.0, "max_x": 1.0, "max_z": 1.0},
            "bounds": {"min_x": 0.0, "min_y": 0.0, "max_x": 1.0, "max_y": 1.0},
        },
        "route_defaults": dict(scene["route"], vehicles=[v[0] for v in VEHICLES]),
        "vehicles": [
            {"key": k, "label": label, "max_slope_deg": mx, "caution_slope_deg": cx}
            for k, label, mx, cx in VEHICLES
        ],
        "heatmap": {
            k: {"blocked_pct": bp, "caution_pct": cp, "free_pct": round(100.0 - bp - cp, 1)}
            for k, bp, cp in [
                ("fire_truck", 18.4, 24.1),
                ("ambulance", 21.2, 26.8),
                ("rescue_atv", 2.6, 9.3),
                ("suv_4x4", 8.9, 17.5),
                ("rescue_chopper", 0.0, 3.1),
            ]
        },
    }
    (out / "scene.json").write_text(json.dumps(payload, indent=1))
    print(f"  wrote {out} ({len(list(out.iterdir()))} files)")


def main():
    print(f"Generating demo bundles in {OUT_DIR} ...")
    for scene in SCENES:
        build_scene(scene)
    print("Done. Serve with: python tools/mock_backend_for_camera_test.py")


if __name__ == "__main__":
    main()
