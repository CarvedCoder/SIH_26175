"""Single-image inference with the frozen flagship.  [DEMO — not citable]

Two tracks (mirrors the problem statement):
  non-georeferenced (PNG/JPG) -> rDSM       : dsm.npy + dsm_preview.png
  georeferenced   (GeoTIFF)   -> metric DSM : + dsm.tif (CRS/transform preserved)

Track-2 anchoring (absolute DSM): pass --anchor-dem <DEM.tif> (bilinear
resample onto the image grid; requires a georeferenced input) or
--ground-elev <metres> (constant datum). Output is labeled

    ANCHORED (not learned)

because anchoring is arithmetic, not a model prediction.

--json-out writes the same scene payload the webapp consumes (grid + RGB +
stats) so the CLI and the FastAPI service stay in lockstep.

NOT an evaluation tool: no gates, no metrics. Citable numbers come from
`evaluate` only.

Usage:
  python model.py infer --config configs/infer.yaml --input IMG_0042.tif
  python model.py infer --input scene.tif --mode tiles --anchor-dem dem.tif
  python model.py infer --input photo.png --json-out payload.json --no-live
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from depthwizard.cli.args import (
    add_config_arg,
    add_device_arg,
    load_config,
    resolve_device,
)
from depthwizard.inference import run_inference

NAME = "infer"
HELP = "image -> AGL/DSM with the flagship (demo path, supports Track-2 anchoring)"


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        NAME,
        help=HELP,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arg(p, "configs/infer.yaml")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="override ckpt path (default: configs/infer.yaml)",
    )
    p.add_argument(
        "--architecture",
        choices=["auto", "calibration_net", "rdah"],
        default=None,
        help="height-model backend: rdah (official RDAH-Net, default) | "
        "calibration_net (legacy Phase-2 net) | auto (detect from the "
        "checkpoint payload)",
    )
    p.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="RDAH only: raw DAv2 depth x scale constant (default 40.0, "
        "the value verified against the released checkpoint)",
    )
    p.add_argument("--input", required=True, help="PNG / JPG / TIF image")
    p.add_argument(
        "--out", default=None, help="output dir (default: outputs/infer/<stem>)"
    )
    p.add_argument("--mode", choices=["auto", "crop", "resize", "tiles"], default=None)
    add_device_arg(p)
    p.add_argument(
        "--dn",
        default=None,
        help="matching raw Dn .npy; default: depth cache lookup, "
        "then LIVE Depth-Anything-V2",
    )
    p.add_argument("--cache-dir", default=None, help="depth cache dir override")
    p.add_argument(
        "--no-live",
        action="store_true",
        help="never run the live DAv2 backbone (fail if no cache)",
    )
    p.add_argument(
        "--backbone", default=None, help="live backbone model id (default from config)"
    )
    p.add_argument(
        "--anchor-dem", default=None, help="DEM/DTM raster for absolute DSM (Track 2)"
    )
    p.add_argument(
        "--ground-elev",
        type=float,
        default=None,
        help="constant ground elevation datum (metres)",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="write the webapp scene payload to this JSON path",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="skip writing dsm/preview files (payload only)",
    )
    p.add_argument(
        "--postprocess",
        default="none",
        help="AGL post-processing preset: none (default, byte-identical "
        "legacy path) | median | guided | bilateral | wls | conf | "
        "semantic | planar | full",
    )
    p.add_argument(
        "--pp-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override one PostProcessConfig field, repeatable "
        "(e.g. --pp-param wls_lambda=2.0 --pp-param guided_radius=8)",
    )
    p.add_argument(
        "--tta",
        action="store_true",
        help="flip/rotate test-time-augmentation ensemble inside the "
        "refinement (~3-4x inference cost; requires the live backbone)",
    )
    return p


def run(args) -> int:
    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    icfg = cfg.get("infer", {})
    mcfg = cfg.get("model", {}) or {}

    # Backend selection (RDAH integration): CLI --architecture > config
    # model.architecture > config infer.architecture > "auto" (detect
    # from the checkpoint payload — never a silent guess).
    architecture = (
        args.architecture
        or mcfg.get("architecture")
        or icfg.get("architecture")
        or "auto"
    )
    if architecture == "auto":
        architecture = None
    depth_scale = (
        args.depth_scale
        if args.depth_scale is not None
        else float(mcfg.get("depth_scale", icfg.get("depth_scale", 40.0)))
    )

    ckpt = (
        args.checkpoint
        or icfg.get("checkpoint")
        or mcfg.get("checkpoint")
    )
    if ckpt is None:
        if architecture == "rdah" or architecture is None:
            # default backend after the RDAH integration: the pretrained
            # official checkpoint (auto-downloads on first use)
            from depthwizard.rdah import RDAH_CKPT_DIR, RDAH_CHECKPOINTS

            ckpt = str(RDAH_CKPT_DIR / RDAH_CHECKPOINTS["track1"]["filename"])
        else:
            ckpt = str(
                Path(paths.get("outputs_dir", "outputs"))
                / "calib_net"
                / "rgb_cos"
                / "best.pt"
            )
    if not Path(ckpt).exists():
        if architecture == "rdah" or (
            architecture is None and "rdah" in str(ckpt)
        ):
            # the released checkpoint is fetched + MD5-verified on demand
            from depthwizard.rdah import ensure_rdah_checkpoint

            try:
                ckpt = str(ensure_rdah_checkpoint(ckpt))
            except Exception as e:  # noqa: BLE001 — loud, actionable error
                print(f"[error] {e}")
                return 1
        else:
            print(
                f"[error] checkpoint not found: {ckpt}\n"
                f"        train one (`python model.py train --out-tag rgb_cos`) or "
                f"pass --checkpoint."
            )
            return 1

    device = resolve_device(args.device or icfg.get("device", "auto"))
    mode = args.mode or icfg.get("mode", "auto")
    cache_dir = args.cache_dir or paths.get("depth_cache_dir")
    backbone_id = args.backbone or icfg.get(
        "backbone", "depth-anything/Depth-Anything-V2-Base-hf"
    )
    input_path = Path(args.input)
    out_dir = (
        Path(args.out)
        if args.out
        else Path(paths.get("outputs_dir", "outputs")) / "infer" / input_path.stem
    )

    pp_params: dict | None = None
    if args.pp_param:
        pp_params = {}
        for kv in args.pp_param:
            key, _, raw = kv.partition("=")
            if not key or not _:
                print(f"[error] --pp-param expects KEY=VALUE, got '{kv}'")
                return 1
            try:
                pp_params[key] = json.loads(raw)
            except json.JSONDecodeError:
                pp_params[key] = raw  # strings pass through unquoted

    payload = run_inference(
        input_path,
        ckpt,
        out_dir=out_dir,
        device=device,
        mode=mode,
        dn_path=args.dn,
        cache_dir=cache_dir,
        live_backbone=(not args.no_live) and bool(icfg.get("live_backbone", True)),
        backbone_id=backbone_id,
        anchor_dem=args.anchor_dem,
        ground_elev=args.ground_elev,
        write_files=(not args.no_write),
        postprocess=args.postprocess,
        postprocess_params=pp_params,
        tta=args.tta,
        architecture=architecture,
        depth_scale=depth_scale,
    )

    if args.json_out:
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"[out] {out_json}")
    return 0
