"""Launch the DepthWizard inference service (FastAPI) for the webapp.

This is the bridge that CONNECTS the frontend to the backend: the Next.js
`/api/predict` route proxies here (DW_API_URL), and this service runs the
SAME inference code path as `python model.py infer` (depthwizard.inference).
One code path, two doors — drift is impossible by construction.

Usage:
  python model.py serve --port 8000                # dev
  uvicorn service.api:app --host 0.0.0.0 --port 8000   # production (Docker)
"""

from __future__ import annotations

import argparse

NAME = "serve"
HELP = "launch the FastAPI inference service used by the webapp"


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        NAME,
        help=HELP,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="dev autoreload")
    return p


def run(args) -> int:
    import uvicorn

    print(f"[i] DepthWizard inference service on http://{args.host}:{args.port}")
    print(
        "[i] endpoints: GET /health, POST /predict (multipart: image, "
        "anchor_dem?, ground_elev?, mode?)"
    )
    uvicorn.run("service.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0
