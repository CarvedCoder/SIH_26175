"""Inspect-models CLI command — ONNX I/O diagnostic.

Wraps depthwizard.disaster.inspect_models so the model signature check
is discoverable from the main CLI (model.py inspect-models) instead of
only as a python -m one-off.
"""

from __future__ import annotations

NAME = "inspect-models"
HELP = "print building/damage ONNX model I/O signatures (no inference)"


def add_parser(sub) -> None:
    sub.add_parser(NAME, help=HELP)


def run(args) -> int:
    from depthwizard.disaster.inspect_models import main

    main()
    return 0
