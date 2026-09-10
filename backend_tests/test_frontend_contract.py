"""Frontend <-> backend contract test (audit P2.4 — the single most
durable regression guard).

Parses the ACTUAL frontend API modules (frontend/src/api/*.js) and asserts
every (method, path) pair they call exists in the live backend OpenAPI
schema. If the frontend gains an API call the backend does not implement —
or a backend route changes shape-breaking — this test fails instead of the
product (the audit's "404 storm" can never silently return).

The frontend's BASE_URL is ``<host>/api/v1``, so frontend paths are
resolved against the /api/v1 prefix.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FRONTEND_API_DIR = Path(__file__).resolve().parents[1] / "frontend" / "src" / "api"

API_PREFIX = "/api/v1"


def _parse_calls(source: str) -> list[tuple[str, str]]:
    """Extract (METHOD, path-template) from every apiFetch() call."""
    calls: list[tuple[str, str]] = []
    idx = 0
    while True:
        start = source.find("apiFetch(", idx)
        if start == -1:
            break
        idx = start + len("apiFetch(")
        # skip the function DEFINITION (async function apiFetch(...))
        prefix = source[max(0, start - 20) : start]
        if "function" in prefix:
            continue

        # --- first argument: quoted string or template literal ---
        while source[idx] in " \n\t":
            idx += 1
        quote = source[idx]
        assert quote in "'\"`", f"unexpected apiFetch argument near {source[idx:idx+30]!r}"
        end = source.index(quote, idx + 1)
        template = source[idx + 1 : end]

        # --- options object (up to the balancing close paren) ---
        depth = 0
        scan = end + 1
        while scan < len(source):
            char = source[scan]
            if char in "({[":
                depth += 1
            elif char in ")}]":
                if depth == 0:
                    break
                depth -= 1
            scan += 1
        options = source[end + 1 : scan]

        method_match = re.search(r"method:\s*'(\w+)'", options)
        method = method_match.group(1).upper() if method_match else "GET"
        calls.append((method, template))
        idx = scan

    return calls


def _collect_frontend_calls() -> set[tuple[str, str]]:
    calls: set[tuple[str, str]] = set()
    for module in sorted(FRONTEND_API_DIR.glob("*.js")):
        source = module.read_text(encoding="utf-8")
        calls.update(_parse_calls(source))

        if module.name == "export.js":
            # export.js navigates the browser directly to download URLs
            for export_type in re.findall(r"/scenes/\$\{id\}/export/(\w+)", source):
                calls.add(("GET", f"/scenes/${{id}}/export/{export_type}"))

        if module.name == "client.js":
            if "`${BASE_URL}/health`" in source:
                calls.add(("GET", "/health"))
    return calls


def _resolve_template(template: str) -> str:
    """Frontend template -> OpenAPI-style path (params become {param})."""
    path = re.sub(r"\$\{[^}]*\}", "{param}", template)
    path = path.split("?")[0]  # query strings are not part of routing
    if not path.startswith(API_PREFIX):
        path = API_PREFIX + path
    return path


def _matches(openapi_path: str, frontend_path: str) -> bool:
    openapi_parts = openapi_path.split("/")
    frontend_parts = frontend_path.split("/")
    if len(openapi_parts) != len(frontend_parts):
        return False
    return all(
        part.startswith("{") or part == frontend_part
        for part, frontend_part in zip(openapi_parts, frontend_parts)
    )


def _openapi_paths(client) -> dict[str, list[str]]:
    spec = client.get("/openapi.json").json()
    return {
        path: [method.upper() for method in ops]
        for path, ops in spec["paths"].items()
    }


@pytest.mark.parametrize(
    "method,template",
    sorted(_collect_frontend_calls()),
)
def test_frontend_call_is_served_by_backend(client, method, template):
    frontend_path = _resolve_template(template)
    paths = _openapi_paths(client)

    candidates = [
        (path, methods)
        for path, methods in paths.items()
        if _matches(path, frontend_path)
    ]
    assert candidates, (
        f"Frontend calls {method} {frontend_path} but the backend "
        f"implements no matching route (404 storm regression)."
    )
    assert any(method in methods for _path, methods in candidates), (
        f"Backend serves {frontend_path} but not with method {method} "
        f"(available: {candidates})."
    )


def test_backend_result_urls_exist_in_schema(client):
    """The canonical /results/* file routes stay in the contract."""
    paths = _openapi_paths(client)
    scene = "/api/v1/scenes/{scene_id}"
    for suffix in (
        "/results",
        "/results/preview",
        "/results/depth",
        "/results/dsm",
        "/results/error-map",
        "/results/reference",
        "/results/minimap",
        "/validation",
        "/validation/error-map",
        "/terrain",
        "/terrain/tiles",
        "/measure/height",
        "/measure/slope",
        "/refine",
        "/process",
        "/validate",
        "/reference",
        "/minimap",
        "/elevation",
    ):
        assert f"{scene}{suffix}" in paths, f"missing route: {scene}{suffix}"
