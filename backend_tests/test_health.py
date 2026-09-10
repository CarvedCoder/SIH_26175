"""Health, CORS, and auth contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Health (Phase 4 contract)
# ---------------------------------------------------------------------------

def test_health_canonical_contract(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]
    # frontend client.js reads model_loaded for the online indicator
    assert isinstance(body["model_loaded"], bool)


def test_health_platform_alias_matches_canonical(client):
    canonical = client.get("/api/v1/health").json()
    alias = client.get("/health")
    assert alias.status_code == 200
    assert alias.json() == canonical


def test_health_reports_model_loaded_false_without_checkpoint(client):
    response = client.get("/api/v1/health")
    assert response.json()["model_loaded"] is False


# ---------------------------------------------------------------------------
# CORS (Phase 17)
# ---------------------------------------------------------------------------

def test_cors_preflight_allowed_origin(client):
    response = client.options(
        "/api/v1/scenes",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_preflight_unlisted_origin_rejected(client):
    response = client.options(
        "/api/v1/scenes",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_cors_simple_request_echoes_origin(client):
    response = client.get(
        "/api/v1/health", headers={"Origin": "http://127.0.0.1:5173"}
    )
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"


# ---------------------------------------------------------------------------
# Auth (Phase 18)
# ---------------------------------------------------------------------------

def test_auth_disabled_by_default_is_explicit_in_settings(client):
    from backend.app.core.security import auth_enabled

    assert auth_enabled() is False


def test_auth_enforced_when_api_key_configured(client, monkeypatch):
    import backend.app.core.config as config_module
    from backend.app.core.security import API_KEY_HEADER

    monkeypatch.setenv("DW_API_KEY", "secret-key-123")
    config_module.get_settings.cache_clear()
    try:
        # missing key -> 401 envelope
        response = client.get("/api/v1/scenes")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

        # wrong key -> 401
        response = client.get(
            "/api/v1/scenes", headers={API_KEY_HEADER: "wrong"}
        )
        assert response.status_code == 401

        # correct key -> passes
        response = client.get(
            "/api/v1/scenes", headers={API_KEY_HEADER: "secret-key-123"}
        )
        assert response.status_code == 200

        # health stays public for infrastructure probes
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/health").status_code == 200
    finally:
        config_module.get_settings.cache_clear()
