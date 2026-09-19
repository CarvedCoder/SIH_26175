"""Authentication tests: Supabase JWT verification + scene ownership.

Covers the migration spec's AUTH matrix:
    * unauthenticated request -> 401
    * invalid/expired JWT     -> 401
    * authenticated user      -> allowed
    * user A cannot access user B's scene -> 403
"""

from __future__ import annotations

import jwt as pyjwt
import pytest

import backend.app.core.config as config_module


SECRET = "test-supabase-jwt-secret"


def make_token(user_id: str, secret: str = SECRET, **overrides) -> str:
    payload = {
        "sub": user_id,
        "email": f"{user_id}@example.com",
        "role": "authenticated",
        "aud": "authenticated",
    }
    payload.update(overrides)
    return pyjwt.encode(payload, secret, algorithm="HS256")


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def supabase_auth(monkeypatch):
    """Enable Supabase auth mode with a known HS256 secret."""
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SECRET)
    monkeypatch.setenv("SUPABASE_URL", "")
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


def test_unauthenticated_request_401(client, supabase_auth):
    response = client.get("/api/v1/scenes")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_invalid_jwt_401(client, supabase_auth):
    response = client.get(
        "/api/v1/scenes", headers=auth_header("not-a-real-token")
    )
    assert response.status_code == 401


def test_wrong_secret_jwt_401(client, supabase_auth):
    response = client.get(
        "/api/v1/scenes", headers=auth_header(make_token("u1", secret="wrong"))
    )
    assert response.status_code == 401


def test_expired_jwt_401(client, supabase_auth):
    import datetime

    expired = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
    token = pyjwt.encode(
        {"sub": "u1", "aud": "authenticated", "role": "authenticated",
         "exp": expired},
        SECRET,
        algorithm="HS256",
    )
    response = client.get("/api/v1/scenes", headers=auth_header(token))
    assert response.status_code == 401


def test_authenticated_user_allowed(client, supabase_auth):
    """Scene creation works with a valid Supabase JWT, and the scene row
    records the JWT sub as the owner (never a payload-supplied id)."""
    import io

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    buffer = io.BytesIO()
    with rasterio.open(
        buffer, "w", driver="GTiff", height=64, width=64, count=3,
        dtype="uint8", crs="EPSG:32617", transform=from_origin(0, 64, 1, 1),
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))
    buffer.seek(0)

    response = client.post(
        "/api/v1/scenes",
        files={"file": ("a.tif", buffer, "image/tiff")},
        headers=auth_header(make_token("user-aaa")),
    )
    assert response.status_code == 200, response.text

    from backend.app.db.database import session_scope
    from backend.app.db.models import SceneRow

    scene_id = response.json()["scene_id"]
    with session_scope() as session:
        row = session.get(SceneRow, scene_id)
        assert row is not None
        assert row.owner_id == "user-aaa"

    # the owner can list and see their scene
    listing = client.get("/api/v1/scenes", headers=auth_header(make_token("user-aaa")))
    assert listing.status_code == 200
    assert scene_id in [s["scene_id"] for s in listing.json()]


def test_other_user_scene_access_403(client, supabase_auth):
    """User B cannot read, process, or export user A's scene."""
    import io

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    buffer = io.BytesIO()
    with rasterio.open(
        buffer, "w", driver="GTiff", height=64, width=64, count=3,
        dtype="uint8", crs="EPSG:32617", transform=from_origin(0, 64, 1, 1),
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))
    buffer.seek(0)

    created = client.post(
        "/api/v1/scenes",
        files={"file": ("a.tif", buffer, "image/tiff")},
        headers=auth_header(make_token("user-aaa")),
    )
    assert created.status_code == 200
    scene_id = created.json()["scene_id"]

    intruder = auth_header(make_token("user-bbb"))
    assert client.get(f"/api/v1/scenes/{scene_id}", headers=intruder).status_code == 403
    assert (
        client.post(f"/api/v1/scenes/{scene_id}/process", json={}, headers=intruder).status_code
        == 403
    )
    assert (
        client.get(f"/api/v1/scenes/{scene_id}/results", headers=intruder).status_code == 403
    )
    assert (
        client.get(f"/api/v1/scenes/{scene_id}/export/dsm", headers=intruder).status_code
        == 403
    )
    assert (
        client.delete(f"/api/v1/scenes/{scene_id}", headers=intruder).status_code == 403
    )

    # and the scene does not appear in the intruder's listing
    listing = client.get("/api/v1/scenes", headers=intruder).json()
    assert scene_id not in [s["scene_id"] for s in listing]


def test_job_ownership_enforced(client, supabase_auth, mock_inference):
    """User B cannot poll or cancel user A's job."""
    import io

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    buffer = io.BytesIO()
    with rasterio.open(
        buffer, "w", driver="GTiff", height=64, width=64, count=3,
        dtype="uint8", crs="EPSG:32617", transform=from_origin(0, 64, 1, 1),
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))
    buffer.seek(0)

    owner = auth_header(make_token("user-aaa"))
    created = client.post(
        "/api/v1/scenes",
        files={"file": ("a.tif", buffer, "image/tiff")},
        headers=owner,
    )
    scene_id = created.json()["scene_id"]
    job = client.post(
        f"/api/v1/scenes/{scene_id}/process", json={}, headers=owner
    ).json()
    job_id = job["job_id"]

    intruder = auth_header(make_token("user-bbb"))
    assert client.get(f"/api/v1/jobs/{job_id}", headers=intruder).status_code == 403
    assert (
        client.post(f"/api/v1/jobs/{job_id}/cancel", headers=intruder).status_code == 403
    )
    # owner still can
    assert client.get(f"/api/v1/jobs/{job_id}", headers=owner).status_code == 200


def test_health_stays_public(client, supabase_auth):
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/health").status_code == 200
