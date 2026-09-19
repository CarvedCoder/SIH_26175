"""Object-storage service tests (local backend + S3 semantics).

Covers: key construction safety, upload/download round-trip and presigned
URL generation/expiry against a FAKE S3 client (no network, no MinIO
process required).
"""

from __future__ import annotations

import pytest

from backend.app.storage.backends import LocalStorage, S3Storage
from backend.app.storage.service import StorageService


# ---------------------------------------------------------------------------
# Key construction (security: no user-controlled path traversal)
# ---------------------------------------------------------------------------

def test_key_construction_is_hierarchical():
    service = StorageService()
    key = service.scene_input_key("user-abc", "scene_0123456789ab", "ortho.tif")
    assert key == "scenes/user-abc/scene_0123456789ab/ortho.tif"
    assert service.output_key("user-abc", "scene_0123456789ab", "dsm.tif") == (
        "results/user-abc/scene_0123456789ab/dsm.tif"
    )


def test_key_construction_rejects_invalid_scene_id():
    service = StorageService()
    with pytest.raises(ValueError):
        service.scene_input_key("user", "../../etc/passwd", "x.tif")
    with pytest.raises(ValueError):
        service.scene_output_prefix("user", "not-a-scene-id")


# ---------------------------------------------------------------------------
# Local backend behavior (dev fallback)
# ---------------------------------------------------------------------------

def test_local_backend_cannot_presign():
    backend = LocalStorage()
    assert backend.presigned_get("any/key") is None


def test_storage_service_local_mode_presign_artifact_returns_none(monkeypatch):
    from backend.app.storage import service as storage_module

    monkeypatch.setattr(storage_module.storage_service, "_backend", LocalStorage())
    assert storage_module.storage_service.is_object_store is False
    assert storage_module.storage_service.presign_artifact(
        "scene_0123456789ab", __import__("pathlib").Path("whatever.png")
    ) is None


# ---------------------------------------------------------------------------
# S3 backend against a fake client (upload/exists/presign/expire/TTL)
# ---------------------------------------------------------------------------

class FakeS3Client:
    """Minimal boto3 S3 client stand-in recording calls."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploaded: list[tuple[str, str]] = []
        self.presign_calls: list[dict] = []

    def head_bucket(self, Bucket):
        if Bucket != "depthwizard":
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    def upload_file(self, Filename, Bucket, Key):
        self.uploaded.append((Bucket, Key))
        self.objects[Key] = open(Filename, "rb").read()

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def generate_presigned_url(self, ClientMethod, Params, ExpiresIn):
        self.presign_calls.append({"params": Params, "expires_in": ExpiresIn})
        return (
            f"https://fake-minio/depthwizard/{Params['Key']}"
            f"?X-Amz-Expires={ExpiresIn}"
        )

    def download_file(self, Bucket, Key, Filename):
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "DownloadFile")
        with open(Filename, "wb") as f:
            f.write(self.objects[Key])

    def list_objects_v2(self, Bucket, Prefix):
        return {
            "Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)]
        }

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)
        return {"Deleted": Delete["Objects"]}


@pytest.fixture()
def s3_service(monkeypatch):
    fake = FakeS3Client()
    backend = S3Storage()
    backend._client = fake
    service = StorageService()
    service._backend = backend
    return service, fake


def test_s3_upload_and_exists(s3_service, tmp_path):
    service, fake = s3_service
    payload = tmp_path / "dsm.tif"
    payload.write_bytes(b"II*\x00 fake tiff")
    key = service.output_key("user-abc", "scene_0123456789ab", "dsm.tif")
    service.backend.upload_file(payload, key)
    assert service.backend.exists(key) is True
    assert service.backend.exists("results/user-abc/scene_0123456789ab/nope.tif") is False
    assert fake.uploaded == [("depthwizard", key)]


def test_s3_presigned_url_carries_configured_ttl(s3_service, tmp_path, monkeypatch):
    import backend.app.core.config as config_module

    service, fake = s3_service
    monkeypatch.setenv("STORAGE_SIGNED_URL_TTL", "42")
    config_module.get_settings.cache_clear()
    try:
        payload = tmp_path / "dsm.tif"
        payload.write_bytes(b"data")
        key = service.output_key("user-abc", "scene_0123456789ab", "dsm.tif")
        service.backend.upload_file(payload, key)
        url = service.backend.presigned_get(key)
        assert url is not None
        assert "X-Amz-Expires=42" in url
        assert fake.presign_calls[0]["expires_in"] == 42
        # the URL must never contain credentials
        assert "MINIO_SECRET" not in url and "secret" not in url
    finally:
        config_module.get_settings.cache_clear()


def test_s3_sync_outputs_registers_artifacts(s3_service, tmp_path, monkeypatch):
    import backend.app.core.paths as paths
    from backend.app.db.database import init_db, reset_engine_for_tests, session_scope
    from backend.app.db.models import SceneRow

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    reset_engine_for_tests()
    init_db()

    scene_id = "scene_0123456789ab"
    out = paths.get_scene_output_dir(scene_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dsm.npy").write_bytes(b"array")
    (out / "dsm_preview.png").write_bytes(b"png")

    with session_scope() as session:
        session.add(
            SceneRow(
                scene_id=scene_id, owner_id="user-abc", filename="o.tif",
                input_ext=".tif", raster_metadata={},
            )
        )

    service, fake = s3_service
    artifacts = service.sync_scene_outputs("user-abc", scene_id)
    assert set(artifacts) == {"dsm.npy", "dsm_preview.png"}
    assert fake.uploaded and all(b == "depthwizard" for b, _ in fake.uploaded)

    # local cache-miss path: remove the local copies, re-materialize
    for child in out.iterdir():
        child.unlink()
    service.ensure_scene_outputs_local("user-abc", scene_id, artifacts)
    assert (out / "dsm.npy").is_file()
    assert (out / "dsm_preview.png").is_file()

    reset_engine_for_tests()


def test_s3_delete_scene_objects(s3_service, tmp_path):
    service, fake = s3_service
    scene_id = "scene_0123456789ab"
    for name in ("dsm.npy", "preview.png"):
        fake.objects[f"results/user-abc/{scene_id}/{name}"] = b"x"
    removed = service.delete_scene_objects("user-abc", scene_id)
    assert removed == 2
    assert not fake.objects
