"""Object-storage service: maps application artifacts to object keys and
keeps the local workspace cache in sync with the durable store.

Design (migration spec sections 5/7):
    * the ML pipeline keeps writing to LOCAL directories (untouched
      inference code);
    * after a job completes, every generated artifact is uploaded to
      the object store and the DB stores its object KEY (never a filesystem path);
    * on read, a missing local artifact is transparently re-downloaded
      from the object store (read-through cache), so a backend restart
      on a fresh machine still serves results;
    * all keys are constructed from validated scene ids + the verified
      owner id — never from arbitrary user input.
"""

from __future__ import annotations

from pathlib import Path

from backend.app.core.logging import logger
from backend.app.core.paths import get_scene_output_dir, valid_scene_id
from backend.app.storage.backends import LocalStorage, S3Storage, build_backend


class StorageService:
    def __init__(self) -> None:
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            self._backend = build_backend()
        return self._backend

    @property
    def is_object_store(self) -> bool:
        return isinstance(self.backend, S3Storage)

    def ensure_bucket(self) -> None:
        """Idempotent; only meaningful for the S3 backend."""
        if isinstance(self.backend, S3Storage):
            self.backend.ensure_bucket()

    # -- key construction (server-only, validated inputs) -----------------

    def scene_input_key(self, owner_id: str, scene_id: str, filename: str) -> str:
        if not valid_scene_id(scene_id):
            raise ValueError("invalid scene id")
        safe_name = Path(filename).name
        return f"scenes/{owner_id}/{scene_id}/{safe_name}"

    def scene_output_prefix(self, owner_id: str, scene_id: str) -> str:
        if not valid_scene_id(scene_id):
            raise ValueError("invalid scene id")
        return f"results/{owner_id}/{scene_id}"

    def output_key(self, owner_id: str, scene_id: str, filename: str) -> str:
        return f"{self.scene_output_prefix(owner_id, scene_id)}/{Path(filename).name}"

    # -- persistence -------------------------------------------------------

    def upload_input(self, owner_id: str, scene_id: str, path: Path) -> str | None:
        """Upload a scene's validated input raster. Returns the key (or
        None with the local backend — the file is already local)."""
        if self.is_object_store:
            key = self.scene_input_key(owner_id, scene_id, path.name)
            self.backend.upload_file(path, key)
            logger.info("uploaded input object %s", key)
            return key
        return None

    def sync_scene_outputs(self, owner_id: str, scene_id: str) -> dict[str, dict]:
        """Upload every artifact in the scene's output dir; return the
        artifacts map for the DB ({name: {key, size}})."""
        output_dir = get_scene_output_dir(scene_id)
        artifacts: dict[str, dict] = {}
        if not output_dir.is_dir():
            return artifacts
        for path in sorted(output_dir.iterdir()):
            if not path.is_file():
                continue
            key = self.output_key(owner_id, scene_id, path.name)
            if self.is_object_store:
                self.backend.upload_file(path, key)
            artifacts[path.name] = {"key": key, "size": path.stat().st_size}
        return artifacts

    def ensure_scene_outputs_local(self, owner_id: str, scene_id: str, artifacts: dict) -> None:
        """Read-through cache: download any DB-registered artifact that is
        missing from the local output dir (e.g. after a restart)."""
        if not self.is_object_store or not artifacts:
            return
        output_dir = get_scene_output_dir(scene_id)
        for name, entry in artifacts.items():
            if not isinstance(entry, dict) or "key" not in entry:
                continue
            local = output_dir / name
            if local.is_file():
                continue
            try:
                self.backend.download_file(entry["key"], local)
            except Exception:
                logger.exception(
                    "failed to materialize artifact %s for scene %s", name, scene_id
                )

    def delete_scene_objects(self, owner_id: str, scene_id: str) -> int:
        if not self.is_object_store:
            return 0
        return self.backend.delete_prefix(self.scene_output_prefix(owner_id, scene_id))

    # -- signed URLs -------------------------------------------------------

    def presign_artifact(self, scene_id: str, path: Path) -> str:
        """Artifact URL for API payloads: an absolute short-lived presigned
        URL when the object store is active (uploads first if needed),
        else None so the caller keeps its legacy relative path. Only call
        AFTER authorization."""
        if not self.is_object_store:
            return None
        owner_id = self._owner_of(scene_id)
        if owner_id is None:
            return None
        url = self.publish_and_presign(owner_id, scene_id, path)
        return url

    def _owner_of(self, scene_id: str) -> str | None:
        try:
            from backend.app.db.database import session_scope
            from backend.app.db.models import SceneRow

            with session_scope() as session:
                row = session.get(SceneRow, scene_id)
                return row.owner_id if row is not None else None
        except Exception:
            logger.exception("owner lookup failed for %s", scene_id)
            return None

    def publish_and_presign(self, owner_id: str, scene_id: str, path: Path) -> str | None:
        """Ensure an artifact is in the object store, then return a
        short-lived GET URL. Only called AFTER authorization. Returns None
        with the local backend or when the local file does not exist yet
        (routes fall back to streaming / on-demand generation)."""
        if not self.is_object_store or not path.is_file():
            return None
        key = self.output_key(owner_id, scene_id, path.name)
        if not self.backend.exists(key):
            self.backend.upload_file(path, key)
        return self.backend.presigned_get(key)

    def presigned_url(self, key: str) -> str | None:
        """Short-lived GET URL (TTL = STORAGE_SIGNED_URL_TTL). Only called
        AFTER authorization; None when the backend cannot presign."""
        if key is None or not self.is_object_store:
            return None
        return self.backend.presigned_get(key)


storage_service = StorageService()
