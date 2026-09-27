"""Local-filesystem ArtifactStore — the development/test implementation.

Production swaps in an S3/RustFS implementation of the same
``domain.protocols.ArtifactStore`` protocol; the contract tests
(backend_tests/test_artifact_store.py) pin the behavior both must
satisfy. Business logic only ever sees artifact KEYS — never paths,
buckets, or URLs.

Key contract: keys are backend-generated, relative, slash-separated
(``scenes/<scene_id>/output/dsm.tif``). Traversal and absolute keys are
rejected — the same defense core/paths.py applies to scene ids.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterator

from backend.app.domain.entities import utc_now
from backend.app.domain.protocols import ArtifactMetadata, ArtifactRef

_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9/_.\-]*$")

_CHUNK = 1024 * 1024


class ArtifactKeyError(ValueError):
    """A storage key violated the backend key contract."""


class ArtifactNotFoundError(KeyError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def validate_key(key: str) -> str:
    """Reject empty/absolute/traversal keys loudly (never trust callers)."""
    if not key or key.startswith("/") or "\\" in key or ".." in key:
        raise ArtifactKeyError(f"illegal artifact key: {key!r}")
    if not _KEY_PATTERN.match(key):
        raise ArtifactKeyError(f"illegal artifact key: {key!r}")
    return key


class LocalArtifactStore:
    """ArtifactStore over a local directory tree (atomic writes)."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # -- mapping ------------------------------------------------------------

    def path_for(self, key: str) -> Path:
        validate_key(key)
        path = (self._root / key).resolve()
        if not str(path).startswith(str(self._root.resolve())):
            raise ArtifactKeyError(f"key escapes the store root: {key!r}")
        return path

    # -- ArtifactStore protocol ----------------------------------------------

    def put_bytes(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> ArtifactRef:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic: readers never see partial artifacts
        return ArtifactRef(key=key)

    def get_bytes(self, key: str) -> bytes:
        path = self.path_for(key)
        if not path.is_file():
            raise ArtifactNotFoundError(key)
        return path.read_bytes()

    def open(self, key: str) -> Iterator[bytes]:
        path = self.path_for(key)
        if not path.is_file():
            raise ArtifactNotFoundError(key)
        with path.open("rb") as f:
            while chunk := f.read(_CHUNK):
                yield chunk

    def exists(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def delete(self, key: str) -> bool:
        path = self.path_for(key)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def metadata(self, key: str) -> ArtifactMetadata:
        path = self.path_for(key)
        if not path.is_file():
            raise ArtifactNotFoundError(key)
        data = path.read_bytes()
        return ArtifactMetadata(
            key=key,
            size=len(data),
            content_type=_guess_content_type(key),
            checksum_sha256=hashlib.sha256(data).hexdigest(),
            created_at=utc_now(),
        )

    def list(self, prefix: str) -> list[str]:
        if prefix and (".." in prefix or prefix.startswith("/")):
            raise ArtifactKeyError(f"illegal artifact key prefix: {prefix!r}")
        base = self._root
        matches = sorted(
            str(p.relative_to(base))
            for p in base.glob(f"{prefix}*" if prefix else "*")
            if p.is_file()
        )
        return matches


def _guess_content_type(key: str) -> str:
    return {
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".npy": "application/x-npy",
        ".json": "application/json",
    }.get(Path(key).suffix.lower(), "application/octet-stream")
