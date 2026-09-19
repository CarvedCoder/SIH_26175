"""ArtifactStore contract tests.

Pins the behavior ANY ArtifactStore implementation must satisfy, so the
LocalArtifactStore (development) and a future S3/MinIO implementation
(production) are interchangeable behind the same protocol
(backend/app/domain/protocols.py).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.domain.protocols import ArtifactStore
from backend.app.infrastructure.storage.local_artifact_store import (
    ArtifactKeyError,
    ArtifactNotFoundError,
    LocalArtifactStore,
)


@pytest.fixture
def store(tmp_path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "artifacts")


def _contract_suite(store: ArtifactStore) -> None:
    """The full behavioral contract, implementation-agnostic."""
    # put/get round-trip
    ref = store.put_bytes("scenes/s1/output/dsm.npy", b"\x00\x01\x02")
    assert ref.key == "scenes/s1/output/dsm.npy"
    assert store.get_bytes(ref.key) == b"\x00\x01\x02"
    assert store.exists(ref.key) is True
    assert store.exists("scenes/s1/output/missing.npy") is False

    # metadata: size + honest sha256 + content type by suffix
    meta = store.metadata(ref.key)
    assert meta.size == 3
    assert meta.checksum_sha256 == hashlib.sha256(b"\x00\x01\x02").hexdigest()
    assert meta.content_type == "application/x-npy"

    # streaming open
    assert b"".join(store.open(ref.key)) == b"\x00\x01\x02"

    # list by prefix
    store.put_bytes("scenes/s1/output/dsm.tif", b"tif")
    keys = store.list("scenes/s1/output/")
    assert keys == ["scenes/s1/output/dsm.npy", "scenes/s1/output/dsm.tif"]

    # atomic overwrite: latest write wins, no partial reads
    store.put_bytes(ref.key, b"newer")
    assert store.get_bytes(ref.key) == b"newer"

    # delete + idempotent-not
    assert store.delete(ref.key) is True
    assert store.exists(ref.key) is False
    assert store.delete(ref.key) is False
    with pytest.raises(ArtifactNotFoundError):
        store.get_bytes(ref.key)


def test_local_artifact_store_contract(tmp_path):
    _contract_suite(LocalArtifactStore(tmp_path / "artifacts"))


def test_store_rejects_traversal_and_absolute_keys(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    for bad in (
        "../escape.npy",
        "/absolute/path.npy",
        "a/../../b.npy",
        "..\\windows.npy",
        "",
    ):
        with pytest.raises(ArtifactKeyError):
            store.put_bytes(bad, b"x")
        with pytest.raises(ArtifactKeyError):
            store.path_for(bad)


def test_local_store_implements_domain_protocol(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    assert isinstance(store, ArtifactStore)


def test_deleted_scenes_leave_no_artifacts(tmp_path):
    """Delete semantics: removing a scene prefix removes every artifact
    under it (business layer composes this; the store stays primitive)."""
    store = LocalArtifactStore(tmp_path / "artifacts")
    store.put_bytes("scenes/s1/output/a.npy", b"1")
    store.put_bytes("scenes/s1/output/b.npy", b"2")
    for key in store.list("scenes/s1/"):
        store.delete(key)
    assert store.list("scenes/s1/") == []
