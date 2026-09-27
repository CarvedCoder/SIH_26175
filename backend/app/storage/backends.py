"""Storage backend interface + implementations.

S3Storage     — RustFS/any S3-compatible endpoint (boto3). Produces
                short-lived presigned GET URLs; credentials never leave
                the server. Data operations use the INTERNAL endpoint
                (fast, Docker-internal); presigned URLs are signed
                against the PUBLIC endpoint so the browser can actually
                reach the objects it is given.
LocalStorage  — development fallback when no object store is configured:
                files already live on local disk, "uploads" are no-ops and
                presigning is unsupported (routes fall back to streaming).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from backend.app.core.config import get_settings
from backend.app.core.logging import logger


def _endpoint_url(endpoint: str, secure: bool) -> str:
    """Accept either "host:port" (scheme chosen via S3_SECURE) or a full
    URL ("http://rustfs:9000") for both endpoint settings."""
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    scheme = "https" if secure else "http"
    return f"{scheme}://{endpoint}"


class StorageBackend(Protocol):
    def upload_file(self, path: Path, key: str) -> None: ...
    def download_file(self, key: str, destination: Path) -> None: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def exists(self, key: str) -> bool: ...
    def presigned_get(self, key: str) -> str | None:
        """Short-lived GET URL, or None when the backend cannot presign."""
        ...


class LocalStorage:
    """No-op backend: the local filesystem IS the store (dev fallback)."""

    def upload_file(self, path: Path, key: str) -> None:
        return None

    def download_file(self, key: str, destination: Path) -> None:
        raise FileNotFoundError(key)

    def delete_prefix(self, prefix: str) -> int:
        return 0

    def exists(self, key: str) -> bool:
        return False

    def presigned_get(self, key: str) -> str | None:
        return None


class S3Storage:
    """RustFS/S3 backend. Clients are created lazily; keys are server-
    constructed only (owner/scene ids are validated before a key is ever
    built), so user-controlled paths can never reach the object store.

    Two endpoints, one store:
      * S3_ENDPOINT (internal) — every data operation (upload, download,
        head, list, delete, bucket create). Inside Docker this is
        http://rustfs:9000; a container-to-container hop never leaves the
        compose network.
      * S3_PUBLIC_ENDPOINT (browser-facing, optional) — presigned URLs are
        SIGNED against this host so the signature matches the Host header
        the browser actually sends (SigV4 covers Host; rewriting the host
        after signing would corrupt the signature). Unset => presign
        against the internal endpoint (host-side dev, where both are
        localhost:9000).
    """

    def __init__(self) -> None:
        self._client = None
        self._presign_client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            s = get_settings()
            self._client = boto3.client(
                "s3",
                endpoint_url=_endpoint_url(s.s3_endpoint, s.s3_secure),
                aws_access_key_id=s.s3_access_key,
                aws_secret_access_key=s.s3_secret_key,
                region_name=s.s3_region or "us-east-1",
                config=Config(signature_version="s3v4"),
            )
        return self._client

    def _get_presign_client(self):
        """Client for presigning only: same credentials, browser-facing
        endpoint (falls back to the internal endpoint when no public
        endpoint is configured)."""
        s = get_settings()
        if not s.s3_public_endpoint:
            return self._get_client()
        if self._presign_client is None:
            import boto3
            from botocore.config import Config

            self._presign_client = boto3.client(
                "s3",
                endpoint_url=_endpoint_url(s.s3_public_endpoint, s.s3_secure),
                aws_access_key_id=s.s3_access_key,
                aws_secret_access_key=s.s3_secret_key,
                region_name=s.s3_region or "us-east-1",
                config=Config(signature_version="s3v4"),
            )
        return self._presign_client

    def ensure_bucket(self) -> None:
        """Idempotent bucket creation (dev convenience; never deletes)."""
        client = self._get_client()
        bucket = get_settings().s3_bucket
        from botocore.exceptions import ClientError

        try:
            client.head_bucket(Bucket=bucket)
        except ClientError:
            client.create_bucket(Bucket=bucket)
            logger.info("created object-storage bucket '%s'", bucket)

    def upload_file(self, path: Path, key: str) -> None:
        client = self._get_client()
        client.upload_file(str(path), get_settings().s3_bucket, key)

    def download_file(self, key: str, destination: Path) -> None:
        client = self._get_client()
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".part")
        client.download_file(get_settings().s3_bucket, key, str(tmp))
        tmp.replace(destination)

    def delete_prefix(self, prefix: str) -> int:
        client = self._get_client()
        bucket = get_settings().s3_bucket
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents", [])
        if not contents:
            return 0
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": obj["Key"]} for obj in contents]},
        )
        return len(contents)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._get_client().head_object(
                Bucket=get_settings().s3_bucket, Key=key
            )
            return True
        except ClientError:
            return False

    def presigned_get(self, key: str) -> str | None:
        client = self._get_presign_client()
        return client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": get_settings().s3_bucket,
                "Key": key,
            },
            ExpiresIn=get_settings().storage_signed_url_ttl,
        )


def build_backend() -> StorageBackend:
    """Backend selection from configuration (single source of truth)."""
    s = get_settings()
    if s.storage_backend == "local":
        logger.info("object storage: LOCAL filesystem (development fallback)")
        return LocalStorage()
    if s.storage_backend not in ("s3", "minio"):  # "minio": legacy alias
        logger.warning(
            "unknown STORAGE_BACKEND %r — falling back to the local "
            "filesystem backend.",
            s.storage_backend,
        )
        return LocalStorage()
    if not s.s3_access_key or not s.s3_secret_key:
        logger.warning(
            "STORAGE_BACKEND=s3 but S3_ACCESS_KEY/S3_SECRET_KEY are "
            "unset — falling back to the local filesystem backend."
        )
        return LocalStorage()
    logger.info("object storage: S3/RustFS endpoint=%s bucket=%s",
                s.s3_endpoint, s.s3_bucket)
    return S3Storage()
