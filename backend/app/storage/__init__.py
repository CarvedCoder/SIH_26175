"""Object-storage abstraction (S3/MinIO + local fallback).

The durable object store is MinIO accessed through the standard S3 API
(boto3). The local filesystem remains the processing workspace: the ML
pipeline keeps writing files locally, and the service uploads final
artifacts to the object store after validation (spec section 7 — the
inference code is untouched).

Key layout (owner = Supabase user id; never derived from user input):
    scenes/{owner_id}/{scene_id}/input.<ext>
    results/{owner_id}/{scene_id}/<artifact filename>
"""

from __future__ import annotations

from backend.app.storage.service import storage_service

__all__ = ["storage_service"]
