"""Shared S3 helpers used by both the sync and async AWS blob stores.

The sync (:mod:`aws_blob_store`) and async (:mod:`async_blob.aws_async_blob_store`)
providers issue the same S3 requests and translate the same S3 responses, so the
header-mapping and error-detection logic lives here once. A change to an S3
header name, a replication-status spelling, or the conditional-write conflict
codes is then made in a single place instead of in lockstep across two files.
"""

from __future__ import annotations

from typing import Any, Optional

from botocore.exceptions import ClientError  # type: ignore[import-not-found]
from multicloud.blob.blob_client.driver import ChecksumAlgorithm, ReplicationStatus
from multicloud.common.exceptions import UnsupportedOperationException


def is_precondition_failed(exc: ClientError) -> bool:
    """Return True if *exc* is an S3 conditional-write precondition failure.

    A failed ``IfMatch`` / ``IfNoneMatch`` condition returns HTTP 412
    ``PreconditionFailed``: the precondition was evaluated and did not hold, so
    the caller lost a compare-and-swap or the object already existed. This is a
    terminal outcome, distinct from the retryable 409 conflict handled by
    :func:`is_conditional_conflict`.
    """
    error_info = exc.response.get("Error", {})
    code = error_info.get("Code", "")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return bool(code == "PreconditionFailed" or status == 412)


# Resource-level 404s that are NOT a missing target object. S3 accepts an
# access-point ARN in ``Bucket`` and returns ``NoSuchAccessPoint`` (HTTP 404) for
# a missing one, just as it returns ``NoSuchBucket`` for a missing bucket. Both
# must stay a genuine not-found rather than be normalized to a precondition
# failure, so they are excluded before the generic 404 check below.
_RESOURCE_NOT_FOUND_CODES = ("NoSuchBucket", "NoSuchAccessPoint")


def is_missing_object(exc: ClientError) -> bool:
    """Return True if *exc* is an S3 missing-object error (``NoSuchKey`` / 404).

    Deliberately excludes resource-level 404s (``NoSuchBucket``,
    ``NoSuchAccessPoint``) so a missing target object can be normalized to a
    precondition failure while a missing bucket or access point stays a genuine
    not-found.
    """
    error_info = exc.response.get("Error", {})
    code = error_info.get("Code", "")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if code in _RESOURCE_NOT_FOUND_CODES:
        return False
    return bool(code in ("NoSuchKey", "404") or status == 404)


def is_conditional_conflict(exc: ClientError) -> bool:
    """Return True if *exc* is an S3 ``ConditionalRequestConflict`` (HTTP 409).

    S3 returns 409 ``ConditionalRequestConflict`` when a concurrent operation
    races a conditional write: the precondition may never have been evaluated
    and the write may not have happened, so S3 documents 409 as a *retry*
    signal rather than a terminal precondition failure. It is deliberately not
    folded into :func:`is_precondition_failed` so callers can tell "retry me"
    apart from "the precondition failed".

    Matched on the ``ConditionalRequestConflict`` code alone, not on HTTP 409:
    other services surface unrelated 409s (e.g. IAM ``LimitExceeded``) that must
    not be reclassified as a conditional-write conflict.
    """
    error_info = exc.response.get("Error", {})
    code = error_info.get("Code", "")
    return bool(code == "ConditionalRequestConflict")


#: PutObject member that carries each algorithm's base64 digest, on both the
#: request (caller-supplied value) and the response (S3-computed value).
CHECKSUM_FIELDS = {
    ChecksumAlgorithm.CRC32C: "ChecksumCRC32C",
    ChecksumAlgorithm.MD5: "ChecksumMD5",
}


def is_bad_digest(exc: ClientError) -> bool:
    """Return True if *exc* is S3 rejecting a body whose checksum did not match.

    S3 returns HTTP 400 ``BadDigest`` when a supplied ``x-amz-checksum-*``
    value differs from the digest it computes over the received body.
    """
    error_info = exc.response.get("Error", {})
    return bool(error_info.get("Code", "") == "BadDigest")


def ensure_etag_quoted(etag: str) -> str:
    """Return *etag* wrapped in the double quotes S3's If-Match header expects.

    S3 stores and compares ETags as quoted strings (e.g. ``"d41d8..."``), but
    downloads expose the ETag with the quotes stripped (see ``_strip_quotes``)
    so callers get a bare hash. Passing that bare value straight back into
    ``IfMatch`` would send an unquoted token that S3 never matches, so the
    natural round-trip ``download().metadata.e_tag`` -> ``if_match=...`` would
    always fail the precondition. Re-quote here so the round-trip works.

    Idempotent: an already-quoted value (a caller who kept S3's raw ETag) is
    returned unchanged rather than double-quoted. Weak validators (``W/"..."``)
    are not used for conditional writes and are left untouched.
    """
    if etag.startswith('"') and etag.endswith('"'):
        return etag
    return f'"{etag}"'


def apply_write_preconditions(
    put_kwargs: dict[str, Any],
    *,
    create_if_absent: bool,
    if_match: Optional[str],
    if_version_match: Optional[str] = None,
) -> None:
    """Add S3 conditional-write headers for create-if-absent / CAS writes.

    ``create_if_absent`` maps to ``IfNoneMatch="*"`` (succeed only when no
    object exists at the key); ``if_match`` maps to ``IfMatch=<etag>`` (succeed
    only when the current object's ETag matches). Both surface an HTTP 412
    PreconditionFailed from S3 when the condition is not met.

    ``if_version_match`` has no S3 equivalent: S3 compare-and-swap is ETag-based
    (``IfMatch``), not version-id-based. A portable caller may set both tokens
    so each provider enforces with the one it supports; when ``if_match`` is
    also given, S3 enforces via ``IfMatch`` and the redundant ``if_version_match``
    is ignored. Only a version-conditioned write with no ``if_match`` cannot be
    expressed against S3, so that case raises UnsupportedOperationException
    rather than silently degrading to an unconditional overwrite.
    """
    if if_version_match is not None and if_match is None:
        raise UnsupportedOperationException(
            "if_version_match is not supported on S3: conditional overwrite is "
            "ETag-based here. Use if_match with the object's ETag instead."
        )
    if create_if_absent:
        put_kwargs["IfNoneMatch"] = "*"
    if if_match is not None:
        put_kwargs["IfMatch"] = ensure_etag_quoted(if_match)


def apply_kms(
    put_kwargs: dict[str, Any],
    *,
    kms_key_id: Optional[str],
    use_kms_managed_key: bool,
) -> None:
    """Add S3 server-side KMS encryption headers.

    An explicit ``kms_key_id`` selects that customer-managed key; setting only
    ``use_kms_managed_key`` requests SSE-KMS with the bucket's default
    (AWS-managed) key. Either path sets ``ServerSideEncryption="aws:kms"``.
    """
    if kms_key_id is not None:
        put_kwargs["ServerSideEncryption"] = "aws:kms"
        put_kwargs["SSEKMSKeyId"] = kms_key_id
    elif use_kms_managed_key:
        put_kwargs["ServerSideEncryption"] = "aws:kms"


# S3 reports replication state via the ReplicationStatus header. Its terminal
# "COMPLETED" spelling is normalized to the common ReplicationStatus.COMPLETE.
# Absent header (unreplicated object) → None.
_REPLICATION_STATUS_MAP = {
    "COMPLETE": ReplicationStatus.COMPLETE,
    "COMPLETED": ReplicationStatus.COMPLETE,
    "PENDING": ReplicationStatus.PENDING,
    "FAILED": ReplicationStatus.FAILED,
    "REPLICA": ReplicationStatus.REPLICA,
}


def map_replication_status(value: Optional[str]) -> Optional[ReplicationStatus]:
    """Map an S3 ReplicationStatus header value to the common enum."""
    if not value:
        return None
    return _REPLICATION_STATUS_MAP.get(value.upper())
