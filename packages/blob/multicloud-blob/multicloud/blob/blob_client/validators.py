"""Input validation for blob storage operations.

Validators raise :class:`InvalidArgumentException` for any input that
fails preconditions. Driver methods invoke them before delegating to
``do_*`` provider hooks so providers don't have to repeat validation.

Generic helpers (:meth:`require_not_blank`, :meth:`require_not_empty`,
:meth:`require_equals_ignore_case`) are intended for reuse — both by
operation-specific validators on this class and by future services.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Collection, Mapping, Optional
from urllib.parse import urlsplit

from multicloud.common.exceptions import InvalidArgumentException

# Error message constants. Kept module-level so tests can import them and
# match against raised messages without duplicating the strings.
INVALID_BUCKET_NAME_MSG = "Bucket name cannot be null or empty"
INVALID_OBJECT_NAME_MSG = "Object name cannot be null or empty"
INVALID_OBJECT_NAME_COLLECTION_MSG = (
    "Collection of object names cannot be null or empty"
)
INVALID_TAGS_COLLECTION_MSG = "Map of tags cannot be null or empty"
MISMATCHED_BUCKET_NAME_MSG = "Bucket names must be equal"
DURATION_INVALID_MSG = (
    "Duration must be a non-null non-zero positive value. Duration was '{0}'"
)
PRESIGNED_URL_OPERATION_INVALID_MSG = (
    "The presigned url operation type must be non-null"
)
INVALID_ENDPOINT_NULL_MSG = "Endpoint cannot be null"
INVALID_ENDPOINT_PROTOCOL_MSG = (
    "Endpoint must have a protocol of http or https. value={0}"
)
INVALID_ENDPOINT_HOSTNAME_MSG = (
    "Endpoint must have a non-empty hostname value. value={0}"
)
INVALID_ENDPOINT_PORT_MSG = (
    "Endpoint must have a non-negative value for the port number. value={0}"
)
INVALID_ENDPOINT_NO_PATH_ALLOWED_MSG = (
    "Endpoint must not have a path specified. value={0}"
)
INVALID_ENDPOINT_NO_QUERY_PARAMS_ALLOWED_MSG = (
    "Endpoint must not have query params specified. value={0}"
)
PRESIGNED_DURATION_TOO_LARGE_MSG = (
    "generate_presigned_url: duration_seconds cannot exceed 604800 (7 days)"
)
INVALID_RANGED_READ_NEGATIVE_BOUNDARIES_MSG = (
    "Ranged read boundaries cannot be negative. start={0} end={1}"
)
INVALID_RANGED_READ_BOUNDARIES_MSG = (
    "Ranged read start cannot be larger than end. start={0} end={1}"
)
INVALID_OBJECT_NAME_TOO_LONG_MSG = "Object name cannot exceed {0} bytes. length={1}"
RETENTION_CONFIG_NULL_MSG = "Retention config cannot be null"
RETENTION_RETAIN_UNTIL_NULL_MSG = "Retention retain_until_date cannot be null"
RETENTION_RETAIN_UNTIL_NAIVE_MSG = (
    "Retention retain_until_date must be timezone-aware (an absolute instant). "
    "value={0}"
)
INVALID_CONTENT_LENGTH_MSG = (
    "content_length must be a non-negative integer when provided. value={0}"
)
UPLOAD_ONLY_PARAM_ON_DOWNLOAD_MSG = (
    "{0} is only valid for UPLOAD presigned URLs, not DOWNLOAD"
)
OBJECT_LOCK_MODE_REQUIRES_DATE_MSG = (
    "object_lock: retain_until_date is required when mode is set"
)
OBJECT_LOCK_DATE_REQUIRES_MODE_MSG = (
    "object_lock: mode is required when retain_until_date is set"
)
OBJECT_LOCK_NAIVE_DATE_MSG = (
    "object_lock: retain_until_date must be a timezone-aware datetime. value={0}"
)
CREATE_IF_ABSENT_WITH_IF_MATCH_MSG = (
    "create_if_absent and if_match are mutually exclusive — "
    "create_if_absent requires the object to be absent, if_match requires "
    "it to be present with a matching ETag"
)
EMPTY_IF_MATCH_MSG = (
    "if_match must be a non-empty ETag; pass None to skip the precondition"
)
CREATE_IF_ABSENT_WITH_IF_VERSION_MATCH_MSG = (
    "create_if_absent and if_version_match are mutually exclusive — "
    "create_if_absent requires the object to be absent, if_version_match "
    "requires it to be present with a matching version"
)
EMPTY_IF_VERSION_MATCH_MSG = (
    "if_version_match must be a non-empty version id; pass None to skip the "
    "precondition"
)
EMPTY_STORAGE_CLASS_MSG = (
    "storage_class must be a non-empty provider storage class; pass None to use "
    "the bucket default"
)


def _is_timezone_aware(value: datetime) -> bool:
    """Return True when *value* carries a concrete UTC offset."""
    return value.tzinfo is not None and value.utcoffset() is not None


# Maximum allowed presigned URL lifetime (7 days, in seconds).
MAX_PRESIGNED_DURATION_SECONDS = 604800

# Maximum object key length enforced at the cloud-agnostic boundary,
# measured in UTF-8 bytes. Supported providers all cap at this value or higher.
MAX_OBJECT_NAME_LENGTH_BYTES = 1024


class BlobStoreValidator:
    """Input validator for blob store operations.

    Raises :class:`InvalidArgumentException` on bad input.
    """

    def require_not_blank(self, value: Optional[str], msg: str) -> None:
        # "blank" = None, "", or whitespace-only. `not value` alone would
        # let "   " through, so strip first.
        if value is None or not value.strip():
            raise InvalidArgumentException(msg)

    def require_not_empty(
        self, collection: Optional[Collection[Any]], msg: str
    ) -> None:
        if collection is None or len(collection) == 0:
            raise InvalidArgumentException(msg)

    def require_equals_ignore_case(
        self, first: Optional[str], second: Optional[str], msg: str
    ) -> None:
        a = first.casefold() if first is not None else None
        b = second.casefold() if second is not None else None
        if a != b:
            raise InvalidArgumentException(msg)

    def validate_key(self, key: str) -> None:
        self.require_not_blank(key, INVALID_OBJECT_NAME_MSG)
        # Length is measured in UTF-8 bytes — cloud providers enforce the
        # limit on the wire, not in code points.
        encoded_len = len(key.encode("utf-8"))
        if encoded_len > MAX_OBJECT_NAME_LENGTH_BYTES:
            raise InvalidArgumentException(
                INVALID_OBJECT_NAME_TOO_LONG_MSG.format(
                    MAX_OBJECT_NAME_LENGTH_BYTES, encoded_len
                )
            )

    def validate_bucket(self, bucket: str) -> None:
        self.require_not_blank(bucket, INVALID_BUCKET_NAME_MSG)

    def validate_keys(self, keys: Collection[str]) -> None:
        self.require_not_empty(keys, INVALID_OBJECT_NAME_COLLECTION_MSG)
        for key in keys:
            self.validate_key(key)

    def validate_tags(self, tags: Mapping[str, str]) -> None:
        self.require_not_empty(tags, INVALID_TAGS_COLLECTION_MSG)

    def validate_duration(self, duration_seconds: Optional[int]) -> None:
        if duration_seconds is None or duration_seconds <= 0:
            raise InvalidArgumentException(
                DURATION_INVALID_MSG.format(duration_seconds)
            )

    def validate_presigned_operation_type(self, operation: Any) -> None:
        if operation is None:
            raise InvalidArgumentException(PRESIGNED_URL_OPERATION_INVALID_MSG)

    def validate_range(self, start: Optional[int], end: Optional[int]) -> None:
        # `start` and `end` are inclusive byte positions (start=0, end=0
        # means "first byte only"), not Python's half-open convention.
        # Either bound may be None (open-ended); both must be non-negative
        # when given; when both given, end >= start.
        if (start is not None and start < 0) or (end is not None and end < 0):
            raise InvalidArgumentException(
                INVALID_RANGED_READ_NEGATIVE_BOUNDARIES_MSG.format(start, end)
            )
        if start is not None and end is not None and end < start:
            raise InvalidArgumentException(
                INVALID_RANGED_READ_BOUNDARIES_MSG.format(start, end)
            )

    def validate_endpoint(self, endpoint: Optional[str], require_port: bool) -> None:
        # Endpoint rules: non-null, http(s) scheme, non-empty hostname,
        # optional positive port, no path, no query string.
        if endpoint is None:
            raise InvalidArgumentException(INVALID_ENDPOINT_NULL_MSG)
        parts = urlsplit(endpoint)
        scheme = parts.scheme
        if not scheme or scheme.lower() not in ("http", "https"):
            raise InvalidArgumentException(INVALID_ENDPOINT_PROTOCOL_MSG.format(scheme))
        host = parts.hostname
        if host is None or not host.strip():
            raise InvalidArgumentException(INVALID_ENDPOINT_HOSTNAME_MSG.format(host))
        if require_port:
            # parts.port raises ValueError for non-numeric or out-of-range
            # ports; translate so callers only see InvalidArgumentException.
            try:
                port = parts.port
            except ValueError as exc:
                raw_port = parts.netloc.rsplit(":", 1)[-1]
                raise InvalidArgumentException(
                    INVALID_ENDPOINT_PORT_MSG.format(raw_port)
                ) from exc
            if port is None or port < 0:
                raise InvalidArgumentException(INVALID_ENDPOINT_PORT_MSG.format(port))
        if parts.path and parts.path != "/":
            raise InvalidArgumentException(
                INVALID_ENDPOINT_NO_PATH_ALLOWED_MSG.format(parts.path)
            )
        if parts.query:
            raise InvalidArgumentException(
                INVALID_ENDPOINT_NO_QUERY_PARAMS_ALLOWED_MSG.format(parts.query)
            )

    def validate_content_length(self, content_length: Optional[int]) -> None:
        if content_length is not None and content_length < 0:
            raise InvalidArgumentException(
                INVALID_CONTENT_LENGTH_MSG.format(content_length)
            )

    def validate_write_preconditions(
        self,
        create_if_absent: bool,
        if_match: Optional[str],
        if_version_match: Optional[str] = None,
    ) -> None:
        # create_if_absent asserts the object is absent; if_match / if_version_match
        # assert it is present with a specific ETag / version. Requesting
        # create_if_absent together with either present-precondition is
        # contradictory, so reject it rather than let a provider surface a
        # confusing native error. if_match and if_version_match MAY be combined:
        # portable callers set both so each provider enforces with the token it
        # supports (ETag on some, version/generation on others).
        if create_if_absent and if_match is not None:
            raise InvalidArgumentException(CREATE_IF_ABSENT_WITH_IF_MATCH_MSG)
        if create_if_absent and if_version_match is not None:
            raise InvalidArgumentException(CREATE_IF_ABSENT_WITH_IF_VERSION_MATCH_MSG)
        # An empty if_match would never match any ETag, so it is a caller
        # mistake rather than a valid "skip the precondition" request (None) —
        # same rule as validate_download.
        if if_match is not None and not if_match.strip():
            raise InvalidArgumentException(EMPTY_IF_MATCH_MSG)
        if if_version_match is not None and not if_version_match.strip():
            raise InvalidArgumentException(EMPTY_IF_VERSION_MATCH_MSG)

    def validate_storage_class(self, storage_class: Optional[str]) -> None:
        # The value is provider-native and passed through unchanged, so only a
        # blank string is rejected here; providers reject unknown classes.
        if storage_class is not None and not storage_class.strip():
            raise InvalidArgumentException(EMPTY_STORAGE_CLASS_MSG)

    def validate_upload(
        self,
        key: str,
        content_length: Optional[int] = None,
        *,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
    ) -> None:
        self.validate_key(key)
        self.validate_content_length(content_length)
        self.validate_write_preconditions(create_if_absent, if_match, if_version_match)

    def validate_upload_from_path(
        self,
        key: str,
        path: Path,
        *,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
    ) -> None:
        self.validate_key(key)
        self.validate_write_preconditions(create_if_absent, if_match, if_version_match)
        if not path.exists():
            raise InvalidArgumentException(f"upload_from_path: file not found: {path}")
        if not path.is_file():
            raise InvalidArgumentException(
                f"upload_from_path: path is not a file: {path}"
            )

    def validate_object_lock(self, object_lock: Any) -> None:
        if object_lock is None:
            return
        # Duck-typed to avoid importing ObjectLockConfiguration (cycle with
        # driver). Only mode and retain_until_date need cross-field checks.
        mode = getattr(object_lock, "mode", None)
        retain_until = getattr(object_lock, "retain_until_date", None)
        if mode is not None and retain_until is None:
            raise InvalidArgumentException(OBJECT_LOCK_MODE_REQUIRES_DATE_MSG)
        if retain_until is not None and mode is None:
            raise InvalidArgumentException(OBJECT_LOCK_DATE_REQUIRES_MODE_MSG)
        if retain_until is not None:
            if not isinstance(retain_until, datetime):
                raise InvalidArgumentException(
                    OBJECT_LOCK_NAIVE_DATE_MSG.format(retain_until)
                )
            if not _is_timezone_aware(retain_until):
                raise InvalidArgumentException(
                    OBJECT_LOCK_NAIVE_DATE_MSG.format(retain_until)
                )
            # No future-date check — that is a server-side policy; defer to it.

    def validate_download(
        self,
        key: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        *,
        if_match: Optional[str] = None,
    ) -> None:
        self.validate_key(key)
        self.validate_range(start, end)
        # An empty if_match would never match any ETag, so it is a caller
        # mistake rather than a valid "skip the precondition" request (None).
        if if_match is not None and not if_match.strip():
            raise InvalidArgumentException(EMPTY_IF_MATCH_MSG)

    def validate_delete(self, key: str) -> None:
        self.validate_key(key)

    # Backward-compat alias for the old name.
    def validate_delete_key(self, key: str) -> None:
        self.validate_delete(key)

    def validate_delete_objects(self, objects: Collection[Any]) -> None:
        """Validate batch delete inputs."""
        if not objects:
            raise InvalidArgumentException(
                "delete_objects: objects collection is required and cannot be empty"
            )
        for i, obj in enumerate(objects):
            key = getattr(obj, "key", "")
            if not key:
                raise InvalidArgumentException(
                    f"delete_objects: object at index {i} has an empty key"
                )

    def validate_copy(
        self,
        src_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> None:
        """Validate copy operation inputs."""
        if not src_key:
            raise InvalidArgumentException("copy: src_key is required")
        if not dest_bucket:
            raise InvalidArgumentException("copy: dest_bucket is required")
        if not dest_key:
            raise InvalidArgumentException("copy: dest_key is required")

    def validate_copy_from(
        self,
        src_bucket: str,
        src_key: str,
        dest_key: str,
    ) -> None:
        """Validate copy_from operation inputs."""
        if not src_bucket:
            raise InvalidArgumentException("copy_from: src_bucket is required")
        if not src_key:
            raise InvalidArgumentException("copy_from: src_key is required")
        if not dest_key:
            raise InvalidArgumentException("copy_from: dest_key is required")

    def validate_get_tags(self, key: str) -> None:
        self.validate_key(key)

    def validate_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        self.validate_key(key)
        self.validate_tags(tags)

    def validate_update_object_retention(self, key: str, retention: Any) -> None:
        # Stateless validation only. mode is an optional sentinel (None =
        # "preserve current mode"), so it is not rejected here; the
        # state-dependent rules run in the provider hook via resolve_and_validate.
        # Duck-typed to avoid importing ObjectRetentionConfig (cycle with driver.py).
        self.validate_key(key)
        if retention is None:
            raise InvalidArgumentException(RETENTION_CONFIG_NULL_MSG)
        retain_until = getattr(retention, "retain_until_date", None)
        if retain_until is None:
            raise InvalidArgumentException(RETENTION_RETAIN_UNTIL_NULL_MSG)
        if not _is_timezone_aware(retain_until):
            raise InvalidArgumentException(
                RETENTION_RETAIN_UNTIL_NAIVE_MSG.format(retain_until)
            )

    def validate_list_page(self, max_results: int) -> None:
        # Only the lower bound is enforced here — the upper bound is
        # provider-specific.
        if max_results < 1:
            raise InvalidArgumentException("max_results must be a positive integer")

    def validate_generate_presigned_url(
        self,
        key: str,
        operation: Any,
        duration_seconds: int,
        *,
        metadata: Optional[Mapping[str, str]] = None,
        tags: Optional[Mapping[str, str]] = None,
        kms_key_id: Optional[str] = None,
    ) -> None:
        self.validate_key(key)
        self.validate_presigned_operation_type(operation)
        # Local import: PresignedOperation lives in driver.py, which imports
        # this module — top-level import would cycle.
        from multicloud.blob.blob_client.driver import PresignedOperation

        if not isinstance(operation, PresignedOperation):
            raise InvalidArgumentException(
                f"generate_presigned_url: operation must be a PresignedOperation "
                f"enum value, got {type(operation).__name__}"
            )
        self.validate_duration(duration_seconds)
        if duration_seconds > MAX_PRESIGNED_DURATION_SECONDS:
            raise InvalidArgumentException(PRESIGNED_DURATION_TOO_LARGE_MSG)

        # Reject upload-only params on DOWNLOAD operations — silently
        # accepting them would mislead users into thinking the URL will
        # enforce those values, which it cannot.
        if operation is PresignedOperation.DOWNLOAD:
            if metadata:
                raise InvalidArgumentException(
                    UPLOAD_ONLY_PARAM_ON_DOWNLOAD_MSG.format("metadata")
                )
            if tags:
                raise InvalidArgumentException(
                    UPLOAD_ONLY_PARAM_ON_DOWNLOAD_MSG.format("tags")
                )
            if kms_key_id:
                raise InvalidArgumentException(
                    UPLOAD_ONLY_PARAM_ON_DOWNLOAD_MSG.format("kms_key_id")
                )

    def validate_multipart_upload(self, mpu: Any, bucket: str) -> None:
        # Duck-typed to avoid importing MultipartUpload (cycle with driver.py).
        self.validate_key(getattr(mpu, "key", ""))
        self.require_equals_ignore_case(
            getattr(mpu, "bucket", None), bucket, MISMATCHED_BUCKET_NAME_MSG
        )
