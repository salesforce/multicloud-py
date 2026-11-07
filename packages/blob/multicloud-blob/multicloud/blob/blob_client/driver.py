"""Driver-level abstractions for blob storage.

The driver layer defines the abstract contract every provider implements.
``BaseBlobStore`` exposes public methods (``upload`` / ``download``
/ ``delete``) that validate input and delegate to ``do_*`` template
methods, which providers override with substrate-specific code.

Users never reach this layer directly — they construct a
:class:`BucketClient` (in ``client.py``), which holds an
``BaseBlobStore`` instance internally.
"""

from __future__ import annotations

import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Callable, Iterator, List, Mapping, Optional, Union

# Re-exported for backward-compat with existing imports.
from multicloud.blob.blob_client.validators import BlobStoreValidator  # noqa: E402
from multicloud.blob.blob_client.validators import UNSUPPORTED_CHECKSUM_ALGORITHM_MSG
from multicloud.common.credentials import CredentialsOverrider
from multicloud.common.exceptions import (
    InvalidArgumentException,
    UnsupportedOperationException,
)
from multicloud.common.provider import Provider, ProviderRegistry
from multicloud.common.proxy import ProxyConfig

# Service-scoped registry. Provider distributions advertise themselves
# under the ``multicloud.blob.providers`` entry-point group; the registry
# imports those modules lazily on first use, at which point each provider's
# ``@BLOB_REGISTRY.register("aws")`` decorator stamps the class.
BLOB_REGISTRY = ProviderRegistry(
    service_name="blob",
    entry_point_group="multicloud.blob.providers",
)


class ChecksumAlgorithm(Enum):
    """Checksum algorithms for validating upload integrity server-side.

    Limited to the algorithms every supported provider validates natively.
    """

    CRC32C = "CRC32C"
    MD5 = "MD5"


@dataclass(frozen=True)
class UploadResponse:
    """Wrapper for upload result data.

    ``checksum_algorithm`` and ``checksum_value`` are set only when the upload
    requested checksum validation; ``checksum_value`` is the base64-encoded
    digest the provider computed over the stored bytes.
    """

    key: str
    e_tag: Optional[str] = None
    version_id: Optional[str] = None
    checksum_algorithm: Optional[ChecksumAlgorithm] = None
    checksum_value: Optional[str] = None


class ReplicationStatus(Enum):
    """Cross-region / cross-bucket replication state of an object.

    Reported on metadata lookups by providers that support replication.
    Providers that do not track replication leave the field ``None``.

    Attributes:
        COMPLETE: The object has been replicated to all configured targets.
        PENDING: Replication is in progress.
        FAILED: Replication failed for at least one target.
        REPLICA: This object is itself a replica created in a destination
            bucket by a replication rule.
    """

    COMPLETE = "COMPLETE"
    PENDING = "PENDING"
    FAILED = "FAILED"
    REPLICA = "REPLICA"


@dataclass(frozen=True)
class BlobMetadata:
    """Metadata returned alongside a download or HEAD-style lookup."""

    key: str
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    e_tag: Optional[str] = None
    version_id: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    last_modified: Optional[datetime] = None
    created_time: Optional[datetime] = None
    replication_status: Optional[ReplicationStatus] = None


@dataclass(frozen=True)
class DownloadResponse:
    """Wrapper object for download result data."""

    key: str
    content: bytes
    metadata: BlobMetadata


class StreamingDownloadResponse:
    """Streaming download response for lazy consumption of large blobs.

    Provides both an iterator interface (for chunk-by-chunk processing)
    and a file-like ``read()`` method. Must be closed after use to release
    the underlying HTTP connection.

    Supports context manager usage::

        with client.download_stream("large-file.bin") as stream:
            for chunk in stream:
                process(chunk)

    Or with explicit read calls::

        with client.download_stream("large-file.bin") as stream:
            while data := stream.read(8192):
                dest.write(data)
    """

    def __init__(
        self,
        key: str,
        metadata: BlobMetadata,
        stream: Iterator[bytes],
        close_func: Optional[Callable[[], None]] = None,
    ) -> None:
        self._key = key
        self._metadata = metadata
        self._stream = stream
        self._close_func = close_func
        self._closed = False
        self._buffer = b""

    @property
    def key(self) -> str:
        """Object key that was downloaded."""
        return self._key

    @property
    def metadata(self) -> BlobMetadata:
        """Metadata for the downloaded object."""
        return self._metadata

    def read(self, size: int = -1) -> bytes:
        """Read up to *size* bytes from the stream.

        Args:
            size: Maximum number of bytes to read. If -1, read all
                remaining content (caution: loads everything into memory).

        Returns:
            Bytes read from the stream. Returns empty bytes when the
            stream is exhausted.

        Raises:
            ValueError: If the stream has been closed.
        """
        if self._closed:
            raise ValueError("I/O operation on closed stream")

        if size == -1:
            # Read all remaining
            chunks = [self._buffer] if self._buffer else []
            for chunk in self._stream:
                chunks.append(chunk)
            self._buffer = b""
            return b"".join(chunks)

        # Fill buffer until we have enough
        while len(self._buffer) < size:
            try:
                self._buffer += next(self._stream)
            except StopIteration:
                break

        result = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return result

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over the stream content in chunks."""
        if self._closed:
            raise ValueError("I/O operation on closed stream")
        if self._buffer:
            yield self._buffer
            self._buffer = b""
        yield from self._stream

    def close(self) -> None:
        """Release the underlying HTTP connection and resources."""
        if not self._closed:
            self._closed = True
            if self._close_func is not None:
                self._close_func()

    def __enter__(self) -> "StreamingDownloadResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        """Whether the stream has been closed."""
        return self._closed


@dataclass(frozen=True)
class BlobInfo:
    """A blob's identity and lightweight metadata from listing operations."""

    key: str
    size: Optional[int] = None
    e_tag: Optional[str] = None
    last_modified: Optional[datetime] = None
    storage_class: Optional[str] = None
    version_id: Optional[str] = None
    is_latest: Optional[bool] = None


@dataclass(frozen=True)
class ListBlobsPageResponse:
    """Single page of blob listing results with pagination support."""

    blobs: List[BlobInfo] = field(default_factory=list)
    common_prefixes: List[str] = field(default_factory=list)
    is_truncated: bool = False
    next_pagination_token: Optional[str] = None


@dataclass(frozen=True)
class BlobIdentifier:
    """Identifies a specific blob (and optionally its version) for batch operations.

    Used by :meth:`BaseBlobStore.delete_objects` to target multiple blobs
    in a single call.

    Attributes:
        key: Object key (path) in the bucket.
        version_id: Specific version to target. If ``None``, targets
            the latest version (or creates a delete marker on versioned buckets).
    """

    key: str
    version_id: Optional[str] = None


@dataclass(frozen=True)
class CopyResponse:
    """Result of a copy operation.

    Attributes:
        key: The key of the copied blob in the destination bucket.
        version_id: Version ID assigned to the new copy (if versioning is
            enabled on the destination bucket).
        e_tag: Entity tag of the new copy.
        last_modified: Timestamp when the copy was created.
    """

    key: str
    version_id: Optional[str] = None
    e_tag: Optional[str] = None
    last_modified: Optional[datetime] = None


@dataclass(frozen=True)
class DeleteError:
    """Represents a single failed deletion within a batch delete.

    Attributes:
        key: Object key that failed to delete.
        version_id: Version ID that failed (if specified in the request).
        code: Provider-specific error code.
        message: Human-readable error description.
    """

    key: str
    version_id: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class DeleteObjectsResponse:
    """Result of a batch delete operation.

    Attributes:
        errors: List of objects that failed to delete. Empty list indicates
            all objects were successfully deleted.
    """

    errors: List[DeleteError] = field(default_factory=list)


class PresignedOperation(Enum):
    """Supported presigned URL operation types."""

    UPLOAD = "upload"
    DOWNLOAD = "download"


@dataclass(frozen=True)
class PresignedUrlResponse:
    """Result of generating a presigned URL.

    Attributes:
        url: The presigned URL string.
        http_method: The HTTP method the consumer must use (e.g. "PUT", "GET").
        required_headers: Headers the consumer MUST include in the HTTP request
            for the presigned URL to work. Includes signed metadata, tagging,
            content-type, and encryption headers. Empty mapping if none required.
            Wrapped in ``MappingProxyType`` to prevent accidental mutation of
            shared state.
    """

    url: str
    http_method: str
    required_headers: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        # Ensure required_headers is truly immutable at runtime.
        if not isinstance(self.required_headers, MappingProxyType):
            object.__setattr__(
                self,
                "required_headers",
                MappingProxyType(dict(self.required_headers)),
            )


class RetentionMode(Enum):
    """Object lock retention mode.

    GOVERNANCE retention can be shortened or removed by sufficiently
    privileged callers; COMPLIANCE retention cannot be shortened or
    removed by anyone until it expires.
    """

    GOVERNANCE = "GOVERNANCE"
    COMPLIANCE = "COMPLIANCE"


@dataclass(frozen=True)
class ObjectLockInfo:
    """Unified read result describing an object's lock state.

    Combines retention and legal-hold state into a single value object
    so callers retrieve everything in one read.

    Attributes:
        mode: Retention mode in effect, or None if no retention is set.
        retain_until_date: Instant until which the object is retained,
            or None if no retention is set.
        legal_hold: True if a legal hold is currently in place.
        event_based_hold: Provider-specific event-based hold flag, or
            None when the provider does not report one. None ("not
            reported") is distinct from False ("reported, and off"), so
            test it with ``is True`` / ``is False`` rather than plain
            truthiness.
    """

    mode: Optional[RetentionMode] = None
    retain_until_date: Optional[datetime] = None
    legal_hold: bool = False
    event_based_hold: Optional[bool] = None


@dataclass(frozen=True)
class ObjectRetentionConfig:
    """Retention settings to apply to a single object.

    Input type for the retention write API. This updates the retention of an
    object that already has retention configured; it does not establish
    retention for the first time.

    Attributes:
        retain_until_date: Instant until which the object must be retained.
            Must be timezone-aware — it denotes an absolute instant and is
            compared against the timezone-aware retain-until the provider reads
            back. A naive (timezone-less) value is rejected with
            InvalidArgumentException.
        mode: Retention mode to apply, or None to preserve the object's
            current mode while changing only the retain-until date.
        bypass_governance: When True, allows shortening an existing
            GOVERNANCE-mode retention, or upgrading GOVERNANCE to COMPLIANCE.
            Ignored for COMPLIANCE, which is immutable.
    """

    retain_until_date: datetime
    mode: Optional[RetentionMode] = None
    bypass_governance: bool = False


@dataclass(frozen=True)
class ObjectLockConfiguration:
    """Object lock settings applied to an object at upload time.

    Bundles first-time retention (mode + retain-until instant) and legal
    hold into a single input value passed to ``upload`` /
    ``upload_from_path``. All fields are optional so callers can set
    retention only, a legal hold only, or both.

    Attributes:
        mode: Retention mode to apply. Requires ``retain_until_date``.
        retain_until_date: Timezone-aware instant until which the object
            must be retained. Requires ``mode``.
        legal_hold: When True, place a legal hold on the object.
        use_event_based_hold: When True, a requested legal hold is applied
            as an event-based hold instead of the default hold mechanism.
            Ignored by providers that do not distinguish the two.
    """

    mode: Optional[RetentionMode] = None
    retain_until_date: Optional[datetime] = None
    legal_hold: bool = False
    use_event_based_hold: bool = False


@dataclass(frozen=True)
class MultipartUpload:
    """Identifies an in-progress multipart upload session.

    Returned by ``initiate_multipart_upload`` and passed to all subsequent
    multipart operations (upload_part, complete, list, abort).
    """

    bucket: str
    key: str
    upload_id: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    content_type: Optional[str] = None
    tags: Mapping[str, str] = field(default_factory=dict)
    object_lock: Optional[ObjectLockConfiguration] = None
    #: Set by the async client for tracing; unused by the sync client.
    correlation_id: Optional[str] = None


@dataclass(frozen=True)
class MultipartPart:
    """A single part to upload within a multipart upload session.

    Accepts either raw bytes or a binary file-like object (BinaryIO).
    When using BinaryIO, ``content_length`` is required.
    """

    part_number: int
    content: Union[bytes, BinaryIO]
    content_length: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.content, bytes):
            if self.content_length is None:
                object.__setattr__(self, "content_length", len(self.content))
            elif self.content_length != len(self.content):
                raise ValueError(
                    f"content_length ({self.content_length}) does not match "
                    f"len(content) ({len(self.content)})"
                )
        elif self.content_length is None:
            raise ValueError(
                "content_length is required when content is a stream (BinaryIO)"
            )


@dataclass(frozen=True)
class UploadPartResponse:
    """Response from uploading an individual part."""

    part_number: int
    e_tag: str
    size_in_bytes: int
    correlation_id: Optional[str] = None


@dataclass(frozen=True)
class MultipartUploadResponse:
    """Response from completing a multipart upload.

    Only ``e_tag`` is set by the synchronous client; the remaining fields are
    populated by the async client.
    """

    e_tag: Optional[str] = None
    key: Optional[str] = None
    version_id: Optional[str] = None
    correlation_id: Optional[str] = None


# Accepted upload payload types: raw bytes or a binary, readable file-like.
UploadPayload = Union[bytes, BinaryIO]


class BaseBlobStore(Provider, ABC):
    """Base class for substrate-specific blob store implementations.

    Providers extend this class and implement :meth:`do_upload` /
    :meth:`do_download` / :meth:`translate_error`. Providers must accept
    a ``credentials_overrider`` constructor kwarg and forward it to
    ``super().__init__`` so it is available for credential resolution.

    All cross-cutting concerns (validation, default arg handling) happen
    in the public ``upload``/``download`` methods so providers don't have
    to repeat them.

    The :meth:`translate_error` method is inherited from :class:`Provider`
    and must be implemented with blob-specific error code mappings.
    """

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        *,
        credentials_overrider: Optional[CredentialsOverrider] = None,
        proxy_config: Optional[ProxyConfig] = None,
        validator: Optional[BlobStoreValidator] = None,
    ) -> None:
        self._validator = validator or BlobStoreValidator()
        self._validator.validate_bucket(bucket)
        self._bucket = bucket
        self._region = region
        self._credentials_overrider = credentials_overrider
        self._proxy_config = proxy_config

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def region(self) -> Optional[str]:
        return self._region

    def upload(
        self,
        key: str,
        payload: UploadPayload,
        *,
        content_type: Optional[str] = None,
        content_length: Optional[int] = None,
        metadata: Optional[Mapping[str, str]] = None,
        tags: Optional[Mapping[str, str]] = None,
        object_lock: Optional[ObjectLockConfiguration] = None,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        checksum_value: Optional[str] = None,
        checksum_algorithm: Optional[Union[ChecksumAlgorithm, str]] = None,
    ) -> UploadResponse:
        """Validate, then delegate to :meth:`do_upload`."""
        self._validator.validate_upload(
            key,
            content_length,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
        )
        self._validator.validate_object_lock(object_lock)
        algorithm = _coerce_checksum_algorithm(checksum_algorithm)
        self._validator.validate_checksum(checksum_value, algorithm)
        payload_bytes = _coerce_to_bytes(payload)
        return self.do_upload(
            key,
            payload_bytes,
            content_type=content_type,
            content_length=content_length,
            metadata=metadata or {},
            tags=tags or {},
            object_lock=object_lock,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
            kms_key_id=kms_key_id,
            use_kms_managed_key=use_kms_managed_key,
            checksum_value=checksum_value,
            checksum_algorithm=algorithm,
        )

    def upload_from_path(
        self,
        key: str,
        path: Union[str, Path, "os.PathLike[str]"],
        *,
        content_type: Optional[str] = None,
        content_length: Optional[int] = None,
        metadata: Optional[Mapping[str, str]] = None,
        object_lock: Optional[ObjectLockConfiguration] = None,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
    ) -> UploadResponse:
        """Upload a local file to blob storage.

        Delegates to :meth:`do_upload_from_path` which providers implement
        using native file-based upload APIs (e.g. boto3 ``upload_file`` with
        automatic multipart, GCS ``upload_from_filename`` with automatic
        resumable upload).

        If *content_type* is not provided, it is auto-detected from the file
        extension using :mod:`mimetypes`. If *content_length* is not provided,
        it is calculated from the file size.

        Args:
            key: Object key (path) in the bucket.
            path: Local file path to upload. Accepts ``str``, :class:`~pathlib.Path`,
                or any :class:`os.PathLike`.
            content_type: MIME type. Auto-detected from file extension if omitted.
            content_length: Size in bytes. Calculated from file size if omitted.
            metadata: User-defined metadata dict.
            object_lock: Object lock configuration (retention mode + expiry
                and/or legal hold) to apply to the object at upload time.
            create_if_absent: Only create when absent; else
                ResourceExistsException. A concurrent write that races the
                condition raises ResourceConflictException (the write may not
                have happened; retry). Mutually exclusive with if_match and
                if_version_match.
            if_match: Only overwrite when the current ETag matches; else
                FailedPreconditionException; a racing write raises
                ResourceConflictException (the write may not have happened;
                retry). Not all providers support ETag-based conditional
                overwrite on upload; unsupported providers raise
                UnsupportedOperationException.
            if_version_match: Only overwrite when the current object version
                matches; else FailedPreconditionException; a racing write raises
                ResourceConflictException (the write may not have happened;
                retry). Not all providers support version-based conditional
                overwrite on upload; unsupported providers raise
                UnsupportedOperationException. May be combined with if_match so
                portable callers get a compare-and-swap overwrite regardless of
                which token a provider enforces.
            kms_key_id: Provider-managed key id for server-side encryption.
            use_kms_managed_key: Request a provider-managed encryption key when
                no kms_key_id is given. A no-op on providers that always
                encrypt at rest.

        Returns:
            UploadResponse containing the key, ETag, and version ID.

        Raises:
            InvalidArgumentException: If key is empty, path does not exist,
                or path is not a regular file.
        """
        resolved = Path(path) if not isinstance(path, Path) else path
        self._validator.validate_upload_from_path(
            key,
            resolved,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
        )
        self._validator.validate_object_lock(object_lock)

        # Auto-detect content_type from file extension if not provided
        if content_type is None:
            guessed, _ = mimetypes.guess_type(str(resolved))
            content_type = guessed or "application/octet-stream"

        # Auto-calculate content_length from file size if not provided
        if content_length is None:
            content_length = resolved.stat().st_size

        return self.do_upload_from_path(
            key,
            resolved,
            content_type=content_type,
            content_length=content_length,
            metadata=metadata or {},
            object_lock=object_lock,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
            kms_key_id=kms_key_id,
            use_kms_managed_key=use_kms_managed_key,
        )

    def download(
        self,
        key: str,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        if_match: Optional[str] = None,
    ) -> DownloadResponse:
        """Validate, then delegate to :meth:`do_download`."""
        self._validator.validate_download(key, start, end, if_match=if_match)
        return self.do_download(
            key,
            version_id=version_id,
            start=start,
            end=end,
            if_match=if_match,
        )

    def download_to_path(
        self,
        key: str,
        destination: Path,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        create_parent_dirs: bool = False,
    ) -> DownloadResponse:
        """Download a blob directly to a file path without loading into memory.

        Writes the object content directly to the local filesystem. For
        large blobs this avoids holding the entire content in memory.

        Conditional reads (if_match) are supported on download() only.

        Args:
            key: Object key (path) in the bucket.
            destination: Local file path to write the downloaded content to.
            version_id: Specific version to download (if versioning enabled).
            start: Start byte position for range download (inclusive).
            end: End byte position for range download (inclusive).
            create_parent_dirs: If True, create parent directories of
                *destination* if they do not exist.

        Returns:
            DownloadResponse with metadata (content field is empty bytes).
        """
        self._validator.validate_download(key, start, end)
        if create_parent_dirs:
            destination.parent.mkdir(parents=True, exist_ok=True)
        return self.do_download_to_path(
            key,
            destination,
            version_id=version_id,
            start=start,
            end=end,
        )

    def download_to_file(
        self,
        key: str,
        file_obj: BinaryIO,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> DownloadResponse:
        """Download a blob directly to a file-like object without loading into memory.

        Writes the object content directly to the provided writable
        binary file object. Useful when the caller already holds an open
        file handle, a ``BytesIO`` buffer, or any object with a
        ``write()`` method.

        Conditional reads (if_match) are supported on download() only.

        Args:
            key: Object key (path) in the bucket.
            file_obj: A writable binary file-like object (must support
                ``write(bytes)``).
            version_id: Specific version to download (if versioning enabled).
            start: Start byte position for range download (inclusive).
            end: End byte position for range download (inclusive).

        Returns:
            DownloadResponse with metadata (content field is empty bytes).
        """
        self._validator.validate_download(key, start, end)
        return self.do_download_to_file(
            key,
            file_obj,
            version_id=version_id,
            start=start,
            end=end,
        )

    def download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> StreamingDownloadResponse:
        """Download a blob as a streaming response for lazy consumption.

        Returns a :class:`StreamingDownloadResponse` that yields content
        on demand without loading the entire blob into memory. The caller
        is responsible for closing the response (preferably via a context
        manager) to release the underlying HTTP connection.

        Conditional reads (if_match) are supported on download() only.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific version to download (if versioning enabled).
            start: Start byte position for range download (inclusive).
            end: End byte position for range download (inclusive).

        Returns:
            StreamingDownloadResponse with an iterator and read() interface.
        """
        self._validator.validate_download(key, start, end)
        return self.do_download_stream(
            key,
            version_id=version_id,
            start=start,
            end=end,
        )

    def delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Delete a single object by ``key``.

        Idempotent — returns silently if the object does not exist.
        Validates input, then delegates to :meth:`do_delete`.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific version to delete. If ``None``, deletes the
                latest version (or creates a delete marker on versioned buckets).
        """
        self._validator.validate_delete(key)
        self.do_delete(key, version_id=version_id)

    def delete_objects(self, objects: List[BlobIdentifier]) -> DeleteObjectsResponse:
        """Delete multiple objects in a single batch operation.

        Validates input, then delegates to :meth:`do_delete_objects`.

        The caller is responsible for chunking to stay within provider
        limits (e.g. 1000 for S3). Exceeding the provider's per-request
        maximum will result in a cloud-level error.

        Args:
            objects: Collection of blob identifiers to delete.

        Returns:
            DeleteObjectsResponse with any per-object errors. An empty
            ``errors`` list means all objects were deleted successfully.
        """
        self._validator.validate_delete_objects(objects)
        return self.do_delete_objects(objects)

    def copy(
        self,
        src_key: str,
        *,
        dest_bucket: str,
        dest_key: str,
        src_version_id: Optional[str] = None,
    ) -> CopyResponse:
        """Copy an object from this bucket to another bucket.

        Performs a server-side copy. Validates input, then delegates to
        :meth:`do_copy`.

        Args:
            src_key: Key of the source object in this bucket.
            dest_bucket: Name of the destination bucket.
            dest_key: Key for the copied object in the destination bucket.
            src_version_id: Specific version to copy. If ``None``, copies
                the latest version.

        Returns:
            CopyResponse with the key, version, ETag, and timestamp of
            the new copy.
        """
        self._validator.validate_copy(src_key, dest_bucket, dest_key)
        return self.do_copy(
            src_key,
            dest_bucket=dest_bucket,
            dest_key=dest_key,
            src_version_id=src_version_id,
        )

    def copy_from(
        self,
        src_bucket: str,
        src_key: str,
        *,
        dest_key: str,
        src_version_id: Optional[str] = None,
    ) -> CopyResponse:
        """Copy an object from a source bucket into this bucket.

        Performs a server-side copy. Validates input, then delegates to
        :meth:`do_copy_from`.

        Args:
            src_bucket: Name of the source bucket.
            src_key: Key of the source object in the source bucket.
            dest_key: Key for the copied object in this bucket.
            src_version_id: Specific version to copy. If ``None``, copies
                the latest version.

        Returns:
            CopyResponse with the key, version, ETag, and timestamp of
            the new copy.
        """
        self._validator.validate_copy_from(src_bucket, src_key, dest_key)
        return self.do_copy_from(
            src_bucket,
            src_key,
            dest_key=dest_key,
            src_version_id=src_version_id,
        )

    def list_page(
        self,
        *,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_results: int = 1000,
        pagination_token: Optional[str] = None,
    ) -> ListBlobsPageResponse:
        """Retrieve a single page of blobs matching the given criteria."""
        self._validator.validate_list_page(max_results)
        return self.do_list_page(
            prefix=prefix,
            delimiter=delimiter,
            max_results=max_results,
            pagination_token=pagination_token,
        )

    def does_object_exist(self, key: str, *, version_id: Optional[str] = None) -> bool:
        """Check whether a blob exists in the bucket."""
        self._validator.validate_key(key)
        return self.do_does_object_exist(key, version_id=version_id)

    def get_metadata(
        self, key: str, *, version_id: Optional[str] = None
    ) -> BlobMetadata:
        """Retrieve object metadata (HEAD-style) without downloading the body.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version. Optional.

        Returns:
            BlobMetadata with key, content_length, content_type, e_tag,
            version_id, custom metadata, last_modified, and created_time.

        Raises:
            InvalidArgumentException: If key is empty.
            ResourceNotFoundException: If the object does not exist.
        """
        self._validator.validate_key(key)
        return self.do_get_metadata(key, version_id=version_id)

    def does_bucket_exist(self) -> bool:
        """Check whether the configured bucket exists."""
        return self.do_does_bucket_exist()

    def get_tags(self, key: str) -> Mapping[str, str]:
        """Retrieve all tags associated with the object."""
        self._validator.validate_get_tags(key)
        return self.do_get_tags(key)

    def set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Set tags on the object, replacing all existing tags."""
        self._validator.validate_set_tags(key, tags)
        self.do_set_tags(key, tags)

    def get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Retrieve the object lock state (retention + legal hold) for an object.

        Returns None when the object has no retention and no legal hold.
        """
        self._validator.validate_key(key)
        return self.do_get_object_lock(key, version_id=version_id)

    def update_legal_hold(
        self, key: str, enabled: bool, *, version_id: Optional[str] = None
    ) -> None:
        """Set or remove a legal hold on an object.

        A legal hold, once placed, prevents the object from being deleted or
        overwritten until the hold is explicitly removed, independent of any
        retention period. Validates input, then delegates to
        :meth:`do_update_legal_hold`.

        Args:
            key: Object key (path) in the bucket.
            enabled: True to place a legal hold, False to remove it.
            version_id: Specific object version. Optional; None targets the
                latest version.
        """
        self._validator.validate_key(key)
        self.do_update_legal_hold(key, enabled, version_id=version_id)

    def update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str] = None,
    ) -> None:
        """Update the retention configuration of a single object.

        The object must already have retention configured; this updates it. It
        does not establish retention for the first time.
        """
        self._validator.validate_update_object_retention(key, retention)
        self.do_update_object_retention(key, retention, version_id=version_id)

    def list_blob_versions(self, key: str) -> Iterator[BlobInfo]:
        """Iterate all versions of a blob in the bucket."""
        self._validator.validate_key(key)
        return self.do_list_blob_versions(key)

    def generate_presigned_url(
        self,
        key: str,
        operation: PresignedOperation,
        *,
        duration_seconds: int = 3600,
        content_type: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
        tags: Optional[Mapping[str, str]] = None,
        kms_key_id: Optional[str] = None,
        content_disposition: Optional[str] = None,
    ) -> PresignedUrlResponse:
        """Generate a presigned URL for uploading or downloading an object.

        Validates input, then delegates to :meth:`do_generate_presigned_url`.

        Args:
            key: Object key (path) in the bucket.
            operation: Whether the URL is for upload or download.
            duration_seconds: How long the URL remains valid (default 3600,
                max 604800 = 7 days).
            content_type: MIME type constraint for uploads. The consumer
                must include this exact Content-Type header.
            metadata: User-defined metadata to associate with the upload.
                Only applicable to UPLOAD operations. The consumer must
                include matching metadata headers.
            tags: Tags to associate with the upload. Only applicable to
                UPLOAD operations. The consumer must include matching
                tagging headers.
            kms_key_id: KMS key identifier for server-side encryption.
                Only applicable to UPLOAD operations. Provider-specific
                format (e.g. AWS ARN/alias). Silently ignored on providers
                that do not support KMS in presigned URLs.
            content_disposition: Content-Disposition header override.
                For DOWNLOAD, controls the filename browsers suggest.
                For UPLOAD, constrains the Content-Disposition header.

        Returns:
            PresignedUrlResponse containing the URL, HTTP method, and
            ``required_headers``. The consumer must send every header in
            ``required_headers`` verbatim: providers may bind these headers
            into the URL signature, so omitting or altering one causes the
            request to be rejected as a signature mismatch.

        Raises:
            InvalidArgumentException: If key is empty, duration is invalid,
                or operation type is invalid.
        """
        self._validator.validate_generate_presigned_url(
            key,
            operation,
            duration_seconds,
            metadata=metadata,
            tags=tags,
            kms_key_id=kms_key_id,
        )
        return self.do_generate_presigned_url(
            key,
            operation,
            duration_seconds=duration_seconds,
            content_type=content_type,
            metadata=metadata or {},
            tags=tags or {},
            kms_key_id=kms_key_id,
            content_disposition=content_disposition,
        )

    # multipart upload operations
    def initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Optional[Mapping[str, str]] = None,
        content_type: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
        object_lock: Optional[ObjectLockConfiguration] = None,
    ) -> MultipartUpload:
        """Initiate a multipart upload session.

        Args:
            key: Object key for the final assembled blob.
            metadata: Optional user-defined metadata.
            content_type: Optional MIME type.
            tags: Optional object tags applied to the final assembled blob.
            object_lock: Optional object lock (retention mode + expiry and/or
                legal hold) applied to the final assembled blob.

        Returns:
            MultipartUpload handle for subsequent part operations.
        """
        self._validator.validate_key(key)
        self._validator.validate_object_lock(object_lock)
        return self.do_initiate_multipart_upload(
            key,
            metadata=metadata or {},
            content_type=content_type,
            tags=tags or {},
            object_lock=object_lock,
        )

    def upload_part(
        self, mpu: MultipartUpload, part: MultipartPart
    ) -> UploadPartResponse:
        """Upload a single part within a multipart upload session.

        Args:
            mpu: The multipart upload handle from initiate.
            part: The part data (part_number + content).

        Returns:
            UploadPartResponse with part_number, e_tag, and size.
        """
        self._validator.validate_multipart_upload(mpu, self._bucket)
        return self.do_upload_part(mpu, part)

    def complete_multipart_upload(
        self, mpu: MultipartUpload, parts: List[UploadPartResponse]
    ) -> MultipartUploadResponse:
        """Complete a multipart upload, assembling all parts into the final blob.

        Args:
            mpu: The multipart upload handle from initiate.
            parts: List of all uploaded part responses.

        Returns:
            MultipartUploadResponse with the final e_tag.
        """
        self._validator.validate_multipart_upload(mpu, self._bucket)
        return self.do_complete_multipart_upload(mpu, parts)

    def list_parts(self, mpu: MultipartUpload) -> List[UploadPartResponse]:
        """List all parts uploaded so far for a multipart upload.

        Args:
            mpu: The multipart upload handle from initiate.

        Returns:
            List of UploadPartResponse for each uploaded part.
        """
        self._validator.validate_multipart_upload(mpu, self._bucket)
        return self.do_list_parts(mpu)

    def abort_multipart_upload(self, mpu: MultipartUpload) -> None:
        """Abort a multipart upload, discarding all uploaded parts.

        Args:
            mpu: The multipart upload handle from initiate.
        """
        self._validator.validate_multipart_upload(mpu, self._bucket)
        self.do_abort_multipart_upload(mpu)

    @abstractmethod
    def do_upload(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: Optional[str],
        content_length: Optional[int],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration],
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        checksum_value: Optional[str] = None,
        checksum_algorithm: Optional[ChecksumAlgorithm] = None,
    ) -> UploadResponse:
        """Provider-specific upload implementation.

        Args:
            create_if_absent: Only create when absent; else
                ResourceExistsException. A racing concurrent write raises
                ResourceConflictException (the write may not have happened;
                retry).
            if_match: Only overwrite when the current ETag matches; else
                FailedPreconditionException; a racing write raises
                ResourceConflictException (the write may not have happened;
                retry). Unsupported providers raise UnsupportedOperationException.
            if_version_match: Only overwrite when the current version matches;
                else FailedPreconditionException; a racing write raises
                ResourceConflictException (the write may not have happened;
                retry). May combine with if_match; unsupported providers raise
                UnsupportedOperationException.
            kms_key_id: Provider-managed key id for server-side encryption.
            use_kms_managed_key: Request a provider-managed key when no
                kms_key_id is given; a no-op where encryption is always on.
            checksum_value: Base64-encoded digest of *payload* for the server
                to validate (already checked for shape; set together with
                checksum_algorithm). A server-side mismatch raises
                InvalidArgumentException with CHECKSUM_MISMATCH_MSG. The
                response reports the provider-computed digest.
            checksum_algorithm: Algorithm of checksum_value.
        """

    @abstractmethod
    def do_upload_from_path(
        self,
        key: str,
        path: Path,
        *,
        content_type: Optional[str],
        content_length: Optional[int],
        metadata: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration],
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
    ) -> UploadResponse:
        """Provider-specific file-path upload implementation.

        Providers should use their native file-based upload APIs for
        optimal performance (e.g. automatic multipart/resumable uploads
        for large files). The file at *path* is guaranteed to exist and
        be a regular file (validated by the caller).

        Args:
            key: Object key (path) in the bucket.
            path: Validated local file path (guaranteed to exist and be a file).
            content_type: MIME type (already resolved — never None here).
            content_length: File size in bytes (already resolved — never None here).
            metadata: User-defined metadata dict (may be empty).
            create_if_absent: Only create when absent; else
                ResourceExistsException. A racing concurrent write raises
                ResourceConflictException (the write may not have happened;
                retry).
            if_match: Only overwrite when the current ETag matches; else
                FailedPreconditionException; a racing write raises
                ResourceConflictException (the write may not have happened;
                retry). Unsupported providers raise UnsupportedOperationException.
            if_version_match: Only overwrite when the current version matches;
                else FailedPreconditionException; a racing write raises
                ResourceConflictException (the write may not have happened;
                retry). May combine with if_match; unsupported providers raise
                UnsupportedOperationException.
            kms_key_id: Provider-managed key id for server-side encryption.
            use_kms_managed_key: Request a provider-managed key when no
                kms_key_id is given; a no-op where encryption is always on.
        """

    @abstractmethod
    def do_download(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
        if_match: Optional[str] = None,
    ) -> DownloadResponse:
        """Provider-specific download implementation."""

    @abstractmethod
    def do_download_to_path(
        self,
        key: str,
        destination: Path,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> DownloadResponse:
        """Provider-specific file-path download implementation.

        Must write the object content directly to *destination* without
        loading the entire content into memory. Returns a
        DownloadResponse with metadata but empty content bytes.
        """

    @abstractmethod
    def do_download_to_file(
        self,
        key: str,
        file_obj: BinaryIO,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> DownloadResponse:
        """Provider-specific file-object download implementation.

        Must write the object content directly to *file_obj* without
        loading the entire content into memory. Returns a
        DownloadResponse with metadata but empty content bytes.
        """

    @abstractmethod
    def do_download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> StreamingDownloadResponse:
        """Provider-specific streaming download implementation.

        Must return a StreamingDownloadResponse that lazily yields
        content chunks without loading the full object into memory.
        The caller is responsible for closing the response.
        """

    @abstractmethod
    def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Provider-specific single-object delete implementation."""

    @abstractmethod
    def do_delete_objects(self, objects: List[BlobIdentifier]) -> DeleteObjectsResponse:
        """Provider-specific batch delete implementation."""

    @abstractmethod
    def do_copy(
        self,
        src_key: str,
        *,
        dest_bucket: str,
        dest_key: str,
        src_version_id: Optional[str],
    ) -> CopyResponse:
        """Provider-specific copy-to-another-bucket implementation."""

    @abstractmethod
    def do_copy_from(
        self,
        src_bucket: str,
        src_key: str,
        *,
        dest_key: str,
        src_version_id: Optional[str],
    ) -> CopyResponse:
        """Provider-specific copy-from-source-bucket implementation."""

    @abstractmethod
    def do_get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Provider-specific object lock state retrieval implementation.

        Returns None when the object has no retention and no legal hold.
        """

    @abstractmethod
    def do_update_legal_hold(
        self, key: str, enabled: bool, *, version_id: Optional[str] = None
    ) -> None:
        """Provider-specific legal hold write implementation.

        Sets a legal hold when ``enabled`` is True, removes it when False.
        """

    @abstractmethod
    def do_get_tags(self, key: str) -> Mapping[str, str]:
        """Provider-specific tag retrieval implementation."""

    @abstractmethod
    def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Provider-specific tag write implementation."""

    @abstractmethod
    def do_update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str],
    ) -> None:
        """Provider-specific object retention write implementation."""

    @abstractmethod
    def do_list_page(
        self,
        *,
        prefix: Optional[str],
        delimiter: Optional[str],
        max_results: int,
        pagination_token: Optional[str],
    ) -> ListBlobsPageResponse:
        """Provider-specific page listing implementation."""

    @abstractmethod
    def do_does_object_exist(self, key: str, *, version_id: Optional[str]) -> bool:
        """Provider-specific object existence check."""

    @abstractmethod
    def do_get_metadata(self, key: str, *, version_id: Optional[str]) -> BlobMetadata:
        """Provider-specific HEAD-style metadata retrieval.

        Must NOT download the object body. Abstract because HEAD-style
        metadata is a core operation every production provider supports
        (like ``do_does_object_exist``); a provider missing it fails fast
        with a clear ``TypeError`` at construction time.
        """

    @abstractmethod
    def do_does_bucket_exist(self) -> bool:
        """Provider-specific bucket existence check."""

    @abstractmethod
    def do_generate_presigned_url(
        self,
        key: str,
        operation: PresignedOperation,
        *,
        duration_seconds: int,
        content_type: Optional[str],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        kms_key_id: Optional[str],
        content_disposition: Optional[str],
    ) -> PresignedUrlResponse:
        """Provider-specific presigned URL generation."""

    def do_list_blob_versions(self, key: str) -> Iterator[BlobInfo]:
        """Provider-specific version listing. Optional — override to support."""
        raise UnsupportedOperationException(
            "list_blob_versions is not supported by this provider"
        )

    def do_initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Mapping[str, str],
        content_type: Optional[str],
        tags: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration] = None,
    ) -> MultipartUpload:
        """Provider-specific multipart upload initiation. Override to support."""
        raise UnsupportedOperationException(
            "initiate_multipart_upload is not supported by this provider"
        )

    def do_upload_part(
        self, mpu: MultipartUpload, part: MultipartPart
    ) -> UploadPartResponse:
        """Provider-specific part upload. Override to support."""
        raise UnsupportedOperationException(
            "upload_part is not supported by this provider"
        )

    def do_complete_multipart_upload(
        self, mpu: MultipartUpload, parts: List[UploadPartResponse]
    ) -> MultipartUploadResponse:
        """Provider-specific multipart completion. Override to support."""
        raise UnsupportedOperationException(
            "complete_multipart_upload is not supported by this provider"
        )

    def do_list_parts(self, mpu: MultipartUpload) -> List[UploadPartResponse]:
        """Provider-specific part listing. Override to support."""
        raise UnsupportedOperationException(
            "list_parts is not supported by this provider"
        )

    def do_abort_multipart_upload(self, mpu: MultipartUpload) -> None:
        """Provider-specific multipart abort. Override to support."""
        raise UnsupportedOperationException(
            "abort_multipart_upload is not supported by this provider"
        )

    def close(self) -> None:
        """Release underlying resources. Default is a no-op."""

    def __enter__(self) -> "BaseBlobStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()


def _coerce_checksum_algorithm(
    value: Optional[Union[ChecksumAlgorithm, str]],
) -> Optional[ChecksumAlgorithm]:
    """Normalize a checksum algorithm given as an enum or a case-insensitive name."""
    if value is None or isinstance(value, ChecksumAlgorithm):
        return value
    try:
        return ChecksumAlgorithm(str(value).upper())
    except ValueError:
        supported = ", ".join(a.value for a in ChecksumAlgorithm)
        raise InvalidArgumentException(
            UNSUPPORTED_CHECKSUM_ALGORITHM_MSG.format(supported, value)
        ) from None


def _coerce_to_bytes(payload: UploadPayload) -> bytes:
    """Normalize a payload to ``bytes`` for providers."""
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    read = getattr(payload, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, str):
            raise InvalidArgumentException(
                "Upload payload must be binary; got a text stream"
            )
        return bytes(data)
    raise InvalidArgumentException(
        f"Unsupported upload payload type: {type(payload).__name__}"
    )


def load_from_path(path: Union[str, os.PathLike]) -> bytes:
    """Convenience loader used by examples/tests."""
    with open(path, "rb") as fh:
        return fh.read()
