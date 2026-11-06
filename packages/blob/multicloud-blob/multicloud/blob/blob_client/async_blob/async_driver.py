"""Async driver-level abstractions for blob storage.

Mirrors the sync :mod:`multicloud.blob.blob_client.driver` but uses
``async def`` coroutines. ``BaseAsyncBlobStore`` validates input and
delegates to ``do_*`` template methods that providers override.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    BinaryIO,
    Callable,
    List,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

from multicloud.blob.blob_client.driver import (
    MultipartPart,
    MultipartUpload,
    MultipartUploadResponse,
    ObjectLockInfo,
    ObjectRetentionConfig,
    ReplicationStatus,
    RetentionMode,
    UploadPartResponse,
)
from multicloud.blob.blob_client.validators import BlobStoreValidator
from multicloud.common.credentials import CredentialsOverrider
from multicloud.common.exceptions import (
    InvalidArgumentException,
    ResourceNotFoundException,
    SubstrateSdkException,
    UnsupportedOperationException,
)
from multicloud.common.provider import Provider, ProviderRegistry
from multicloud.common.proxy import ProxyConfig

ASYNC_BLOB_REGISTRY = ProviderRegistry(
    service_name="async_blob",
    entry_point_group="multicloud.blob.async_providers",
)

DEFAULT_MAX_CONCURRENCY = 10
"""Default maximum number of concurrent file transfers for directory operations."""

MAX_REPORTED_DELETE_FAILURES = 100
"""Cap on per-object failures quoted in a directory-delete error message.

A prefix whose every object is undeletable would otherwise build an
unbounded error string, so the message quotes at most this many and
reports the remainder as a count.
"""

_DeleteUnit = TypeVar("_DeleteUnit")


class BlobAsyncError(SubstrateSdkException):
    """Transport failure during an async blob operation."""

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[Exception] = None,
    ) -> None:
        self.message = message
        self.cause = cause
        super().__init__(message)


class BlobNotFoundError(BlobAsyncError, ResourceNotFoundException):
    """The requested blob was not found in the bucket.

    Also a ResourceNotFoundException so that a caller writing one
    ``except ResourceNotFoundException`` handles a missing blob the same way
    whichever provider raised it, and the same way the synchronous client
    reports it.
    """

    def __init__(
        self,
        message: str,
        *,
        key: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.key = key
        super().__init__(message, cause=cause)


@dataclass(frozen=True)
class AsyncTransferConfig:
    """Configuration for async transfer operations.

    Fields below are reserved for the upcoming multipart upload and
    streaming download phases; they are not yet honored by the
    single-shot upload path shipped in this release.
    """

    max_concurrency: int = 50
    multipart_threshold: int = 8 * 1024 * 1024  # 8 MB
    multipart_chunksize: int = 8 * 1024 * 1024  # 8 MB
    download_chunk_size: int = 1024 * 1024  # 1 MB


@dataclass(frozen=True)
class AsyncUploadResponse:
    """Result of an async upload operation."""

    key: str
    e_tag: Optional[str] = None
    version_id: Optional[str] = None


AsyncUploadPayload = Union[bytes, bytearray, memoryview, BinaryIO, Path]


@dataclass(frozen=True)
class AsyncDownloadRequest:
    """Request parameters for an async download operation.

    Attributes:
        key: Object key (path) in the bucket.
        version_id: Specific object version to download. Optional.
            Provider support varies -- implementations that do not support
            versioning will raise InvalidArgumentException.
        start: First byte offset (inclusive) for a range download, or
            None for a full download. When only ``start`` is set,
            produces an open-ended range (``bytes=start-``).
        end: Last byte offset (inclusive, per RFC 7233) for a range
            download, or None. When only ``end`` is set, it is
            interpreted as a suffix-range meaning "last N bytes"
            (``bytes=-N``), NOT an absolute position.
        if_match: Only download when the current ETag matches; else
            FailedPreconditionException. Empty/whitespace is rejected as
            a caller mistake. None skips the precondition.
    """

    key: str
    version_id: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    if_match: Optional[str] = None


@dataclass(frozen=True)
class BlobMetadata:
    """Metadata about a blob object.

    Attributes:
        key: Object key (path) in the bucket.
        content_length: Number of bytes actually returned (for both
            full and range downloads). Always equals ``len(content)``.
        content_type: MIME type of the object.
        e_tag: Entity tag. Format is provider-specific and should not
            be compared across providers — AWS returns a hex MD5 (quotes
            stripped), GCS returns a base64-encoded opaque token.
        version_id: Object version (S3 VersionId or GCS generation).
        metadata: User-defined metadata key-value pairs.
        last_modified: Last-modified timestamp as a timezone-aware
            datetime, or None if the provider did not report it.
        created_time: Object creation timestamp as a timezone-aware
            datetime, or None if the provider did not report it. Some
            providers expose no distinct creation time; in those cases
            ``created_time`` mirrors ``last_modified``.
        replication_status: Cross-region replication status, or None if the
            provider does not report one.
    """

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
class AsyncDownloadResponse:
    """Result of an async download operation."""

    key: str
    content: bytes
    metadata: BlobMetadata


class AsyncStreamingDownloadResponse:
    """A blob's content, consumed lazily instead of held in memory.

    Offers both an async iterator over chunks and an awaitable file-like
    ``read``, and must be closed to release the underlying connection::

        async with await client.download_stream("large.bin") as stream:
            async for chunk in stream:
                process(chunk)

    or::

        async with await client.download_stream("large.bin") as stream:
            while data := await stream.read(8192):
                destination.write(data)

    Mixing the two is supported: whatever ``read`` has buffered is handed
    back before iteration continues from the connection.
    """

    def __init__(
        self,
        key: str,
        metadata: BlobMetadata,
        stream: AsyncIterator[bytes],
        close_func: Optional[Callable[[], Awaitable[None]]] = None,
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

    @property
    def closed(self) -> bool:
        """Whether the stream has been closed."""
        return self._closed

    async def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes, or everything remaining if ``size`` is negative.

        Returns exactly ``size`` bytes until the object is exhausted, pulling
        further chunks as needed and keeping any surplus for the next call.
        Returns empty bytes once exhausted.

        Args:
            size: Maximum number of bytes to return. Any negative value reads
                the remainder, as :mod:`io` does, which puts it all in memory.

        Raises:
            ValueError: If the stream is closed.
        """
        if self._closed:
            raise ValueError("I/O operation on closed stream")

        # Any negative size means read-all, matching io. Treating only -1 that
        # way would make read(-2) return a silently truncated slice.
        if size < 0:
            chunks = [self._buffer] if self._buffer else []
            async for chunk in self._stream:
                chunks.append(chunk)
            self._buffer = b""
            return b"".join(chunks)

        while len(self._buffer) < size:
            try:
                self._buffer += await self._stream.__anext__()
            except StopAsyncIteration:
                break

        result = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return result

    def __aiter__(self) -> "AsyncStreamingDownloadResponse":
        if self._closed:
            raise ValueError("I/O operation on closed stream")
        return self

    async def __anext__(self) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed stream")
        if self._buffer:
            buffered, self._buffer = self._buffer, b""
            return buffered
        return await self._stream.__anext__()

    async def close(self) -> None:
        """Release the underlying connection. Safe to call more than once."""
        if not self._closed:
            self._closed = True
            if self._close_func is not None:
                await self._close_func()

    async def __aenter__(self) -> "AsyncStreamingDownloadResponse":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()


@dataclass(frozen=True)
class BlobInfo:
    """A blob's identity and lightweight metadata from listing operations.

    Mirrors the synchronous :class:`multicloud.blob.blob_client.driver.BlobInfo`.
    Populated from a single listing entry; fields the provider does not
    report stay ``None``.

    Attributes:
        key: Object key (path) in the bucket.
        size: Object size in bytes, or None if not reported.
        e_tag: Entity tag; provider-specific opaque format, not comparable
            across providers, or None if not reported.
        last_modified: Last-modified timestamp as a timezone-aware
            datetime, or None if the provider did not report it.
        storage_class: Provider storage class/tier (e.g. "STANDARD"), or None.
        version_id: Provider-specific version identifier, or None if the
            provider did not report version information.
        is_latest: Whether this entry is the latest version, or None if the
            provider did not report version state.
    """

    key: str
    size: Optional[int] = None
    e_tag: Optional[str] = None
    last_modified: Optional[datetime] = None
    storage_class: Optional[str] = None
    version_id: Optional[str] = None
    is_latest: Optional[bool] = None


@dataclass(frozen=True)
class ListBlobsPageResponse:
    """Single page of blob listing results with pagination support.

    Mirrors the synchronous
    :class:`multicloud.blob.blob_client.driver.ListBlobsPageResponse`.

    Attributes:
        blobs: The blobs on this page (empty when nothing matched).
        common_prefixes: Prefixes rolled up by ``delimiter`` (empty when no
            delimiter was supplied or none matched).
        is_truncated: True if more pages remain. When True,
            ``next_pagination_token`` carries the token for the next page.
        next_pagination_token: Opaque token to pass back as
            ``pagination_token`` to fetch the next page, or None on the
            last page.
    """

    blobs: List[BlobInfo] = field(default_factory=list)
    common_prefixes: List[str] = field(default_factory=list)
    is_truncated: bool = False
    next_pagination_token: Optional[str] = None


@dataclass(frozen=True)
class PresignedUrlResult:
    """Result of an async presigned URL generation.

    Attributes:
        url: The presigned URL string for direct blob access.
        expiration: UTC datetime when the URL expires.
        correlation_id: Unique identifier for tracing/correlation.
        http_method: The HTTP method the consumer must use (``"GET"`` or
            ``"PUT"``).
        required_headers: Headers the consumer MUST include in the HTTP
            request for the presigned URL to work. Includes signed metadata,
            content-type, and encryption headers. Empty mapping if none
            required.
    """

    url: str
    expiration: datetime
    correlation_id: str
    http_method: str = "GET"
    required_headers: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class BlobIdentifier:
    """Identifies a blob for bulk delete operations.

    Attributes:
        key: Object key (path) in the bucket. Required.
        version_id: Specific object version to delete. Optional.
            If None, targets the latest version (or inserts a delete
            marker on versioned buckets).
    """

    key: str
    version_id: Optional[str] = None


@dataclass(frozen=True)
class DeletedBlob:
    """A successfully deleted blob entry in a bulk delete response.

    Attributes:
        key: Object key that was deleted.
        version_id: Version that was deleted (if versioned bucket).
    """

    key: str
    version_id: Optional[str] = None


@dataclass(frozen=True)
class DeleteError:
    """A blob that failed to delete in a bulk delete operation.

    Attributes:
        key: Object key that failed.
        version_id: Version that was targeted (if specified).
        code: Provider-specific error code (e.g. "AccessDenied").
        message: Human-readable error description.
    """

    key: str
    version_id: Optional[str] = None
    code: str = ""
    message: str = ""


@dataclass(frozen=True)
class BulkDeleteResponse:
    """Result of a bulk delete operation.

    Reports per-object outcomes. The operation does NOT raise on
    partial failure — inspect ``errors`` for failed items.

    Attributes:
        deleted: List of successfully deleted objects.
        errors: List of objects that failed to delete.
    """

    deleted: list[DeletedBlob] = field(default_factory=list)
    errors: list[DeleteError] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """True if any objects failed to delete."""
        return len(self.errors) > 0


@dataclass(frozen=True)
class CopyRequest:
    """Request parameters for an async blob copy operation.

    Supports same-bucket copy (rename/move pattern) and cross-bucket copy.
    Metadata override replaces the source object's metadata on the copy.

    Attributes:
        source_bucket: Bucket containing the source object.
        source_key: Key of the source object to copy.
        destination_bucket: Bucket for the copied object.
        destination_key: Key for the copied object.
        source_version_id: Specific version of the source to copy (optional).
        metadata: If provided, replaces the source object's metadata on the
            destination. If None, the source metadata is preserved.
    """

    source_bucket: str
    source_key: str
    destination_bucket: str
    destination_key: str
    source_version_id: Optional[str] = None
    metadata: Optional[Mapping[str, str]] = None


@dataclass(frozen=True)
class CopyResponse:
    """Result of an async blob copy operation.

    Attributes:
        key: Destination key where the object was copied to.
        version_id: Version ID of the newly created copy (if versioning
            is enabled on the destination bucket). None otherwise.
        e_tag: Entity tag (checksum) of the new copy.
        last_modified: ISO-8601 timestamp of when the copy was created.
            Corresponds to S3 CopyObjectResult.LastModified or GCS
            timeCreated/updated. None if not available from the provider.
        correlation_id: Unique identifier for tracking this operation.
    """

    key: str
    version_id: Optional[str] = None
    e_tag: Optional[str] = None
    last_modified: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass(frozen=True)
class FailedAsyncBlobUpload:
    """A single failed upload within an async directory upload operation."""

    source: Path
    exception: Exception


@dataclass(frozen=True)
class AsyncDirectoryUploadResponse:
    """Result of an async directory upload operation.

    Attributes:
        total_files: Total number of files attempted.
        successful_uploads: Number of files successfully uploaded.
        failed_transfers: List of per-file failures with source and exception.
        total_bytes_transferred: Total bytes successfully uploaded.
    """

    total_files: int = 0
    successful_uploads: int = 0
    failed_transfers: List[FailedAsyncBlobUpload] = field(default_factory=list)
    total_bytes_transferred: int = 0

    @property
    def has_failures(self) -> bool:
        """True if any uploads failed."""
        return len(self.failed_transfers) > 0


#: Part numbers are one-based everywhere. The upper bound is not: each
#: provider caps it differently, so it is enforced by the provider.
MIN_PART_NUMBER = 1


@dataclass(frozen=True)
class FailedAsyncBlobDownload:
    """A single failed download within an async directory download operation."""

    destination: Path
    exception: Exception


@dataclass(frozen=True)
class AsyncDirectoryDownloadResponse:
    """Result of an async directory download operation.

    Attributes:
        total_files: Total number of files attempted.
        successful_downloads: Number of files successfully downloaded.
        failed_transfers: List of per-file failures with destination and exception.
        total_bytes_transferred: Total bytes successfully downloaded.
    """

    total_files: int = 0
    successful_downloads: int = 0
    failed_transfers: List[FailedAsyncBlobDownload] = field(default_factory=list)
    total_bytes_transferred: int = 0

    @property
    def has_failures(self) -> bool:
        """True if any downloads failed."""
        return len(self.failed_transfers) > 0


def build_range_header(start: Optional[int], end: Optional[int]) -> Optional[str]:
    """Build HTTP Range header for partial downloads.

    Cloud-agnostic utility used by all providers to construct
    RFC 7233 byte-range headers from optional start/end offsets.

    Args:
        start: Starting byte offset (inclusive), or None for suffix range.
        end: Ending byte offset (inclusive), or None for open-ended range.

    Returns:
        Range header string in ``bytes=N-M`` format, or None if both are None.

    Examples:
        >>> build_range_header(0, 499)
        'bytes=0-499'
        >>> build_range_header(100, None)
        'bytes=100-'
        >>> build_range_header(None, 500)
        'bytes=-500'
        >>> build_range_header(None, None) is None
        True
    """
    if start is None and end is None:
        return None
    if start is None:
        return f"bytes=-{end}"
    if end is None:
        return f"bytes={start}-"
    return f"bytes={start}-{end}"


def parse_rfc3339_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse an RFC 3339 / ISO 8601 timestamp into a timezone-aware datetime.

    Cloud-agnostic utility for providers that return timestamps as strings
    (e.g. ``"2024-01-15T12:00:00.123Z"`` or ``"2024-01-15T12:00:00.123+00:00"``).
    Python 3.10's :meth:`datetime.fromisoformat` does not accept a trailing
    ``"Z"``, so it is normalized to ``"+00:00"`` first.

    Args:
        value: RFC 3339 timestamp string, or None.

    Returns:
        A timezone-aware :class:`datetime`, or None if ``value`` is falsy or
        cannot be parsed. ``last_modified`` / ``created_time`` are optional, so
        parsing is best-effort and never raises.
    """
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


class BaseAsyncBlobStore(Provider, ABC):
    """Abstract base class for async blob store implementations.

    Follows the template method pattern: public methods validate input
    and delegate to abstract ``do_*`` hooks that providers implement.

    Providers must accept a ``credentials_overrider`` constructor kwarg and
    forward it to ``super().__init__`` so it is available for credential
    resolution.
    """

    _MIN_EXPIRATION_SECONDS = 1
    _MAX_EXPIRATION_SECONDS = 604800  # 7 days
    _VALID_OPERATIONS = ("GET", "PUT")

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        *,
        endpoint_url: Optional[str] = None,
        transfer_config: Optional[AsyncTransferConfig] = None,
        credentials_overrider: Optional[CredentialsOverrider] = None,
        proxy_config: Optional[ProxyConfig] = None,
        validator: Optional[BlobStoreValidator] = None,
    ) -> None:
        self._validator = validator or BlobStoreValidator()
        self._validator.validate_bucket(bucket)
        self._bucket = bucket
        self._region = region
        self._endpoint_url = endpoint_url
        self._transfer_config = transfer_config or AsyncTransferConfig()
        self._credentials_overrider = credentials_overrider
        self._proxy_config = proxy_config

    @property
    def bucket(self) -> str:
        """The bucket this store operates on."""
        return self._bucket

    @property
    def region(self) -> Optional[str]:
        """Cloud region (provider-interpreted)."""
        return self._region

    @property
    def transfer_config(self) -> AsyncTransferConfig:
        """Transfer configuration for this store."""
        return self._transfer_config

    # Upload

    async def upload(
        self,
        key: str,
        payload: AsyncUploadPayload,
        *,
        content_type: Optional[str] = None,
        content_length: Optional[int] = None,
        metadata: Optional[Mapping[str, str]] = None,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        storage_class: Optional[str] = None,
    ) -> AsyncUploadResponse:
        """Upload data to the object identified by ``key``.

        Args:
            key: Object key (path) in the bucket.
            payload: Content to upload -- bytes, Path, or BinaryIO.
            content_type: MIME type. Auto-detected from Path extension
                if None. Falls back to "application/octet-stream".
            content_length: Size in bytes. Calculated from payload if omitted.
            metadata: User-defined metadata dict attached to the object.
            create_if_absent: Only create when absent; else
                ResourceExistsException. A concurrent write that races the
                condition raises ResourceConflictException (the write may not
                have happened; retry). Mutually exclusive with if_match and
                if_version_match.
            if_match: Only overwrite when the current ETag matches; else
                FailedPreconditionException; a racing write raises
                ResourceConflictException (the write may not have happened;
                retry). Not all providers support conditional overwrite on
                upload; unsupported providers raise UnsupportedOperationException.
            if_version_match: Only overwrite when the current version id
                matches; else FailedPreconditionException; a racing write raises
                ResourceConflictException (the write may not have happened;
                retry). Not all providers support version-based conditional
                overwrite; unsupported providers raise UnsupportedOperationException.
            kms_key_id: Provider key id for server-side encryption.
            use_kms_managed_key: Request a provider-managed encryption key when
                no kms_key_id is given. A no-op on providers that always
                encrypt at rest.
            storage_class: Provider-native storage class for the object (for
                example ``"STANDARD_IA"`` or ``"NEARLINE"``), passed through
                unchanged. None uses the bucket default. A blank value, or a
                class the provider does not recognise, raises
                InvalidArgumentException.

        Returns:
            AsyncUploadResponse with key, version_id, e_tag.

        Raises:
            InvalidArgumentException: If key is empty, content_length is
                negative, or payload unsupported.
        """
        self._validator.validate_upload(
            key,
            content_length,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
        )
        self._validator.validate_storage_class(storage_class)

        payload_bytes, resolved_content_type = await self._prepare_payload(
            payload, content_type
        )
        resolved_length = (
            content_length if content_length is not None else len(payload_bytes)
        )

        return await self.do_upload(
            key,
            payload_bytes,
            content_type=resolved_content_type,
            content_length=resolved_length,
            metadata=metadata or {},
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
            kms_key_id=kms_key_id,
            use_kms_managed_key=use_kms_managed_key,
            storage_class=storage_class,
        )

    async def upload_from_path(
        self,
        key: str,
        path: Union[str, Path, "os.PathLike[str]"],
        *,
        content_type: Optional[str] = None,
        content_length: Optional[int] = None,
        metadata: Optional[Mapping[str, str]] = None,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        storage_class: Optional[str] = None,
    ) -> AsyncUploadResponse:
        """Upload a local file to the object identified by ``key``.

        Rejects a missing or non-regular file before any request is sent, and
        auto-detects the content type from the file extension when one is not
        given.

        Args:
            key: Object key (path) in the bucket.
            path: Local file to upload.
            content_type: MIME type. Detected from the extension if omitted.
            content_length: Size in bytes. Taken from the file if omitted.
            metadata: User-defined metadata to attach.
            create_if_absent: Only create; fail if the key already exists.
            if_match: Only write if the current entity tag matches.
            if_version_match: Only write if the current version matches.
            kms_key_id: Customer-managed encryption key.
            use_kms_managed_key: Encrypt with a provider-managed key.
            storage_class: Provider-native storage class, or None for the
                bucket default. See :meth:`upload`.

        Returns:
            AsyncUploadResponse with key, version_id, e_tag.

        Raises:
            InvalidArgumentException: If key is empty, the file does not
                exist, or it is not a regular file.
        """
        # Checked before touching the filesystem, and separately from the
        # path validator, which takes no length. Without it this method would
        # accept a negative length that upload() rejects.
        self._validator.validate_content_length(content_length)
        self._validator.validate_storage_class(storage_class)

        resolved = path if isinstance(path, Path) else Path(os.fspath(path))

        def _validate_and_read() -> bytes:
            self._validator.validate_upload_from_path(
                key,
                resolved,
                create_if_absent=create_if_absent,
                if_match=if_match,
                if_version_match=if_version_match,
            )
            return resolved.read_bytes()

        # Stat and read are both filesystem calls, so they share one hop off
        # the event loop rather than blocking it.
        payload_bytes = await asyncio.to_thread(_validate_and_read)

        if content_type is None:
            guessed, _ = mimetypes.guess_type(str(resolved))
            content_type = guessed or "application/octet-stream"

        return await self.do_upload(
            key,
            payload_bytes,
            content_type=content_type,
            content_length=(
                content_length if content_length is not None else len(payload_bytes)
            ),
            metadata=metadata or {},
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
            kms_key_id=kms_key_id,
            use_kms_managed_key=use_kms_managed_key,
            storage_class=storage_class,
        )

    # Download

    async def download(self, request: AsyncDownloadRequest) -> AsyncDownloadResponse:
        """Download the object identified by ``request.key``.

        Args:
            request: AsyncDownloadRequest containing key and optional
                version_id, start, end for range downloads.

        Returns:
            AsyncDownloadResponse with key, content bytes, and metadata.

        Raises:
            InvalidArgumentException: If key is empty or range params invalid.
        """
        self._validator.validate_download(
            request.key,
            request.start,
            request.end,
            if_match=request.if_match,
        )

        return await self.do_download(request)

    async def download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> AsyncStreamingDownloadResponse:
        """Open the object for lazy reading instead of loading it into memory.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version. Optional.
            start: First byte to read, inclusive. Optional.
            end: Last byte to read, inclusive. Optional.

        Returns:
            An AsyncStreamingDownloadResponse the caller must close.

        Raises:
            InvalidArgumentException: If key is empty or the range is invalid.
        """
        self._validator.validate_download(key, start, end)
        return await self.do_download_stream(
            key, version_id=version_id, start=start, end=end
        )

    async def get_tags(self, key: str) -> Mapping[str, str]:
        """Retrieve the tags attached to the object identified by ``key``.

        Args:
            key: Object key (path) in the bucket.

        Returns:
            A mapping of tag keys to values. Empty if the object has no tags.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        self._validator.validate_get_tags(key)
        return await self.do_get_tags(key)

    async def set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Replace all tags on the object identified by ``key``.

        Args:
            key: Object key (path) in the bucket.
            tags: Mapping of tag keys to values. Replaces any existing tags.

        Raises:
            InvalidArgumentException: If key is empty or tags is empty.
        """
        self._validator.validate_set_tags(key, tags)
        await self.do_set_tags(key, tags)

    # Metadata

    async def get_metadata(
        self, key: str, *, version_id: Optional[str] = None
    ) -> BlobMetadata:
        """Retrieve object metadata (HEAD-style) without downloading the body.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version. Optional.

        Returns:
            BlobMetadata with key, size (``content_length``), content_type,
            e_tag, version_id, custom metadata, last_modified, and created_time.

        Raises:
            InvalidArgumentException: If key is empty.
            BlobNotFoundError: If the object does not exist.
        """
        self._validator.validate_key(key)
        return await self.do_get_metadata(key, version_id=version_id)

    # Object lock

    async def get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Retrieve the object lock state (retention + legal hold) for an object.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version. Optional; None targets the
                latest version.

        Returns:
            ObjectLockInfo, or None when the object has no retention and no
            legal hold.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        self._validator.validate_key(key)
        return await self.do_get_object_lock(key, version_id=version_id)

    async def update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str] = None,
    ) -> None:
        """Update the retention configuration of a single object.

        The object must already have retention configured; this updates it. It
        does not establish retention for the first time.

        Args:
            key: Object key (path) in the bucket.
            retention: The retention change to apply.
            version_id: Specific object version. Optional; None targets the
                latest version.

        Raises:
            InvalidArgumentException: If key is empty, the config is missing,
                or its retain-until instant is missing or timezone-naive.
            FailedPreconditionException: If the object has no retention yet, or
                the change is not permitted from its current state.
        """
        self._validator.validate_update_object_retention(key, retention)
        await self.do_update_object_retention(key, retention, version_id=version_id)

    async def update_legal_hold(
        self, key: str, enabled: bool, *, version_id: Optional[str] = None
    ) -> None:
        """Set or remove a legal hold on an object.

        A legal hold, once placed, prevents the object from being deleted or
        overwritten until the hold is explicitly removed, independent of any
        retention period.

        Args:
            key: Object key (path) in the bucket.
            enabled: True to place a legal hold, False to remove it.
            version_id: Specific object version. Optional; None targets the
                latest version.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        self._validator.validate_key(key)
        await self.do_update_legal_hold(key, enabled, version_id=version_id)

    # Delete

    async def delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Delete a single object. Idempotent — does not raise if the object does not exist.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version to delete. Optional.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        self._validator.validate_delete(key)
        await self.do_delete(key, version_id=version_id)

    async def delete_objects(
        self,
        identifiers: list[BlobIdentifier],
    ) -> BulkDeleteResponse:
        """Delete multiple objects. Does NOT raise on partial failure.

        Non-existent objects are reported as successfully deleted
        (idempotent — deleting a non-existent object is a successful no-op).

        Args:
            identifiers: List of BlobIdentifiers to delete.

        Returns:
            BulkDeleteResponse with deleted and errors lists.

        Raises:
            InvalidArgumentException: If identifiers is empty or invalid.
        """
        self._validator.validate_delete_objects(identifiers)
        return await self.do_delete_objects(identifiers)

    # Copy

    async def copy(self, request: CopyRequest) -> CopyResponse:
        """Copy a blob from one location to another.

        Supports same-bucket copy (move/rename pattern) and cross-bucket copy.

        Args:
            request: CopyRequest with source and destination details.

        Returns:
            CopyResponse with destination key, version_id, e_tag, and
            correlation_id for tracking.

        Raises:
            InvalidArgumentException: If required fields are empty.
        """
        if not request.source_bucket:
            raise InvalidArgumentException("copy: source_bucket is required")
        if not request.source_key:
            raise InvalidArgumentException("copy: source_key is required")
        if not request.destination_bucket:
            raise InvalidArgumentException("copy: destination_bucket is required")
        if not request.destination_key:
            raise InvalidArgumentException("copy: destination_key is required")

        correlation_id = str(uuid.uuid4())
        return await self.do_copy(request, correlation_id=correlation_id)

    # Multipart upload

    async def initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Optional[Mapping[str, str]] = None,
        content_type: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
    ) -> MultipartUpload:
        """Start a multipart upload and return its handle."""
        self._validator.validate_key(key)
        correlation_id = str(uuid.uuid4())
        return await self.do_initiate_multipart_upload(
            key,
            metadata=metadata or {},
            content_type=content_type,
            tags=tags or {},
            correlation_id=correlation_id,
        )

    async def upload_part(
        self, mpu: MultipartUpload, part: MultipartPart
    ) -> UploadPartResponse:
        """Upload one part of an in-progress multipart upload."""
        self._validator.validate_multipart_upload(mpu, self._bucket)
        self._validate_part_number(part.part_number)
        correlation_id = str(uuid.uuid4())
        return await self.do_upload_part(mpu, part, correlation_id=correlation_id)

    async def complete_multipart_upload(
        self, mpu: MultipartUpload, parts: List[UploadPartResponse]
    ) -> MultipartUploadResponse:
        """Assemble the uploaded parts into the final object."""
        self._validator.validate_multipart_upload(mpu, self._bucket)
        if not parts:
            raise InvalidArgumentException(
                "complete_multipart_upload: parts is required"
            )
        seen: set[int] = set()
        for part in parts:
            self._validate_part_number(part.part_number)
            if part.part_number in seen:
                raise InvalidArgumentException(
                    f"complete_multipart_upload: part_number "
                    f"{part.part_number} appears more than once"
                )
            seen.add(part.part_number)
        correlation_id = str(uuid.uuid4())
        return await self.do_complete_multipart_upload(
            mpu, parts, correlation_id=correlation_id
        )

    async def list_parts(self, mpu: MultipartUpload) -> List[UploadPartResponse]:
        """List the parts uploaded so far, in ascending part-number order."""
        self._validator.validate_multipart_upload(mpu, self._bucket)
        correlation_id = str(uuid.uuid4())
        return await self.do_list_parts(mpu, correlation_id=correlation_id)

    async def abort_multipart_upload(self, mpu: MultipartUpload) -> None:
        """Discard an in-progress multipart upload and its parts."""
        self._validator.validate_multipart_upload(mpu, self._bucket)
        correlation_id = str(uuid.uuid4())
        await self.do_abort_multipart_upload(mpu, correlation_id=correlation_id)

    @staticmethod
    def _validate_part_number(part_number: int) -> None:
        """Reject a part number no provider could accept.

        Lower bound only; the ceiling differs per provider.
        """
        if part_number < MIN_PART_NUMBER:
            raise InvalidArgumentException(
                f"part_number must be {MIN_PART_NUMBER} or greater, "
                f"got {part_number}"
            )

    # Presigned URL

    async def generate_presigned_url(
        self,
        key: str,
        operation: str = "GET",
        *,
        expiration_seconds: int = 3600,
        version_id: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
        tags: Optional[Mapping[str, str]] = None,
        kms_key_id: Optional[str] = None,
        content_disposition: Optional[str] = None,
    ) -> PresignedUrlResult:
        """Generate a presigned URL for GET or PUT operations.

        This is a synchronous operation (no network I/O) -- the URL is signed
        locally and returned immediately.

        Args:
            key: Object key (path) in the bucket.
            operation: HTTP method - either "GET" (download) or "PUT" (upload).
                Defaults to "GET".
            expiration_seconds: URL validity duration in seconds.
                Must be between 1 and 604800 (7 days). Defaults to 3600 (1 hour).
            version_id: Specific object version (GET only). Optional.
            content_type: MIME type for PUT operations. Optional but recommended.
            metadata: Custom metadata for PUT operations. Optional.
            tags: Object tags for PUT operations. Optional.
            kms_key_id: KMS key ARN for server-side encryption (PUT). Optional.
            content_disposition: Content-Disposition header value. Optional.

        Returns:
            PresignedUrlResult with url, expiration, correlation_id, http_method,
            and required_headers.

        Raises:
            InvalidArgumentException: If key is empty, operation is not
                GET/PUT, or expiration_seconds is out of range.
        """
        if not key:
            raise InvalidArgumentException("key is required")

        if operation not in self._VALID_OPERATIONS:
            raise InvalidArgumentException(
                f"Unsupported operation '{operation}'. "
                f"Valid operations: {', '.join(self._VALID_OPERATIONS)}"
            )

        if not (
            self._MIN_EXPIRATION_SECONDS
            <= expiration_seconds
            <= self._MAX_EXPIRATION_SECONDS
        ):
            raise InvalidArgumentException(
                f"expiration_seconds must be between "
                f"{self._MIN_EXPIRATION_SECONDS} and {self._MAX_EXPIRATION_SECONDS}, "
                f"got {expiration_seconds}"
            )

        correlation_id = str(uuid.uuid4())

        # Normalize metadata and tags to empty dicts if None
        metadata = metadata or {}
        tags = tags or {}

        return await self.do_generate_presigned_url(
            key,
            operation,
            expiration_seconds=expiration_seconds,
            version_id=version_id,
            content_type=content_type,
            metadata=metadata,
            tags=tags,
            kms_key_id=kms_key_id,
            content_disposition=content_disposition,
            correlation_id=correlation_id,
        )

    # Directory operations
    #
    # Prefix contract: matched verbatim, so ``"logs"`` also matches
    # ``"logs.txt"``; empty or None means every object in the bucket. Follows
    # the reference SDK, which never validates the prefix.
    #
    # ``download_directory`` is the exception -- it rejects a blank prefix and
    # appends ``"/"``. Kept, because normalizing is what stops the same call
    # selecting different objects on different providers.

    async def upload_directory(
        self,
        local_source_directory: str,
        *,
        prefix: str = "",
        include_sub_folders: bool = True,
        follow_symbolic_links: bool = False,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        tags: Optional[Mapping[str, str]] = None,
    ) -> "AsyncDirectoryUploadResponse":
        """Validate and delegate to :meth:`do_upload_directory`.

        The prefix follows the contract stated above the directory operations:
        matched verbatim, never normalized, and empty means the bucket root.

        Args:
            local_source_directory: Path to the local directory to upload.
            prefix: Key prefix to prepend to all uploaded objects.
            include_sub_folders: Whether to recurse into subdirectories.
            follow_symbolic_links: Whether to follow symlinks.
            max_concurrency: Maximum concurrent upload coroutines.
            tags: Tags to apply to all uploaded objects.

        Returns:
            AsyncDirectoryUploadResponse with results and failures.

        Raises:
            InvalidArgumentException: If the source directory is invalid.
        """
        if not local_source_directory or not local_source_directory.strip():
            raise InvalidArgumentException(
                "upload_directory: local_source_directory is required"
            )
        source = Path(local_source_directory)
        if not source.exists():
            raise InvalidArgumentException(
                f"upload_directory: source directory does not exist: {source}"
            )
        if not source.is_dir():
            raise InvalidArgumentException(
                f"upload_directory: source path is not a directory: {source}"
            )
        if max_concurrency <= 0:
            raise InvalidArgumentException(
                f"upload_directory: max_concurrency must be > 0, got {max_concurrency}"
            )
        return await self.do_upload_directory(
            local_source_directory,
            prefix=prefix,
            include_sub_folders=include_sub_folders,
            follow_symbolic_links=follow_symbolic_links,
            max_concurrency=max_concurrency,
            tags=tags,
        )

    async def download_directory(
        self,
        prefix: str,
        local_destination_directory: str,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        prefixes_to_exclude: Optional[List[str]] = None,
    ) -> "AsyncDirectoryDownloadResponse":
        """Validate and delegate to :meth:`do_download_directory`.

        Unlike its siblings, this rejects a blank prefix and appends a trailing
        ``"/"`` when one is missing, so it addresses a directory rather than a
        raw key prefix. See the prefix contract above the directory operations
        for why this one differs.

        Args:
            prefix: Key prefix identifying the "directory" to download. Must be
                non-blank; normalized to end with ``"/"``.
            local_destination_directory: Local path to write files into.
            max_concurrency: Maximum concurrent download coroutines.
            prefixes_to_exclude: Prefixes to skip during download.

        Returns:
            AsyncDirectoryDownloadResponse with results and failures.

        Raises:
            InvalidArgumentException: If prefix or destination is empty.
        """
        if not prefix or not prefix.strip():
            raise InvalidArgumentException("download_directory: prefix is required")
        if not local_destination_directory or not local_destination_directory.strip():
            raise InvalidArgumentException(
                "download_directory: local_destination_directory is required"
            )
        if max_concurrency <= 0:
            raise InvalidArgumentException(
                f"download_directory: max_concurrency must be > 0, got {max_concurrency}"
            )
        # Normalize prefix to end with "/" for consistent listing behavior
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        return await self.do_download_directory(
            prefix,
            local_destination_directory,
            max_concurrency=max_concurrency,
            prefixes_to_exclude=prefixes_to_exclude,
        )

    async def delete_directory(
        self,
        prefix: Optional[str],
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        """Validate and delegate to :meth:`do_delete_directory`.

        Deletes every object whose key begins with ``prefix``. Matching a
        prefix that contains no objects is a successful no-op, and objects
        that disappear while the operation is running are treated as already
        deleted.

        The prefix is matched verbatim and is never normalized. A prefix with
        no trailing ``"/"`` therefore also matches sibling keys that merely
        share its leading characters -- ``"logs"`` matches ``"logs.txt"`` and
        ``"logs-archive/x"`` as well as ``"logs/x"``. Pass a trailing ``"/"``
        to target only the directory.

        An empty or ``None`` prefix matches **every object in the bucket**
        and deletes all of them.

        Only the current version of each object is deleted. On a
        version-enabled bucket prior versions are retained, so storage is not
        necessarily reclaimed.

        The operation is not atomic. Objects written under the prefix while it
        is running may survive.

        Args:
            prefix: Key prefix identifying the "directory" to delete. Empty or
                None deletes every object in the bucket.
            max_concurrency: Maximum concurrent delete coroutines.

        Raises:
            InvalidArgumentException: If max_concurrency is not positive.
            SubstrateSdkException: If any matched object could not be deleted.
        """
        if max_concurrency <= 0:
            raise InvalidArgumentException(
                f"delete_directory: max_concurrency must be > 0, got {max_concurrency}"
            )
        await self.do_delete_directory(prefix, max_concurrency=max_concurrency)

    # Listing

    async def list_page(
        self,
        *,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_results: int = 1000,
        pagination_token: Optional[str] = None,
    ) -> ListBlobsPageResponse:
        """Retrieve a single page of blobs matching the given criteria.

        Fetches exactly one page and returns it together with a
        continuation token for manual pagination -- pass
        ``page.next_pagination_token`` back as ``pagination_token`` to
        fetch the next page. Iterate until ``is_truncated`` is False.

        Args:
            prefix: Restrict results to keys beginning with this prefix.
            delimiter: Group keys sharing a common substring up to the
                delimiter into ``common_prefixes`` (e.g. "/" for a
                directory-style listing).
            max_results: Maximum number of blobs to return on this page.
                Must be a positive integer. Defaults to 1000.
            pagination_token: Opaque token from a prior page's
                ``next_pagination_token`` to continue listing. None starts
                from the beginning.

        Returns:
            ListBlobsPageResponse with the page's blobs, common_prefixes,
            is_truncated flag, and next_pagination_token. When no objects
            match, ``blobs`` is empty and ``is_truncated`` is False.

        Raises:
            InvalidArgumentException: If max_results is less than 1.
        """
        self._validator.validate_list_page(max_results)
        return await self.do_list_page(
            prefix=prefix,
            delimiter=delimiter,
            max_results=max_results,
            pagination_token=pagination_token,
        )

    def list_blob_versions(self, key: str) -> AsyncIterator[BlobInfo]:
        """Iterate every version of one key.

        Deliberately not a coroutine: the key is validated now and the pages
        are fetched as they are consumed, so callers write
        ``async for info in store.list_blob_versions(key)`` without an
        intervening await.

        Args:
            key: Object key (path) in the bucket.

        Returns:
            An async iterator of BlobInfo, one per version. A key with no
            versions yields nothing rather than raising.

            The order is whatever the provider returns and is not normalised,
            so do not rely on it. Sort on ``last_modified`` if order matters.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        self._validator.validate_key(key)
        return self.do_list_blob_versions(key)

    # Existence checks

    async def does_object_exist(
        self, key: str, *, version_id: Optional[str] = None
    ) -> bool:
        """Check if an object exists in the bucket.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version to check. Optional.
                Provider support varies -- implementations that do not support
                versioning will raise InvalidArgumentException.

        Returns:
            True if the object exists, False otherwise.

        Raises:
            InvalidArgumentException: If key is empty or version_id unsupported.
        """
        self._validator.validate_key(key)
        return await self.do_does_object_exist(key, version_id=version_id)

    async def does_bucket_exist(self) -> bool:
        """Check if the bucket exists.

        Returns:
            True if the bucket exists, False otherwise.
        """
        return await self.do_does_bucket_exist()

    # Abstract methods

    @abstractmethod
    async def do_upload(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: Optional[str],
        content_length: int,
        metadata: Mapping[str, str],
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        storage_class: Optional[str] = None,
    ) -> AsyncUploadResponse:
        """Provider-specific upload implementation.

        ``storage_class`` is provider-native, already validated as non-blank,
        or None for the bucket default.
        """

    @abstractmethod
    async def do_download(self, request: AsyncDownloadRequest) -> AsyncDownloadResponse:
        """Provider-specific download implementation."""

    @abstractmethod
    async def do_copy(
        self, request: CopyRequest, *, correlation_id: str
    ) -> CopyResponse:
        """Provider-specific copy implementation.

        Args:
            request: Validated CopyRequest.
            correlation_id: Unique correlation ID for this operation.

        Returns:
            CopyResponse with destination details.
        """

    @abstractmethod
    async def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Provider-specific single object delete. Must be idempotent."""

    @abstractmethod
    async def do_delete_objects(
        self,
        identifiers: list[BlobIdentifier],
    ) -> BulkDeleteResponse:
        """Provider-specific bulk delete implementation."""

    @abstractmethod
    async def do_get_tags(self, key: str) -> Mapping[str, str]:
        """Provider-specific tag retrieval implementation."""

    @abstractmethod
    async def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Provider-specific tag replacement implementation."""

    @abstractmethod
    async def do_upload_directory(
        self,
        local_source_directory: str,
        *,
        prefix: str = "",
        include_sub_folders: bool = True,
        follow_symbolic_links: bool = False,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        tags: Optional[Mapping[str, str]] = None,
    ) -> "AsyncDirectoryUploadResponse":
        """Provider-specific async directory upload."""

    @abstractmethod
    async def do_download_directory(
        self,
        prefix: str,
        local_destination_directory: str,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        prefixes_to_exclude: Optional[List[str]] = None,
    ) -> "AsyncDirectoryDownloadResponse":
        """Provider-specific async directory download."""

    @abstractmethod
    async def do_delete_directory(
        self,
        prefix: Optional[str],
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        """Provider-specific async directory (prefix) delete.

        Implementations must enumerate every object under the prefix,
        exhausting pagination, and must treat an object that is already
        absent as successfully deleted.
        """

    @abstractmethod
    async def do_get_metadata(
        self, key: str, *, version_id: Optional[str] = None
    ) -> BlobMetadata:
        """Provider-specific HEAD-style metadata retrieval.

        Must NOT download the object body. Must raise BlobNotFoundError if
        the object does not exist. Abstract because HEAD-style metadata is a
        core operation every production provider supports (like
        do_does_object_exist) -- a provider missing it fails fast with a
        clear TypeError at construction time.
        """

    async def do_get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Provider-specific object lock state retrieval.

        Must return None when the object carries neither retention nor a
        legal hold, so callers can distinguish "unlocked" from a lock whose
        fields happen to be empty.

        Not abstract so existing providers remain instantiable; providers
        that support object lock override this.
        """
        raise UnsupportedOperationException(
            "get_object_lock is not supported by this provider"
        )

    async def do_update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str] = None,
    ) -> None:
        """Provider-specific object retention write.

        Implementations must read the object's current retention, resolve the
        request against the shared rules table, and write back the resolved
        mode, so that a config mode of None preserves whatever is already set.

        Not abstract so existing providers remain instantiable; providers
        that support object lock override this.
        """
        raise UnsupportedOperationException(
            "update_object_retention is not supported by this provider"
        )

    async def do_update_legal_hold(
        self, key: str, enabled: bool, *, version_id: Optional[str] = None
    ) -> None:
        """Provider-specific legal hold write.

        Sets a hold when ``enabled`` is True and removes it when False. On a
        provider that expresses holds through more than one flag, removal must
        clear all of them, so that a released object is not still held by a
        flag the caller cannot see.

        Not abstract so existing providers remain instantiable; providers
        that support object lock override this.
        """
        raise UnsupportedOperationException(
            "update_legal_hold is not supported by this provider"
        )

    async def do_download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> AsyncStreamingDownloadResponse:
        """Provider-specific lazy download.

        Implementations must not read the body before returning, and must
        supply a close function that releases the connection, since the
        caller decides how much of the object to consume.

        Not abstract so existing providers remain instantiable; providers
        that support streaming override this.
        """
        raise UnsupportedOperationException(
            "download_stream is not supported by this provider"
        )

    def do_list_blob_versions(self, key: str) -> AsyncIterator[BlobInfo]:
        """Provider-specific version listing.

        Implementations return an async iterator and must exhaust pagination as
        it is consumed rather than up front, so a key with many versions does
        not have to be held in memory. A key with no versions yields nothing.

        Deliberately not an async generator function here: one would return an
        iterator and defer this refusal to the first iteration, when the caller
        has no reason to expect it.

        Not abstract so existing providers remain instantiable; providers that
        support version listing override this.
        """
        raise UnsupportedOperationException(
            "list_blob_versions is not supported by this provider"
        )

    @abstractmethod
    async def do_list_page(
        self,
        *,
        prefix: Optional[str],
        delimiter: Optional[str],
        max_results: int,
        pagination_token: Optional[str],
    ) -> ListBlobsPageResponse:
        """Provider-specific single-page listing implementation.

        Must fetch exactly one page (honoring ``max_results`` as the page
        cap), map the provider's listing response onto
        :class:`ListBlobsPageResponse`, and surface the provider's
        continuation token as ``next_pagination_token``. When no objects
        match, return an empty page (``blobs=[]``, ``is_truncated=False``,
        ``next_pagination_token=None``).
        """

    @abstractmethod
    async def do_does_object_exist(
        self, key: str, *, version_id: Optional[str]
    ) -> bool:
        """Provider-specific object existence check."""

    @abstractmethod
    async def do_does_bucket_exist(self) -> bool:
        """Provider-specific bucket existence check."""

    async def do_generate_presigned_url(
        self,
        key: str,
        operation: str,
        *,
        expiration_seconds: int,
        version_id: Optional[str],
        content_type: Optional[str],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        kms_key_id: Optional[str],
        content_disposition: Optional[str],
        correlation_id: str,
    ) -> PresignedUrlResult:
        """Provider-specific presigned URL generation.

        Not abstract so existing providers remain backward compatible;
        providers that support presigned URLs (AWS, GCP) override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support presigned URL generation"
        )

    # Multipart hooks, concrete so a provider without multipart still
    # instantiates.

    async def do_initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Mapping[str, str],
        content_type: Optional[str],
        tags: Mapping[str, str],
        correlation_id: str,
    ) -> MultipartUpload:
        """Provider-specific multipart initiation. Override to support."""
        raise UnsupportedOperationException(
            "initiate_multipart_upload is not supported by this provider"
        )

    async def do_upload_part(
        self, mpu: MultipartUpload, part: MultipartPart, *, correlation_id: str
    ) -> UploadPartResponse:
        """Provider-specific part upload. Override to support."""
        raise UnsupportedOperationException(
            "upload_part is not supported by this provider"
        )

    async def do_complete_multipart_upload(
        self,
        mpu: MultipartUpload,
        parts: List[UploadPartResponse],
        *,
        correlation_id: str,
    ) -> MultipartUploadResponse:
        """Provider-specific multipart completion. Override to support."""
        raise UnsupportedOperationException(
            "complete_multipart_upload is not supported by this provider"
        )

    async def do_list_parts(
        self, mpu: MultipartUpload, *, correlation_id: str
    ) -> List[UploadPartResponse]:
        """Provider-specific part listing. Override to support."""
        raise UnsupportedOperationException(
            "list_parts is not supported by this provider"
        )

    async def do_abort_multipart_upload(
        self, mpu: MultipartUpload, *, correlation_id: str
    ) -> None:
        """Provider-specific multipart abort. Override to support."""
        raise UnsupportedOperationException(
            "abort_multipart_upload is not supported by this provider"
        )

    @abstractmethod
    async def close(self) -> None:
        """Release underlying resources (sessions, connections)."""

    # Context manager

    async def __aenter__(self) -> "BaseAsyncBlobStore":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # Internal

    async def _prepare_payload(
        self,
        payload: AsyncUploadPayload,
        content_type: Optional[str],
    ) -> tuple[bytes, Optional[str]]:
        """Normalize payload to bytes and resolve content type."""
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload), content_type or "application/octet-stream"

        if isinstance(payload, Path):
            if content_type is None:
                guessed, _ = mimetypes.guess_type(str(payload))
                content_type = guessed or "application/octet-stream"
            data = await asyncio.to_thread(payload.read_bytes)
            return data, content_type

        read_fn = getattr(payload, "read", None)
        if callable(read_fn):
            data = await asyncio.to_thread(read_fn)
            if isinstance(data, str):
                raise InvalidArgumentException(
                    "Upload payload must be binary; got a text stream"
                )
            return bytes(data), content_type or "application/octet-stream"

        raise InvalidArgumentException(
            f"Unsupported upload payload type: {type(payload).__name__}. "
            f"Expected bytes, Path, or BinaryIO."
        )

    # Shared helpers for directory operations

    @staticmethod
    def collect_files(
        source_dir: Path, include_sub_folders: bool, follow_symlinks: bool
    ) -> List[Path]:
        """Collect files from a directory for upload (blocking, run in thread).

        Simple file traversal helper shared across providers to avoid
        duplicating os.walk logic.

        Args:
            source_dir: Base directory to scan.
            include_sub_folders: Whether to recurse into subdirectories.
            follow_symlinks: Whether to follow symbolic links.

        Returns:
            List of Path objects representing files to upload.
        """
        files: List[Path] = []
        if include_sub_folders:
            for dirpath, _, filenames in os.walk(
                source_dir, followlinks=follow_symlinks
            ):
                for fname in filenames:
                    fp = Path(dirpath) / fname
                    if not fp.is_symlink() or follow_symlinks:
                        files.append(fp)
        else:
            files = [
                p
                for p in source_dir.iterdir()
                if p.is_file() and (follow_symlinks or not p.is_symlink())
            ]
        return files

    @staticmethod
    async def run_upload_tasks(
        files: List[Path],
        source_dir: Path,
        prefix: str,
        max_concurrency: int,
        upload_fn: Callable[[Path, str, bytes], Awaitable[None]],
    ) -> "AsyncDirectoryUploadResponse":
        """Run concurrent file uploads with bounded concurrency.

        Shared orchestration helper used by providers. Each provider passes
        a cloud-specific ``upload_fn(file_path, blob_key, data)`` coroutine.

        Args:
            files: Files to upload (from collect_files).
            source_dir: Resolved base directory (for computing relative keys).
            prefix: Key prefix to prepend to relative paths.
            max_concurrency: Maximum concurrent upload coroutines.
            upload_fn: Provider-specific async callable that performs the
                actual cloud upload. Receives (file_path, blob_key, data).

        Returns:
            AsyncDirectoryUploadResponse with totals and per-file failures.
        """
        if not files:
            return AsyncDirectoryUploadResponse(
                total_files=0,
                successful_uploads=0,
                failed_transfers=[],
                total_bytes_transferred=0,
            )

        queue: asyncio.Queue[Path] = asyncio.Queue()
        for fp in files:
            queue.put_nowait(fp)

        failed: List[FailedAsyncBlobUpload] = []
        successful = 0
        total_bytes = 0
        lock = asyncio.Lock()

        async def _worker() -> None:
            nonlocal successful, total_bytes
            while True:
                try:
                    file_path = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                relative = file_path.relative_to(source_dir).as_posix()
                blob_key = f"{prefix}{relative}" if prefix else relative
                try:
                    data = await asyncio.to_thread(file_path.read_bytes)
                    await upload_fn(file_path, blob_key, data)
                    async with lock:
                        successful += 1
                        total_bytes += len(data)
                except Exception as exc:
                    async with lock:
                        failed.append(
                            FailedAsyncBlobUpload(source=file_path, exception=exc)
                        )

        workers = [
            asyncio.create_task(_worker())
            for _ in range(min(max_concurrency, len(files)))
        ]
        await asyncio.gather(*workers)

        return AsyncDirectoryUploadResponse(
            total_files=len(files),
            successful_uploads=successful,
            failed_transfers=failed,
            total_bytes_transferred=total_bytes,
        )

    @staticmethod
    async def run_download_tasks(
        objects: List[str],
        prefix: str,
        dest_dir: Path,
        max_concurrency: int,
        download_fn: Callable[[str, Path], Awaitable[bytes]],
    ) -> "AsyncDirectoryDownloadResponse":
        """Run concurrent file downloads with bounded concurrency.

        Shared orchestration helper used by providers. Each provider passes
        a cloud-specific ``download_fn(blob_key, local_path)`` coroutine
        that returns the downloaded bytes.

        Args:
            objects: List of blob keys to download.
            prefix: Key prefix to strip when computing local relative paths.
            dest_dir: Resolved local destination directory.
            max_concurrency: Maximum concurrent download coroutines.
            download_fn: Provider-specific async callable that downloads the
                blob and returns its content as bytes. Receives
                (blob_key, local_path).

        Returns:
            AsyncDirectoryDownloadResponse with totals and per-file failures.
        """
        if not objects:
            return AsyncDirectoryDownloadResponse(
                total_files=0,
                successful_downloads=0,
                failed_transfers=[],
                total_bytes_transferred=0,
            )

        queue: asyncio.Queue[str] = asyncio.Queue()
        for key in objects:
            queue.put_nowait(key)

        resolved_dest = dest_dir.resolve()
        failed: List[FailedAsyncBlobDownload] = []
        successful = 0
        total_bytes = 0
        lock = asyncio.Lock()

        async def _worker() -> None:
            nonlocal successful, total_bytes
            while True:
                try:
                    blob_key = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                relative = (
                    blob_key[len(prefix) :] if blob_key.startswith(prefix) else blob_key
                )
                local_path = (dest_dir / relative).resolve()
                if not local_path.is_relative_to(resolved_dest):
                    async with lock:
                        failed.append(
                            FailedAsyncBlobDownload(
                                destination=dest_dir / relative,
                                exception=ValueError(
                                    f"Path traversal detected in key: {blob_key}"
                                ),
                            )
                        )
                    continue
                try:
                    await asyncio.to_thread(
                        local_path.parent.mkdir, parents=True, exist_ok=True
                    )
                    body = await download_fn(blob_key, local_path)
                    await asyncio.to_thread(local_path.write_bytes, body)
                    async with lock:
                        successful += 1
                        total_bytes += len(body)
                except Exception as exc:
                    async with lock:
                        failed.append(
                            FailedAsyncBlobDownload(
                                destination=local_path, exception=exc
                            )
                        )

        workers = [
            asyncio.create_task(_worker())
            for _ in range(min(max_concurrency, len(objects)))
        ]
        await asyncio.gather(*workers)

        return AsyncDirectoryDownloadResponse(
            total_files=len(objects),
            successful_downloads=successful,
            failed_transfers=failed,
            total_bytes_transferred=total_bytes,
        )

    @staticmethod
    async def run_delete_tasks(
        units: AsyncIterator[_DeleteUnit],
        max_concurrency: int,
        delete_fn: Callable[[_DeleteUnit], Awaitable[List[str]]],
        *,
        operation: str = "delete_directory",
    ) -> None:
        """Run concurrent delete work with bounded concurrency.

        Shared orchestration helper used by providers. Each provider supplies
        an async iterator of work units and a coroutine that deletes one unit.
        A unit is whatever grouping the provider's API rewards: a batch of
        keys where it offers a batch delete, or a single key where it does not.

        A fixed pool of at most ``max_concurrency`` workers pulls units from the
        iterator as slots free up, so every slot stays busy for the whole run
        rather than idling at a per-group barrier. Units are pulled one at a
        time, and only failure descriptions up to the reporting cap are
        retained, so neither resident memory nor the number of live tasks grows
        with the size of the directory.

        The iterator is always drained to exhaustion before any failure is
        reported, so a listing that is still paginating can never have its
        outstanding deletes abandoned.

        Args:
            units: Async iterator yielding work units to delete.
            max_concurrency: Maximum concurrent delete coroutines.
            delete_fn: Provider-specific async callable that deletes one unit
                and returns a list of human-readable failure descriptions --
                empty when the whole unit succeeded. Returning failures rather
                than raising lets a provider whose API reports per-key errors
                in a successful response surface them the same way as one that
                raises.
            operation: Public method name, used to prefix the error message.

        Raises:
            BlobAsyncError: If any unit reported a failure. Raised once, after
                all units have been attempted, so a single undeletable object
                does not strand the rest of the directory.
        """
        failures: List[str] = []
        failure_count = 0
        first_cause: Optional[Exception] = None
        exhausted = False
        listing_error: Optional[BaseException] = None
        pull_lock = asyncio.Lock()

        def record(description: str, cause: Optional[BaseException] = None) -> None:
            """Tally a failure, retaining only as many details as we will quote."""
            nonlocal failure_count, first_cause
            failure_count += 1
            if len(failures) < MAX_REPORTED_DELETE_FAILURES:
                failures.append(description)
            if first_cause is None and isinstance(cause, Exception):
                first_cause = cause

        async def pull() -> tuple[Optional[_DeleteUnit], bool]:
            """Take the next unit, serialized because generators aren't reentrant."""
            nonlocal exhausted, listing_error
            async with pull_lock:
                if exhausted:
                    return None, False
                try:
                    return await units.__anext__(), True
                except StopAsyncIteration:
                    exhausted = True
                    return None, False
                except Exception as exc:
                    # A listing failure is a whole-operation failure, not a
                    # per-unit one, so stop pulling and re-raise it later.
                    # Assigned at most once: `exhausted` blocks re-entry.
                    exhausted = True
                    listing_error = exc
                    return None, False

        async def worker() -> None:
            while True:
                unit, ok = await pull()
                if not ok:
                    return
                try:
                    unit_failures = await delete_fn(unit)
                except Exception as exc:
                    record(str(exc), cause=exc)
                    continue
                for description in unit_failures:
                    record(description)

        workers = [
            asyncio.create_task(worker()) for _ in range(max(1, max_concurrency))
        ]
        await asyncio.gather(*workers)

        if listing_error is not None:
            raise listing_error

        if failure_count == 0:
            return

        remainder = failure_count - len(failures)
        detail = "; ".join(failures)
        if remainder > 0:
            detail = f"{detail}; and {remainder} more"
        raise BlobAsyncError(
            f"{operation}: failed to delete {failure_count} object(s): {detail}",
            cause=first_cause,
        )
