"""User-facing :class:`AsyncBucketClient` -- the public async entry point.

Mirrors the sync :class:`BucketClient` architecture. User code stays
cloud-agnostic; only the ``provider`` kwarg names a cloud.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import AsyncIterator, BinaryIO, List, Mapping, Optional, Union

from multicloud.common.credentials import CredentialsOverrider
from multicloud.common.exceptions import SubstrateSdkException
from multicloud.common.proxy import ProxyConfig

from .async_driver import (
    ASYNC_BLOB_REGISTRY,
    DEFAULT_MAX_CONCURRENCY,
    AsyncDirectoryDownloadResponse,
    AsyncDirectoryUploadResponse,
    AsyncDownloadRequest,
    AsyncDownloadResponse,
    AsyncStreamingDownloadResponse,
    AsyncTransferConfig,
    AsyncUploadPayload,
    AsyncUploadResponse,
    BaseAsyncBlobStore,
    BlobAsyncError,
    BlobIdentifier,
    BlobInfo,
    BlobMetadata,
    BulkDeleteResponse,
    CopyRequest,
    CopyResponse,
    ListBlobsPageResponse,
    MultipartPart,
    MultipartUpload,
    MultipartUploadResponse,
    ObjectLockInfo,
    ObjectRetentionConfig,
    PresignedUrlResult,
    UploadPartResponse,
)


class AsyncBucketClient:
    """Cloud-agnostic async blob storage client.

    Construction takes a ``provider`` id and unified options; the
    underlying provider class is selected from the async registry.

    Every method here is awaited except :meth:`list_blob_versions`, which
    hands back an async iterator directly so callers write
    ``async for info in client.list_blob_versions(key)``.

    Args:
        provider: Provider id -- ``"aws"``, ``"gcp"``, or ``"ali"``.
        bucket: Bucket name.
        region: Region (interpreted per provider). Optional.
        endpoint: Endpoint override URL. Optional.
        transfer_config: Transfer configuration. Uses defaults if omitted.
        credentials_overrider: Override default credential resolution.
            Optional.
    """

    def __init__(
        self,
        *,
        provider: str,
        bucket: str,
        region: Optional[str] = None,
        endpoint: Optional[str] = None,
        transfer_config: Optional[AsyncTransferConfig] = None,
        credentials_overrider: Optional[CredentialsOverrider] = None,
        proxy_config: Optional[ProxyConfig] = None,
    ) -> None:
        store_cls = ASYNC_BLOB_REGISTRY.get(provider)
        kwargs: dict = {"bucket": bucket}
        if region is not None:
            kwargs["region"] = region
        if endpoint is not None:
            kwargs["endpoint_url"] = endpoint
        if transfer_config is not None:
            kwargs["transfer_config"] = transfer_config
        if credentials_overrider is not None:
            kwargs["credentials_overrider"] = credentials_overrider
        if proxy_config is not None:
            kwargs["proxy_config"] = proxy_config
        self._store: BaseAsyncBlobStore = store_cls(**kwargs)

    @property
    def bucket(self) -> str:
        """The bucket this client operates on."""
        return self._store.bucket

    @property
    def provider(self) -> str:
        """The provider id (e.g. 'aws', 'gcp')."""
        return self._store.get_provider_id()

    # Download

    async def download(
        self,
        key: str,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        if_match: Optional[str] = None,
    ) -> AsyncDownloadResponse:
        """Download the object identified by ``key`` into memory.

        Returns an :class:`AsyncDownloadResponse` containing the blob
        content as ``bytes`` in the ``content`` field.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version to download. Optional.
            start: First byte of a range download (inclusive). Optional.
            end: Last byte of a range download (inclusive). Optional.
            if_match: Only download when the current ETag matches; else
                FailedPreconditionException. Optional.

        Returns:
            AsyncDownloadResponse with key, content bytes, and metadata.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        request = AsyncDownloadRequest(
            key=key,
            version_id=version_id,
            start=start,
            end=end,
            if_match=if_match,
        )
        try:
            return await self._store.download(request)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def download_to_path(
        self,
        key: str,
        destination: Union[str, Path, os.PathLike],
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        create_parent_dirs: bool = False,
    ) -> AsyncDownloadResponse:
        """Download the object to a local file path.

        Args:
            key: Object key (path) in the bucket.
            destination: Local file path to write the blob content.
            version_id: Specific object version to download. Optional.
            start: First byte of a range download (inclusive). Optional.
            end: Last byte of a range download (inclusive). Optional.
            create_parent_dirs: If True, create parent directories if
                they don't exist.

        Returns:
            AsyncDownloadResponse with download metadata.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        request = AsyncDownloadRequest(
            key=key,
            version_id=version_id,
            start=start,
            end=end,
        )
        try:
            response = await self._store.download(request)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

        dest_path = Path(destination)

        def _write_file() -> None:
            if create_parent_dirs:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(response.content)

        try:
            await asyncio.to_thread(_write_file)
        except OSError as exc:
            raise BlobAsyncError(
                f"Failed to write file {destination}: {exc}",
                cause=exc,
            ) from exc
        return response

    async def download_to_file(
        self,
        key: str,
        file_obj: BinaryIO,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> AsyncDownloadResponse:
        """Download the object to a file-like object.

        Args:
            key: Object key (path) in the bucket.
            file_obj: Binary file-like object to write the blob content.
            version_id: Specific object version to download. Optional.
            start: First byte of a range download (inclusive). Optional.
            end: Last byte of a range download (inclusive). Optional.

        Returns:
            AsyncDownloadResponse with download metadata.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        request = AsyncDownloadRequest(
            key=key,
            version_id=version_id,
            start=start,
            end=end,
        )
        try:
            response = await self._store.download(request)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

        def _write_to_file() -> None:
            file_obj.write(response.content)

        try:
            await asyncio.to_thread(_write_to_file)
        except OSError as exc:
            raise BlobAsyncError(
                f"Failed to write to file object: {exc}",
                cause=exc,
            ) from exc
        return response

    async def download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> AsyncStreamingDownloadResponse:
        """Open the object for lazy reading instead of loading it into memory.

        Use this instead of :meth:`download` when the object may be too large
        to hold, or when it can be processed incrementally.

        The returned stream holds a connection until it is closed, so close it
        even if only part of the object is read. ``async with`` does that::

            async with await client.download_stream("large.bin") as stream:
                async for chunk in stream:
                    process(chunk)

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version. Optional.
            start: First byte to read, inclusive. Optional. Omitting it while
                giving ``end`` reads the last ``end`` bytes.
            end: Last byte to read, inclusive. Optional.

        Returns:
            An AsyncStreamingDownloadResponse supporting async iteration, an
            awaitable ``read``, and async context management.

        Raises:
            InvalidArgumentException: If key is empty or the range is invalid.
            SubstrateSdkException: On provider error.
        """
        try:
            return await self._store.download_stream(
                key, version_id=version_id, start=start, end=end
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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
            content_type: MIME type (auto-detected from Path extension
                if not specified). Falls back to "application/octet-stream".
            content_length: Explicit size in bytes. Calculated if omitted.
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
            InvalidArgumentException: If key is empty or payload unsupported.
        """
        try:
            return await self._store.upload(
                key,
                payload,
                content_type=content_type,
                content_length=content_length,
                metadata=metadata,
                create_if_absent=create_if_absent,
                if_match=if_match,
                if_version_match=if_version_match,
                kms_key_id=kms_key_id,
                use_kms_managed_key=use_kms_managed_key,
                storage_class=storage_class,
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def upload_from_path(
        self,
        key: str,
        path: Union[str, Path, os.PathLike],
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

        Convenience over :meth:`upload` for a file on disk: the path may be a
        string as well as a Path, a missing or non-regular file is rejected up
        front rather than surfacing as a transport error, and the content type
        is detected from the extension when not given. Reading is done off the
        event loop.

        Note:
            The file is read into memory before being sent, so use the
            multipart API for uploads too large to buffer. ``tags`` and
            ``object_lock`` are not accepted here, matching :meth:`upload`.

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
        try:
            return await self._store.upload_from_path(
                key,
                path,
                content_type=content_type,
                content_length=content_length,
                metadata=metadata,
                create_if_absent=create_if_absent,
                if_match=if_match,
                if_version_match=if_version_match,
                kms_key_id=kms_key_id,
                use_kms_managed_key=use_kms_managed_key,
                storage_class=storage_class,
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    # Delete

    async def delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Delete a single object. Idempotent — does not raise if the object does not exist.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version to delete. Optional.
                If None, targets the latest version.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        try:
            await self._store.delete(key, version_id=version_id)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def delete_objects(
        self,
        identifiers: list[BlobIdentifier],
    ) -> BulkDeleteResponse:
        """Delete multiple objects. Does NOT raise on partial failure.

        Non-existent objects are reported as successfully deleted
        (idempotent semantics). Inspect ``BulkDeleteResponse.errors``
        for any objects that genuinely failed.

        Args:
            identifiers: List of BlobIdentifiers to delete.

        Returns:
            BulkDeleteResponse with per-object outcomes.

        Raises:
            InvalidArgumentException: If identifiers is empty or invalid.
        """
        try:
            return await self._store.delete_objects(identifiers)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    # Tags

    async def get_tags(self, key: str) -> Mapping[str, str]:
        """Retrieve the tags attached to the object identified by ``key``.

        Args:
            key: Object key (path) in the bucket.

        Returns:
            A mapping of tag keys to values. Empty if the object has no tags.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        try:
            return await self._store.get_tags(key)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Replace all tags on the object identified by ``key``.

        Args:
            key: Object key (path) in the bucket.
            tags: Mapping of tag keys to values. Replaces any existing tags.

        Raises:
            InvalidArgumentException: If key is empty or tags is empty.
        """
        try:
            await self._store.set_tags(key, tags)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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
            SubstrateSdkException: On other provider errors.
        """
        try:
            return await self._store.get_metadata(key, version_id=version_id)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    # Object lock

    async def get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Retrieve the retention and legal-hold state of an object.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version. Optional; None targets the
                latest version.

        Returns:
            ObjectLockInfo describing the lock, or None when the object has
            neither retention nor a legal hold.

            ``event_based_hold`` is a tri-state: None means the provider does
            not report one, so test it with ``is True`` / ``is False`` rather
            than plain truthiness.

        Raises:
            InvalidArgumentException: If key is empty.
            SubstrateSdkException: On provider error, including a bucket that
                does not have object lock enabled.
        """
        try:
            return await self._store.get_object_lock(key, version_id=version_id)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str] = None,
    ) -> None:
        """Update the retention configuration of a single object.

        The object must ALREADY have retention configured. This changes an
        existing retention; it does not establish one, and a call against an
        object without retention is refused.

        Shortening or relaxing an existing retention is restricted: an
        immutable-mode retention can only be extended, and a mutable one can
        be shortened only with ``bypass_governance``.

        Args:
            key: Object key (path) in the bucket.
            retention: The retention change to apply. ``retain_until_date``
                must be timezone-aware.
            version_id: Specific object version. Optional; None targets the
                latest version.

        Raises:
            InvalidArgumentException: If key is empty, the config is missing,
                or its retain-until instant is missing or timezone-naive.
            FailedPreconditionException: If the object has no retention yet, or
                the change is not permitted from its current state.
            SubstrateSdkException: On provider error.
        """
        try:
            await self._store.update_object_retention(
                key, retention, version_id=version_id
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def update_legal_hold(
        self, key: str, enabled: bool, *, version_id: Optional[str] = None
    ) -> None:
        """Set or remove a legal hold on an object.

        A legal hold prevents the object from being deleted or overwritten
        until the hold is explicitly removed. On providers with a dedicated
        legal-hold flag this is independent of any retention period; on
        providers where the legal-hold and retention mechanisms share the same
        underlying flag, the two concerns may be entangled.

        Provider notes:
            - Placing a hold may require object lock / versioning to be
              enabled on the bucket for some providers; the call errors
              otherwise.
            - Some providers guard the write with an optimistic-concurrency
              check, so a concurrent modification can surface as a translated
              error on those providers only.

        Args:
            key: Object key (path) in the bucket.
            enabled: True to place a legal hold, False to remove it.
            version_id: Specific object version. Optional; None targets the
                latest version.

        Raises:
            InvalidArgumentException: If key is empty.
            SubstrateSdkException: On provider error.
        """
        try:
            await self._store.update_legal_hold(key, enabled, version_id=version_id)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    # Copy

    async def copy(
        self,
        src_key: str,
        *,
        dest_bucket: str,
        dest_key: str,
        src_version_id: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> CopyResponse:
        """Copy a blob from this bucket to another bucket.

        Args:
            src_key: Source object key in this bucket.
            dest_bucket: Destination bucket name.
            dest_key: Key for the copied object.
            src_version_id: Source version to copy; latest if None.
            metadata: Replacement metadata; source metadata kept if None.

        Returns:
            CopyResponse for the new copy.

        Raises:
            SubstrateSdkException: A subclass instance on failure.
        """
        request = CopyRequest(
            source_bucket=self._store.bucket,
            source_key=src_key,
            destination_bucket=dest_bucket,
            destination_key=dest_key,
            source_version_id=src_version_id,
            metadata=metadata,
        )
        try:
            return await self._store.copy(request)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def copy_from(
        self,
        src_bucket: str,
        src_key: str,
        *,
        dest_key: str,
        src_version_id: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> CopyResponse:
        """Copy a blob from another bucket into this bucket.

        Args:
            src_bucket: Source bucket name.
            src_key: Source object key in the source bucket.
            dest_key: Key for the copied object in this bucket.
            src_version_id: Source version to copy; latest if None.
            metadata: Replacement metadata; source metadata kept if None.

        Returns:
            CopyResponse for the new copy.

        Raises:
            SubstrateSdkException: A subclass instance on failure.
        """
        request = CopyRequest(
            source_bucket=src_bucket,
            source_key=src_key,
            destination_bucket=self._store.bucket,
            destination_key=dest_key,
            source_version_id=src_version_id,
            metadata=metadata,
        )
        try:
            return await self._store.copy(request)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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

        Returns a time-limited URL that grants temporary access to an object
        without requiring credentials. Useful for direct browser uploads/downloads.

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
            PresignedUrlResult with:
                - url: The presigned URL string
                - expiration: Absolute expiration datetime (UTC)
                - correlation_id: Unique request identifier
                - http_method: The HTTP method (GET/PUT)
                - required_headers: Headers that must be included in the request

        Raises:
            InvalidArgumentException: If key is empty, operation is not
                GET/PUT, or expiration is out of range.
            SubstrateSdkException: On provider error (bucket not found, etc.).
        """
        try:
            return await self._store.generate_presigned_url(
                key,
                operation,
                expiration_seconds=expiration_seconds,
                version_id=version_id,
                content_type=content_type,
                metadata=metadata,
                tags=tags,
                kms_key_id=kms_key_id,
                content_disposition=content_disposition,
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    # Directory operations

    async def upload_directory(
        self,
        local_source_directory: str,
        *,
        prefix: str = "",
        include_sub_folders: bool = True,
        follow_symbolic_links: bool = False,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        tags: Optional[Mapping[str, str]] = None,
    ) -> AsyncDirectoryUploadResponse:
        """Upload a local directory to cloud storage asynchronously.

        Uploads all files in the specified local directory to the bucket,
        preserving the directory structure as key prefixes. Operations are
        parallelized using asyncio with bounded concurrency.

        Args:
            local_source_directory: Absolute or relative path to the local
                directory to upload.
            prefix: Key prefix to prepend to all uploaded objects.
            include_sub_folders: Whether to recurse into subdirectories.
            follow_symbolic_links: Whether to follow symlinks during traversal.
            max_concurrency: Maximum number of concurrent upload coroutines.
                Defaults to 10.
            tags: Tags to apply to all uploaded objects. Encoding is
                provider-specific. Optional.

        Returns:
            AsyncDirectoryUploadResponse with total files, successful
            uploads, per-file failures, and bytes transferred.

        Raises:
            InvalidArgumentException: If the source directory doesn't exist
                or is not a directory.
        """
        try:
            return await self._store.upload_directory(
                local_source_directory,
                prefix=prefix,
                include_sub_folders=include_sub_folders,
                follow_symbolic_links=follow_symbolic_links,
                max_concurrency=max_concurrency,
                tags=tags,
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def download_directory(
        self,
        prefix: str,
        local_destination_directory: str,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        prefixes_to_exclude: Optional[List[str]] = None,
    ) -> AsyncDirectoryDownloadResponse:
        """Download all objects under a prefix to a local directory asynchronously.

        Downloads all objects matching the specified prefix into the local
        directory, preserving the key structure as subdirectories. Operations
        are parallelized using asyncio with bounded concurrency.

        Args:
            prefix: Key prefix identifying the "directory" to download.
            local_destination_directory: Local path to write files into.
            max_concurrency: Maximum concurrent download coroutines.
                Defaults to 10.
            prefixes_to_exclude: Prefixes to skip during download. Optional.

        Returns:
            AsyncDirectoryDownloadResponse with total files, successful
            downloads, per-file failures, and bytes transferred.

        Raises:
            InvalidArgumentException: If the prefix or destination is empty.
        """
        try:
            return await self._store.download_directory(
                prefix,
                local_destination_directory,
                max_concurrency=max_concurrency,
                prefixes_to_exclude=prefixes_to_exclude,
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def delete_directory(
        self,
        prefix: Optional[str],
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        """Delete all objects under a prefix asynchronously.

        Deletes every object whose key begins with the prefix. Deletes are
        parallelized using asyncio with bounded concurrency. A prefix matching
        no objects is a successful no-op, and objects already absent count as
        deleted.

        The prefix is matched verbatim and is never normalized, so a prefix
        without a trailing ``"/"`` also matches sibling keys sharing its
        leading characters -- ``"logs"`` matches ``"logs.txt"`` and
        ``"logs-archive/x"`` as well as ``"logs/x"``. Pass a trailing ``"/"``
        to target only the directory.

        Warning:
            An empty or ``None`` prefix matches every object in the bucket and
            deletes all of them.

        Only the current version of each object is deleted, so on a
        version-enabled bucket earlier versions are retained and storage is
        not necessarily reclaimed. The operation is not atomic: objects written
        under the prefix while it runs may survive.

        Args:
            prefix: Key prefix identifying the "directory" to delete. Empty or
                None deletes every object in the bucket.
            max_concurrency: Maximum concurrent delete coroutines.
                Defaults to 10.

        Raises:
            InvalidArgumentException: If max_concurrency is not positive.
            SubstrateSdkException: If any matched object could not be deleted.
        """
        try:
            await self._store.delete_directory(
                prefix,
                max_concurrency=max_concurrency,
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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

        Fetches exactly one page of listings and returns it alongside a
        continuation token for manual pagination. To iterate, feed the
        returned ``next_pagination_token`` back in as ``pagination_token``
        until ``is_truncated`` is ``False``.

        Args:
            prefix: Restrict results to keys beginning with this prefix.
                Optional.
            delimiter: Group keys sharing a common prefix up to the next
                delimiter into ``common_prefixes`` (e.g. ``"/"`` for a
                directory-style listing). Optional.
            max_results: Maximum number of blobs to return in this page.
                Defaults to 1000. Must be a positive integer.
            pagination_token: Continuation token from a previous page's
                ``next_pagination_token``. Omit for the first page.

        Returns:
            ListBlobsPageResponse with ``blobs``, ``common_prefixes``,
            ``is_truncated``, and ``next_pagination_token``. Empty results
            yield empty lists, ``is_truncated=False``, and a ``None`` token.

        Raises:
            InvalidArgumentException: If max_results is not a positive integer.
            SubstrateSdkException: On provider error (bucket not found, etc.).
        """
        try:
            return await self._store.list_page(
                prefix=prefix,
                delimiter=delimiter,
                max_results=max_results,
                pagination_token=pagination_token,
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def list_blob_versions(self, key: str) -> AsyncIterator[BlobInfo]:
        """Iterate every version of one key.

        Not a coroutine, so no await is needed to obtain the iterator::

            async for info in client.list_blob_versions("report.pdf"):
                print(info.version_id, info.size)

        Pages are fetched as they are consumed, so a key with many versions is
        never held in memory all at once.

        Args:
            key: Object key (path) in the bucket.

        Returns:
            An async iterator of BlobInfo, one per version. A key that does not
            exist yields nothing rather than raising.

            The order is the provider's own and is not normalised, so do not
            rely on it. What is guaranteed: ``is_latest`` identifies the
            current version on every provider, and ``last_modified`` is
            populated on every entry, so chronology is recoverable by sorting.

        Raises:
            InvalidArgumentException: If key is empty.
            SubstrateSdkException: On provider error. Because the listing is
                lazy, this surfaces from the loop rather than from this call --
                including a failure on the very first page.
        """
        try:
            return self._store.list_blob_versions(key)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    # Existence checks

    async def does_object_exist(
        self, key: str, *, version_id: Optional[str] = None
    ) -> bool:
        """Check if an object exists in the bucket.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version to check. Optional.
                Provider support varies.

        Returns:
            True if the object exists, False otherwise.

        Raises:
            InvalidArgumentException: If key is empty or version_id unsupported.
        """
        try:
            return await self._store.does_object_exist(key, version_id=version_id)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def does_bucket_exist(self) -> bool:
        """Check if the configured bucket exists.

        Returns:
            True if the bucket exists, False otherwise.
        """
        try:
            return await self._store.does_bucket_exist()
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    # Multipart upload

    async def initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Optional[Mapping[str, str]] = None,
        content_type: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
    ) -> MultipartUpload:
        """Start a multipart upload.

        Args:
            key: Object key the assembled parts will be written to.
            metadata: User metadata to set on the finished object.
            content_type: MIME type of the finished object.
            tags: Tags to set on the finished object.

        Returns:
            A handle identifying the session, to pass to the calls below.

        Raises:
            InvalidArgumentException: If the key is empty.
            SubstrateSdkException: If the provider rejects the request.

        The handle is only meaningful to the client that created it. Whether a
        stale or hand-built one is detected is provider-dependent: some report
        it on the next call, others accept it and orphan the part, so do not
        rely on either.
        """
        try:
            return await self._store.initiate_multipart_upload(
                key, metadata=metadata, content_type=content_type, tags=tags
            )
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def upload_part(
        self, mpu: MultipartUpload, part: MultipartPart
    ) -> UploadPartResponse:
        """Upload one part.

        Args:
            mpu: Handle from :meth:`initiate_multipart_upload`.
            part: Part number and content. Part numbers start at 1; the
                maximum is provider-specific.

        Returns:
            The part's ETag, which completion needs.

        Raises:
            InvalidArgumentException: If the part number is below 1, above the
                provider's ceiling, or the content length does not match the
                bytes read.
            SubstrateSdkException: If the upload is unknown to the provider, or
                the request fails.
        """
        try:
            return await self._store.upload_part(mpu, part)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def complete_multipart_upload(
        self, mpu: MultipartUpload, parts: List[UploadPartResponse]
    ) -> MultipartUploadResponse:
        """Assemble the uploaded parts into the final object.

        Args:
            mpu: Handle from :meth:`initiate_multipart_upload`.
            parts: The results of the part uploads. Order does not matter;
                parts are assembled by part number, and a repeated part number
                is rejected.

        Raises:
            InvalidArgumentException: If ``parts`` is empty or repeats a part
                number.
            SubstrateSdkException: If the upload is unknown to the provider, a
                part is missing, or a part is below the provider's minimum size.

        How much of each entry is used varies by provider: all of them need the
        part number, and some also verify the ETag. Pass the responses you got
        from :meth:`upload_part` unmodified rather than reconstructing them.
        """
        try:
            return await self._store.complete_multipart_upload(mpu, parts)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def list_parts(self, mpu: MultipartUpload) -> List[UploadPartResponse]:
        """List the parts uploaded so far for an in-progress upload.

        Args:
            mpu: Handle from :meth:`initiate_multipart_upload`.

        Returns:
            The parts the provider currently holds, ascending by part number.
            Every page is fetched, so a long upload is not truncated.

        Raises:
            SubstrateSdkException: If the request fails.

        An upload the provider has no record of is not reported uniformly: some
        raise, others return an empty list, so do not rely on either. Which
        fields are populated also varies; the part number is always set.
        """
        try:
            return await self._store.list_parts(mpu)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    async def abort_multipart_upload(self, mpu: MultipartUpload) -> None:
        """Discard an upload and its parts.

        Idempotent, so it is safe in a ``finally``. Without it, the parts of an
        abandoned upload are retained and billed.

        Raises:
            SubstrateSdkException: If the parts could not be discarded, for
                instance when the credentials do not permit it. An upload that
                is already gone is not an error.
        """
        try:
            await self._store.abort_multipart_upload(mpu)
        except SubstrateSdkException:
            raise
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    # Lifecycle

    async def close(self) -> None:
        """Release underlying resources (HTTP sessions, connections)."""
        await self._store.close()

    async def __aenter__(self) -> "AsyncBucketClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
