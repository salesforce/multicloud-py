"""User-facing :class:`BucketClient` — the public entry point for blob ops.

Three-layer architecture:

1. **Client** (this file) — what users import; provider-agnostic API.
2. **Driver** (:class:`BaseBlobStore` in ``driver.py``) — internal abstract
   contract; defines validated public methods + provider hooks.
3. **Provider** (e.g. ``AwsBlobStore``, ``GcpBlobStore``) — concrete impls
   that callers never reference directly.

User code stays cloud-agnostic; only the ``provider`` kwarg names a cloud::

    from multicloud.blob.blob_client import BucketClient

    with BucketClient(
        provider="aws", bucket="my-bucket", region="us-west-2"
    ) as client:
        client.upload("hello.txt", b"hi")
        resp = client.download("hello.txt")
        client.delete("hello.txt")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, Iterator, List, Mapping, Optional, Union

from multicloud.blob.blob_client.driver import (
    BLOB_REGISTRY,
    BaseBlobStore,
    BlobIdentifier,
    BlobInfo,
    BlobMetadata,
    ChecksumAlgorithm,
    CopyResponse,
    DeleteObjectsResponse,
    DownloadResponse,
    ListBlobsPageResponse,
    MultipartPart,
    MultipartUpload,
    MultipartUploadResponse,
    ObjectLockConfiguration,
    ObjectLockInfo,
    ObjectRetentionConfig,
    PresignedOperation,
    PresignedUrlResponse,
    StreamingDownloadResponse,
    UploadPartResponse,
    UploadResponse,
)
from multicloud.common.credentials import CredentialsOverrider
from multicloud.common.proxy import ProxyConfig


class BucketClient:
    """Cloud-agnostic blob storage client.

    The user-facing API surface. Construction takes a ``provider`` id and
    a unified set of cloud-agnostic options; the underlying provider class
    is selected from the registry. Operations (``upload``, ``download``,
    ``delete``) take cloud-agnostic value objects and never reference a
    provider.

    Args:
        provider: Provider id — ``"aws"``, ``"gcp"``, or ``"ali"``.
        bucket: Bucket name.
        region: Region (interpreted per provider). Optional; falls back to
            provider defaults (env vars, IAM role, ADC).
        endpoint: Endpoint override URL. Optional.
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
        credentials_overrider: Optional[CredentialsOverrider] = None,
        proxy_config: Optional[ProxyConfig] = None,
    ) -> None:
        store_cls = BLOB_REGISTRY.get(provider)
        kwargs: dict = {"bucket": bucket}
        if region is not None:
            kwargs["region"] = region
        if endpoint is not None:
            kwargs["endpoint_url"] = endpoint
        if credentials_overrider is not None:
            kwargs["credentials_overrider"] = credentials_overrider
        if proxy_config is not None:
            kwargs["proxy_config"] = proxy_config
        self._store: BaseBlobStore = store_cls(**kwargs)

    @property
    def bucket(self) -> str:
        return self._store.bucket

    @property
    def provider(self) -> str:
        return self._store.get_provider_id()

    def upload(
        self,
        key: str,
        payload: Union[bytes, BinaryIO],
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
        """Upload ``payload`` to the object identified by ``key``.

        Args:
            key: Object key (path) in the bucket.
            payload: Content to upload - raw bytes or a binary file-like object.
            content_type: MIME type (e.g., "text/plain", "application/json").
            content_length: Size in bytes (optional, calculated automatically if omitted).
            metadata: User-defined metadata dict (keys and values are strings).
            tags: Object tags dict (keys and values are strings). Encoding is
                provider-specific — providers that lack native tagging encode
                tags in object metadata. Provider-specific limits on tag
                count, key length, and value length apply.
            object_lock: Object lock configuration (retention mode + expiry
                and/or legal hold) to apply to the object at upload time. On
                some providers a legal hold is applied by a follow-up write
                after the object is created; if that write fails the error is
                raised and the object is left WITHOUT the hold.
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
            kms_key_id: Provider-managed key id for server-side encryption.
            use_kms_managed_key: Request a provider-managed encryption key when
                no kms_key_id is given. A no-op on providers that always
                encrypt at rest.
            checksum_value: Base64-encoded digest of ``payload`` (for example
                ``base64.b64encode(hashlib.md5(payload).digest())``) that the
                provider validates against the bytes it receives. Requires
                checksum_algorithm. A mismatch raises InvalidArgumentException
                and the object is not written.
            checksum_algorithm: Algorithm of checksum_value, as a
                :class:`ChecksumAlgorithm` or its case-insensitive name
                (``"CRC32C"`` or ``"MD5"``). Requires checksum_value.

        Returns:
            UploadResponse containing the key, ETag, and version ID. When a
            checksum was supplied, it also carries the algorithm and the
            provider-computed checksum value.
        """
        try:
            return self._store.upload(
                key,
                payload,
                content_type=content_type,
                content_length=content_length,
                metadata=metadata,
                tags=tags,
                object_lock=object_lock,
                create_if_absent=create_if_absent,
                if_match=if_match,
                if_version_match=if_version_match,
                kms_key_id=kms_key_id,
                use_kms_managed_key=use_kms_managed_key,
                checksum_value=checksum_value,
                checksum_algorithm=checksum_algorithm,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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
        """Upload a local file to the object identified by ``key``.

        Convenience method that reads a file from the local filesystem and
        uploads its content. Auto-detects ``content_type`` from the file
        extension (via :mod:`mimetypes`) and ``content_length`` from the
        file size when not explicitly provided.

        Note:
            ``tags`` is not supported on this method yet.

        Args:
            key: Object key (path) in the bucket.
            path: Local file path to upload. Accepts a string, :class:`~pathlib.Path`,
                or any :class:`os.PathLike`.
            content_type: MIME type (e.g., "text/plain"). Auto-detected from file
                extension if omitted.
            content_length: Size in bytes. Calculated from file size if omitted.
            metadata: User-defined metadata dict (keys and values are strings).
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
                retry). Not all providers support conditional overwrite on
                upload; unsupported providers raise UnsupportedOperationException.
            if_version_match: Only overwrite when the current version id
                matches; else FailedPreconditionException; a racing write raises
                ResourceConflictException (the write may not have happened;
                retry). Not all providers support version-based conditional
                overwrite; unsupported providers raise UnsupportedOperationException.
            kms_key_id: Provider-managed key id for server-side encryption.
            use_kms_managed_key: Request a provider-managed encryption key when
                no kms_key_id is given. A no-op on providers that always
                encrypt at rest.

        Returns:
            UploadResponse containing the key, ETag, and version ID.

        Raises:
            InvalidArgumentException: If key is empty, file does not exist,
                or path is not a regular file.
        """
        try:
            return self._store.upload_from_path(
                key,
                path,
                content_type=content_type,
                content_length=content_length,
                metadata=metadata,
                object_lock=object_lock,
                create_if_absent=create_if_absent,
                if_match=if_match,
                if_version_match=if_version_match,
                kms_key_id=kms_key_id,
                use_kms_managed_key=use_kms_managed_key,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def download(
        self,
        key: str,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        if_match: Optional[str] = None,
    ) -> DownloadResponse:
        """Download the object identified by ``key``.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific version to download (if versioning is enabled).
            start: Start byte position for range download (inclusive).
            end: End byte position for range download (inclusive).
            if_match: Only download when the current ETag matches; else
                FailedPreconditionException.

        Returns:
            DownloadResponse containing the key, content bytes, and metadata.
        """
        try:
            return self._store.download(
                key,
                version_id=version_id,
                start=start,
                end=end,
                if_match=if_match,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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
        """Download the object directly to a file path.

        Writes content directly to the local filesystem without loading
        the entire blob into memory. Suitable for large objects.

        Conditional reads (if_match) are supported on download() only;
        use it when you need an ETag precondition.

        Args:
            key: Object key (path) in the bucket.
            destination: Local file path to write the content to.
            version_id: Specific version to download (if versioning is enabled).
            start: Start byte position for range download (inclusive).
            end: End byte position for range download (inclusive).
            create_parent_dirs: If True, create parent directories of
                *destination* if they do not exist.

        Returns:
            DownloadResponse with metadata (content field is empty bytes).
        """
        try:
            return self._store.download_to_path(
                key,
                destination,
                version_id=version_id,
                start=start,
                end=end,
                create_parent_dirs=create_parent_dirs,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def download_to_file(
        self,
        key: str,
        file_obj: BinaryIO,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> DownloadResponse:
        """Download the object directly to a file-like object.

        Writes content directly to the provided writable binary file
        object without loading the entire blob into memory.

        Conditional reads (if_match) are supported on download() only;
        use it when you need an ETag precondition.

        Args:
            key: Object key (path) in the bucket.
            file_obj: A writable binary file-like object (must support
                ``write(bytes)``).
            version_id: Specific version to download (if versioning is enabled).
            start: Start byte position for range download (inclusive).
            end: End byte position for range download (inclusive).

        Returns:
            DownloadResponse with metadata (content field is empty bytes).
        """
        try:
            return self._store.download_to_file(
                key,
                file_obj,
                version_id=version_id,
                start=start,
                end=end,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> StreamingDownloadResponse:
        """Download the object as a streaming response for lazy consumption.

        Returns a streaming response that yields content on demand
        without loading the entire blob into memory. The caller must
        close the response (preferably via a context manager) to release
        the underlying HTTP connection.

        Conditional reads (if_match) are supported on download() only;
        use it when you need an ETag precondition.

        Example::

            with client.download_stream("large-file.bin") as stream:
                for chunk in stream:
                    process(chunk)

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific version to download (if versioning is enabled).
            start: Start byte position for range download (inclusive).
            end: End byte position for range download (inclusive).

        Returns:
            StreamingDownloadResponse with an iterator and read() interface.
        """
        try:
            return self._store.download_stream(
                key,
                version_id=version_id,
                start=start,
                end=end,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Delete the object identified by ``key``.

        Idempotent — returns silently if the object does not exist.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific version to delete. If ``None``, deletes the
                latest version (or creates a delete marker on versioned buckets).
        """
        try:
            self._store.delete(key, version_id=version_id)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def delete_objects(self, objects: List[BlobIdentifier]) -> DeleteObjectsResponse:
        """Delete multiple objects in a single batch operation.

        The caller is responsible for chunking to stay within provider
        limits (e.g. 1000 for S3).

        Args:
            objects: Collection of blob identifiers to delete. Each
                identifier specifies a key and an optional version ID.

        Returns:
            DeleteObjectsResponse with any per-object errors. An empty
            ``errors`` list means all objects were deleted successfully.

        Raises:
            InvalidArgumentException: If objects is empty or contains
                entries with empty keys.
        """
        try:
            return self._store.delete_objects(objects)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def copy(
        self,
        src_key: str,
        *,
        dest_bucket: str,
        dest_key: str,
        src_version_id: Optional[str] = None,
    ) -> CopyResponse:
        """Copy an object from this bucket to another bucket.

        Performs a server-side copy — no data is transferred through the
        client.

        Args:
            src_key: Key of the source object in this bucket.
            dest_bucket: Name of the destination bucket.
            dest_key: Key for the copied object in the destination bucket.
            src_version_id: Specific version of the source to copy. If
                ``None``, copies the latest version.

        Returns:
            CopyResponse with the key, version ID, ETag, and last-modified
            timestamp of the new copy.

        Raises:
            InvalidArgumentException: If any required parameter is empty.
        """
        try:
            return self._store.copy(
                src_key,
                dest_bucket=dest_bucket,
                dest_key=dest_key,
                src_version_id=src_version_id,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def copy_from(
        self,
        src_bucket: str,
        src_key: str,
        *,
        dest_key: str,
        src_version_id: Optional[str] = None,
    ) -> CopyResponse:
        """Copy an object from a source bucket into this bucket.

        Performs a server-side copy — no data is transferred through the
        client.

        Args:
            src_bucket: Name of the source bucket.
            src_key: Key of the source object in the source bucket.
            dest_key: Key for the copied object in this bucket.
            src_version_id: Specific version of the source to copy. If
                ``None``, copies the latest version.

        Returns:
            CopyResponse with the key, version ID, ETag, and last-modified
            timestamp of the new copy.

        Raises:
            InvalidArgumentException: If any required parameter is empty.
        """
        try:
            return self._store.copy_from(
                src_bucket,
                src_key,
                dest_key=dest_key,
                src_version_id=src_version_id,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def list_page(
        self,
        *,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_results: int = 1000,
        pagination_token: Optional[str] = None,
    ) -> ListBlobsPageResponse:
        """Retrieve a single page of blobs matching the given criteria.

        Args:
            prefix: Filter results to keys starting with this prefix.
            delimiter: Character to group keys into common prefixes
                (usually "/").
            max_results: Maximum number of blobs to return (1-1000,
                default 1000).
            pagination_token: Continuation token from a previous
                response for pagination.

        Returns:
            ListBlobsPageResponse with blobs, common_prefixes,
            is_truncated, and next_pagination_token.
        """
        try:
            return self._store.list_page(
                prefix=prefix,
                delimiter=delimiter,
                max_results=max_results,
                pagination_token=pagination_token,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def does_object_exist(self, key: str, *, version_id: Optional[str] = None) -> bool:
        """Check whether a blob exists in the bucket.

        Args:
            key: Object key to check.
            version_id: Optional version ID for versioned buckets.

        Returns:
            True if the object exists, False if not found.
            Other errors (auth, server) propagate as translated
            exceptions.
        """
        try:
            return self._store.does_object_exist(key, version_id=version_id)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def get_metadata(
        self, key: str, *, version_id: Optional[str] = None
    ) -> BlobMetadata:
        """Retrieve object metadata (HEAD-style) without downloading the body.

        Args:
            key: Object key (path) in the bucket.
            version_id: Optional version ID for versioned buckets.

        Returns:
            BlobMetadata with key, content_length, content_type, e_tag,
            version_id, custom metadata, last_modified, and created_time.

        Raises:
            InvalidArgumentException: If key is empty.
            ResourceNotFoundException: If the object does not exist.
        """
        try:
            return self._store.get_metadata(key, version_id=version_id)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def does_bucket_exist(self) -> bool:
        """Check whether the configured bucket exists.

        Returns:
            True if the bucket exists, False if not found.
            Other errors (auth, server) propagate as translated
            exceptions.
        """
        try:
            return self._store.does_bucket_exist()
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def get_tags(self, key: str) -> Mapping[str, str]:
        """Retrieve all tags associated with the object.

        Args:
            key: Object key (path) in the bucket.

        Returns:
            A mapping of tag keys to tag values. Returns an empty dict
            if the object exists but has no tags.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        try:
            return self._store.get_tags(key)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Set tags on the object, replacing all existing tags.

        Args:
            key: Object key (path) in the bucket.
            tags: Mapping of tag keys to tag values. Replaces all
                existing tags on the object.

        Raises:
            InvalidArgumentException: If key is empty or tags is empty.
        """
        try:
            self._store.set_tags(key, tags)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str] = None,
    ) -> None:
        """Update the retention configuration of a single object.

        The object must already have retention configured; this updates it and
        does not establish retention for the first time.

        Args:
            key: Object key (path) in the bucket.
            retention: Retention settings to apply (retain-until date, an
                optional mode where None preserves the current mode, plus an
                optional flag to bypass an existing governance-mode retention).
            version_id: Optional version identifier to target a specific
                version of the object.

        Raises:
            InvalidArgumentException: If key is empty or the retention
                config is missing a retain-until date.
        """
        try:
            self._store.update_object_retention(key, retention, version_id=version_id)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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

        The returned URL can be used by any HTTP client (no SDK or credentials
        required) to perform the specified operation within the expiration window.

        Args:
            key: Object key (path) in the bucket.
            operation: Whether to generate an UPLOAD (PUT) or DOWNLOAD (GET) URL.
            duration_seconds: URL validity duration in seconds (default 3600 = 1 hour).
            content_type: MIME type (upload only). Callers must send this header.
            metadata: User-defined metadata (upload only). Callers must send the
                corresponding provider-specific headers.
            tags: Object tags (upload only). Encoding is provider-specific.
            kms_key_id: KMS key for server-side encryption (upload only, provider-
                dependent — silently ignored on providers that do not support it).
            content_disposition: Content-Disposition header value. For uploads, the
                header is baked into the URL; for downloads, it overrides the
                response header.

        Returns:
            PresignedUrlResponse containing the URL, HTTP method, and any
            headers the caller must include when using the URL.
        """
        try:
            return self._store.generate_presigned_url(
                key,
                operation,
                duration_seconds=duration_seconds,
                content_type=content_type,
                metadata=metadata,
                tags=tags,
                kms_key_id=kms_key_id,
                content_disposition=content_disposition,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Retrieve the object lock state (retention + legal hold) for an object.

        Args:
            key: Object key (path) in the bucket.
            version_id: Specific object version. Optional; None targets the
                latest version.

        Returns:
            ObjectLockInfo with retention mode/date, legal hold, and
            event-based hold, or None when the object has no retention and
            no legal hold.

        Raises:
            InvalidArgumentException: If key is empty.
        """
        try:
            return self._store.get_object_lock(key, version_id=version_id)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def update_legal_hold(
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
        """
        try:
            self._store.update_legal_hold(key, enabled, version_id=version_id)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def list_blob_versions(self, key: str) -> Iterator[BlobInfo]:
        """Iterate all versions of a blob in the bucket.

        Args:
            key: Object key whose versions to list.

        Returns:
            Iterator yielding BlobInfo for each version.

        Raises:
            UnsupportedOperationException: If the provider does not
                support version listing.
        """
        try:
            return self._store.list_blob_versions(key)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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
        try:
            return self._store.initiate_multipart_upload(
                key,
                metadata=metadata,
                content_type=content_type,
                tags=tags,
                object_lock=object_lock,
            )
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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
        try:
            return self._store.upload_part(mpu, part)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

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
        try:
            return self._store.complete_multipart_upload(mpu, parts)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def list_parts(self, mpu: MultipartUpload) -> List[UploadPartResponse]:
        """List all parts uploaded so far for a multipart upload.

        Args:
            mpu: The multipart upload handle from initiate.

        Returns:
            List of UploadPartResponse for each uploaded part.
        """
        try:
            return self._store.list_parts(mpu)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def abort_multipart_upload(self, mpu: MultipartUpload) -> None:
        """Abort a multipart upload, discarding all uploaded parts.

        Args:
            mpu: The multipart upload handle from initiate.
        """
        try:
            self._store.abort_multipart_upload(mpu)
        except Exception as exc:
            raise self._store.translate_error(exc) from exc

    def close(self) -> None:
        """Release underlying resources."""
        self._store.close()

    def __enter__(self) -> "BucketClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
