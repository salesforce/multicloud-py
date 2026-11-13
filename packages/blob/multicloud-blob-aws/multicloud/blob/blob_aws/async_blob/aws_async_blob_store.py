"""AWS S3 async implementation of :class:`BaseAsyncBlobStore`.

Translates S3/botocore errors into the cloud-agnostic exception
hierarchy so callers can write a single ``except`` block that works
across providers.
"""

from __future__ import annotations

import asyncio
import mimetypes
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, AsyncIterator, List, Mapping, Optional
from urllib.parse import quote

from multicloud.aws_common.error_mapping import AWS_THROTTLING_CODES
from multicloud.blob.blob_aws._aws_common import (
    apply_kms,
    apply_write_preconditions,
    ensure_etag_quoted,
    is_conditional_conflict,
    is_missing_object,
    is_precondition_failed,
    map_replication_status,
)
from multicloud.blob.blob_client.async_blob.async_driver import (
    ASYNC_BLOB_REGISTRY,
    DEFAULT_MAX_CONCURRENCY,
    AsyncDirectoryDownloadResponse,
    AsyncDirectoryUploadResponse,
    AsyncDownloadRequest,
    AsyncDownloadResponse,
    AsyncStreamingDownloadResponse,
    AsyncTransferConfig,
    AsyncUploadResponse,
    BaseAsyncBlobStore,
    BlobAsyncError,
    BlobIdentifier,
    BlobInfo,
    BlobMetadata,
    BlobNotFoundError,
    BulkDeleteResponse,
    CopyRequest,
    CopyResponse,
    DeletedBlob,
    DeleteError,
    FailedAsyncBlobDownload,
    FailedAsyncBlobUpload,
    ListBlobsPageResponse,
    MultipartPart,
    MultipartUpload,
    MultipartUploadResponse,
    ObjectLockInfo,
    ObjectRetentionConfig,
    PresignedUrlResult,
    RetentionMode,
    UploadPartResponse,
    build_range_header,
)
from multicloud.blob.blob_client.retention_rules import resolve_and_validate
from multicloud.common.credentials import CredentialsOverrider
from multicloud.common.exceptions import (
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceConflictException,
    ResourceExhaustedException,
    ResourceExistsException,
    ResourceNotFoundException,
    SubstrateSdkException,
    UnAuthorizedException,
    UnknownException,
)
from multicloud.common.proxy import ProxyConfig

try:
    # Presence check: importing aiobotocore fails module load (and is recorded
    # by the entry-point loader) when the ``async`` extra is not installed, so
    # the missing dependency surfaces at construction rather than first call.
    import aiobotocore.session  # type: ignore[import-not-found]  # noqa: F401
    from botocore.exceptions import (  # type: ignore[import-not-found]
        BotoCoreError,
        ClientError,
        NoCredentialsError,
    )
    from multicloud.aws_common.aws_client_config import s3_client_config

    # Async credential resolver shared across AWS async providers. Aliased as
    # ``get_aio_session`` so it can be patched on this module in unit tests.
    from multicloud.aws_common.aws_credentials import get_aio_session
except ImportError as _exc:
    raise ImportError(
        "aiobotocore is required for async AWS blob operations. "
        "Install it with: pip install 'multicloud-py-blob-aws[async]'"
    ) from _exc

_STREAM_CHUNK_SIZE = 8192
"""Chunk size for streamed reads, matching the synchronous provider."""

_MAX_KEYS_PER_DELETE = 1000
"""S3 DeleteObjects accepts at most 1000 keys per request.

Exceeding it fails the whole request with a MalformedXML error rather than
anything that names the real cause, so batches are chunked client-side.
"""


@ASYNC_BLOB_REGISTRY.register("aws")
class AwsAsyncBlobStore(BaseAsyncBlobStore):
    """Async S3-backed blob store using aiobotocore.

    Args:
        bucket: S3 bucket name.
        region: AWS region (e.g. "us-west-2"). Falls back to env/config.
        endpoint_url: S3 endpoint override (for LocalStack, MinIO, etc.).
        transfer_config: Async transfer configuration.
        credentials_overrider: Override default credential resolution.
        _session: Pre-created aiobotocore session (for testing).
    """

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        *,
        endpoint_url: Optional[str] = None,
        transfer_config: Optional["AsyncTransferConfig"] = None,
        credentials_overrider: Optional[CredentialsOverrider] = None,
        proxy_config: Optional[ProxyConfig] = None,
        _session: Any = None,
    ) -> None:
        super().__init__(
            bucket=bucket,
            region=region,
            endpoint_url=endpoint_url,
            transfer_config=transfer_config,
            credentials_overrider=credentials_overrider,
            proxy_config=proxy_config,
        )
        # Session resolution is async (source-credential lookup for the role
        # paths), so it is deferred to the first API call in _ensure_client.
        # An injected session takes precedence over credentials_overrider.
        self._session: Any = _session
        self._client: Any = None
        self._client_cm: Any = None
        self._client_lock = asyncio.Lock()

    async def _ensure_client(self) -> None:
        """Lazily create the S3 async client on first use."""
        if self._client is not None:
            return
        async with self._client_lock:
            if self._client is None:
                if self._session is None:
                    self._session = await get_aio_session(
                        self._credentials_overrider, self._region
                    )
                # Pin SigV4: presigned URLs for KMS-encrypted buckets require
                # it, and botocore still downgrades to SigV2 in some legacy
                # regions unless set at client construction.
                kwargs: dict[str, Any] = {
                    "config": s3_client_config(self._proxy_config)
                }
                if self._region:
                    kwargs["region_name"] = self._region
                if self._endpoint_url:
                    kwargs["endpoint_url"] = self._endpoint_url
                self._client_cm = self._session.create_client("s3", **kwargs)
                self._client = await self._client_cm.__aenter__()

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        """Translate AWS SDK exceptions to common hierarchy."""
        if isinstance(exc, SubstrateSdkException):
            return exc

        if isinstance(exc, NoCredentialsError):
            return UnAuthorizedException(str(exc))

        if isinstance(exc, ClientError):
            error_info = exc.response.get("Error", {})
            code = error_info.get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            # Checked before the status branches below: AWS reports throttling
            # with HTTP 400 on some services, which would otherwise be read as
            # an invalid argument.
            if code in AWS_THROTTLING_CODES or status == 429:
                return ResourceExhaustedException(str(exc))

            if (
                code in ("NoSuchBucket", "NoSuchKey", "NoSuchUpload", "404")
                or status == 404
            ):
                return ResourceNotFoundException(str(exc))
            if code in ("AccessDenied", "Forbidden", "403") or status == 403:
                return UnAuthorizedException(str(exc))
            if code in ("InvalidArgument", "InvalidRequest", "400") or status == 400:
                return InvalidArgumentException(str(exc))
            # An IfMatch/IfNoneMatch condition that was evaluated and failed
            # returns 412 PreconditionFailed -> FailedPreconditionException.
            # (The create-if-absent IfNoneMatch="*" conflict also lands here but
            # is re-labelled to ResourceExistsException in do_upload before
            # reaching this generic translation.)
            if is_precondition_failed(exc):
                return FailedPreconditionException(str(exc))
            # A concurrent write racing a conditional write returns
            # ConditionalRequestConflict (HTTP 409 on S3). Match the error
            # code, not HTTP 409 in general — other AWS APIs reuse 409.
            # Surface it as ResourceConflictException rather than a terminal
            # failure.
            if is_conditional_conflict(exc):
                return ResourceConflictException(str(exc))
            return UnknownException(str(exc))

        if isinstance(exc, BotoCoreError):
            return InvalidArgumentException(str(exc))

        return BlobAsyncError(str(exc), cause=exc)

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
    ) -> "AsyncUploadResponse":
        """Upload bytes to S3 using put_object.

        Exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()

        put_kwargs: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
            "Body": payload,
        }
        if content_type is not None:
            put_kwargs["ContentType"] = content_type
        if content_length is not None:
            put_kwargs["ContentLength"] = content_length
        if metadata:
            put_kwargs["Metadata"] = dict(metadata)
        if storage_class is not None:
            put_kwargs["StorageClass"] = storage_class
        apply_write_preconditions(
            put_kwargs,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
        )
        apply_kms(
            put_kwargs,
            kms_key_id=kms_key_id,
            use_kms_managed_key=use_kms_managed_key,
        )

        resp = await self._put_object_conditional(
            put_kwargs, create_if_absent, if_match
        )

        return AsyncUploadResponse(
            key=key,
            e_tag=_strip_quotes(resp.get("ETag")),
            version_id=resp.get("VersionId"),
        )

    async def _put_object_conditional(
        self,
        put_kwargs: dict[str, Any],
        create_if_absent: bool,
        if_match: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call ``put_object``, disambiguating conditional-write outcomes.

        S3 returns HTTP 412 PreconditionFailed for both ``IfNoneMatch="*"``
        (object exists) and ``IfMatch`` (ETag mismatch); only create-if-absent is
        re-labelled to ResourceExistsException here. An ``IfMatch`` overwrite of a
        missing object returns 404, which is re-labelled to
        FailedPreconditionException so an absent target fails the precondition
        rather than surfacing as not-found. Everything else (including the
        retryable ConditionalRequestConflict) propagates for the client layer to
        translate.
        """
        if not create_if_absent and if_match is None:
            return await self._client.put_object(**put_kwargs)  # type: ignore[no-any-return]
        try:
            return await self._client.put_object(**put_kwargs)  # type: ignore[no-any-return]
        except ClientError as exc:
            if create_if_absent and is_precondition_failed(exc):
                raise ResourceExistsException(str(exc)) from exc
            if if_match is not None and is_missing_object(exc):
                raise FailedPreconditionException(str(exc)) from exc
            raise

    async def do_download(self, request: AsyncDownloadRequest) -> AsyncDownloadResponse:
        """Download bytes from S3 using get_object.

        Raises:
            BlobNotFoundError: If the object does not exist (404).

        Other exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()

        get_kwargs: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": request.key,
        }
        if request.version_id is not None:
            get_kwargs["VersionId"] = request.version_id

        range_header = build_range_header(request.start, request.end)
        if range_header:
            get_kwargs["Range"] = range_header

        # Conditional read: IfMatch returns 412 on ETag mismatch. Re-quote via
        # the shared helper so the natural round-trip (download().metadata.e_tag
        # -> if_match) works; download strips the quotes S3's header requires.
        if request.if_match is not None:
            get_kwargs["IfMatch"] = ensure_etag_quoted(request.if_match)

        try:
            resp = await self._client.get_object(**get_kwargs)
            body = await resp["Body"].read()
        except ClientError as exc:
            # Control-flow: distinguish 404 from other errors
            error_code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error_code in ("NoSuchKey", "NoSuchBucket", "404") or status == 404:
                raise BlobNotFoundError(
                    f"Blob not found: {request.key}",
                    key=request.key,
                    cause=exc,
                ) from exc
            raise

        metadata = BlobMetadata(
            key=request.key,
            # Use len(body) for uniform semantics across providers —
            # S3 ContentLength is the range size for 206 responses,
            # so len(body) == ContentLength anyway, and aligns with GCP.
            content_length=len(body),
            content_type=resp.get("ContentType"),
            # S3 ETags are hex MD5 wrapped in quotes (e.g. "d41d8...").
            # Strip quotes so callers get a bare hash string.
            e_tag=_strip_quotes(resp.get("ETag")),
            version_id=resp.get("VersionId"),
            metadata=resp.get("Metadata", {}),
            replication_status=map_replication_status(resp.get("ReplicationStatus")),
        )

        return AsyncDownloadResponse(
            key=request.key,
            content=body,
            metadata=metadata,
        )

    async def do_download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> AsyncStreamingDownloadResponse:
        """Stream an object out of S3 without reading it into memory.

        get_object returns the body as a stream that is read on demand, so
        only the response headers have arrived when this returns. The client
        is held for the store's lifetime, so the body stays usable after this
        call, but it pins a pooled connection until it is closed.

        Raises:
            BlobNotFoundError: If the object does not exist (404).

        Other exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()

        get_kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if version_id is not None:
            get_kwargs["VersionId"] = version_id
        range_header = build_range_header(start, end)
        if range_header:
            get_kwargs["Range"] = range_header

        try:
            resp = await self._client.get_object(**get_kwargs)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error_code in ("NoSuchKey", "NoSuchBucket", "404") or status == 404:
                raise BlobNotFoundError(
                    f"Blob not found: {key}", key=key, cause=exc
                ) from exc
            raise

        body = resp["Body"]
        metadata = BlobMetadata(
            key=key,
            # ContentLength is the size of what this response carries, which
            # for a ranged read is the range rather than the whole object.
            content_length=resp.get("ContentLength"),
            content_type=resp.get("ContentType"),
            e_tag=_strip_quotes(resp.get("ETag")),
            version_id=resp.get("VersionId"),
            metadata=resp.get("Metadata", {}),
            replication_status=map_replication_status(resp.get("ReplicationStatus")),
        )

        async def chunks() -> AsyncIterator[bytes]:
            # Translated here because the client's catch wraps only the call
            # that opens the stream. A body that stops short of its declared
            # length, or times out mid-read, raises native errors that would
            # otherwise reach the caller untranslated.
            try:
                async for chunk in body.iter_chunks(_STREAM_CHUNK_SIZE):
                    yield chunk
            except SubstrateSdkException:
                raise
            except Exception as exc:
                raise self.translate_error(exc) from exc

        async def close() -> None:
            # release() rather than close(): it returns the connection to the
            # pool when the body is exhausted, and degrades to a close by
            # itself when it is not, so an abandoned stream is still safe.
            body.release()

        return AsyncStreamingDownloadResponse(
            key=key,
            metadata=metadata,
            stream=chunks(),
            close_func=close,
        )

    async def do_copy(
        self, request: CopyRequest, *, correlation_id: str
    ) -> CopyResponse:
        """Copy an object within S3 using copy_object.

        Supports same-bucket and cross-bucket copy. When metadata is
        provided in the request, uses MetadataDirective=REPLACE to
        override the source object's metadata.

        Raises:
            ResourceNotFoundException: If source object does not exist (404).

        Other exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()

        copy_source: dict[str, Any] = {
            "Bucket": request.source_bucket,
            "Key": request.source_key,
        }
        if request.source_version_id:
            copy_source["VersionId"] = request.source_version_id

        copy_kwargs: dict[str, Any] = {
            "Bucket": request.destination_bucket,
            "Key": request.destination_key,
            "CopySource": copy_source,
        }

        if request.metadata is not None:
            copy_kwargs["Metadata"] = dict(request.metadata)
            copy_kwargs["MetadataDirective"] = "REPLACE"

        try:
            resp = await self._client.copy_object(**copy_kwargs)
        except ClientError as exc:
            # Control-flow: distinguish 404 from other errors
            error_code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error_code in ("NoSuchKey", "NoSuchBucket", "404") or status == 404:
                raise ResourceNotFoundException(
                    f"Source object not found: "
                    f"{request.source_bucket}/{request.source_key}"
                ) from exc
            raise

        copy_result = resp.get("CopyObjectResult", {})
        last_modified = copy_result.get("LastModified")
        last_modified_str: Optional[str] = None
        if last_modified is not None:
            last_modified_str = (
                last_modified.isoformat()
                if hasattr(last_modified, "isoformat")
                else str(last_modified)
            )
        return CopyResponse(
            key=request.destination_key,
            version_id=resp.get("VersionId"),
            e_tag=_strip_quotes(copy_result.get("ETag")),
            last_modified=last_modified_str,
            correlation_id=correlation_id,
        )

    async def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Delete a single S3 object. Idempotent — S3 returns 204 even if key doesn't exist.

        Exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()

        delete_kwargs: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
        }
        if version_id is not None:
            delete_kwargs["VersionId"] = version_id

        await self._client.delete_object(**delete_kwargs)

    async def do_delete_objects(
        self,
        identifiers: list[BlobIdentifier],
    ) -> "BulkDeleteResponse":
        """Batch delete using S3 delete_objects API (max 1000 per call).

        S3 returns structured partial-failure information: a Deleted list
        and an Errors list. Non-existent objects appear in the Deleted
        list (idempotent).
        """
        await self._ensure_client()

        all_deleted: list[DeletedBlob] = []
        all_errors: list[DeleteError] = []

        # S3 delete_objects supports max 1000 objects per call
        BATCH_SIZE = 1000
        items = list(identifiers)

        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i : i + BATCH_SIZE]
            objects: list[dict[str, str]] = []
            for item in batch:
                obj: dict[str, str] = {"Key": item.key}
                if item.version_id is not None:
                    obj["VersionId"] = item.version_id
                objects.append(obj)

            try:
                resp = await self._client.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": objects},
                )

                for d in resp.get("Deleted", []):
                    all_deleted.append(
                        DeletedBlob(
                            key=d["Key"],
                            version_id=d.get("VersionId"),
                        )
                    )

                for e in resp.get("Errors", []):
                    all_errors.append(
                        DeleteError(
                            key=e["Key"],
                            version_id=e.get("VersionId"),
                            code=e.get("Code", ""),
                            message=e.get("Message", ""),
                        )
                    )

            except (ClientError, BotoCoreError, NoCredentialsError) as exc:
                # Entire batch failed — report each item as errored
                for item in batch:
                    all_errors.append(
                        DeleteError(
                            key=item.key,
                            version_id=item.version_id,
                            code="BatchFailed",
                            message=str(exc),
                        )
                    )
            except Exception as exc:
                # Unexpected error — record batch items as failed and continue
                # so that already-accumulated results from prior chunks are not lost.
                for item in batch:
                    all_errors.append(
                        DeleteError(
                            key=item.key,
                            version_id=item.version_id,
                            code="BatchFailed",
                            message=str(exc),
                        )
                    )

        return BulkDeleteResponse(deleted=all_deleted, errors=all_errors)

    async def do_get_tags(self, key: str) -> Mapping[str, str]:
        """Retrieve object tags using S3 GetObjectTagging API."""
        await self._ensure_client()
        resp = await self._client.get_object_tagging(Bucket=self._bucket, Key=key)
        return {tag["Key"]: tag["Value"] for tag in resp.get("TagSet", [])}

    async def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Replace all object tags using S3 PutObjectTagging API."""
        await self._ensure_client()
        tag_set = [{"Key": k, "Value": v} for k, v in tags.items()]
        await self._client.put_object_tagging(
            Bucket=self._bucket,
            Key=key,
            Tagging={"TagSet": tag_set},
        )

    async def do_upload_directory(
        self,
        local_source_directory: str,
        *,
        prefix: str = "",
        include_sub_folders: bool = True,
        follow_symbolic_links: bool = False,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        tags: Optional[Mapping[str, str]] = None,
    ) -> AsyncDirectoryUploadResponse:
        """Upload a local directory to S3 with bounded concurrency."""
        await self._ensure_client()

        source_dir = Path(local_source_directory).resolve()
        files_to_upload = await asyncio.to_thread(
            self.collect_files, source_dir, include_sub_folders, follow_symbolic_links
        )

        async def _put_object(file_path: Path, blob_key: str, data: bytes) -> None:
            put_kwargs: dict[str, Any] = {
                "Bucket": self._bucket,
                "Key": blob_key,
                "Body": data,
            }
            content_type, _ = mimetypes.guess_type(str(file_path))
            put_kwargs["ContentType"] = content_type or "application/octet-stream"
            if tags:
                put_kwargs["Tagging"] = "&".join(
                    f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in tags.items()
                )
            await self._client.put_object(**put_kwargs)

        return await self.run_upload_tasks(
            files_to_upload, source_dir, prefix, max_concurrency, _put_object
        )

    async def do_download_directory(
        self,
        prefix: str,
        local_destination_directory: str,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        prefixes_to_exclude: Optional[List[str]] = None,
    ) -> AsyncDirectoryDownloadResponse:
        """Download all objects under a prefix from S3 to a local directory."""
        await self._ensure_client()

        dest_dir = Path(local_destination_directory)
        await asyncio.to_thread(dest_dir.mkdir, parents=True, exist_ok=True)

        # Paginated listing
        objects_to_download: List[str] = []
        exclusion_prefixes = tuple(prefixes_to_exclude) if prefixes_to_exclude else ()
        continuation_token: Optional[str] = None

        while True:
            list_kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            resp = await self._client.list_objects_v2(**list_kwargs)

            for obj in resp.get("Contents", []):
                key = obj["Key"]
                size = obj.get("Size", 0)
                if key.endswith("/") and size == 0:
                    continue
                if exclusion_prefixes and key.startswith(exclusion_prefixes):
                    continue
                objects_to_download.append(key)

            if resp.get("IsTruncated"):
                continuation_token = resp.get("NextContinuationToken")
            else:
                break

        async def _get_object(s3_key: str, _local_path: Path) -> bytes:
            get_resp = await self._client.get_object(Bucket=self._bucket, Key=s3_key)
            return await get_resp["Body"].read()

        return await self.run_download_tasks(
            objects_to_download, prefix, dest_dir, max_concurrency, _get_object
        )

    async def do_delete_directory(
        self,
        prefix: Optional[str],
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        """Delete every S3 object under a prefix using batched delete_objects.

        Lists with no Delimiter so the whole subtree is matched, and streams
        each page straight into a DeleteObjects batch rather than accumulating
        every key first, so memory stays flat regardless of directory size.

        DeleteObjects reports per-key failures inside an HTTP 200 response
        instead of raising, so the Errors array is collected and surfaced --
        ignoring it would silently leave objects behind. Quiet mode is enabled
        because only those errors are needed, not the ~1000 success entries.
        """
        await self._ensure_client()

        async def batches() -> AsyncIterator[List[dict[str, str]]]:
            continuation_token: Optional[str] = None
            while True:
                list_kwargs: dict[str, Any] = {"Bucket": self._bucket}
                if prefix:
                    list_kwargs["Prefix"] = prefix
                if continuation_token:
                    list_kwargs["ContinuationToken"] = continuation_token

                resp = await self._client.list_objects_v2(**list_kwargs)

                keys = [{"Key": obj["Key"]} for obj in resp.get("Contents", [])]
                for start in range(0, len(keys), _MAX_KEYS_PER_DELETE):
                    yield keys[start : start + _MAX_KEYS_PER_DELETE]

                if not resp.get("IsTruncated"):
                    return
                continuation_token = resp.get("NextContinuationToken")

        async def delete_batch(batch: List[dict[str, str]]) -> List[str]:
            resp = await self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": batch, "Quiet": True},
            )
            return [
                f"{err.get('Key')}: {err.get('Code', '')} "
                f"{err.get('Message', '')}".strip()
                for err in resp.get("Errors", [])
            ]

        await self.run_delete_tasks(batches(), max_concurrency, delete_batch)

    async def do_does_object_exist(
        self, key: str, *, version_id: Optional[str]
    ) -> bool:
        """Check if an object exists using head_object.

        Only handles 404/NoSuchKey → False. All other exceptions propagate
        to the client layer for translation.

        Returns:
            True if the object exists, False if 404 or NoSuchKey.
        """
        await self._ensure_client()
        head_kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if version_id is not None:
            head_kwargs["VersionId"] = version_id
        try:
            await self._client.head_object(**head_kwargs)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey") or status == 404:
                return False
            raise

    async def do_initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Mapping[str, str],
        content_type: Optional[str],
        tags: Mapping[str, str],
        correlation_id: str,
    ) -> MultipartUpload:
        """Start a multipart upload with the S3 CreateMultipartUpload API."""
        await self._ensure_client()

        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if metadata:
            kwargs["Metadata"] = dict(metadata)
        if content_type:
            kwargs["ContentType"] = content_type
        if tags:
            kwargs["Tagging"] = "&".join(
                f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in tags.items()
            )

        resp = await self._client.create_multipart_upload(**kwargs)
        return MultipartUpload(
            bucket=self._bucket,
            key=key,
            upload_id=resp["UploadId"],
            metadata=dict(metadata),
            content_type=content_type,
            tags=dict(tags),
            correlation_id=correlation_id,
        )

    async def do_upload_part(
        self, mpu: MultipartUpload, part: MultipartPart, *, correlation_id: str
    ) -> UploadPartResponse:
        """Upload one part with the S3 UploadPart API."""
        await self._ensure_client()

        content = part.content
        if isinstance(content, bytes):
            payload = content
        else:
            payload = await asyncio.to_thread(content.read)

        if part.content_length is not None and len(payload) != part.content_length:
            raise InvalidArgumentException(
                f"part {part.part_number}: read {len(payload)} bytes but "
                f"content_length is {part.content_length}; a stream cannot be "
                f"re-read after a previous attempt consumed it"
            )

        resp = await self._client.upload_part(
            Bucket=mpu.bucket,
            Key=mpu.key,
            UploadId=mpu.upload_id,
            PartNumber=part.part_number,
            Body=payload,
            ContentLength=len(payload),
        )
        return UploadPartResponse(
            part_number=part.part_number,
            e_tag=_strip_quotes(resp.get("ETag")) or "",
            size_in_bytes=len(payload),
            correlation_id=correlation_id,
        )

    async def do_complete_multipart_upload(
        self,
        mpu: MultipartUpload,
        parts: List[UploadPartResponse],
        *,
        correlation_id: str,
    ) -> MultipartUploadResponse:
        """Assemble the parts with the S3 CompleteMultipartUpload API."""
        await self._ensure_client()

        ordered = sorted(parts, key=lambda p: p.part_number)
        resp = await self._client.complete_multipart_upload(
            Bucket=mpu.bucket,
            Key=mpu.key,
            UploadId=mpu.upload_id,
            MultipartUpload={
                "Parts": [
                    {"PartNumber": p.part_number, "ETag": p.e_tag} for p in ordered
                ]
            },
        )
        return MultipartUploadResponse(
            e_tag=_strip_quotes(resp.get("ETag")),
            key=resp.get("Key", mpu.key),
            version_id=resp.get("VersionId"),
            correlation_id=correlation_id,
        )

    async def do_list_parts(
        self, mpu: MultipartUpload, *, correlation_id: str
    ) -> List[UploadPartResponse]:
        """List uploaded parts with the S3 ListParts API.

        Follows the truncation marker to exhaustion; a page caps at 1000 parts
        while an upload may hold up to 10000.
        """
        await self._ensure_client()

        collected: List[UploadPartResponse] = []
        kwargs: dict[str, Any] = {
            "Bucket": mpu.bucket,
            "Key": mpu.key,
            "UploadId": mpu.upload_id,
        }
        while True:
            resp = await self._client.list_parts(**kwargs)
            collected.extend(
                UploadPartResponse(
                    part_number=p["PartNumber"],
                    e_tag=_strip_quotes(p.get("ETag", "")) or "",
                    size_in_bytes=p.get("Size", 0),
                    correlation_id=correlation_id,
                )
                for p in resp.get("Parts", [])
            )
            if not resp.get("IsTruncated", False):
                break
            kwargs["PartNumberMarker"] = resp["NextPartNumberMarker"]
        return sorted(collected, key=lambda p: p.part_number)

    async def do_abort_multipart_upload(
        self, mpu: MultipartUpload, *, correlation_id: str
    ) -> None:
        """Discard the upload with the S3 AbortMultipartUpload API."""
        await self._ensure_client()

        try:
            await self._client.abort_multipart_upload(
                Bucket=mpu.bucket, Key=mpu.key, UploadId=mpu.upload_id
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "NoSuchUpload":
                raise

    async def do_does_bucket_exist(self) -> bool:
        """Check if the bucket exists using head_bucket.

        Only handles 404/NoSuchBucket → False. All other exceptions propagate
        to the client layer for translation.

        Returns:
            True if the bucket exists, False if 404 or NoSuchBucket.
        """
        await self._ensure_client()
        try:
            await self._client.head_bucket(Bucket=self._bucket)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchBucket") or status == 404:
                return False
            raise

    async def do_list_page(
        self,
        *,
        prefix: Optional[str],
        delimiter: Optional[str],
        max_results: int,
        pagination_token: Optional[str],
    ) -> ListBlobsPageResponse:
        """List a single page of objects using S3 list_objects_v2.

        Mirrors the synchronous ``AwsBlobStore.do_list_page`` field mapping:
        ``Contents[].Key`` -> key, ``Size`` -> size, ``ETag`` (quotes
        stripped) -> e_tag, ``LastModified`` -> last_modified,
        ``StorageClass`` -> storage_class. ``version_id`` stays None
        (``list_objects_v2`` reports no version info), consistent with the
        GCP store. ``CommonPrefixes[].Prefix`` -> common_prefixes;
        ``IsTruncated`` -> is_truncated; ``NextContinuationToken`` ->
        next_pagination_token. Exceptions propagate to the client layer for
        translation.
        """
        await self._ensure_client()

        kwargs: dict[str, Any] = {"Bucket": self._bucket, "MaxKeys": max_results}
        if prefix is not None:
            kwargs["Prefix"] = prefix
        if delimiter is not None:
            kwargs["Delimiter"] = delimiter
        if pagination_token is not None:
            kwargs["ContinuationToken"] = pagination_token

        resp = await self._client.list_objects_v2(**kwargs)

        blobs = [
            BlobInfo(
                key=obj["Key"],
                size=obj.get("Size"),
                # S3 ETags are hex MD5 wrapped in quotes; strip them so
                # callers get a bare hash string (matches do_get_metadata).
                e_tag=_strip_quotes(obj.get("ETag")),
                last_modified=obj.get("LastModified"),
                storage_class=obj.get("StorageClass"),
            )
            for obj in resp.get("Contents", [])
        ]
        common_prefixes = [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]
        return ListBlobsPageResponse(
            blobs=blobs,
            common_prefixes=common_prefixes,
            is_truncated=resp.get("IsTruncated", False),
            next_pagination_token=resp.get("NextContinuationToken"),
        )

    async def do_list_blob_versions(self, key: str) -> AsyncIterator[BlobInfo]:
        """Yield every version of one key from ListObjectVersions.

        The API is prefix-based, so entries whose key merely starts with the
        requested one are filtered out. Pages are fetched as they are consumed.

        Delete markers are not versions of the object's content and are left
        out, so a key whose current state is a delete marker still yields the
        versions underneath it.
        """
        await self._ensure_client()
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": key}

        while True:
            # Translated here rather than left to the client. Calling an async
            # generator function runs none of this body, so the client's catch
            # has already returned by the time the first page is fetched and
            # cannot see any failure, on this page or a later one.
            try:
                resp = await self._client.list_object_versions(**kwargs)
            except (ClientError, BotoCoreError, NoCredentialsError) as exc:
                raise self.translate_error(exc) from exc

            for version in resp.get("Versions", []):
                if version.get("Key") != key:
                    continue
                yield BlobInfo(
                    key=version["Key"],
                    size=version.get("Size"),
                    e_tag=_strip_quotes(version.get("ETag")),
                    last_modified=version.get("LastModified"),
                    storage_class=version.get("StorageClass"),
                    version_id=version.get("VersionId"),
                    is_latest=version.get("IsLatest"),
                )

            if not resp.get("IsTruncated", False):
                break
            kwargs["KeyMarker"] = resp["NextKeyMarker"]
            kwargs["VersionIdMarker"] = resp["NextVersionIdMarker"]

    async def do_get_metadata(
        self, key: str, *, version_id: Optional[str] = None
    ) -> BlobMetadata:
        """Retrieve object metadata using head_object (no body download).

        Raises:
            BlobNotFoundError: If the object does not exist (404/NoSuchKey).

        All other exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()
        head_kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if version_id is not None:
            head_kwargs["VersionId"] = version_id
        try:
            resp = await self._client.head_object(**head_kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("NoSuchKey", "NoSuchBucket", "404") or status == 404:
                raise BlobNotFoundError(
                    f"Blob not found: {key}",
                    key=key,
                    cause=exc,
                ) from exc
            raise

        # S3 HEAD reports the true object size in ContentLength (no body is
        # transferred). ETags are hex MD5 wrapped in quotes; strip them.
        last_modified = resp.get("LastModified")
        return BlobMetadata(
            key=key,
            content_length=resp.get("ContentLength"),
            content_type=resp.get("ContentType"),
            e_tag=_strip_quotes(resp.get("ETag")),
            version_id=resp.get("VersionId"),
            metadata=resp.get("Metadata", {}),
            last_modified=last_modified,
            # S3 exposes no distinct creation timestamp on HEAD; mirror
            # last_modified into created_time.
            created_time=last_modified,
            replication_status=map_replication_status(resp.get("ReplicationStatus")),
        )

    async def do_get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Retrieve object lock state via GetObjectRetention + GetObjectLegalHold.

        S3 exposes retention and legal hold as two independent sub-resources,
        each fetched with a separate call. Either call raises
        ``NoSuchObjectLockConfiguration`` when *that* piece is not configured on
        the object, independently of the other. This is expected for a normal
        object that has only one (or neither) of the two set, so each call is
        wrapped to treat that error as "this piece is absent" rather than a
        failure. Any other native error (missing object, bucket without Object
        Lock enabled, access denied) propagates to the client layer for
        translation.

        Returns ``None`` when the object has neither retention nor legal hold.
        """
        await self._ensure_client()
        version_kwargs = {"VersionId": version_id} if version_id is not None else {}

        async def read_retention() -> (
            tuple[Optional[RetentionMode], Optional[datetime]]
        ):
            try:
                resp = await self._client.get_object_retention(
                    Bucket=self._bucket, Key=key, **version_kwargs
                )
            except ClientError as exc:
                if _is_no_object_lock_error(exc):
                    return None, None
                raise
            retention = resp.get("Retention", {})
            if not retention.get("Mode"):
                return None, None
            return RetentionMode(retention["Mode"]), retention.get("RetainUntilDate")

        async def read_legal_hold() -> bool:
            try:
                resp = await self._client.get_object_legal_hold(
                    Bucket=self._bucket, Key=key, **version_kwargs
                )
            except ClientError as exc:
                if _is_no_object_lock_error(exc):
                    return False
                raise
            # Absent Status is not a hold: S3 marks the field optional, so
            # only an explicit ON counts.
            return resp.get("LegalHold", {}).get("Status") == "ON"

        # Read in sequence rather than gathered. Overlapping the two round
        # trips is tempting and was tried, but gathering mocked coroutines
        # leaves unittest.mock reporting an un-awaited coroutine on the error
        # paths, and the suite is otherwise warning-free. Revisit with a real
        # client in the loop if this read ever shows up as a latency problem.
        mode, retain_until = await read_retention()
        legal_hold = await read_legal_hold()

        if mode is None and not legal_hold:
            return None

        return ObjectLockInfo(
            mode=mode,
            retain_until_date=retain_until,
            legal_hold=legal_hold,
            # S3 has no event-based hold; None means "not reported", which is
            # distinct from a reported-and-off False.
            event_based_hold=None,
        )

    async def do_update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str] = None,
    ) -> None:
        """Update object retention using PutObjectRetention.

        Reads the object's current retention first (a
        ``NoSuchObjectLockConfiguration`` means "no current retention"),
        resolves the effective mode against the shared rules table, then writes
        it back. The resolved mode is used so a ``None`` config mode preserves
        whatever is currently on the object.

        The write depends on what the read returns, so the two are ordered
        rather than independent.
        """
        await self._ensure_client()
        version_kwargs = {"VersionId": version_id} if version_id is not None else {}

        current_mode: Optional[RetentionMode] = None
        current_retain_until: Optional[datetime] = None
        try:
            resp = await self._client.get_object_retention(
                Bucket=self._bucket, Key=key, **version_kwargs
            )
            current = resp.get("Retention", {})
            if current.get("Mode"):
                current_mode = RetentionMode(current["Mode"])
                current_retain_until = current.get("RetainUntilDate")
        except ClientError as exc:
            if not _is_no_object_lock_error(exc):
                raise

        resolved_mode = resolve_and_validate(
            current_mode, current_retain_until, retention
        )

        kwargs: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
            "Retention": {
                "Mode": resolved_mode.value,
                "RetainUntilDate": retention.retain_until_date,
            },
        }
        if version_id is not None:
            kwargs["VersionId"] = version_id
        if retention.bypass_governance:
            kwargs["BypassGovernanceRetention"] = True
        await self._client.put_object_retention(**kwargs)

    async def do_update_legal_hold(
        self, key: str, enabled: bool, *, version_id: Optional[str] = None
    ) -> None:
        """Set or remove a legal hold via PutObjectLegalHold.

        S3 models legal hold as an ON/OFF status on the object; setting it to
        OFF is how a hold is removed. Requires the bucket to have Object Lock
        enabled -- otherwise S3 raises an error that propagates to the client
        layer for translation.
        """
        await self._ensure_client()
        version_kwargs = {"VersionId": version_id} if version_id is not None else {}
        await self._client.put_object_legal_hold(
            Bucket=self._bucket,
            Key=key,
            LegalHold={"Status": "ON" if enabled else "OFF"},
            **version_kwargs,
        )

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
        """Generate a presigned URL for S3 using aiobotocore.

        aiobotocore's generate_presigned_url is a coroutine, so it must be
        awaited. Signing itself is local (no network I/O), so no thread
        offload is needed once awaited.

        The client is pinned to SigV4 (``AWS4-HMAC-SHA256``) at construction,
        which KMS-encrypted buckets require and which folds any
        KMS/metadata/tag headers into the signature — callers must replay
        every ``required_headers`` entry verbatim.
        """
        await self._ensure_client()

        if operation == "PUT":
            # version_id is not forwarded to a PUT: put_object has no
            # VersionId parameter (S3 assigns the version on write).
            return await self._generate_presigned_upload(
                key,
                expiration_seconds,
                content_type,
                metadata,
                tags,
                kms_key_id,
                content_disposition,
                correlation_id,
            )
        else:  # GET
            return await self._generate_presigned_download(
                key,
                expiration_seconds,
                version_id,
                content_disposition,
                correlation_id,
            )

    async def _generate_presigned_upload(
        self,
        key: str,
        expiration_seconds: int,
        content_type: Optional[str],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        kms_key_id: Optional[str],
        content_disposition: Optional[str],
        correlation_id: str,
    ) -> PresignedUrlResult:
        """Generate presigned PUT URL for uploads.

        Every parameter signed into the URL under SigV4 (metadata, tagging,
        server-side encryption, content-disposition) becomes part of
        X-Amz-SignedHeaders and must be replayed by the caller as a request
        header, so each is mirrored into required_headers.
        """
        params: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
        }
        required_headers_dict: dict[str, str] = {}

        if content_type:
            params["ContentType"] = content_type
            required_headers_dict["Content-Type"] = content_type

        if metadata:
            params["Metadata"] = dict(metadata)
            for meta_key, meta_value in metadata.items():
                required_headers_dict[f"x-amz-meta-{meta_key}"] = meta_value

        if tags:
            tag_str = "&".join(
                f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in tags.items()
            )
            params["Tagging"] = tag_str
            required_headers_dict["x-amz-tagging"] = tag_str

        if kms_key_id:
            params["ServerSideEncryption"] = "aws:kms"
            params["SSEKMSKeyId"] = kms_key_id
            required_headers_dict["x-amz-server-side-encryption"] = "aws:kms"
            required_headers_dict["x-amz-server-side-encryption-aws-kms-key-id"] = (
                kms_key_id
            )

        if content_disposition:
            params["ContentDisposition"] = content_disposition
            required_headers_dict["Content-Disposition"] = content_disposition

        url = await self._client.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=expiration_seconds,
        )

        expiration = datetime.now(timezone.utc) + timedelta(seconds=expiration_seconds)

        return PresignedUrlResult(
            url=url,
            expiration=expiration,
            correlation_id=correlation_id,
            http_method="PUT",
            required_headers=MappingProxyType(required_headers_dict),
        )

    async def _generate_presigned_download(
        self,
        key: str,
        expiration_seconds: int,
        version_id: Optional[str],
        content_disposition: Optional[str],
        correlation_id: str,
    ) -> PresignedUrlResult:
        """Generate presigned GET URL for downloads."""
        params: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
        }
        if version_id:
            params["VersionId"] = version_id
        if content_disposition:
            params["ResponseContentDisposition"] = content_disposition

        url = await self._client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=expiration_seconds,
        )

        expiration = datetime.now(timezone.utc) + timedelta(seconds=expiration_seconds)

        return PresignedUrlResult(
            url=url,
            expiration=expiration,
            correlation_id=correlation_id,
            http_method="GET",
            required_headers=MappingProxyType({}),
        )

    async def close(self) -> None:
        """Release the aiobotocore client connection."""
        if self._client_cm is not None:
            await self._client_cm.__aexit__(None, None, None)
            self._client = None
            self._client_cm = None


def _strip_quotes(value: Optional[str]) -> Optional[str]:
    """Strip surrounding quotes from ETag values."""
    if value is None:
        return None
    return value.strip('"')


def _is_no_object_lock_error(exc: ClientError) -> bool:
    """Return True when a GetObjectRetention/GetObjectLegalHold call reports that
    the queried piece (retention or legal hold) is simply not configured on the
    object. S3 signals this with the ``NoSuchObjectLockConfiguration`` error
    code, which is expected for objects that have only one (or neither) of the
    two lock sub-resources set."""
    code = exc.response.get("Error", {}).get("Code", "")
    return code == "NoSuchObjectLockConfiguration"
