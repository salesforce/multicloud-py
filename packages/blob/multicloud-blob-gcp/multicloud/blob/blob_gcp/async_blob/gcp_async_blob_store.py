"""GCP Cloud Storage async implementation of :class:`BaseAsyncBlobStore`.

Translates GCS/aiohttp errors into the cloud-agnostic exception
hierarchy so callers can write a single ``except`` block that works
across providers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, AsyncIterator, List, Mapping, Optional

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
    parse_rfc3339_timestamp,
)
from multicloud.blob.blob_client.retention_rules import resolve_and_validate
from multicloud.blob.blob_gcp._gcp_common import resolve_generation_match
from multicloud.common.exceptions import (
    DeadlineExceededException,
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceExhaustedException,
    ResourceExistsException,
    ResourceNotFoundException,
    SubstrateSdkException,
    UnAuthorizedException,
    UnknownException,
)

try:
    import aiohttp  # type: ignore[import-not-found]
    from gcloud.aio.storage import Storage  # type: ignore[import-not-found]
except ImportError as _exc:
    raise ImportError(
        "gcloud-aio-storage is required for async GCP blob operations. "
        "Install it with: pip install 'python-multicloud-sdk[async-gcp]' "
        "or: pip install gcloud-aio-storage"
    ) from _exc

try:
    import google.auth.transport.requests  # type: ignore[import-not-found]
    from google.cloud import storage as sync_storage  # type: ignore[import-not-found]
    from google.cloud.storage import transfer_manager  # type: ignore[import-not-found]
    from google.oauth2 import service_account  # type: ignore[import-not-found]
except ImportError as _exc:
    raise ImportError(
        "google-cloud-storage is required for GCP directory operations. "
        "Install it with: pip install 'python-multicloud-sdk[gcp]' "
        "or: pip install google-cloud-storage"
    ) from _exc

from multicloud.common.credentials import CredentialsOverrider
from multicloud.common.proxy import ProxyConfig
from multicloud.gcp_common.gcp_credentials import DEFAULT_SCOPES, get_token

_logger = logging.getLogger(__name__)

_ERR_CLIENT_NOT_INITIALIZED = "Storage client not initialized"

_STREAM_CHUNK_SIZE = 8192
"""Chunk size for streamed reads, matching the synchronous provider."""


def _to_rfc3339(value: datetime) -> str:
    """Render a timezone-aware instant the way the GCS JSON API expects.

    Millisecond precision, because GCS truncates anything finer and echoes it
    back truncated, which would otherwise read as a changed value.
    """
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# GCS has no native object tagging API. Tags are stored as custom metadata
# entries with this prefix, mirroring the synchronous GCP provider.
_GCP_TAG_PREFIX = "gcp-tag-"

# Multipart is emulated: each part is a scratch object under this infix, and
# compose assembles them. Compose takes at most 32 sources per call, and folding
# batches iteratively caps a single upload at 1024 parts.
_MULTIPART_INFIX = "__multipart__"
_GCS_MAX_COMPOSE_SOURCES = 32
_GCS_COMPOSE_MAX_PARTS = 1024

# GCS caps a listing page at 1000 items regardless of a larger maxResults,
# so callers must follow nextPageToken rather than trust one response.
_MAX_KEYS_PER_LIST_PAGE = 1000


@ASYNC_BLOB_REGISTRY.register("gcp")
class GcpAsyncBlobStore(BaseAsyncBlobStore):
    """Async GCS-backed blob store using gcloud-aio-storage.

    Args:
        bucket: GCS bucket name.
        region: GCS location (informational, not required for API calls).
        endpoint_url: Endpoint override (for emulators).
        transfer_config: Async transfer configuration.
        service_file: Path to service account JSON key file. If None,
            uses Application Default Credentials (ADC).
        credentials_overrider: Override default credential resolution.
        _storage: Pre-created Storage instance (for testing).
    """

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        *,
        endpoint_url: Optional[str] = None,
        transfer_config: Optional["AsyncTransferConfig"] = None,
        service_file: Optional[str] = None,
        credentials_overrider: Optional[CredentialsOverrider] = None,
        proxy_config: Optional[ProxyConfig] = None,
        _storage: Any = None,
    ) -> None:
        super().__init__(
            bucket=bucket,
            region=region,
            endpoint_url=endpoint_url,
            transfer_config=transfer_config,
            credentials_overrider=credentials_overrider,
            proxy_config=proxy_config,
        )
        self._service_file = service_file
        self._storage: Optional[Storage] = _storage
        self._owns_storage = _storage is None
        # Token owned by this store (SESSION/ASSUME_ROLE overrides). Storage.close()
        # closes only its session, so we close the token ourselves.
        self._token: Any = None
        # Set when this store creates the HTTP session itself, which happens
        # only for a proxy override; it must then be closed here.
        self._owned_session: Any = None
        self._client_lock = asyncio.Lock()

    async def _ensure_client(self) -> None:
        """Lazily create the GCS async client on first use."""
        if self._storage is not None:
            return
        async with self._client_lock:
            if self._storage is None:
                kwargs: dict[str, Any] = {}
                if self._service_file:
                    kwargs["service_file"] = self._service_file
                if self._endpoint_url:
                    kwargs["api_root"] = self._endpoint_url
                # An overrider-resolved token takes precedence over ADC. A None
                # token (unsupported override type) falls through to ADC.
                token = get_token(self._credentials_overrider, scopes=DEFAULT_SCOPES)
                if token is not None:
                    self._token = token
                    kwargs["token"] = token
                session = self._proxy_session()
                if session is not None:
                    self._owned_session = session
                    kwargs["session"] = session
                self._storage = Storage(**kwargs)

    def _proxy_session(self) -> Any:
        """Session carrying the proxy policy, or None to accept the default.

        aiohttp ignores the ambient proxy unless told to trust the environment,
        so honouring the setting means opting in rather than switching off.
        """
        if self._proxy_config is None:
            return None
        return aiohttp.ClientSession(
            proxy=self._proxy_config.endpoint,
            trust_env=self._proxy_config.use_environment_variable_proxy_values,
        )

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        """Translate GCP/aiohttp exceptions to common hierarchy."""
        if isinstance(exc, SubstrateSdkException):
            return exc

        if isinstance(exc, aiohttp.ClientResponseError):
            status = exc.status
            if status == 404:
                return ResourceNotFoundException(str(exc))
            if status in (401, 403):
                return UnAuthorizedException(str(exc))
            if status == 400:
                return InvalidArgumentException(str(exc))
            # 429 means the request was rate limited; 504 is reported as a
            # deadline. 503 falls through: an unavailable backend is not the
            # caller exceeding a quota.
            if status == 429:
                return ResourceExhaustedException(str(exc))
            if status == 504:
                return DeadlineExceededException(str(exc))
            # A conditional write whose precondition is not met returns 412.
            # (The create-if-absent ifGenerationMatch=0 conflict also returns
            # 412 but is re-labelled to ResourceExistsException in do_upload
            # before reaching here.)
            if status == 412:
                return FailedPreconditionException(str(exc))
            return UnknownException(str(exc))

        if isinstance(exc, aiohttp.ClientError):
            return BlobAsyncError(f"Network error: {exc}", cause=exc)

        return BlobAsyncError(str(exc), cause=exc)

    @staticmethod
    def _resolve_upload_parameters(
        *,
        create_if_absent: bool,
        if_match: Optional[str],
        if_version_match: Optional[str],
        kms_key_id: Optional[str],
    ) -> dict[str, str]:
        """Adapt shared write-precondition policy onto GCS JSON-API query params.

        Policy (ETag-only reject, generation mapping) lives in
        :func:`resolve_generation_match`. This adapter emits
        ``ifGenerationMatch`` / ``kmsKeyName`` query strings for
        ``gcloud-aio-storage``.
        """
        parameters: dict[str, str] = {}
        generation = resolve_generation_match(
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
        )
        if generation is not None:
            parameters["ifGenerationMatch"] = str(generation)
        if kms_key_id is not None:
            parameters["kmsKeyName"] = kms_key_id
        return parameters

    async def _upload_conditional(
        self, upload_kwargs: dict[str, Any], create_if_absent: bool
    ) -> Any:
        """Call ``storage.upload``, disambiguating a create-if-absent conflict.

        With ``ifGenerationMatch=0`` GCS returns HTTP 412 when a live object
        already exists at the key; surface that as ResourceExistsException so
        callers can distinguish it from other precondition failures. Every
        other error propagates to the client layer for translation. This local
        catch exists solely for that semantic disambiguation and re-raises
        everything else untouched.
        """
        assert self._storage is not None  # for type checker
        if not create_if_absent:
            return await self._storage.upload(**upload_kwargs)
        try:
            return await self._storage.upload(**upload_kwargs)
        except aiohttp.ClientResponseError as exc:
            if exc.status == 412:
                raise ResourceExistsException(str(exc)) from exc
            raise

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
        """Upload bytes to GCS using gcloud-aio-storage.

        Exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        upload_kwargs: dict[str, Any] = {
            "bucket": self._bucket,
            "object_name": key,
            "file_data": payload,
        }
        if content_type:
            upload_kwargs["content_type"] = content_type
        # ``metadata`` here is the object resource sent with the upload; user
        # metadata and the storage class are both fields of it.
        resource: dict[str, Any] = {}
        if metadata:
            resource["metadata"] = dict(metadata)
        if storage_class is not None:
            # gcloud-aio-storage lower-cases resource keys up to the first
            # hyphen, so "storageClass" would be sent as "storageclass" and
            # ignored. The hyphenated form is rewritten to "storageClass".
            resource["Storage-Class"] = storage_class
        if resource:
            upload_kwargs["metadata"] = resource
        # An explicit kms_key_id selects a customer-managed key. use_kms_managed_key
        # with no explicit key is a no-op on GCS: objects are always encrypted at
        # rest with Google-managed keys by default.
        if use_kms_managed_key:
            _logger.debug(
                "use_kms_managed_key is a no-op on GCS (objects are always "
                "encrypted at rest) and was ignored"
            )
        parameters = self._resolve_upload_parameters(
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
            kms_key_id=kms_key_id,
        )
        if parameters:
            upload_kwargs["parameters"] = parameters

        resp = await self._upload_conditional(upload_kwargs, create_if_absent)

        e_tag: Optional[str] = None
        version_id: Optional[str] = None

        if isinstance(resp, dict):
            e_tag = resp.get("etag") or resp.get("md5Hash")
            version_id = resp.get("generation")

        return AsyncUploadResponse(
            key=key,
            e_tag=e_tag,
            version_id=version_id,
        )

    async def do_download(self, request: AsyncDownloadRequest) -> AsyncDownloadResponse:
        """Download blob content from GCS using gcloud-aio-storage.

        **Known limitation (TOCTOU):** gcloud-aio-storage's ``download()``
        returns only raw bytes - response headers (content-type, etag,
        generation) are not accessible. Metadata is therefore fetched via
        a separate ``download_metadata()`` call. Between these two calls
        the object could be mutated or deleted, so the returned metadata
        may not correspond exactly to the downloaded content. This is an
        inherent limitation of the gcloud-aio-storage library. For
        consistency-critical workloads, verify the etag after download.

        Args:
            request: Download request containing key and optional byte range.

        Returns:
            AsyncDownloadResponse with content and metadata.

        Raises:
            BlobNotFoundError: If the blob does not exist (404).

        Other exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        if request.version_id is not None:
            raise InvalidArgumentException(
                "version_id (generation) is not supported for GCP downloads "
                "via gcloud-aio-storage. Use the REST API directly for "
                "generation-pinned reads."
            )

        headers: dict[str, str] = {}
        range_header = build_range_header(request.start, request.end)
        if range_header:
            headers["Range"] = range_header

        # Conditional read: the If-Match header returns HTTP 412 on ETag
        # mismatch, translated to FailedPreconditionException in the client
        # layer. Mirrors the sync provider's if_etag_match precondition.
        if request.if_match is not None:
            headers["If-Match"] = request.if_match

        try:
            body = await self._storage.download(
                self._bucket,
                request.key,
                headers=headers if headers else None,
            )
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
                raise BlobNotFoundError(
                    f"Blob not found: {request.key}",
                    key=request.key,
                    cause=exc,
                ) from exc
            raise

        try:
            metadata_dict = await self._storage.download_metadata(
                self._bucket,
                request.key,
            )
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
                metadata_dict = {}
            else:
                raise

        metadata = BlobMetadata(
            key=request.key,
            content_length=len(body),
            content_type=metadata_dict.get("contentType"),
            e_tag=metadata_dict.get("etag"),
            version_id=(
                str(metadata_dict.get("generation"))
                if metadata_dict.get("generation")
                else None
            ),
            metadata=metadata_dict.get("metadata", {}) or {},
            # replication_status is left None: GCS has no per-object
            # replication-status field (bucket replication is reported at the
            # bucket level). None means "no per-object signal", NOT
            # "unreplicated" — unlike the AWS x-amz-replication-status header.
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
        """Stream an object out of GCS without reading it into memory.

        The object resource is fetched first for the metadata fields, since the
        streaming call surfaces only response headers. That leaves a window in
        which the object could change between the two requests, the same
        limitation the buffered download carries.

        A generation is pinned through the library's private streaming helper,
        because its public wrapper fixes the query parameters and leaves no way
        to add one. That dependency is pinned in uv.lock; a release that renames
        it needs this call site updated.

        Raises:
            InvalidArgumentException: If version_id is not an integer generation.
            BlobNotFoundError: If the object does not exist (404).
        """
        resource = await self._fetch_object_resource(key, version_id)

        headers: dict[str, str] = {}
        range_header = build_range_header(start, end)
        if range_header:
            headers["Range"] = range_header

        parameters = {"alt": "media"}
        # Always pinned to the generation the metadata came from, not just when
        # the caller asked for a version. Without it the returned metadata
        # describes the object as of the first request while the bytes come from
        # whatever is live at the second, with nothing to detect the mismatch.
        generation = resource.get("generation")
        if generation is not None:
            parameters["generation"] = str(generation)

        assert self._storage is not None  # established by _fetch_object_resource
        try:
            stream = await self._storage._download_stream(
                self._bucket, key, params=parameters, headers=headers or None
            )
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
                raise BlobNotFoundError(
                    f"Blob not found: {key}", key=key, cause=exc
                ) from exc
            raise

        metadata = BlobMetadata(
            key=key,
            # Read off the response, not the object resource, so a ranged read
            # reports the bytes this stream will yield. Taken from the header
            # directly because the library's accessor reports a missing
            # Content-Length as 0, which would read as a real length -- GCS
            # omits it and chunks the body when transcoding a compressed object.
            content_length=_coerce_int(stream._response.headers.get("content-length")),
            content_type=resource.get("contentType"),
            e_tag=resource.get("etag"),
            version_id=str(generation) if generation else None,
            metadata=resource.get("metadata", {}) or {},
            last_modified=parse_rfc3339_timestamp(resource.get("updated")),
            created_time=parse_rfc3339_timestamp(resource.get("timeCreated")),
        )

        async def chunks() -> AsyncIterator[bytes]:
            # Translated here because the client's catch wraps only the call
            # that opens the stream, so a body truncated part-way through would
            # otherwise reach the caller untranslated.
            try:
                while data := await stream.read(_STREAM_CHUNK_SIZE):
                    yield data
            except SubstrateSdkException:
                raise
            except Exception as exc:
                raise self.translate_error(exc) from exc

        async def close() -> None:
            # release() rather than close(): it returns the connection to the
            # pool once the body is exhausted, and degrades to a close by itself
            # when it is not, so an abandoned stream is still safe.
            stream._response.release()

        return AsyncStreamingDownloadResponse(
            key=key,
            metadata=metadata,
            stream=chunks(),
            close_func=close,
        )

    async def do_copy(
        self, request: CopyRequest, *, correlation_id: str
    ) -> CopyResponse:
        """Copy an object within GCS using storage.copy().

        Supports same-bucket and cross-bucket copy. When metadata is
        provided in the request, performs a best-effort metadata patch
        after the copy (GCS copy doesn't support inline metadata override).
        A warning is logged if the metadata patch fails.

        Raises:
            InvalidArgumentException: If source_version_id is provided
                (gcloud-aio-storage does not support sourceGeneration on copy).
            ResourceNotFoundException: If source object does not exist (404).

        Other exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()

        # gcloud-aio-storage does not support a sourceGeneration parameter
        # on copy(). Raise rather than silently ignoring so callers know
        # they would get the latest generation instead of the requested one.
        if request.source_version_id is not None:
            raise InvalidArgumentException(
                "source_version_id (generation) is not supported for GCP copies "
                "via gcloud-aio-storage. Use the REST API directly for "
                "generation-pinned copies."
            )

        try:
            resp = await self._storage.copy(
                request.source_bucket,
                request.source_key,
                request.destination_bucket,
                new_name=request.destination_key,
            )
        except aiohttp.ClientResponseError as exc:
            # Control-flow: distinguish 404 from other errors
            if exc.status == 404:
                raise ResourceNotFoundException(
                    f"Source object not found: "
                    f"{request.source_bucket}/{request.source_key}"
                ) from exc
            raise

        # Extract metadata from copy response
        resource = resp.get("resource", resp) if isinstance(resp, dict) else {}
        e_tag = resource.get("etag") or resource.get("md5Hash")
        version_id = (
            str(resource.get("generation")) if resource.get("generation") else None
        )
        # Extract last_modified from GCS timeCreated or updated field
        last_modified = resource.get("timeCreated") or resource.get("updated")

        # Best-effort metadata override: GCS copy doesn't support inline
        # metadata replacement, so we patch after the copy.
        if request.metadata is not None:
            try:
                await self._storage.patch_metadata(
                    request.destination_bucket,
                    request.destination_key,
                    metadata={"metadata": dict(request.metadata)},
                )
            except aiohttp.ClientError as exc:
                # Best-effort: log warning so failure is diagnosable,
                # but don't fail the copy itself.
                _logger.warning(
                    "copy: best-effort metadata patch failed for %s/%s: %s",
                    request.destination_bucket,
                    request.destination_key,
                    exc,
                )

        return CopyResponse(
            key=request.destination_key,
            version_id=version_id,
            e_tag=e_tag,
            last_modified=last_modified,
            correlation_id=correlation_id,
        )

    async def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Delete a single GCS blob. Idempotent - 404 is treated as success.

        Only the 404 control-flow is handled here (idempotent semantics).
        All other exceptions propagate to the client layer for translation.
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        delete_kwargs: dict[str, Any] = {}
        if version_id is not None:
            delete_kwargs["params"] = {"generation": version_id}

        try:
            await self._storage.delete(
                self._bucket,
                key,
                **delete_kwargs,
            )
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
                return  # Idempotent: not-found is success
            raise

    async def do_delete_objects(
        self,
        identifiers: list[BlobIdentifier],
    ) -> "BulkDeleteResponse":
        """Batch delete GCS blobs using concurrent individual deletes.

        GCS does not have a native multi-object delete API like S3.
        We process identifiers in chunks (max 1000) and use
        asyncio.gather with a semaphore to bound concurrency within
        each chunk. Each coroutine returns its result (no shared
        mutable state) for thread-safety under free-threaded Python.
        404 responses are treated as successful deletes (idempotent).
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        sem = asyncio.Semaphore(self._transfer_config.max_concurrency)

        async def _delete_one(
            ident: BlobIdentifier,
        ) -> DeletedBlob | DeleteError:
            async with sem:
                delete_kwargs: dict[str, Any] = {}
                if ident.version_id is not None:
                    delete_kwargs["params"] = {"generation": ident.version_id}
                try:
                    await self._storage.delete(
                        self._bucket,
                        ident.key,
                        **delete_kwargs,
                    )
                    return DeletedBlob(key=ident.key, version_id=ident.version_id)
                except aiohttp.ClientResponseError as exc:
                    if exc.status == 404:
                        return DeletedBlob(key=ident.key, version_id=ident.version_id)
                    return DeleteError(
                        key=ident.key,
                        version_id=ident.version_id,
                        code=str(exc.status),
                        message=exc.message,
                    )
                except aiohttp.ClientError as exc:
                    return DeleteError(
                        key=ident.key,
                        version_id=ident.version_id,
                        code="NetworkError",
                        message=str(exc),
                    )
                except Exception as exc:
                    return DeleteError(
                        key=ident.key,
                        version_id=ident.version_id,
                        code="Unknown",
                        message=str(exc),
                    )

        all_deleted: list[DeletedBlob] = []
        all_errors: list[DeleteError] = []

        BATCH_SIZE = 1000
        items = list(identifiers)

        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i : i + BATCH_SIZE]
            results = await asyncio.gather(*[_delete_one(ident) for ident in batch])
            for result in results:
                if isinstance(result, DeletedBlob):
                    all_deleted.append(result)
                else:
                    all_errors.append(result)

        return BulkDeleteResponse(deleted=all_deleted, errors=all_errors)

    async def do_get_tags(self, key: str) -> Mapping[str, str]:
        """Retrieve tags from GCS object metadata (gcp-tag-* entries).

        GCS has no native object tagging API. Tags are stored as custom
        metadata entries with a ``gcp-tag-`` prefix.
        """
        await self._ensure_client()
        assert self._storage is not None  # for type checker

        obj_metadata = await self._storage.download_metadata(self._bucket, key)

        custom = obj_metadata.get("metadata") or {}
        return {
            k[len(_GCP_TAG_PREFIX) :]: v
            for k, v in custom.items()
            if k.startswith(_GCP_TAG_PREFIX) and v is not None
        }

    async def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Store tags as gcp-tag-* metadata entries on a GCS object.

        Replaces all existing tag entries while preserving non-tag
        metadata. Existing tag keys are set to ``None`` so the GCS
        metadata PATCH removes them, then the new tags are added.

        Uses ``ifMetagenerationMatch`` on the patch so a concurrent
        metadata change between the read and the patch is rejected by
        GCS (412 Precondition Failed) instead of silently lost.
        """
        await self._ensure_client()
        assert self._storage is not None  # for type checker

        obj_metadata = await self._storage.download_metadata(self._bucket, key)

        existing = obj_metadata.get("metadata") or {}
        merged: dict[str, Optional[str]] = dict(existing)

        for k in list(merged.keys()):
            if k.startswith(_GCP_TAG_PREFIX):
                merged[k] = None

        for tag_key, tag_value in tags.items():
            merged[f"{_GCP_TAG_PREFIX}{tag_key}"] = tag_value

        params: dict[str, str] = {}
        metageneration = obj_metadata.get("metageneration")
        if metageneration is not None:
            params["ifMetagenerationMatch"] = str(metageneration)
        else:
            _logger.warning(
                "set_tags on %s/%s: object metadata has no metageneration; "
                "applying tags without optimistic concurrency control "
                "(last-write-wins).",
                self._bucket,
                key,
            )

        await self._storage.patch_metadata(
            self._bucket, key, {"metadata": merged}, params=params
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
        """Upload a local directory to GCS using transfer_manager.

        Delegates to ``transfer_manager.upload_many_from_filenames`` which
        handles concurrency, retries, and checksums natively.
        """
        source_dir = Path(local_source_directory).resolve()
        files_to_upload = await asyncio.to_thread(
            self.collect_files, source_dir, include_sub_folders, follow_symbolic_links
        )

        if not files_to_upload:
            return AsyncDirectoryUploadResponse(
                total_files=0,
                successful_uploads=0,
                failed_transfers=[],
                total_bytes_transferred=0,
            )

        filenames = [
            str(fp.relative_to(source_dir).as_posix()) for fp in files_to_upload
        ]

        # Custom metadata is a blob *attribute*, assigned after the blob object
        # is constructed -- `Bucket.blob()` accepts no `metadata` argument, so
        # routing it through `blob_constructor_kwargs` raises TypeError and
        # fails every file in the upload.
        additional_blob_attributes: dict[str, Any] = {}
        if tags:
            additional_blob_attributes["metadata"] = {
                f"{_GCP_TAG_PREFIX}{k}": v for k, v in tags.items()
            }

        def _run_upload() -> list:
            client = sync_storage.Client()
            bucket = client.bucket(self._bucket)
            return transfer_manager.upload_many_from_filenames(
                bucket,
                filenames,
                source_directory=str(source_dir),
                blob_name_prefix=prefix,
                max_workers=max_concurrency,
                additional_blob_attributes=additional_blob_attributes,
                raise_exception=False,
                worker_type="thread",
            )

        results = await asyncio.to_thread(_run_upload)

        successful = 0
        total_bytes = 0
        failed: List[FailedAsyncBlobUpload] = []

        for file_path, result in zip(files_to_upload, results):
            if isinstance(result, Exception):
                failed.append(FailedAsyncBlobUpload(source=file_path, exception=result))
            else:
                successful += 1
                total_bytes += file_path.stat().st_size

        return AsyncDirectoryUploadResponse(
            total_files=len(files_to_upload),
            successful_uploads=successful,
            failed_transfers=failed,
            total_bytes_transferred=total_bytes,
        )

    async def do_download_directory(
        self,
        prefix: str,
        local_destination_directory: str,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        prefixes_to_exclude: Optional[List[str]] = None,
    ) -> AsyncDirectoryDownloadResponse:
        """Download objects under a prefix using transfer_manager.

        Delegates to ``transfer_manager.download_many_to_path`` which
        handles concurrency, path traversal protection, and checksums
        natively.
        """
        dest_dir = Path(local_destination_directory)
        await asyncio.to_thread(dest_dir.mkdir, parents=True, exist_ok=True)

        exclusion_prefixes = tuple(prefixes_to_exclude) if prefixes_to_exclude else ()

        def _list_and_download() -> tuple:
            client = sync_storage.Client()
            bucket = client.bucket(self._bucket)

            blobs = list(client.list_blobs(self._bucket, prefix=prefix))
            blob_names: List[str] = []
            for blob in blobs:
                key = blob.name
                if key.endswith("/") and (blob.size is None or blob.size == 0):
                    continue
                if exclusion_prefixes and key.startswith(exclusion_prefixes):
                    continue
                blob_names.append(key)

            if not blob_names:
                return blob_names, []

            # blob_name_prefix is PREPENDED to each entry in blob_names to
            # form the full GCS object key for download. Since blob_names
            # already contain the full key (from list_blobs), we strip the
            # prefix so that prefix + relative_name = full key.
            relative_names = [
                name[len(prefix) :] if name.startswith(prefix) else name
                for name in blob_names
            ]

            results = transfer_manager.download_many_to_path(
                bucket,
                relative_names,
                destination_directory=str(dest_dir),
                blob_name_prefix=prefix,
                max_workers=max_concurrency,
                create_directories=True,
                raise_exception=False,
                worker_type="thread",
            )
            return blob_names, results

        blob_names, results = await asyncio.to_thread(_list_and_download)

        if not blob_names:
            return AsyncDirectoryDownloadResponse(
                total_files=0,
                successful_downloads=0,
                failed_transfers=[],
                total_bytes_transferred=0,
            )

        successful = 0
        total_bytes = 0
        failed: List[FailedAsyncBlobDownload] = []

        for blob_name, result in zip(blob_names, results):
            relative = (
                blob_name[len(prefix) :] if blob_name.startswith(prefix) else blob_name
            )
            local_path = dest_dir / relative

            if isinstance(result, (Exception, UserWarning)):
                exc = (
                    result
                    if isinstance(result, Exception)
                    else RuntimeError(str(result))
                )
                failed.append(
                    FailedAsyncBlobDownload(destination=local_path, exception=exc)
                )
            else:
                successful += 1
                if local_path.exists():
                    total_bytes += local_path.stat().st_size

        return AsyncDirectoryDownloadResponse(
            total_files=len(blob_names),
            successful_downloads=successful,
            failed_transfers=failed,
            total_bytes_transferred=total_bytes,
        )

    async def do_delete_directory(
        self,
        prefix: Optional[str],
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        """Delete every GCS object under a prefix using concurrent deletes.

        GCS has no multi-object delete, so each object is deleted individually
        and throughput comes from concurrency alone. Keys are yielded lazily
        page by page, so memory stays flat regardless of directory size.

        Listing omits the delimiter so the whole subtree is matched, and
        follows nextPageToken to exhaustion -- GCS caps a page at 1000 items,
        so a single request would silently miss a larger directory.

        A 404 counts as a successful delete, matching the idempotent semantics
        of the single-object delete.
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        storage = self._storage

        async def keys() -> AsyncIterator[str]:
            page_token: Optional[str] = None
            while True:
                params: dict[str, str] = {"maxResults": str(_MAX_KEYS_PER_LIST_PAGE)}
                if prefix:
                    params["prefix"] = prefix
                if page_token is not None:
                    params["pageToken"] = page_token

                resp = await storage.list_objects(self._bucket, params=params)

                for item in resp.get("items", []):
                    name = item.get("name")
                    if name:
                        yield name

                page_token = resp.get("nextPageToken")
                if not page_token:
                    return

        async def delete_one(key: str) -> List[str]:
            try:
                await storage.delete(self._bucket, key)
                return []
            except aiohttp.ClientResponseError as exc:
                if exc.status == 404:
                    return []
                # Raise rather than return so the original error survives as the
                # aggregate's cause; returning a string would discard it.
                raise BlobAsyncError(
                    f"{key}: {exc.status} {exc.message}".strip(), cause=exc
                ) from exc
            except aiohttp.ClientError as exc:
                raise BlobAsyncError(f"{key}: {exc}", cause=exc) from exc

        await self.run_delete_tasks(keys(), max_concurrency, delete_one)

    async def do_does_object_exist(
        self, key: str, *, version_id: Optional[str]
    ) -> bool:
        """Check if an object exists using download_metadata.

        Only handles 404 → False. All other exceptions propagate to
        the client layer for translation.

        Args:
            key: Object key.
            version_id: GCS generation (must be an integer string). Optional.

        Returns:
            True if the object exists, False if 404.

        Raises:
            InvalidArgumentException: If version_id is not an integer string.
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError("Storage client not initialized")
        if version_id is not None:
            try:
                int(version_id)
            except ValueError as exc:
                raise InvalidArgumentException(
                    f"GCP version_id must be an integer generation; got {version_id!r}"
                ) from exc
            # gcloud-aio-storage's download_metadata() doesn't accept a
            # ``params`` kwarg, so we call _download directly to pass the
            # generation query parameter.  This is a private method — if a
            # future gcloud-aio-storage release renames/removes it, this
            # call site will need updating.  Pinned dependency version in
            # uv.lock mitigates surprise breakage.
            try:
                await self._storage._download(
                    self._bucket,
                    key,
                    params={"alt": "json", "generation": version_id},
                )
                return True
            except aiohttp.ClientResponseError as exc:
                if exc.status == 404:
                    return False
                raise
        else:
            try:
                await self._storage.download_metadata(self._bucket, key)
                return True
            except aiohttp.ClientResponseError as exc:
                if exc.status == 404:
                    return False
                raise

    def _part_key(self, mpu: MultipartUpload, part_number: int) -> str:
        return f"{mpu.key}{_MULTIPART_INFIX}{mpu.upload_id}__{part_number}"

    def _part_prefix(self, mpu: MultipartUpload) -> str:
        return f"{mpu.key}{_MULTIPART_INFIX}{mpu.upload_id}__"

    async def do_initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Mapping[str, str],
        content_type: Optional[str],
        tags: Mapping[str, str],
        correlation_id: str,
    ) -> MultipartUpload:
        """Begin a multipart upload; the id is minted locally, not by GCS."""
        return MultipartUpload(
            bucket=self._bucket,
            key=key,
            upload_id=uuid.uuid4().hex,
            metadata=dict(metadata),
            content_type=content_type,
            tags=dict(tags),
            correlation_id=correlation_id,
        )

    async def do_upload_part(
        self, mpu: MultipartUpload, part: MultipartPart, *, correlation_id: str
    ) -> UploadPartResponse:
        """Store one part as its own object, to be composed on completion."""
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        if part.part_number > _GCS_COMPOSE_MAX_PARTS:
            raise InvalidArgumentException(
                f"multipart upload supports at most {_GCS_COMPOSE_MAX_PARTS} "
                f"parts, got part number {part.part_number}"
            )

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

        resp = await self._storage.upload(
            bucket=mpu.bucket,
            object_name=self._part_key(mpu, part.part_number),
            file_data=payload,
        )

        e_tag = ""
        if isinstance(resp, dict):
            e_tag = resp.get("etag") or resp.get("md5Hash") or ""
        return UploadPartResponse(
            part_number=part.part_number,
            e_tag=e_tag,
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
        """Compose the stored parts, folding in batches of 32 sources."""
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        if len(parts) > _GCS_COMPOSE_MAX_PARTS:
            raise InvalidArgumentException(
                f"multipart upload supports at most {_GCS_COMPOSE_MAX_PARTS} "
                f"parts, got {len(parts)}"
            )

        ordered = sorted(parts, key=lambda p: p.part_number)
        sources = [self._part_key(mpu, p.part_number) for p in ordered]
        intermediates: List[str] = []
        try:
            while len(sources) > _GCS_MAX_COMPOSE_SOURCES:
                batch, sources = (
                    sources[:_GCS_MAX_COMPOSE_SOURCES],
                    sources[_GCS_MAX_COMPOSE_SOURCES:],
                )
                intermediate = (
                    f"{mpu.key}{_MULTIPART_INFIX}{mpu.upload_id}"
                    f"__intermediate_{len(intermediates)}"
                )
                await self._storage.compose(
                    bucket=mpu.bucket,
                    object_name=intermediate,
                    source_object_names=batch,
                )
                intermediates.append(intermediate)
                sources.insert(0, intermediate)

            compose_kwargs: dict[str, Any] = {
                "bucket": mpu.bucket,
                "object_name": mpu.key,
                "source_object_names": sources,
            }
            if mpu.content_type:
                compose_kwargs["content_type"] = mpu.content_type
            resp = await self._storage.compose(**compose_kwargs)
        finally:
            await self._delete_objects_quietly(
                mpu.bucket,
                [self._part_key(mpu, p.part_number) for p in ordered] + intermediates,
            )

        if mpu.metadata or mpu.tags:
            merged = dict(mpu.metadata)
            merged.update({f"{_GCP_TAG_PREFIX}{k}": v for k, v in mpu.tags.items()})
            await self._storage.patch_metadata(
                bucket=mpu.bucket, object_name=mpu.key, metadata={"metadata": merged}
            )

        e_tag = None
        version_id = None
        if isinstance(resp, dict):
            e_tag = resp.get("etag") or resp.get("md5Hash")
            version_id = resp.get("generation")
        return MultipartUploadResponse(
            e_tag=e_tag,
            key=mpu.key,
            version_id=version_id,
            correlation_id=correlation_id,
        )

    async def do_list_parts(
        self, mpu: MultipartUpload, *, correlation_id: str
    ) -> List[UploadPartResponse]:
        """List the stored parts, following nextPageToken to exhaustion.

        Skips the intermediates left by a batched compose: their names carry a
        non-numeric suffix, so they are not parts the caller uploaded.
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        prefix = self._part_prefix(mpu)
        collected: List[UploadPartResponse] = []
        page_token: Optional[str] = None
        while True:
            params: dict[str, str] = {
                "prefix": prefix,
                "maxResults": str(_MAX_KEYS_PER_LIST_PAGE),
            }
            if page_token:
                params["pageToken"] = page_token
            listing = await self._storage.list_objects(self._bucket, params=params)
            for item in listing.get("items", []) or []:
                suffix = item.get("name", "")[len(prefix) :]
                try:
                    part_number = int(suffix)
                except ValueError:
                    # Compose intermediates share the prefix but are not parts.
                    continue
                collected.append(
                    UploadPartResponse(
                        part_number=part_number,
                        e_tag=item.get("etag") or "",
                        size_in_bytes=_coerce_int(item.get("size")) or 0,
                        correlation_id=correlation_id,
                    )
                )
            page_token = listing.get("nextPageToken")
            if not page_token:
                break
        return sorted(collected, key=lambda p: p.part_number)

    async def do_abort_multipart_upload(
        self, mpu: MultipartUpload, *, correlation_id: str
    ) -> None:
        """Delete every stored part, following nextPageToken to exhaustion."""
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        prefix = self._part_prefix(mpu)
        page_token: Optional[str] = None
        while True:
            params: dict[str, str] = {
                "prefix": prefix,
                "maxResults": str(_MAX_KEYS_PER_LIST_PAGE),
            }
            if page_token:
                params["pageToken"] = page_token
            listing = await self._storage.list_objects(self._bucket, params=params)
            keys = [item["name"] for item in listing.get("items", []) or []]
            await self._delete_scratch_objects(mpu.bucket, keys)
            page_token = listing.get("nextPageToken")
            if not page_token:
                return

    async def _delete_scratch_objects(self, bucket: str, keys: List[str]) -> None:
        """Delete each key, tolerating a 404 and propagating anything else."""
        if self._storage is None:
            return
        for key in keys:
            try:
                await self._storage.delete(bucket, key)
            except aiohttp.ClientResponseError as exc:
                if exc.status != 404:
                    raise

    async def _delete_objects_quietly(self, bucket: str, keys: List[str]) -> None:
        """Delete each key after a successful compose, warning on failure."""
        if self._storage is None:
            return
        for key in keys:
            try:
                await self._storage.delete(bucket, key)
            except aiohttp.ClientResponseError as exc:
                if exc.status != 404:
                    _logger.warning(
                        "multipart cleanup: %s left behind (HTTP %s)", key, exc.status
                    )
            except Exception as exc:  # noqa: BLE001
                _logger.warning("multipart cleanup: %s left behind (%s)", key, exc)

    async def do_does_bucket_exist(self) -> bool:
        """Check if the bucket exists using get_bucket_metadata.

        Uses bucket metadata GET (requires only storage.buckets.get
        permission), matching the semantics of S3 head_bucket.

        Only handles 404 → False. All other exceptions propagate to
        the client layer for translation.

        Returns:
            True if the bucket exists, False if 404.
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError("Storage client not initialized")
        try:
            await self._storage.get_bucket_metadata(self._bucket)
            return True
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
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
        """List a single page of objects using the GCS list_objects JSON API.

        gcloud-aio-storage's ``list_objects`` performs a single HTTP GET and
        returns the raw GCS JSON payload (``{items, prefixes, nextPageToken}``),
        so this maps that payload directly onto ``ListBlobsPageResponse`` — no
        thread offload to the sync SDK is needed (unlike directory ops / URL
        signing). Exceptions propagate to the client layer for translation.

        Field mapping (GCS ``items[]`` entry -> :class:`BlobInfo`):
        ``name`` -> key, ``size`` (decimal string) -> size (coerced to int),
        ``etag`` -> e_tag, ``updated`` (RFC 3339) -> last_modified,
        ``storageClass`` -> storage_class. ``version_id`` is left None to
        match the sync GCP store and async AWS (uniform cross-provider
        result). ``prefixes`` -> common_prefixes; ``nextPageToken`` ->
        next_pagination_token (and is_truncated when present).
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        # GCS list params are all strings on the query string.
        params: dict[str, str] = {"maxResults": str(max_results)}
        if prefix is not None:
            params["prefix"] = prefix
        if delimiter is not None:
            params["delimiter"] = delimiter
        if pagination_token is not None:
            params["pageToken"] = pagination_token

        resp = await self._storage.list_objects(self._bucket, params=params)

        blobs = [
            BlobInfo(
                key=item.get("name"),
                # GCS reports "size" as a decimal string; coerce defensively.
                size=_coerce_int(item.get("size")),
                e_tag=item.get("etag"),
                last_modified=parse_rfc3339_timestamp(item.get("updated")),
                storage_class=item.get("storageClass"),
            )
            for item in resp.get("items", [])
        ]
        common_prefixes = list(resp.get("prefixes", []) or [])
        next_page_token = resp.get("nextPageToken")
        return ListBlobsPageResponse(
            blobs=blobs,
            common_prefixes=common_prefixes,
            is_truncated=bool(next_page_token),
            next_pagination_token=next_page_token,
        )

    async def _fetch_object_resource(
        self, key: str, version_id: Optional[str]
    ) -> dict[str, Any]:
        """Fetch an object's JSON resource, optionally pinned to a generation.

        Non-versioned reads use ``download_metadata`` (a metadata-only JSON
        GET). Versioned reads pin the generation via a raw ``_download`` with
        ``params={"alt": "json", "generation": ...}`` and parse the JSON bytes,
        because ``download_metadata`` accepts no ``params`` kwarg. That private
        method dependency mirrors do_does_object_exist (pinned in uv.lock); if a
        future gcloud-aio-storage release renames or removes it, this call site
        needs updating.

        Raises:
            InvalidArgumentException: If version_id is not an integer generation.
            BlobNotFoundError: If the object does not exist (404).
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        if version_id is not None:
            try:
                int(version_id)
            except ValueError as exc:
                raise InvalidArgumentException(
                    f"GCP version_id must be an integer generation; got {version_id!r}"
                ) from exc
            try:
                raw = await self._storage._download(
                    self._bucket,
                    key,
                    params={"alt": "json", "generation": version_id},
                )
            except aiohttp.ClientResponseError as exc:
                if exc.status == 404:
                    raise BlobNotFoundError(
                        f"Blob not found: {key}", key=key, cause=exc
                    ) from exc
                raise
            parsed: dict[str, Any] = json.loads(raw.decode())
            return parsed

        try:
            return await self._storage.download_metadata(self._bucket, key)
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
                raise BlobNotFoundError(
                    f"Blob not found: {key}", key=key, cause=exc
                ) from exc
            raise

    async def do_list_blob_versions(self, key: str) -> AsyncIterator[BlobInfo]:
        """Yield every generation of one key from the versioned object listing.

        The listing is prefix-based, so entries whose name merely starts with
        the requested key are filtered out. Pages are followed as they are
        consumed rather than gathered up front.

        A live generation is one with no deletion timestamp, which is what
        ``is_latest`` reports.
        """
        await self._ensure_client()
        if self._storage is None:
            raise RuntimeError(_ERR_CLIENT_NOT_INITIALIZED)

        params: dict[str, str] = {
            "prefix": key,
            "versions": "true",
            "maxResults": str(_MAX_KEYS_PER_LIST_PAGE),
        }

        while True:
            # Translated here rather than left to the client. Calling an async
            # generator function runs none of this body, so the client's catch
            # has already returned by the time the first page is fetched and
            # cannot see any failure, on this page or a later one.
            try:
                resp = await self._storage.list_objects(self._bucket, params=params)
            except aiohttp.ClientError as exc:
                raise self.translate_error(exc) from exc

            for item in resp.get("items", []):
                if item.get("name") != key:
                    continue
                generation = item.get("generation")
                yield BlobInfo(
                    key=item["name"],
                    size=_coerce_int(item.get("size")),
                    e_tag=item.get("etag"),
                    last_modified=parse_rfc3339_timestamp(item.get("updated")),
                    storage_class=item.get("storageClass"),
                    version_id=str(generation) if generation is not None else None,
                    is_latest=item.get("timeDeleted") is None,
                )

            next_token = resp.get("nextPageToken")
            if not next_token:
                break
            params["pageToken"] = next_token

    async def do_get_metadata(
        self, key: str, *, version_id: Optional[str] = None
    ) -> BlobMetadata:
        """Retrieve object metadata (HEAD-style) without downloading the body.

        Raises:
            InvalidArgumentException: If version_id is not an integer generation.
            BlobNotFoundError: If the object does not exist (404).

        All other exceptions propagate to the client layer for translation.
        """
        metadata_dict = await self._fetch_object_resource(key, version_id)

        # GCS reports "size" as a decimal string; coerce defensively.
        size_raw = metadata_dict.get("size")
        try:
            content_length = int(size_raw) if size_raw is not None else None
        except (TypeError, ValueError):
            content_length = None

        generation = metadata_dict.get("generation")
        return BlobMetadata(
            key=key,
            content_length=content_length,
            content_type=metadata_dict.get("contentType"),
            e_tag=metadata_dict.get("etag"),
            version_id=str(generation) if generation else None,
            metadata=metadata_dict.get("metadata", {}) or {},
            last_modified=parse_rfc3339_timestamp(metadata_dict.get("updated")),
            created_time=parse_rfc3339_timestamp(metadata_dict.get("timeCreated")),
            # replication_status is left None: GCS has no per-object
            # replication-status field (bucket replication is reported at the
            # bucket level). None means "no per-object signal", NOT
            # "unreplicated" — unlike the AWS x-amz-replication-status header.
        )

    async def do_get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Read object lock state off the object's JSON resource.

        Retention (mode + retain-until) and the two hold flags all live on the
        object resource, so a single metadata GET answers the whole question.
        Retention counts as a lock only when it carries an actual value: a
        ``Locked`` retention maps to COMPLIANCE, anything else to GOVERNANCE. A
        legal hold is reported when either the temporary or the event-based
        hold is set, since either alone keeps the object undeletable.

        GCS omits a hold flag entirely when it is off, so an absent key reads
        as False rather than "not reported".

        Returns ``None`` when the object has neither retention nor a hold.

        Raises:
            BlobNotFoundError: If the object does not exist (404).
        """
        resource = await self._fetch_object_resource(key, version_id)

        retention = resource.get("retention") or {}
        retention_mode = retention.get("mode")
        retain_until = parse_rfc3339_timestamp(retention.get("retainUntilTime"))
        has_retention = retention_mode is not None or retain_until is not None

        event_based_hold = bool(resource.get("eventBasedHold"))
        legal_hold = event_based_hold or bool(resource.get("temporaryHold"))

        if not has_retention and not legal_hold:
            return None

        mode: Optional[RetentionMode] = None
        if has_retention:
            mode = (
                RetentionMode.COMPLIANCE
                if retention_mode == "Locked"
                else RetentionMode.GOVERNANCE
            )

        return ObjectLockInfo(
            mode=mode,
            retain_until_date=retain_until,
            legal_hold=legal_hold,
            event_based_hold=event_based_hold,
        )

    def _read_modify_write_guard(
        self, resource: dict[str, Any], key: str, operation: str
    ) -> dict[str, str]:
        """Preconditions that tie a write to the read it was decided from.

        Both halves are needed. The metageneration alone does not identify a
        version -- it restarts at 1 for each new generation -- so an overwrite
        landing between the read and the write produces a fresh object whose
        metageneration matches the one read, and the write is applied to a
        version the caller never saw. Pinning the generation as well makes that
        case a precondition failure.

        A field the provider did not return is reported rather than dropped
        silently, since losing a guard weakens the write without failing it.
        """
        parameters: dict[str, str] = {}
        for field, parameter in (
            ("generation", "ifGenerationMatch"),
            ("metageneration", "ifMetagenerationMatch"),
        ):
            value = resource.get(field)
            if value is not None:
                parameters[parameter] = str(value)
            else:
                _logger.warning(
                    "%s on %s/%s: object metadata has no %s; applying the "
                    "change without optimistic concurrency control "
                    "(last-write-wins).",
                    operation,
                    self._bucket,
                    key,
                    field,
                )
        return parameters

    async def do_update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str] = None,
    ) -> None:
        """Update object retention by patching the object resource.

        GCS names its retention modes ``Locked`` (irreversible) and
        ``Unlocked`` (reducible by privileged callers). The object's current
        retention is read first, resolved against the shared rules table, and
        the resolved mode written back, so a ``None`` config mode preserves
        whatever is already set.

        Relaxing an Unlocked retention -- shortening it, or sealing it to
        Locked -- is refused unless the override is sent, so the override is
        forwarded from ``bypass_governance``.
        """
        resource = await self._fetch_object_resource(key, version_id)

        current = resource.get("retention") or {}
        current_mode: Optional[RetentionMode] = None
        current_retain_until: Optional[datetime] = None
        if current.get("mode") is not None:
            current_mode = (
                RetentionMode.COMPLIANCE
                if current["mode"] == "Locked"
                else RetentionMode.GOVERNANCE
            )
            current_retain_until = parse_rfc3339_timestamp(
                current.get("retainUntilTime")
            )

        resolved_mode = resolve_and_validate(
            current_mode, current_retain_until, retention
        )

        # The rules above were resolved against the retention just read, so the
        # write must not land on anything else -- including a shorten that was
        # only approved because a since-replaced value permitted it.
        parameters = self._read_modify_write_guard(
            resource, key, "update_object_retention"
        )
        if version_id is not None:
            parameters["generation"] = version_id
        if retention.bypass_governance:
            parameters["overrideUnlockedRetention"] = "true"

        assert self._storage is not None  # established by _fetch_object_resource
        await self._storage.patch_metadata(
            self._bucket,
            key,
            {
                "retention": {
                    "mode": (
                        "Locked"
                        if resolved_mode == RetentionMode.COMPLIANCE
                        else "Unlocked"
                    ),
                    "retainUntilTime": _to_rfc3339(retention.retain_until_date),
                }
            },
            params=parameters,
        )

    async def do_update_legal_hold(
        self, key: str, enabled: bool, *, version_id: Optional[str] = None
    ) -> None:
        """Set or remove a legal hold via the object's hold flags.

        GCS has no single "legal hold" property; it exposes two independent
        hold flags -- the temporary hold and the event-based hold -- either of
        which keeps the object undeletable regardless of any retention policy.
        Because either flag alone holds the object, the two directions are
        asymmetric:

        - Placing a hold sets a single flag: the event-based hold if one is
          already active on the object, otherwise the temporary hold. One flag
          is sufficient, so there is no need to add a second, redundant one.
        - Releasing a hold clears *both* flags. An object can carry both at
          once, and clearing only one would leave it held by the other while
          reporting success, contradicting the read side, which reports a
          legal hold when *either* flag is set.

        The object's current state is read first -- to decide which flag to set
        on enable, and to capture the metageneration for the write. The
        read-modify-write is guarded with ``ifMetagenerationMatch`` so a
        concurrent modification between the two surfaces as a precondition
        failure rather than clobbering an unrelated change.

        Retention-intent caveat: GCS exposes no metadata for *why* an
        event-based hold was placed, so releasing a legal hold clears an active
        event-based hold even when it was set for retention rather than
        legal-hold reasons. Making the two concerns fully orthogonal would be a
        larger contract change.
        """
        resource = await self._fetch_object_resource(key, version_id)

        if enabled:
            # One flag is enough to hold the object: reuse an active
            # event-based hold, otherwise set the temporary hold.
            body: dict[str, Any] = (
                {"eventBasedHold": True}
                if resource.get("eventBasedHold")
                else {"temporaryHold": True}
            )
        else:
            # Clear both: either flag left set would keep the object held and
            # contradict how the lock read reports it.
            body = {"eventBasedHold": False, "temporaryHold": False}

        # Which flag to write was decided from the read above, so the write is
        # tied to that exact object. This applies to releasing as much as to
        # placing: a release that lands on a replacement silently leaves the
        # object the caller meant to unhold still held.
        parameters = self._read_modify_write_guard(resource, key, "update_legal_hold")
        if version_id is not None:
            parameters["generation"] = version_id

        assert self._storage is not None  # established by _fetch_object_resource
        await self._storage.patch_metadata(self._bucket, key, body, params=parameters)

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
        """Generate a V4 signed URL for GCS.

        gcloud-aio-storage cannot sign URLs, so signing uses the sync
        google-cloud-storage SDK (imported as ``sync_storage``) off the event
        loop via asyncio.to_thread. kms_key_id is ignored — GCS does not bind
        customer-managed encryption keys in signed URLs.
        """
        url, required_headers_dict = await asyncio.to_thread(
            self._sign_url_sync,
            key,
            operation,
            expiration_seconds,
            version_id,
            content_type,
            metadata,
            tags,
            content_disposition,
        )

        expiration = datetime.now(timezone.utc) + timedelta(seconds=expiration_seconds)

        return PresignedUrlResult(
            url=url,
            expiration=expiration,
            correlation_id=correlation_id,
            http_method=operation,
            required_headers=MappingProxyType(required_headers_dict),
        )

    def _make_sync_client(self) -> "sync_storage.Client":
        """Build a sync google-cloud-storage client for V4 URL signing.

        gcloud-aio-storage cannot sign URLs, so the presigned path falls back to
        the sync SDK. When a service-account key file is configured we build the
        client from it (that key is itself a local V4 signer); otherwise we let
        the client resolve Application Default Credentials.

        A bare ``sync_storage.Client()`` raises ``OSError`` when ADC carry no
        default project (e.g. ``gcloud auth application-default login`` without a
        quota project) — even though V4 URL signing is purely cryptographic and
        needs no project at all. Passing ``project=None`` sets ``no_project`` and
        skips that lookup, so signing works with projectless ADC.

        The signing path reads the resolved credentials back off the returned
        client (``client._credentials``) rather than resolving them separately.
        That keeps signing hermetic and testable: patching ``sync_storage.Client``
        to return a client whose credential has a local signer exercises real
        offline V4 signing with no ADC lookup and no network I/O.
        """
        if self._service_file:
            credentials = service_account.Credentials.from_service_account_file(
                self._service_file, scopes=DEFAULT_SCOPES
            )
            return sync_storage.Client(
                credentials=credentials, project=credentials.project_id
            )
        return sync_storage.Client(project=None)

    def _sign_url_sync(
        self,
        key: str,
        operation: str,
        expiration_seconds: int,
        version_id: Optional[str],
        content_type: Optional[str],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        content_disposition: Optional[str],
    ) -> tuple[str, dict[str, str]]:
        """Synchronous V4 URL signing using the google-cloud-storage SDK.

        Called via asyncio.to_thread from do_generate_presigned_url. Returns
        the signed URL together with the headers the caller must replay: any
        header folded into the V4 signature (metadata, tags, content
        disposition) must be sent on the request or GCS rejects it with 403
        SignatureDoesNotMatch.
        """
        client = self._make_sync_client()
        credentials = client._credentials
        bucket = client.bucket(self._bucket)
        blob = bucket.blob(key)
        expiration = timedelta(seconds=expiration_seconds)

        # Determine how to sign, mirroring the synchronous GCP provider. A
        # credential with a local private key (service-account key file) signs
        # V4 URLs directly. Otherwise (impersonated SA, ADC, GCE metadata) the
        # library must sign via the IAM signBlob API, which needs a fresh access
        # token plus the signer's service-account email — without these it falls
        # back to raw ``sign_bytes`` and fails.
        sign_kwargs: dict[str, Any] = {}
        has_local_signer = (
            hasattr(credentials, "signer")
            and hasattr(credentials.signer, "sign")
            and hasattr(credentials.signer, "key_id")
            and credentials.signer.key_id is not None
        )
        if not has_local_signer:
            # ADC resolved by the client (e.g. via ``gcloud auth
            # application-default login``) default to no explicit scopes, which
            # the IAM signBlob call rejects with "insufficient authentication
            # scopes". Re-scope to cloud-platform so the access token can drive
            # signBlob. ``google.auth.default`` is never called here, so the
            # signing path needs no ADC and stays offline under test.
            if getattr(credentials, "requires_scopes", False) or not getattr(
                credentials, "scopes", None
            ):
                if hasattr(credentials, "with_scopes"):
                    credentials = credentials.with_scopes(DEFAULT_SCOPES)
            if not credentials.valid:
                credentials.refresh(google.auth.transport.requests.Request())
            sign_kwargs["service_account_email"] = getattr(
                credentials, "service_account_email", None
            )
            sign_kwargs["access_token"] = credentials.token

        if operation == "PUT":
            headers: dict[str, str] = {}
            required_headers: dict[str, str] = {}

            if content_type:
                required_headers["Content-Type"] = content_type

            for meta_key, meta_value in metadata.items():
                header_key = f"x-goog-meta-{meta_key}"
                headers[header_key] = meta_value
                required_headers[header_key] = meta_value

            # GCS has no native object tagging; tags round-trip as
            # x-goog-meta-gcp-tag-* metadata, matching the sync provider.
            for tag_key, tag_value in tags.items():
                header_key = f"x-goog-meta-{_GCP_TAG_PREFIX}{tag_key}"
                headers[header_key] = tag_value
                required_headers[header_key] = tag_value

            if content_disposition:
                headers["Content-Disposition"] = content_disposition
                required_headers["Content-Disposition"] = content_disposition

            url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="PUT",
                headers=headers or None,
                content_type=content_type,
                **sign_kwargs,
            )
            return url, required_headers

        # GET: response overrides ride as query parameters, not as signed
        # request headers, so callers need no extra headers.
        kwargs: dict[str, Any] = {
            "version": "v4",
            "expiration": expiration,
            "method": "GET",
        }
        if version_id:
            kwargs["query_parameters"] = {"generation": version_id}
        if content_disposition:
            kwargs["response_disposition"] = content_disposition
        kwargs.update(sign_kwargs)

        url = blob.generate_signed_url(**kwargs)
        return url, {}

    async def close(self) -> None:
        """Release the underlying aiohttp session."""
        if self._storage is not None and self._owns_storage:
            await self._storage.close()
            self._storage = None
        # Storage.close() closes only its session; close our owned token too.
        if self._token is not None:
            await self._token.close()
            self._token = None
        if self._owned_session is not None:
            await self._owned_session.close()
            self._owned_session = None


def _coerce_int(value: Any) -> Optional[int]:
    """Coerce a GCS numeric-string field (e.g. ``size``) to int.

    GCS reports ``size`` as a decimal string in list/metadata JSON. Parsing
    is best-effort — a missing or malformed value yields None rather than
    raising, mirroring the defensive coercion in ``do_get_metadata``.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
