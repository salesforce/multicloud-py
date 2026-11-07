"""In-memory async blob store registered as provider ``"benchfake"``.

The async twin of :mod:`benchmarks.blob._sync_fakes`. A live benchmark
mixes SDK code, the provider SDK, the event loop, network, and the cloud;
if latency moves you cannot tell which layer changed. This store isolates
the async client -> driver path: every ``do_*`` is a dict read/write, so a
slower result here means that path got slower.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

from multicloud.blob.blob_client.async_blob.async_driver import (
    ASYNC_BLOB_REGISTRY,
    DEFAULT_MAX_CONCURRENCY,
    AsyncDirectoryDownloadResponse,
    AsyncDirectoryUploadResponse,
    AsyncDownloadRequest,
    AsyncDownloadResponse,
    AsyncUploadResponse,
    BaseAsyncBlobStore,
    BlobIdentifier,
    BlobInfo,
    BlobMetadata,
    BlobNotFoundError,
    BulkDeleteResponse,
    CopyRequest,
    CopyResponse,
    DeletedBlob,
    ListBlobsPageResponse,
    MultipartPart,
    MultipartUpload,
    MultipartUploadResponse,
    UploadPartResponse,
)
from multicloud.common.exceptions import SubstrateSdkException, UnknownException

# Stable ETag returned for every stored object. A constant keeps the store
# cheap so the measured cost is the SDK code above this hook, not hashing.
_E_TAG = "bench-etag"

# Keys deleted per batch by the directory-delete helper. Mirrors the shape of
# a provider that chunks a large delete, so the base run_delete_tasks fan-out
# is exercised the way a real backend drives it.
_DELETE_BATCH = 1000

# Stored record: (content bytes, content type, user metadata).
_Record = Tuple[bytes, Optional[str], Dict[str, str]]


@ASYNC_BLOB_REGISTRY.register("benchfake")
class AsyncBenchBlobStore(BaseAsyncBlobStore):
    """In-memory async store for ``benchfake``: dict-backed ``do_*``, no network."""

    provider_id = "benchfake"

    def __init__(self, bucket: str = "bench-bucket", **kwargs: Any) -> None:
        super().__init__(bucket=bucket, **kwargs)
        self._objects: Dict[str, _Record] = {}
        # In-progress multipart sessions: upload_id -> {part_number: bytes}.
        self._multipart: Dict[str, Dict[int, bytes]] = {}
        # Monotonic counter for unique upload ids (no clock/random needed).
        self._mpu_counter = 0

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        if isinstance(exc, SubstrateSdkException):
            return exc
        return UnknownException(str(exc))

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
    ) -> AsyncUploadResponse:
        self._objects[key] = (payload, content_type, dict(metadata))
        return AsyncUploadResponse(key=key, e_tag=_E_TAG)

    async def do_download(self, request: AsyncDownloadRequest) -> AsyncDownloadResponse:
        content, content_type, metadata = self._objects[request.key]
        return AsyncDownloadResponse(
            key=request.key,
            content=content,
            metadata=BlobMetadata(
                key=request.key,
                content_length=len(content),
                content_type=content_type,
                e_tag=_E_TAG,
                metadata=metadata,
            ),
        )

    async def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        self._objects.pop(key, None)

    async def do_copy(
        self, request: CopyRequest, *, correlation_id: str
    ) -> CopyResponse:
        content, content_type, metadata = self._objects[request.source_key]
        # The in-memory store collapses all buckets to one dict; the request's
        # destination_bucket is accepted for API-shape parity but the copy
        # lands in this store.
        new_metadata = (
            dict(request.metadata) if request.metadata is not None else dict(metadata)
        )
        self._objects[request.destination_key] = (content, content_type, new_metadata)
        return CopyResponse(
            key=request.destination_key,
            e_tag=_E_TAG,
            correlation_id=correlation_id,
        )

    async def do_delete_objects(
        self, identifiers: List[BlobIdentifier]
    ) -> BulkDeleteResponse:
        deleted = []
        for ident in identifiers:
            self._objects.pop(ident.key, None)
            deleted.append(DeletedBlob(key=ident.key, version_id=ident.version_id))
        # The in-memory store never fails a delete, so no per-key errors.
        return BulkDeleteResponse(deleted=deleted, errors=[])

    async def do_get_tags(self, key: str) -> Mapping[str, str]:
        # Tags are not modeled separately from user metadata in the fake.
        return dict(self._objects[key][2])

    async def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        content, content_type, _ = self._objects[key]
        self._objects[key] = (content, content_type, dict(tags))

    async def do_get_metadata(
        self, key: str, *, version_id: Optional[str] = None
    ) -> BlobMetadata:
        record = self._objects.get(key)
        if record is None:
            raise BlobNotFoundError(f"Blob not found: {key}", key=key)
        content, content_type, metadata = record
        return BlobMetadata(
            key=key,
            content_length=len(content),
            content_type=content_type,
            e_tag=_E_TAG,
            metadata=metadata,
        )

    async def do_list_page(
        self,
        *,
        prefix: Optional[str],
        delimiter: Optional[str],
        max_results: int,
        pagination_token: Optional[str],
    ) -> ListBlobsPageResponse:
        # Sort for a stable page order so a paginate-until-done loop is
        # deterministic; the token is the offset of the next page.
        keys = sorted(
            k for k in self._objects if prefix is None or k.startswith(prefix)
        )
        offset = int(pagination_token) if pagination_token else 0
        window = keys[offset : offset + max_results]
        blobs = [
            BlobInfo(key=k, size=len(self._objects[k][0]), e_tag=_E_TAG) for k in window
        ]
        next_offset = offset + len(window)
        is_truncated = next_offset < len(keys)
        return ListBlobsPageResponse(
            blobs=blobs,
            is_truncated=is_truncated,
            next_pagination_token=str(next_offset) if is_truncated else None,
        )

    async def do_does_object_exist(
        self, key: str, *, version_id: Optional[str]
    ) -> bool:
        return key in self._objects

    async def do_does_bucket_exist(self) -> bool:
        return True

    # Directory operations. These drive the base-class fan-out helpers
    # (collect_files / run_upload_tasks / run_download_tasks /
    # run_delete_tasks) exactly as a real provider does, so the benchmark
    # measures that shared orchestration rather than a bespoke loop.

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
        source_dir = Path(local_source_directory).resolve()
        files = self.collect_files(
            source_dir, include_sub_folders, follow_symbolic_links
        )

        async def _put(_file_path: Path, blob_key: str, data: bytes) -> None:
            self._objects[blob_key] = (data, None, {})

        return await self.run_upload_tasks(
            files, source_dir, prefix, max_concurrency, _put
        )

    async def do_download_directory(
        self,
        prefix: str,
        local_destination_directory: str,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        prefixes_to_exclude: Optional[List[str]] = None,
    ) -> AsyncDirectoryDownloadResponse:
        dest_dir = Path(local_destination_directory)
        exclusions = tuple(prefixes_to_exclude) if prefixes_to_exclude else ()
        objects = [
            k
            for k in sorted(self._objects)
            if k.startswith(prefix)
            and not (k.endswith("/") and len(self._objects[k][0]) == 0)
            and not (exclusions and k.startswith(exclusions))
        ]

        async def _get(blob_key: str, _local_path: Path) -> bytes:
            return self._objects[blob_key][0]

        return await self.run_download_tasks(
            objects, prefix, dest_dir, max_concurrency, _get
        )

    async def do_delete_directory(
        self,
        prefix: Optional[str],
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        async def batches() -> AsyncIterator[List[str]]:
            keys = [
                k for k in sorted(self._objects) if not prefix or k.startswith(prefix)
            ]
            for start in range(0, len(keys), _DELETE_BATCH):
                yield keys[start : start + _DELETE_BATCH]

        async def _delete_batch(batch: List[str]) -> List[str]:
            for k in batch:
                self._objects.pop(k, None)
            # The in-memory store never fails a delete, so no failure strings.
            return []

        await self.run_delete_tasks(batches(), max_concurrency, _delete_batch)

    # Multipart hooks. Concrete on the base (they raise
    # UnsupportedOperationException), so they must be overridden to benchmark.

    async def do_initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Mapping[str, str],
        content_type: Optional[str],
        tags: Mapping[str, str],
        correlation_id: str,
    ) -> MultipartUpload:
        self._mpu_counter += 1
        upload_id = f"bench-mpu-{self._mpu_counter}"
        self._multipart[upload_id] = {}
        return MultipartUpload(
            bucket=self._bucket,
            key=key,
            upload_id=upload_id,
            metadata=dict(metadata),
            content_type=content_type,
            tags=dict(tags),
            correlation_id=correlation_id,
        )

    async def do_upload_part(
        self, mpu: MultipartUpload, part: MultipartPart, *, correlation_id: str
    ) -> UploadPartResponse:
        content = part.content
        if not isinstance(content, bytes):
            content = content.read()
        self._multipart[mpu.upload_id][part.part_number] = content
        return UploadPartResponse(
            part_number=part.part_number,
            e_tag=_E_TAG,
            size_in_bytes=len(content),
            correlation_id=correlation_id,
        )

    async def do_complete_multipart_upload(
        self,
        mpu: MultipartUpload,
        parts: List[UploadPartResponse],
        *,
        correlation_id: str,
    ) -> MultipartUploadResponse:
        stored = self._multipart.pop(mpu.upload_id, {})
        # Assemble parts in ascending part-number order into the final object.
        assembled = b"".join(stored[n] for n in sorted(stored))
        self._objects[mpu.key] = (assembled, mpu.content_type, dict(mpu.metadata))
        return MultipartUploadResponse(
            e_tag=_E_TAG, key=mpu.key, correlation_id=correlation_id
        )

    async def do_list_parts(
        self, mpu: MultipartUpload, *, correlation_id: str
    ) -> List[UploadPartResponse]:
        stored = self._multipart.get(mpu.upload_id, {})
        return [
            UploadPartResponse(
                part_number=n,
                e_tag=_E_TAG,
                size_in_bytes=len(stored[n]),
                correlation_id=correlation_id,
            )
            for n in sorted(stored)
        ]

    async def do_abort_multipart_upload(
        self, mpu: MultipartUpload, *, correlation_id: str
    ) -> None:
        self._multipart.pop(mpu.upload_id, None)

    async def close(self) -> None:
        # No sessions or connections to release; the store is pure memory.
        return None
