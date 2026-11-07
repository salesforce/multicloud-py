"""In-memory blob store registered as provider ``"benchfake"``.

A live benchmark mixes SDK code, the provider SDK, network, and the cloud.
If latency jumps, you cannot tell which layer changed. This store exists to
isolate the client → driver path: ``do_*`` is a dict write/read, so no
credentials or network are required, and a slower result here means that
path got slower.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from multicloud.blob.blob_client.driver import (
    BLOB_REGISTRY,
    BaseBlobStore,
    BlobIdentifier,
    BlobInfo,
    BlobMetadata,
    CopyResponse,
    DeleteObjectsResponse,
    DownloadResponse,
    ListBlobsPageResponse,
    MultipartPart,
    MultipartUpload,
    MultipartUploadResponse,
    UploadPartResponse,
    UploadResponse,
)
from multicloud.common.exceptions import SubstrateSdkException, UnknownException

# Stable ETag returned for every stored object. A constant keeps the store
# cheap so the measured cost is the SDK code above this hook, not hashing.
_E_TAG = "bench-etag"

# Stored record: (content bytes, content type, user metadata).
_Record = Tuple[bytes, Optional[str], Dict[str, str]]


@BLOB_REGISTRY.register("benchfake")
class BenchBlobStore(BaseBlobStore):
    """In-memory store for ``benchfake``: dict-backed ``do_*``, no network."""

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

    def do_upload(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: Optional[str],
        content_length: Optional[int],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        object_lock: Any,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
    ) -> UploadResponse:
        self._objects[key] = (payload, content_type, dict(metadata))
        return UploadResponse(key=key, e_tag=_E_TAG)

    def do_download(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
        if_match: Optional[str] = None,
    ) -> DownloadResponse:
        content, content_type, metadata = self._objects[key]
        return DownloadResponse(
            key=key,
            content=content,
            metadata=BlobMetadata(
                key=key,
                content_length=len(content),
                content_type=content_type,
                e_tag=_E_TAG,
                metadata=metadata,
            ),
        )

    def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        self._objects.pop(key, None)

    def do_copy(
        self,
        src_key: str,
        *,
        dest_bucket: str,
        dest_key: str,
        src_version_id: Optional[str],
    ) -> CopyResponse:
        content, content_type, metadata = self._objects[src_key]
        # The in-memory store collapses all buckets to one dict; dest_bucket
        # is accepted for API-shape parity but the copy lands in this store.
        self._objects[dest_key] = (content, content_type, dict(metadata))
        return CopyResponse(key=dest_key, e_tag=_E_TAG)

    def do_list_page(
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

    def do_does_object_exist(self, key: str, *, version_id: Optional[str]) -> bool:
        return key in self._objects

    def do_get_metadata(self, key: str, *, version_id: Optional[str]) -> BlobMetadata:
        content, content_type, metadata = self._objects[key]
        return BlobMetadata(
            key=key,
            content_length=len(content),
            content_type=content_type,
            e_tag=_E_TAG,
            metadata=metadata,
        )

    def do_does_bucket_exist(self) -> bool:
        return True

    def do_delete_objects(self, objects: List[BlobIdentifier]) -> DeleteObjectsResponse:
        for obj in objects:
            self._objects.pop(obj.key, None)
        # The in-memory store never fails a delete, so no per-key errors.
        return DeleteObjectsResponse(errors=[])

    def do_initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Mapping[str, str],
        content_type: Optional[str],
        tags: Mapping[str, str],
        object_lock: Any = None,
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
            object_lock=object_lock,
        )

    def do_upload_part(
        self, mpu: MultipartUpload, part: MultipartPart
    ) -> UploadPartResponse:
        content = part.content
        if not isinstance(content, bytes):
            content = content.read()
        self._multipart[mpu.upload_id][part.part_number] = content
        return UploadPartResponse(
            part_number=part.part_number,
            e_tag=_E_TAG,
            size_in_bytes=len(content),
        )

    def do_complete_multipart_upload(
        self, mpu: MultipartUpload, parts: List[UploadPartResponse]
    ) -> MultipartUploadResponse:
        stored = self._multipart.pop(mpu.upload_id, {})
        # Assemble parts in ascending part-number order into the final object.
        assembled = b"".join(stored[n] for n in sorted(stored))
        self._objects[mpu.key] = (assembled, mpu.content_type, dict(mpu.metadata))
        return MultipartUploadResponse(e_tag=_E_TAG, key=mpu.key)

    def do_list_parts(self, mpu: MultipartUpload) -> List[UploadPartResponse]:
        stored = self._multipart.get(mpu.upload_id, {})
        return [
            UploadPartResponse(
                part_number=n, e_tag=_E_TAG, size_in_bytes=len(stored[n])
            )
            for n in sorted(stored)
        ]

    def do_abort_multipart_upload(self, mpu: MultipartUpload) -> None:
        self._multipart.pop(mpu.upload_id, None)

    # Remaining driver hooks are not exercised by benchmarks. Stub them so the
    # class is concrete without implying support for the operation.
    def do_upload_from_path(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_download_to_path(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_download_to_file(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_download_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_copy_from(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_get_object_lock(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_update_legal_hold(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_get_tags(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_set_tags(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_update_object_retention(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def do_generate_presigned_url(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
