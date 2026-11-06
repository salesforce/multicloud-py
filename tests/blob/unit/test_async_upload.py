"""Unit tests for the async blob upload feature.

Tests cover:
- AsyncBucketClient upload with bytes, Path, and BinaryIO
- BaseAsyncBlobStore validation and payload normalization
- Content-type auto-detection
- Metadata passing
- Error handling with BlobAsyncError
- AWS provider (AwsAsyncBlobStore) with mocked aiobotocore
- GCP provider (GcpAsyncBlobStore) with mocked gcloud-aio-storage
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from multicloud.blob.blob_client.async_blob.async_client import AsyncBucketClient
from multicloud.blob.blob_client.async_blob.async_driver import (
    ASYNC_BLOB_REGISTRY,
    AsyncTransferConfig,
    AsyncUploadResponse,
    BaseAsyncBlobStore,
    BlobAsyncError,
)
from multicloud.common.exceptions import (
    InvalidArgumentException,
    SubstrateSdkException,
    UnknownException,
)


class InMemoryAsyncBlobStore(BaseAsyncBlobStore):
    """A minimal in-memory async blob store for testing the base class."""

    provider_id = "inmemory"

    def __init__(self, bucket: str = "test-bucket", **kwargs):
        super().__init__(bucket=bucket, **kwargs)
        self.stored: dict[str, tuple[bytes, dict]] = {}
        self.upload_calls: list[dict] = []
        self.tags: dict[str, dict] = {}
        self._should_fail: Optional[Exception] = None

    def set_failure(self, exc: Exception) -> None:
        """Configure the next upload to raise the given exception."""
        self._should_fail = exc

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
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
        storage_class: Optional[str] = None,
    ) -> "AsyncUploadResponse":
        if self._should_fail:
            exc = self._should_fail
            self._should_fail = None
            if isinstance(exc, SubstrateSdkException):
                raise exc
            raise BlobAsyncError(
                str(exc),
                cause=exc,
            ) from exc

        self.stored[key] = (payload, dict(metadata))
        self.upload_calls.append(
            {
                "key": key,
                "content_type": content_type,
                "content_length": content_length,
                "metadata": dict(metadata),
                "create_if_absent": create_if_absent,
                "if_match": if_match,
                "if_version_match": if_version_match,
                "storage_class": storage_class,
            }
        )
        return AsyncUploadResponse(
            key=key,
            e_tag="mock-etag-123",
            version_id="v1",
        )

    async def do_download(self, request):
        """Minimal download stub for upload-focused tests."""
        from multicloud.blob.blob_client.async_blob.async_driver import (
            BlobNotFoundError,
        )

        raise BlobNotFoundError(f"Not found: {request.key}", key=request.key)

    async def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        pass

    async def do_delete_objects(self, identifiers):
        from multicloud.blob.blob_client.async_blob.async_driver import (
            BulkDeleteResponse,
            DeletedBlob,
        )

        return BulkDeleteResponse(
            deleted=[
                DeletedBlob(key=i.key, version_id=i.version_id) for i in identifiers
            ],
            errors=[],
        )

    async def do_copy(self, request, *, correlation_id):
        """Minimal copy stub for upload-focused tests."""
        from multicloud.blob.blob_client.async_blob.async_driver import CopyResponse

        return CopyResponse(key=request.destination_key, correlation_id=correlation_id)

    async def do_get_tags(self, key: str) -> Mapping[str, str]:
        return dict(self.tags.get(key, {}))

    async def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        self.tags[key] = dict(tags)

    async def do_upload_directory(self, local_source_directory, **kwargs):
        raise NotImplementedError

    async def do_download_directory(
        self, prefix, local_destination_directory, **kwargs
    ):
        raise NotImplementedError

    async def do_delete_directory(self, prefix, **kwargs):
        raise NotImplementedError

    async def do_does_object_exist(
        self, key: str, *, version_id: Optional[str] = None
    ) -> bool:
        return True

    async def do_get_metadata(self, key, *, version_id=None):
        from multicloud.blob.blob_client.async_blob.async_driver import BlobMetadata

        return BlobMetadata(key=key)

    async def do_does_bucket_exist(self) -> bool:
        return True

    async def do_list_page(
        self, *, prefix=None, delimiter=None, max_results=1000, pagination_token=None
    ):
        from multicloud.blob.blob_client.async_blob.async_driver import (
            ListBlobsPageResponse,
        )

        return ListBlobsPageResponse()

    async def close(self) -> None:
        pass


@pytest.fixture
def store():
    """A fresh in-memory async blob store."""
    return InMemoryAsyncBlobStore()


@pytest.fixture
def tmp_file(tmp_path):
    """A temporary file with known content for Path uploads."""
    f = tmp_path / "test-upload.txt"
    f.write_text("hello world from file")
    return f


@pytest.fixture
def tmp_json_file(tmp_path):
    """A temporary JSON file for content-type detection testing."""
    f = tmp_path / "data.json"
    f.write_text('{"key": "value"}')
    return f


class TestUploadBytes:
    """Upload with bytes payload."""

    @pytest.mark.asyncio
    async def test_upload_bytes_success(self, store):
        result = await store.upload("hello.txt", b"hello world")

        assert result.key == "hello.txt"
        assert result.e_tag == "mock-etag-123"
        assert result.version_id == "v1"

    @pytest.mark.asyncio
    async def test_upload_bytes_stores_content(self, store):
        await store.upload("key1", b"content-data")

        assert store.stored["key1"][0] == b"content-data"

    @pytest.mark.asyncio
    async def test_upload_bytes_default_content_type(self, store):
        await store.upload("key1", b"data")

        assert store.upload_calls[0]["content_type"] == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_upload_bytes_explicit_content_type(self, store):
        await store.upload("key1", b"data", content_type="text/plain")

        assert store.upload_calls[0]["content_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_upload_bytes_calculates_content_length(self, store):
        await store.upload("key1", b"12345")

        assert store.upload_calls[0]["content_length"] == 5

    @pytest.mark.asyncio
    async def test_upload_bytes_explicit_content_length(self, store):
        await store.upload("key1", b"12345", content_length=99)

        assert store.upload_calls[0]["content_length"] == 99

    @pytest.mark.asyncio
    async def test_upload_bytes_with_metadata(self, store):
        meta = {"author": "test", "version": "1.0"}
        await store.upload("key1", b"data", metadata=meta)

        assert store.upload_calls[0]["metadata"] == meta

    @pytest.mark.asyncio
    async def test_upload_bytes_no_metadata_is_empty_dict(self, store):
        await store.upload("key1", b"data")

        assert store.upload_calls[0]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_upload_bytearray(self, store):
        result = await store.upload("key1", bytearray(b"bytearray-data"))

        assert result.key == "key1"
        assert store.stored["key1"][0] == b"bytearray-data"

    @pytest.mark.asyncio
    async def test_upload_memoryview(self, store):
        data = b"memoryview-data"
        result = await store.upload("key1", memoryview(data))

        assert result.key == "key1"
        assert store.stored["key1"][0] == b"memoryview-data"


class TestUploadPath:
    """Upload with Path payload (local file)."""

    @pytest.mark.asyncio
    async def test_upload_path_reads_file(self, store, tmp_file):
        result = await store.upload("remote/file.txt", tmp_file)

        assert result.key == "remote/file.txt"
        assert store.stored["remote/file.txt"][0] == b"hello world from file"

    @pytest.mark.asyncio
    async def test_upload_path_auto_detects_content_type_txt(self, store, tmp_file):
        await store.upload("file.txt", tmp_file)

        assert store.upload_calls[0]["content_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_upload_path_auto_detects_content_type_json(
        self, store, tmp_json_file
    ):
        await store.upload("data.json", tmp_json_file)

        assert store.upload_calls[0]["content_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_upload_path_explicit_content_type_overrides(self, store, tmp_file):
        await store.upload("file.txt", tmp_file, content_type="application/xml")

        assert store.upload_calls[0]["content_type"] == "application/xml"

    @pytest.mark.asyncio
    async def test_upload_path_calculates_content_length(self, store, tmp_file):
        await store.upload("file.txt", tmp_file)

        # "hello world from file" = 21 bytes
        assert store.upload_calls[0]["content_length"] == 21


class TestUploadStream:
    """Upload with BinaryIO payload (file-like stream)."""

    @pytest.mark.asyncio
    async def test_upload_stream_reads_all(self, store):
        stream = io.BytesIO(b"stream-content")
        result = await store.upload("stream.bin", stream)

        assert result.key == "stream.bin"
        assert store.stored["stream.bin"][0] == b"stream-content"

    @pytest.mark.asyncio
    async def test_upload_stream_default_content_type(self, store):
        stream = io.BytesIO(b"data")
        await store.upload("file.bin", stream)

        assert store.upload_calls[0]["content_type"] == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_upload_stream_explicit_content_type(self, store):
        stream = io.BytesIO(b"data")
        await store.upload("file.bin", stream, content_type="image/png")

        assert store.upload_calls[0]["content_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_upload_text_stream_raises(self, store):
        """A text-mode stream should raise InvalidArgumentException."""
        stream = io.StringIO("text data")
        with pytest.raises(InvalidArgumentException, match="binary"):
            await store.upload("key", stream)


class TestUploadValidation:
    """Input validation for upload."""

    @pytest.mark.asyncio
    async def test_empty_key_raises(self, store):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            await store.upload("", b"data")

    @pytest.mark.asyncio
    async def test_negative_content_length_raises(self, store):
        with pytest.raises(
            InvalidArgumentException, match="content_length must be a non-negative"
        ):
            await store.upload("key.txt", b"data", content_length=-1)

    @pytest.mark.asyncio
    async def test_unsupported_payload_type_raises(self, store):
        with pytest.raises(
            InvalidArgumentException, match="Unsupported upload payload"
        ):
            await store.upload("key", 12345)  # type: ignore

    @pytest.mark.asyncio
    async def test_none_payload_raises(self, store):
        with pytest.raises(
            InvalidArgumentException, match="Unsupported upload payload"
        ):
            await store.upload("key", None)  # type: ignore

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank", ["", "   "])
    async def test_blank_storage_class_raises_before_delegating(self, store, blank):
        with pytest.raises(InvalidArgumentException, match="storage_class"):
            await store.upload("key", b"data", storage_class=blank)
        assert store.upload_calls == []

    @pytest.mark.asyncio
    async def test_storage_class_forwarded(self, store):
        await store.upload("key", b"data", storage_class="NEARLINE")
        assert store.upload_calls[0]["storage_class"] == "NEARLINE"

    @pytest.mark.asyncio
    async def test_storage_class_defaults_to_none(self, store):
        await store.upload("key", b"data")
        assert store.upload_calls[0]["storage_class"] is None

    def test_empty_bucket_raises_invalid_argument(self):
        """Empty bucket string raises InvalidArgumentException."""
        with pytest.raises(
            InvalidArgumentException, match="Bucket name cannot be null or empty"
        ):
            InMemoryAsyncBlobStore(bucket="")


class TestBaseAsyncBlobStoreProperties:
    """Property accessors on BaseAsyncBlobStore."""

    def test_region_property_returns_region(self):
        store = InMemoryAsyncBlobStore(bucket="b", region="eu-west-1")
        assert store.region == "eu-west-1"

    def test_region_property_returns_none_when_unset(self):
        store = InMemoryAsyncBlobStore(bucket="b")
        assert store.region is None

    def test_transfer_config_property_returns_config(self):
        cfg = AsyncTransferConfig(max_concurrency=7)
        store = InMemoryAsyncBlobStore(bucket="b", transfer_config=cfg)
        assert store.transfer_config is cfg
        assert store.transfer_config.max_concurrency == 7

    def test_transfer_config_defaults_when_not_provided(self):
        store = InMemoryAsyncBlobStore(bucket="b")
        assert store.transfer_config is not None
        assert store.transfer_config.max_concurrency == 50


class TestUploadErrorHandling:
    """Error handling and BlobAsyncError wrapping."""

    @pytest.mark.asyncio
    async def test_transport_error_raises_blob_async_error(self, store):
        store.set_failure(RuntimeError("connection reset"))

        with pytest.raises(BlobAsyncError) as exc_info:
            await store.upload("key", b"data")

        err = exc_info.value
        assert "connection reset" in str(err)
        assert err.cause is not None

    @pytest.mark.asyncio
    async def test_sdk_exception_reraises_directly(self, store):
        """SubstrateSdkException subclasses pass through without wrapping."""
        from multicloud.common.exceptions import ResourceNotFoundException

        store.set_failure(ResourceNotFoundException("not found"))

        with pytest.raises(ResourceNotFoundException):
            await store.upload("key", b"data")


class TestAsyncTransferConfig:
    """Transfer configuration defaults and customization."""

    def test_default_values(self):
        cfg = AsyncTransferConfig()
        assert cfg.max_concurrency == 50
        assert cfg.multipart_threshold == 8 * 1024 * 1024
        assert cfg.multipart_chunksize == 8 * 1024 * 1024
        assert cfg.download_chunk_size == 1024 * 1024

    def test_custom_values(self):
        cfg = AsyncTransferConfig(max_concurrency=10, multipart_threshold=1024)
        assert cfg.max_concurrency == 10
        assert cfg.multipart_threshold == 1024

    def test_is_frozen(self):
        cfg = AsyncTransferConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.max_concurrency = 99  # type: ignore


class TestContextManager:
    """Async context manager lifecycle."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, store):
        async with store as s:
            result = await s.upload("key", b"data")
            assert result.key == "key"

    @pytest.mark.asyncio
    async def test_close_is_called_on_exit(self):
        store = InMemoryAsyncBlobStore()
        store.close = AsyncMock()  # type: ignore

        async with store:
            pass

        store.close.assert_called_once()


class TestTagsDriver:
    """Driver-layer validation and delegation for get_tags / set_tags."""

    @pytest.mark.asyncio
    async def test_set_then_get_tags_round_trip(self, store):
        await store.set_tags("key", {"env": "prod", "team": "multicloud"})
        result = await store.get_tags("key")

        assert result == {"env": "prod", "team": "multicloud"}

    @pytest.mark.asyncio
    async def test_get_tags_empty_key_raises(self, store):
        with pytest.raises(InvalidArgumentException, match="Object name cannot"):
            await store.get_tags("")

    @pytest.mark.asyncio
    async def test_set_tags_empty_key_raises(self, store):
        with pytest.raises(InvalidArgumentException, match="Object name cannot"):
            await store.set_tags("", {"env": "prod"})

    @pytest.mark.asyncio
    async def test_set_tags_empty_tags_raises(self, store):
        with pytest.raises(InvalidArgumentException, match="Map of tags cannot"):
            await store.set_tags("key", {})

    @pytest.mark.asyncio
    async def test_get_tags_empty_when_untagged(self, store):
        assert await store.get_tags("never-tagged") == {}
