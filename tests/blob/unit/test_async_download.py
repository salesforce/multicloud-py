"""Unit tests for async blob download — verifies AsyncBucketClient download methods.

Tests cover:
- AsyncDownloadResponse and BlobMetadata dataclass models
- BlobNotFoundError exception hierarchy
- download() — in-memory content fetch
- download_to_path() — writes blob to local file
- download_to_file() — writes blob to file-like object
- Range downloads (start, end, both)
- Missing key error propagation
- Integration: upload → download round-trip with content verification
"""

from __future__ import annotations

import asyncio
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest
from multicloud.blob.blob_client.async_blob.async_client import AsyncBucketClient
from multicloud.blob.blob_client.async_blob.async_driver import (
    ASYNC_BLOB_REGISTRY,
    AsyncDownloadRequest,
    AsyncDownloadResponse,
    AsyncUploadResponse,
    BaseAsyncBlobStore,
    BlobAsyncError,
    BlobMetadata,
    BlobNotFoundError,
)
from multicloud.common.exceptions import InvalidArgumentException


class InMemoryAsyncBlobStore(BaseAsyncBlobStore):
    """In-memory test provider for async blob operations.

    Provides a simple dict-based storage backend that mimics cloud blob
    behavior (404s, range requests, metadata) without external dependencies.
    """

    provider_id = "memory-download-test"

    def __init__(self, **kwargs) -> None:
        bucket = kwargs.get("bucket", "test-bucket")
        region = kwargs.get("region")
        endpoint_url = kwargs.get("endpoint_url")
        transfer_config = kwargs.get("transfer_config")
        super().__init__(
            bucket=bucket,
            region=region,
            endpoint_url=endpoint_url,
            transfer_config=transfer_config,
        )
        # key -> (content: bytes, metadata_dict: dict)
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.closed = False

    def translate_error(self, exc: Exception) -> Any:
        from multicloud.common.exceptions import UnknownException

        return UnknownException(str(exc))

    async def do_download(self, request: AsyncDownloadRequest) -> AsyncDownloadResponse:
        """Fetch object from in-memory store; raise BlobNotFoundError on miss."""
        if request.key not in self.objects:
            raise BlobNotFoundError(
                f"Key not found: {request.key}",
                key=request.key,
            )

        content, meta = self.objects[request.key]

        # Honor the conditional-read precondition: a non-matching if_match
        # surfaces the cloud-agnostic FailedPreconditionException, mirroring
        # a provider's 412 → translate_error path.
        if request.if_match is not None and request.if_match != meta.get("e_tag"):
            from multicloud.common.exceptions import FailedPreconditionException

            raise FailedPreconditionException(
                f"if_match precondition failed for {request.key}"
            )

        # Handle range request (inclusive bounds).
        if request.start is not None or request.end is not None:
            start = request.start or 0
            end = (request.end + 1) if request.end is not None else len(content)
            content = content[start:end]

        metadata = BlobMetadata(
            key=request.key,
            content_length=len(content),
            content_type=meta.get("content_type"),
            e_tag=meta.get("e_tag"),
            version_id=request.version_id,
            metadata=meta,
        )
        return AsyncDownloadResponse(
            key=request.key,
            content=content,
            metadata=metadata,
        )

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
        """Store object in-memory."""
        self.objects[key] = (
            payload,
            {
                "content_type": content_type or "application/octet-stream",
                "e_tag": "mock-etag",
            },
        )
        return AsyncUploadResponse(key=key, e_tag="mock-etag")

    async def do_delete(self, key: str, *, version_id=None) -> None:
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
        """Minimal copy stub for download-focused tests."""
        from multicloud.blob.blob_client.async_blob.async_driver import CopyResponse

        return CopyResponse(key=request.destination_key, correlation_id=correlation_id)

    async def do_get_tags(self, key: str) -> Mapping[str, str]:
        return {}

    async def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        pass

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
        """Mark store as closed."""
        self.closed = True


# Register the in-memory provider so AsyncBucketClient can resolve it.
ASYNC_BLOB_REGISTRY.register("memory-download-test")(InMemoryAsyncBlobStore)


@pytest.fixture
def client():
    """AsyncBucketClient with in-memory test provider."""
    return AsyncBucketClient(
        provider="memory-download-test",
        bucket="test-bucket",
    )


@pytest.fixture
def populated_client(client):
    """Client with pre-populated test objects."""
    store = client._store
    store.objects["hello.txt"] = (b"hello world", {"content_type": "text/plain"})
    store.objects["data.bin"] = (
        b"0123456789",
        {"content_type": "application/octet-stream"},
    )
    store.objects["empty.txt"] = (b"", {"content_type": "text/plain"})
    store.objects["doc.pdf"] = (
        b"x" * 1024,
        {"content_type": "application/pdf", "e_tag": "abc123"},
    )
    return client


class TestAsyncDownloadResponse:
    """Tests for :class:`AsyncDownloadResponse` — the return type of async downloads."""

    def test_instantiation_with_required_fields(self):
        """Dataclass can be constructed with all required fields."""
        metadata = BlobMetadata(key="test.txt", content_length=11)
        response = AsyncDownloadResponse(
            key="test.txt",
            content=b"hello world",
            metadata=metadata,
        )
        assert response.key == "test.txt"
        assert response.content == b"hello world"
        assert response.metadata.key == "test.txt"
        assert response.metadata.content_length == 11

    def test_frozen_dataclass_prevents_mutation(self):
        """Frozen dataclass prevents field reassignment after construction."""
        metadata = BlobMetadata(key="test.txt", content_length=5)
        response = AsyncDownloadResponse(
            key="test.txt",
            content=b"hello",
            metadata=metadata,
        )
        with pytest.raises(AttributeError):
            response.key = "modified.txt"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            response.content = b"modified"  # type: ignore[misc]

    def test_metadata_contains_expected_fields(self):
        """Metadata field surfaces content_type, e_tag, version_id."""
        metadata = BlobMetadata(
            key="doc.pdf",
            content_length=1024,
            content_type="application/pdf",
            e_tag="abc123",
            version_id="v2",
        )
        response = AsyncDownloadResponse(
            key="doc.pdf",
            content=b"x" * 1024,
            metadata=metadata,
        )
        assert response.metadata.content_type == "application/pdf"
        assert response.metadata.e_tag == "abc123"
        assert response.metadata.version_id == "v2"


class TestBlobMetadata:
    """Tests for :class:`BlobMetadata` — metadata returned with downloads."""

    def test_default_metadata_mapping_is_empty(self):
        """Default metadata field is an empty dict."""
        meta = BlobMetadata(key="test.txt")
        assert meta.metadata == {}

    def test_all_fields_optional_except_key(self):
        """Only key is required; all other fields default to None/empty."""
        meta = BlobMetadata(key="minimal.txt")
        assert meta.key == "minimal.txt"
        assert meta.content_length is None
        assert meta.content_type is None
        assert meta.e_tag is None
        assert meta.version_id is None
        assert meta.metadata == {}


class TestBlobNotFoundError:
    """Tests for :class:`BlobNotFoundError` — 404 error specialization."""

    def test_subclass_of_blob_async_error(self):
        """BlobNotFoundError is a BlobAsyncError subclass."""
        assert issubclass(BlobNotFoundError, BlobAsyncError)

    def test_subclass_of_substrate_sdk_exception(self):
        """BlobNotFoundError is a SubstrateSdkException subclass (transitive)."""
        from multicloud.common.exceptions import SubstrateSdkException

        assert issubclass(BlobNotFoundError, SubstrateSdkException)

    def test_instantiation_with_key(self):
        """Exception can be raised with a key for debugging."""
        err = BlobNotFoundError("Object missing", key="missing.txt")
        assert "Object missing" in str(err)
        assert err.key == "missing.txt"

    def test_cause_chaining(self):
        """Exception supports cause chaining."""
        root = IOError("Connection reset")
        err = BlobNotFoundError("Not found", key="k", cause=root)
        assert err.cause is root

    def test_raised_and_caught_as_blob_async_error(self):
        """BlobNotFoundError can be caught as BlobAsyncError."""
        with pytest.raises(BlobAsyncError):
            raise BlobNotFoundError("Not found", key="gone.txt")


class TestAsyncDownloadRequest:
    """Tests for :class:`AsyncDownloadRequest` — download parameters."""

    def test_required_key_field(self):
        """Key is the only required field."""
        req = AsyncDownloadRequest(key="test.txt")
        assert req.key == "test.txt"
        assert req.version_id is None
        assert req.start is None
        assert req.end is None
        assert req.if_match is None

    def test_all_fields(self):
        """All fields can be set."""
        req = AsyncDownloadRequest(
            key="file.bin",
            version_id="v2",
            start=100,
            end=200,
            if_match="etag-9",
        )
        assert req.key == "file.bin"
        assert req.version_id == "v2"
        assert req.start == 100
        assert req.end == 200
        assert req.if_match == "etag-9"

    def test_frozen(self):
        """Request is immutable."""
        req = AsyncDownloadRequest(key="test.txt")
        with pytest.raises(AttributeError):
            req.key = "changed"  # type: ignore[misc]


class TestAsyncBucketClientDownload:
    """Tests for ``AsyncBucketClient.download()`` — in-memory content fetch."""

    @pytest.mark.asyncio
    async def test_download_returns_content_in_memory(self, populated_client):
        """Basic download fetches content as bytes."""
        response = await populated_client.download("hello.txt")

        assert response.key == "hello.txt"
        assert response.content == b"hello world"
        assert response.metadata.content_length == 11
        assert response.metadata.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_download_with_version_id(self, populated_client):
        """Version ID is passed through to provider."""
        response = await populated_client.download("doc.pdf", version_id="v1")
        assert response.metadata.version_id == "v1"

    @pytest.mark.asyncio
    async def test_download_with_range_start_and_end(self, populated_client):
        """Range download (start/end) extracts a slice."""
        response = await populated_client.download("data.bin", start=2, end=5)
        assert response.content == b"2345"
        assert response.metadata.content_length == 4

    @pytest.mark.asyncio
    async def test_download_with_start_only(self, populated_client):
        """Range with start only fetches from start to end."""
        response = await populated_client.download("data.bin", start=7)
        assert response.content == b"789"

    @pytest.mark.asyncio
    async def test_download_with_end_only(self, populated_client):
        """Range with end only fetches from beginning to end (inclusive)."""
        response = await populated_client.download("data.bin", end=3)
        assert response.content == b"0123"

    @pytest.mark.asyncio
    async def test_download_empty_blob(self, populated_client):
        """Download of empty blob returns empty bytes."""
        response = await populated_client.download("empty.txt")
        assert response.content == b""
        assert response.metadata.content_length == 0

    @pytest.mark.asyncio
    async def test_download_missing_key_raises_blob_not_found(self, client):
        """Missing key raises BlobNotFoundError."""
        with pytest.raises(BlobNotFoundError) as exc_info:
            await client.download("missing.txt")

        assert "missing.txt" in str(exc_info.value)
        assert exc_info.value.key == "missing.txt"

    @pytest.mark.asyncio
    async def test_download_empty_key_raises_invalid_argument(self, client):
        """Empty key raises InvalidArgumentException."""
        with pytest.raises(InvalidArgumentException):
            await client.download("")

    @pytest.mark.asyncio
    async def test_download_negative_start_raises_invalid_argument(self, client):
        """Negative start raises InvalidArgumentException."""
        with pytest.raises(
            InvalidArgumentException, match="boundaries cannot be negative"
        ):
            await client.download("file.txt", start=-1)

    @pytest.mark.asyncio
    async def test_download_negative_end_raises_invalid_argument(self, client):
        """Negative end raises InvalidArgumentException."""
        with pytest.raises(
            InvalidArgumentException, match="boundaries cannot be negative"
        ):
            await client.download("file.txt", end=-1)

    @pytest.mark.asyncio
    async def test_download_start_greater_than_end_raises_invalid_argument(
        self, client
    ):
        """start > end raises InvalidArgumentException."""
        with pytest.raises(
            InvalidArgumentException, match="start cannot be larger than end"
        ):
            await client.download("file.txt", start=100, end=50)

    @pytest.mark.asyncio
    async def test_download_if_match_matching_etag_returns_content(
        self, populated_client
    ):
        """A matching if_match precondition returns the object content."""
        response = await populated_client.download("doc.pdf", if_match="abc123")
        assert response.content == b"x" * 1024
        assert response.metadata.e_tag == "abc123"

    @pytest.mark.asyncio
    async def test_download_if_match_mismatch_raises_failed_precondition(
        self, populated_client
    ):
        """A non-matching if_match raises FailedPreconditionException."""
        from multicloud.common.exceptions import FailedPreconditionException

        with pytest.raises(FailedPreconditionException):
            await populated_client.download("doc.pdf", if_match="wrong-etag")

    @pytest.mark.asyncio
    async def test_download_empty_if_match_raises_invalid_argument(self, client):
        """Empty/whitespace if_match is a validation error, not a skipped precondition."""
        with pytest.raises(InvalidArgumentException):
            await client.download("file.txt", if_match="   ")


class TestAsyncBucketClientDownloadToPath:
    """Tests for ``download_to_path()`` — writes to filesystem."""

    @pytest.mark.asyncio
    async def test_download_to_path_writes_file(self, populated_client):
        """Content is written to destination file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "test.txt"
            response = await populated_client.download_to_path(
                "hello.txt", destination=dest
            )

            assert dest.exists()
            assert dest.read_bytes() == b"hello world"
            assert response.key == "hello.txt"
            assert response.metadata.content_length == 11

    @pytest.mark.asyncio
    async def test_download_to_path_creates_parent_dirs(self, populated_client):
        """Parent directories are created when create_parent_dirs=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "a" / "b" / "c" / "nested.txt"
            assert not dest.parent.exists()

            await populated_client.download_to_path(
                "hello.txt",
                destination=dest,
                create_parent_dirs=True,
            )

            assert dest.exists()
            assert dest.read_bytes() == b"hello world"

    @pytest.mark.asyncio
    async def test_download_to_path_missing_key_raises(self, client):
        """Missing key raises BlobNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "missing.txt"
            with pytest.raises(BlobNotFoundError):
                await client.download_to_path("missing.txt", destination=dest)

    @pytest.mark.asyncio
    async def test_download_to_path_string_destination(self, populated_client):
        """Destination can be a string, not just Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_str = str(Path(tmpdir) / "str-dest.txt")
            await populated_client.download_to_path("hello.txt", destination=dest_str)
            assert Path(dest_str).read_bytes() == b"hello world"

    @pytest.mark.asyncio
    async def test_download_to_path_overwrites_existing_file(self, populated_client):
        """Existing file is overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "overwrite.txt"
            dest.write_bytes(b"old content")

            await populated_client.download_to_path("hello.txt", destination=dest)
            assert dest.read_bytes() == b"hello world"

    @pytest.mark.asyncio
    async def test_download_to_path_with_range(self, populated_client):
        """Range download writes partial content to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "partial.bin"
            await populated_client.download_to_path(
                "data.bin", destination=dest, start=0, end=4
            )
            assert dest.read_bytes() == b"01234"

    @pytest.mark.asyncio
    async def test_download_to_path_io_error_wrapped(self, populated_client):
        """File I/O errors are wrapped in BlobAsyncError."""
        # Write to a path where parent doesn't exist and
        # create_parent_dirs=False (default)
        dest = Path("/nonexistent-dir-abc123/file.txt")
        with pytest.raises(BlobAsyncError, match="Failed to write file"):
            await populated_client.download_to_path("hello.txt", destination=dest)


class TestAsyncBucketClientDownloadToFile:
    """Tests for ``download_to_file()`` — writes to file-like object."""

    @pytest.mark.asyncio
    async def test_download_to_file_writes_to_bytesio(self, populated_client):
        """Content is written to BytesIO target."""
        buffer = BytesIO()
        response = await populated_client.download_to_file("hello.txt", file_obj=buffer)

        assert buffer.getvalue() == b"hello world"
        assert response.key == "hello.txt"
        assert response.metadata.content_length == 11

    @pytest.mark.asyncio
    async def test_download_to_file_missing_key_raises(self, client):
        """Missing key raises BlobNotFoundError."""
        buffer = BytesIO()
        with pytest.raises(BlobNotFoundError):
            await client.download_to_file("missing.txt", file_obj=buffer)

    @pytest.mark.asyncio
    async def test_download_to_file_with_real_file(self, populated_client):
        """Content can be written to a real file handle."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            await populated_client.download_to_file("hello.txt", file_obj=tmp)

        try:
            assert tmp_path.read_bytes() == b"hello world"
        finally:
            tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_download_to_file_range_request(self, populated_client):
        """Range download writes partial content to file-like object."""
        buffer = BytesIO()
        await populated_client.download_to_file(
            "data.bin", file_obj=buffer, start=3, end=6
        )
        assert buffer.getvalue() == b"3456"

    @pytest.mark.asyncio
    async def test_download_to_file_empty_blob(self, populated_client):
        """Empty blob writes zero bytes to file object."""
        buffer = BytesIO()
        response = await populated_client.download_to_file("empty.txt", file_obj=buffer)
        assert buffer.getvalue() == b""
        assert response.metadata.content_length == 0


class TestAsyncDownloadIntegration:
    """Integration tests covering upload → download workflows."""

    @pytest.mark.asyncio
    async def test_upload_download_roundtrip_bytes(self, client):
        """Upload then download verifies content integrity (bytes)."""
        payload = b"roundtrip content verification"
        await client.upload("roundtrip.txt", payload)

        response = await client.download("roundtrip.txt")
        assert response.content == payload

    @pytest.mark.asyncio
    async def test_upload_download_roundtrip_to_path(self, client):
        """Upload then download_to_path verifies file content."""
        payload = b"file roundtrip data"
        await client.upload("path-rt.txt", payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "downloaded.txt"
            await client.download_to_path("path-rt.txt", destination=dest)
            assert dest.read_bytes() == payload

    @pytest.mark.asyncio
    async def test_upload_download_roundtrip_to_file(self, client):
        """Upload then download_to_file verifies stream content."""
        payload = b"stream roundtrip data"
        await client.upload("file-rt.txt", payload)

        buffer = BytesIO()
        await client.download_to_file("file-rt.txt", file_obj=buffer)
        assert buffer.getvalue() == payload

    @pytest.mark.asyncio
    async def test_multiple_uploads_and_downloads(self, client):
        """Multiple sequential operations in a single client session."""
        async with AsyncBucketClient(
            provider="memory-download-test", bucket="test-bucket"
        ) as c:
            await c.upload("file1.txt", b"content1")
            await c.upload("file2.txt", b"content2")
            await c.upload("file3.txt", b"content3")

            r1 = await c.download("file1.txt")
            r2 = await c.download("file2.txt")
            r3 = await c.download("file3.txt")

            assert r1.content == b"content1"
            assert r2.content == b"content2"
            assert r3.content == b"content3"

    @pytest.mark.asyncio
    async def test_concurrent_downloads(self, client):
        """Concurrent downloads via asyncio.gather work correctly."""
        # Upload test data
        for i in range(5):
            await client.upload(f"concurrent-{i}.txt", f"data-{i}".encode())

        # Download concurrently
        results = await asyncio.gather(
            *[client.download(f"concurrent-{i}.txt") for i in range(5)]
        )

        for i, response in enumerate(results):
            assert response.content == f"data-{i}".encode()

    @pytest.mark.asyncio
    async def test_download_after_overwrite(self, client):
        """Download returns latest content after overwrite."""
        await client.upload("mutable.txt", b"version 1")
        await client.upload("mutable.txt", b"version 2")

        response = await client.download("mutable.txt")
        assert response.content == b"version 2"
