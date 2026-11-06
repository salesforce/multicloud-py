"""Unit tests for AsyncBucketClient — the user-facing async API.

Tests validate the client:
- Resolves providers from the async registry
- Delegates upload calls to the underlying store
- Supports all upload payload types through to the store
- Provides async context manager lifecycle
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Mapping, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from multicloud.blob.blob_client.async_blob.async_client import AsyncBucketClient
from multicloud.blob.blob_client.async_blob.async_driver import (
    ASYNC_BLOB_REGISTRY,
    AsyncUploadResponse,
    BaseAsyncBlobStore,
    BlobAsyncError,
)
from multicloud.common.exceptions import SubstrateSdkException, UnknownException


class FakeAsyncBlobStore(BaseAsyncBlobStore):
    """Fake provider registered for testing AsyncBucketClient."""

    provider_id = "fake-test"

    def __init__(self, **kwargs):
        # Accept all kwargs the client might pass
        bucket = kwargs.get("bucket", "test")
        region = kwargs.get("region")
        endpoint_url = kwargs.get("endpoint_url")
        transfer_config = kwargs.get("transfer_config")
        credentials_overrider = kwargs.get("credentials_overrider")
        super().__init__(
            bucket=bucket,
            region=region,
            endpoint_url=endpoint_url,
            transfer_config=transfer_config,
            credentials_overrider=credentials_overrider,
        )
        self.uploads: list[dict] = []
        self.closed = False
        self.tags: dict[str, dict] = {}
        self.set_tags_calls: list[dict] = []
        self.get_metadata_calls: list[dict] = []
        self.list_page_calls: list[dict] = []

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        if isinstance(exc, SubstrateSdkException):
            return exc
        return UnknownException(str(exc))

    async def do_download(self, request):
        """Minimal download implementation for upload-focused tests."""
        from multicloud.blob.blob_client.async_blob.async_driver import (
            AsyncDownloadResponse,
            BlobMetadata,
            BlobNotFoundError,
        )

        raise BlobNotFoundError(f"Not found: {request.key}", key=request.key)

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
        self.uploads.append(
            {
                "key": key,
                "payload": payload,
                "content_type": content_type,
                "content_length": content_length,
                "metadata": dict(metadata),
                "create_if_absent": create_if_absent,
                "if_match": if_match,
                "if_version_match": if_version_match,
                "kms_key_id": kms_key_id,
                "use_kms_managed_key": use_kms_managed_key,
                "storage_class": storage_class,
            }
        )
        return AsyncUploadResponse(
            key=key,
            e_tag="fake-etag",
            version_id="fake-v1",
        )

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
        """Minimal copy stub for client-focused tests."""
        from multicloud.blob.blob_client.async_blob.async_driver import CopyResponse

        return CopyResponse(key=request.destination_key, correlation_id=correlation_id)

    async def do_get_tags(self, key: str) -> Mapping[str, str]:
        return dict(self.tags.get(key, {}))

    async def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        self.set_tags_calls.append({"key": key, "tags": dict(tags)})
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

        self.get_metadata_calls.append({"key": key, "version_id": version_id})
        return BlobMetadata(
            key=key,
            content_length=10,
            content_type="text/plain",
            e_tag="fake-etag",
            version_id=version_id,
        )

    async def do_does_bucket_exist(self) -> bool:
        return True

    async def do_list_page(
        self, *, prefix=None, delimiter=None, max_results=1000, pagination_token=None
    ):
        from multicloud.blob.blob_client.async_blob.async_driver import (
            ListBlobsPageResponse,
        )

        self.list_page_calls.append(
            {
                "prefix": prefix,
                "delimiter": delimiter,
                "max_results": max_results,
                "pagination_token": pagination_token,
            }
        )
        return ListBlobsPageResponse()

    async def close(self) -> None:
        self.closed = True


# Register the fake provider for tests
ASYNC_BLOB_REGISTRY.register("fake-test")(FakeAsyncBlobStore)


@pytest.fixture
def client():
    """AsyncBucketClient using the fake-test provider."""
    return AsyncBucketClient(
        provider="fake-test",
        bucket="my-bucket",
        region="us-east-1",
    )


@pytest.fixture
def tmp_file(tmp_path):
    """Temporary file for Path upload testing."""
    f = tmp_path / "upload-me.txt"
    f.write_text("file content here")
    return f


class TestAsyncBucketClientConstruction:
    """Client construction and property access."""

    def test_bucket_property(self, client):
        assert client.bucket == "my-bucket"

    def test_provider_property(self, client):
        assert client.provider == "fake-test"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="No provider registered"):
            AsyncBucketClient(provider="nonexistent", bucket="b")

    def test_endpoint_passed_to_store(self):
        c = AsyncBucketClient(
            provider="fake-test",
            bucket="b",
            endpoint="http://localhost:9000",
        )
        assert c._store._endpoint_url == "http://localhost:9000"

    def test_transfer_config_passed_to_store(self):
        from multicloud.blob.blob_client.async_blob.async_driver import (
            AsyncTransferConfig,
        )

        cfg = AsyncTransferConfig(max_concurrency=10)
        c = AsyncBucketClient(
            provider="fake-test",
            bucket="b",
            transfer_config=cfg,
        )
        assert c._store._transfer_config is cfg
        assert c._store._transfer_config.max_concurrency == 10

    def test_credentials_overrider_passed_to_store(self):
        from multicloud.common.credentials import (
            CredentialsOverrider,
            CredentialsType,
            StsCredentials,
        )

        overrider = CredentialsOverrider(
            credentials_type=CredentialsType.SESSION,
            session_credentials=StsCredentials(
                access_key_id="AKID",
                access_key_secret="SECRET",
                security_token="TOKEN",
            ),
        )
        c = AsyncBucketClient(
            provider="fake-test",
            bucket="b",
            credentials_overrider=overrider,
        )
        assert c._store._credentials_overrider is overrider

    def test_credentials_overrider_defaults_to_none(self):
        c = AsyncBucketClient(provider="fake-test", bucket="b")
        assert c._store._credentials_overrider is None


class TestAsyncBucketClientUpload:
    """Upload method delegation to the store."""

    @pytest.mark.asyncio
    async def test_upload_bytes(self, client):
        result = await client.upload("key.txt", b"hello")

        assert result.key == "key.txt"
        assert result.e_tag == "fake-etag"
        assert result.version_id == "fake-v1"

    @pytest.mark.asyncio
    async def test_upload_path(self, client, tmp_file):
        result = await client.upload("remote.txt", tmp_file)

        assert result.key == "remote.txt"
        store = client._store
        assert store.uploads[0]["payload"] == b"file content here"

    @pytest.mark.asyncio
    async def test_upload_stream(self, client):
        stream = io.BytesIO(b"stream-bytes")
        result = await client.upload("stream.bin", stream)

        assert result.key == "stream.bin"
        store = client._store
        assert store.uploads[0]["payload"] == b"stream-bytes"

    @pytest.mark.asyncio
    async def test_upload_with_content_type(self, client):
        await client.upload("key", b"data", content_type="text/csv")

        store = client._store
        assert store.uploads[0]["content_type"] == "text/csv"

    @pytest.mark.asyncio
    async def test_upload_with_metadata(self, client):
        meta = {"team": "multicloud", "env": "dev"}
        await client.upload("key", b"data", metadata=meta)

        store = client._store
        assert store.uploads[0]["metadata"] == meta

    @pytest.mark.asyncio
    async def test_upload_with_content_length(self, client):
        await client.upload("key", b"12345", content_length=5)

        store = client._store
        assert store.uploads[0]["content_length"] == 5

    @pytest.mark.asyncio
    async def test_upload_forwards_storage_class(self, client):
        await client.upload("k", b"data", storage_class="STANDARD_IA")
        store = client._store
        assert store.uploads[0]["storage_class"] == "STANDARD_IA"

    @pytest.mark.asyncio
    async def test_upload_forwards_preconditions_and_kms(self, client):
        await client.upload(
            "key",
            b"data",
            create_if_absent=True,
            kms_key_id="arn:key/abc",
            use_kms_managed_key=True,
        )

        store = client._store
        assert store.uploads[0]["create_if_absent"] is True
        assert store.uploads[0]["kms_key_id"] == "arn:key/abc"
        assert store.uploads[0]["use_kms_managed_key"] is True

    @pytest.mark.asyncio
    async def test_upload_forwards_if_match(self, client):
        await client.upload("key", b"data", if_match="etag-abc")

        store = client._store
        assert store.uploads[0]["if_match"] == "etag-abc"


class TestAsyncBucketClientTags:
    """get_tags / set_tags delegation to the store."""

    @pytest.mark.asyncio
    async def test_set_then_get_tags_round_trip(self, client):
        await client.set_tags("key.txt", {"env": "prod", "team": "multicloud"})
        result = await client.get_tags("key.txt")

        assert result == {"env": "prod", "team": "multicloud"}

    @pytest.mark.asyncio
    async def test_set_tags_delegates_to_store(self, client):
        await client.set_tags("key.txt", {"env": "dev"})

        store = client._store
        assert store.set_tags_calls[0] == {"key": "key.txt", "tags": {"env": "dev"}}

    @pytest.mark.asyncio
    async def test_get_tags_empty_when_no_tags(self, client):
        result = await client.get_tags("untagged")

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_tags_empty_key_raises(self, client):
        from multicloud.common.exceptions import InvalidArgumentException

        with pytest.raises(InvalidArgumentException):
            await client.get_tags("")

    @pytest.mark.asyncio
    async def test_set_tags_empty_tags_raises(self, client):
        from multicloud.common.exceptions import InvalidArgumentException

        with pytest.raises(InvalidArgumentException):
            await client.set_tags("key.txt", {})

    @pytest.mark.asyncio
    async def test_get_tags_translates_native_store_error(self, client):
        # A native (untranslated) error from the store is caught and routed
        # through translate_error at the client layer.
        client._store.get_tags = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(UnknownException):
            await client.get_tags("key.txt")

    @pytest.mark.asyncio
    async def test_set_tags_translates_native_store_error(self, client):
        client._store.set_tags = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(UnknownException):
            await client.set_tags("key.txt", {"env": "prod"})

    @pytest.mark.asyncio
    async def test_get_tags_passes_through_substrate_exception(self, client):
        from multicloud.common.exceptions import ResourceNotFoundException

        client._store.get_tags = AsyncMock(
            side_effect=ResourceNotFoundException("missing")
        )

        with pytest.raises(ResourceNotFoundException, match="missing"):
            await client.get_tags("key.txt")


class TestAsyncBucketClientGetMetadata:
    """get_metadata delegation to the store."""

    @pytest.mark.asyncio
    async def test_get_metadata_delegates_and_returns(self, client):
        from multicloud.blob.blob_client.async_blob.async_driver import BlobMetadata

        result = await client.get_metadata("key.txt")

        assert isinstance(result, BlobMetadata)
        assert result.key == "key.txt"
        assert result.content_length == 10
        assert result.content_type == "text/plain"
        assert result.e_tag == "fake-etag"
        assert client._store.get_metadata_calls[0] == {
            "key": "key.txt",
            "version_id": None,
        }

    @pytest.mark.asyncio
    async def test_get_metadata_threads_version_id(self, client):
        result = await client.get_metadata("key.txt", version_id="v9")

        assert result.version_id == "v9"
        assert client._store.get_metadata_calls[0] == {
            "key": "key.txt",
            "version_id": "v9",
        }

    @pytest.mark.asyncio
    async def test_get_metadata_empty_key_raises(self, client):
        from multicloud.common.exceptions import InvalidArgumentException

        with pytest.raises(InvalidArgumentException):
            await client.get_metadata("")

    @pytest.mark.asyncio
    async def test_get_metadata_passes_through_blob_not_found(self, client):
        from multicloud.blob.blob_client.async_blob.async_driver import (
            BlobNotFoundError,
        )

        client._store.get_metadata = AsyncMock(
            side_effect=BlobNotFoundError("missing", key="key.txt")
        )

        with pytest.raises(BlobNotFoundError) as excinfo:
            await client.get_metadata("key.txt")
        assert excinfo.value.key == "key.txt"

    @pytest.mark.asyncio
    async def test_get_metadata_passes_through_substrate_exception(self, client):
        from multicloud.common.exceptions import ResourceNotFoundException

        client._store.get_metadata = AsyncMock(
            side_effect=ResourceNotFoundException("missing")
        )

        with pytest.raises(ResourceNotFoundException, match="missing"):
            await client.get_metadata("key.txt")

    @pytest.mark.asyncio
    async def test_get_metadata_translates_native_store_error(self, client):
        # A native (untranslated) error from the store is caught and routed
        # through translate_error at the client layer.
        client._store.get_metadata = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(UnknownException):
            await client.get_metadata("key.txt")


class TestAsyncBucketClientListPage:
    """list_page delegation to the store."""

    @pytest.mark.asyncio
    async def test_list_page_delegates_with_defaults(self, client):
        from multicloud.blob.blob_client.async_blob.async_driver import (
            ListBlobsPageResponse,
        )

        result = await client.list_page()

        assert isinstance(result, ListBlobsPageResponse)
        assert client._store.list_page_calls[0] == {
            "prefix": None,
            "delimiter": None,
            "max_results": 1000,
            "pagination_token": None,
        }

    @pytest.mark.asyncio
    async def test_list_page_forwards_all_params(self, client):
        await client.list_page(
            prefix="photos/",
            delimiter="/",
            max_results=50,
            pagination_token="tok-1",
        )

        assert client._store.list_page_calls[0] == {
            "prefix": "photos/",
            "delimiter": "/",
            "max_results": 50,
            "pagination_token": "tok-1",
        }

    @pytest.mark.asyncio
    async def test_list_page_passes_through_substrate_exception(self, client):
        from multicloud.common.exceptions import ResourceNotFoundException

        client._store.list_page = AsyncMock(
            side_effect=ResourceNotFoundException("missing")
        )

        with pytest.raises(ResourceNotFoundException, match="missing"):
            await client.list_page()

    @pytest.mark.asyncio
    async def test_list_page_translates_native_store_error(self, client):
        # A native (untranslated) error from the store is caught and routed
        # through translate_error at the client layer.
        client._store.list_page = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(UnknownException):
            await client.list_page()


class TestAsyncBucketClientLifecycle:
    """Context manager and close behavior."""

    @pytest.mark.asyncio
    async def test_close_delegates_to_store(self, client):
        await client.close()
        assert client._store.closed is True

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exit(self):
        async with AsyncBucketClient(provider="fake-test", bucket="b") as c:
            await c.upload("key", b"data")

        assert c._store.closed is True

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exception(self):
        client = AsyncBucketClient(provider="fake-test", bucket="b")
        try:
            async with client:
                raise ValueError("boom")
        except ValueError:
            pass

        assert client._store.closed is True


class TestAsyncBucketClientErrorHandling:
    """Error handling in AsyncBucketClient.upload()."""

    @pytest.mark.asyncio
    async def test_substrate_sdk_exception_reraises_without_wrapping(self):
        """SubstrateSdkException from store.upload reraises directly."""
        from multicloud.common.exceptions import ResourceNotFoundException

        client = AsyncBucketClient(provider="fake-test", bucket="b")
        # Patch the store's upload to raise a SubstrateSdkException
        client._store.upload = AsyncMock(
            side_effect=ResourceNotFoundException("not found")
        )

        with pytest.raises(ResourceNotFoundException, match="not found"):
            await client.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_exceptions_from_store_propagate(self):
        """Exceptions from the underlying store propagate without re-wrapping."""
        client = AsyncBucketClient(provider="fake-test", bucket="b")
        client._store.upload = AsyncMock(
            side_effect=BlobAsyncError(
                "network failure",
                cause=RuntimeError("conn reset"),
            )
        )

        with pytest.raises(BlobAsyncError) as exc_info:
            await client.upload("key", b"data")

        assert "network failure" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raw_exception_translated_via_translate_error(self):
        """Non-SDK exceptions are translated via store.translate_error()."""
        client = AsyncBucketClient(provider="fake-test", bucket="b")
        raw_exc = RuntimeError("connection reset by peer")
        client._store.upload = AsyncMock(side_effect=raw_exc)

        with pytest.raises(UnknownException) as exc_info:
            await client.upload("key", b"data")

        assert "connection reset by peer" in str(exc_info.value)


class TestAsyncBucketClientListParts:
    """The list_parts forwarder on AsyncBucketClient.

    The driver and providers are covered elsewhere; these pin the client
    wrapper itself, which is the one layer without direct coverage.
    """

    @staticmethod
    def _mpu() -> Any:
        from multicloud.blob.blob_client import MultipartUpload

        return MultipartUpload(bucket="b", key="k", upload_id="u")

    @pytest.mark.asyncio
    async def test_returns_what_the_store_returns(self):
        from multicloud.blob.blob_client import UploadPartResponse

        client = AsyncBucketClient(provider="fake-test", bucket="b")
        parts = [UploadPartResponse(part_number=1, e_tag="a", size_in_bytes=10)]
        client._store.list_parts = AsyncMock(return_value=parts)

        assert await client.list_parts(self._mpu()) == parts

    @pytest.mark.asyncio
    async def test_substrate_sdk_exception_reraises_unwrapped(self):
        """An already-typed error keeps its type and identity."""
        from multicloud.common.exceptions import ResourceNotFoundException

        client = AsyncBucketClient(provider="fake-test", bucket="b")
        original = ResourceNotFoundException("no such upload")
        client._store.list_parts = AsyncMock(side_effect=original)

        with pytest.raises(ResourceNotFoundException) as exc_info:
            await client.list_parts(self._mpu())

        assert exc_info.value is original
        # Re-raised as-is. Routing it through translate_error would return the
        # same object but chain it to itself, so __cause__ pins the branch.
        assert exc_info.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_raw_exception_translated_via_translate_error(self):
        """A native error is translated and keeps the original as its cause."""
        client = AsyncBucketClient(provider="fake-test", bucket="b")
        raw_exc = RuntimeError("connection reset by peer")
        client._store.list_parts = AsyncMock(side_effect=raw_exc)

        with pytest.raises(UnknownException) as exc_info:
            await client.list_parts(self._mpu())

        assert exc_info.value.__cause__ is raw_exc
