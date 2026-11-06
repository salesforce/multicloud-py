"""Unit tests for GcpAsyncBlobStore.upload() with mocked gcloud-aio-storage.

Tests validate the GCP async provider correctly:
- Creates and manages the gcloud-aio-storage client lifecycle
- Passes correct parameters to Storage.upload()
- Translates GCP/aiohttp errors to BlobAsyncError
- Handles GCS response metadata (etag, generation)
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

pytest.importorskip(
    "gcloud.aio.storage",
    reason="gcloud-aio-storage not installed (optional async-gcp dependency)",
)

import multicloud.blob.blob_gcp.async_blob.gcp_async_blob_store as _gcp_store_mod
from multicloud.blob.blob_client.async_blob.async_driver import (
    ASYNC_BLOB_REGISTRY,
    AsyncUploadResponse,
    BlobAsyncError,
)
from multicloud.blob.blob_gcp.async_blob.gcp_async_blob_store import GcpAsyncBlobStore
from multicloud.common.exceptions import (
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceExistsException,
    ResourceNotFoundException,
    UnAuthorizedException,
    UnknownException,
    UnsupportedOperationException,
)


def _make_mock_storage(
    upload_response: dict | None = None, upload_side_effect: Any = None
):
    """Create a mock gcloud.aio.storage.Storage instance."""
    mock_storage = AsyncMock()
    if upload_response is not None:
        mock_storage.upload.return_value = upload_response
    if upload_side_effect is not None:
        mock_storage.upload.side_effect = upload_side_effect
    mock_storage.close = AsyncMock()
    return mock_storage


def _make_client_response_error(status: int, message: str = "error"):
    """Create a mock aiohttp.ClientResponseError."""
    mock_request_info = MagicMock()
    mock_request_info.real_url = "https://storage.googleapis.com/..."
    return aiohttp.ClientResponseError(
        request_info=mock_request_info,
        history=(),
        status=status,
        message=message,
    )


@pytest.fixture
def mock_storage():
    """A mock Storage that returns success."""
    return _make_mock_storage(
        upload_response={
            "name": "test-key",
            "bucket": "test-bucket",
            "generation": "1234567890",
            "etag": "CJDe9fH3x4cDEAE=",
            "md5Hash": "rL0Y20zC+Fzt72VPzMSk2A==",
        }
    )


@pytest.fixture
def gcp_store(mock_storage):
    """GcpAsyncBlobStore with mocked storage."""
    return GcpAsyncBlobStore(
        bucket="test-bucket",
        region="us-central1",
        _storage=mock_storage,
    )


class TestGcpAsyncUploadSuccess:
    """Successful upload scenarios."""

    @pytest.mark.asyncio
    async def test_upload_bytes_returns_response(self, gcp_store):
        result = await gcp_store.upload("my-key", b"hello")

        assert isinstance(result, AsyncUploadResponse)
        assert result.key == "my-key"
        assert result.e_tag == "CJDe9fH3x4cDEAE="
        assert result.version_id == "1234567890"

    @pytest.mark.asyncio
    async def test_upload_passes_bucket_and_key(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="my-bucket", _storage=mock_storage)
        await store.upload("path/to/obj", b"data")

        call_kwargs = mock_storage.upload.call_args[1]
        assert call_kwargs["bucket"] == "my-bucket"
        assert call_kwargs["object_name"] == "path/to/obj"

    @pytest.mark.asyncio
    async def test_upload_passes_content_type(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        await store.upload("key", b"data", content_type="image/png")

        call_kwargs = mock_storage.upload.call_args[1]
        assert call_kwargs["content_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_upload_passes_metadata(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        await store.upload("key", b"data", metadata={"env": "staging"})

        call_kwargs = mock_storage.upload.call_args[1]
        assert call_kwargs["metadata"] == {"metadata": {"env": "staging"}}

    @pytest.mark.asyncio
    async def test_upload_passes_file_data(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        await store.upload("key", b"my-content")

        call_kwargs = mock_storage.upload.call_args[1]
        assert call_kwargs["file_data"] == b"my-content"

    @pytest.mark.asyncio
    async def test_upload_falls_back_to_md5hash_for_etag(self):
        mock = _make_mock_storage(
            upload_response={
                "name": "key",
                "md5Hash": "abc123hash",
                "generation": "gen-1",
            }
        )
        store = GcpAsyncBlobStore(bucket="b", _storage=mock)

        result = await store.upload("key", b"data")
        assert result.e_tag == "abc123hash"


class TestGcpAsyncUploadKmsEncryption:
    """Server-side KMS encryption on upload."""

    @pytest.mark.asyncio
    async def test_kms_key_id_sets_kms_key_name_param(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        key = "projects/p/locations/l/keyRings/r/cryptoKeys/k"
        await store.upload("key", b"data", kms_key_id=key)

        call_kwargs = mock_storage.upload.call_args[1]
        assert call_kwargs["parameters"] == {"kmsKeyName": key}

    @pytest.mark.asyncio
    async def test_use_kms_managed_key_alone_is_noop(self, mock_storage):
        # GCS always encrypts at rest with Google-managed keys; requesting a
        # managed key without an explicit id sets no query parameters.
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        await store.upload("key", b"data", use_kms_managed_key=True)

        call_kwargs = mock_storage.upload.call_args[1]
        assert "parameters" not in call_kwargs


class TestGcpAsyncUploadStorageClass:
    """Storage class selection on upload."""

    @staticmethod
    def _wire_resource(call_kwargs: dict[str, Any]) -> dict[str, Any]:
        # gcloud-aio-storage rewrites resource keys before sending them, so
        # assert on the JSON field name GCS actually receives.
        from gcloud.aio.storage import Storage

        return {
            Storage._format_metadata_key(k): v
            for k, v in call_kwargs["metadata"].items()
        }

    @pytest.mark.asyncio
    async def test_storage_class_sent_as_storage_class_field(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        await store.upload("key", b"data", storage_class="NEARLINE")

        call_kwargs = mock_storage.upload.call_args[1]
        assert self._wire_resource(call_kwargs) == {"storageClass": "NEARLINE"}

    @pytest.mark.asyncio
    async def test_storage_class_combined_with_user_metadata(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        await store.upload(
            "key", b"data", metadata={"env": "staging"}, storage_class="COLDLINE"
        )

        call_kwargs = mock_storage.upload.call_args[1]
        assert self._wire_resource(call_kwargs) == {
            "metadata": {"env": "staging"},
            "storageClass": "COLDLINE",
        }

    @pytest.mark.asyncio
    async def test_no_storage_class_sends_no_resource(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        await store.upload("key", b"data")

        assert "metadata" not in mock_storage.upload.call_args[1]

    @pytest.mark.asyncio
    async def test_blank_storage_class_rejected_before_request(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        with pytest.raises(InvalidArgumentException, match="storage_class"):
            await store.upload("key", b"data", storage_class=" ")
        mock_storage.upload.assert_not_called()


class TestGcpAsyncUploadWritePreconditions:
    """Conditional writes: create_if_absent supported, if_match unsupported."""

    @pytest.mark.asyncio
    async def test_create_if_absent_sets_if_generation_match_zero(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        await store.upload("key", b"data", create_if_absent=True)

        call_kwargs = mock_storage.upload.call_args[1]
        assert call_kwargs["parameters"] == {"ifGenerationMatch": "0"}

    @pytest.mark.asyncio
    async def test_if_match_raises_unsupported_operation(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        with pytest.raises(UnsupportedOperationException):
            await store.upload("key", b"data", if_match="etag-123")

    @pytest.mark.asyncio
    async def test_if_version_match_maps_to_if_generation_match(self, mock_storage):
        # The version id GCS returns is the object generation, so a
        # version-conditioned overwrite maps to ifGenerationMatch=<generation>.
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        await store.upload("key", b"data", if_version_match="42")

        call_kwargs = mock_storage.upload.call_args[1]
        assert call_kwargs["parameters"] == {"ifGenerationMatch": "42"}

    @pytest.mark.asyncio
    async def test_if_version_match_non_numeric_rejected(self, mock_storage):
        # A non-numeric version token is a caller mistake.
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        with pytest.raises(InvalidArgumentException, match="numeric GCS generation"):
            await store.upload("key", b"data", if_version_match="not-a-number")

    @pytest.mark.asyncio
    async def test_if_match_and_if_version_match_together_uses_generation(
        self, mock_storage
    ):
        # A portable caller may set both tokens so each provider enforces with
        # the one it supports. On GCS that is the generation (if_version_match);
        # the redundant if_match must be ignored rather than trigger the
        # "unsupported" rejection that fires when if_match arrives alone.
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        await store.upload("key", b"data", if_match="etag-123", if_version_match="42")

        call_kwargs = mock_storage.upload.call_args[1]
        assert call_kwargs["parameters"] == {"ifGenerationMatch": "42"}

    @pytest.mark.asyncio
    async def test_no_preconditions_omits_parameters(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        await store.upload("key", b"data")

        call_kwargs = mock_storage.upload.call_args[1]
        assert "parameters" not in call_kwargs

    @pytest.mark.asyncio
    async def test_create_if_absent_conflict_raises_resource_exists(self):
        """A 412 on create_if_absent is re-labelled ResourceExistsException."""
        exc = _make_client_response_error(412, "Precondition Failed")
        mock = _make_mock_storage(upload_side_effect=exc)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock)

        with pytest.raises(ResourceExistsException):
            await store.upload("key", b"data", create_if_absent=True)

    def test_translate_error_maps_precondition_failed(self, mock_storage):
        """A 412 maps to FailedPreconditionException at the client layer."""
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        result = store.translate_error(_make_client_response_error(412))

        assert isinstance(result, FailedPreconditionException)

    @pytest.mark.asyncio
    async def test_if_version_match_on_missing_object_maps_to_failed_precondition(self):
        """Parity with the AWS if_match-on-missing case: a version-conditioned
        overwrite whose target does not exist yields the precondition-failed
        outcome. GCS returns 412 (not 404), so the store propagates it raw and
        the client layer translates it to FailedPreconditionException."""
        import aiohttp

        exc = _make_client_response_error(412, "Precondition Failed")
        mock = _make_mock_storage(upload_side_effect=exc)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock)

        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await store.upload("key", b"data", if_version_match="7")
        assert isinstance(
            store.translate_error(excinfo.value), FailedPreconditionException
        )

    @pytest.mark.asyncio
    async def test_create_if_absent_combined_with_kms(self, mock_storage):
        """create_if_absent and KMS coexist in the same upload parameters dict."""
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        key = "projects/p/locations/l/keyRings/r/cryptoKeys/k"

        await store.upload("key", b"data", create_if_absent=True, kms_key_id=key)

        call_kwargs = mock_storage.upload.call_args[1]
        assert call_kwargs["parameters"] == {
            "ifGenerationMatch": "0",
            "kmsKeyName": key,
        }


class TestGcpAsyncUploadErrors:
    """Error handling and translation.

    Provider do_upload does NOT catch/translate — exceptions propagate to
    the client layer. These tests verify raw propagation + translate_error
    mapping (called by AsyncBucketClient).
    """

    @pytest.mark.asyncio
    async def test_client_response_error_propagates_to_client_layer(self):
        """aiohttp.ClientResponseError propagates raw."""
        import aiohttp

        exc = _make_client_response_error(404, "Not Found")
        mock = _make_mock_storage(upload_side_effect=exc)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock)

        with pytest.raises(aiohttp.ClientResponseError):
            await store.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_403_propagates_to_client_layer(self):
        """403 ClientResponseError propagates raw."""
        import aiohttp

        exc = _make_client_response_error(403, "Forbidden")
        mock = _make_mock_storage(upload_side_effect=exc)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock)

        with pytest.raises(aiohttp.ClientResponseError):
            await store.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_400_propagates_to_client_layer(self):
        """400 ClientResponseError propagates raw."""
        import aiohttp

        exc = _make_client_response_error(400, "Bad Request")
        mock = _make_mock_storage(upload_side_effect=exc)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock)

        with pytest.raises(aiohttp.ClientResponseError):
            await store.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_500_propagates_to_client_layer(self):
        """500 ClientResponseError propagates raw."""
        import aiohttp

        exc = _make_client_response_error(500, "Internal Server Error")
        mock = _make_mock_storage(upload_side_effect=exc)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock)

        with pytest.raises(aiohttp.ClientResponseError):
            await store.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_network_error_propagates_to_client_layer(self):
        """aiohttp.ClientConnectorError propagates raw."""
        import aiohttp

        exc = aiohttp.ClientConnectorError(
            connection_key=MagicMock(), os_error=OSError("Connection refused")
        )
        mock = _make_mock_storage(upload_side_effect=exc)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock)

        with pytest.raises(aiohttp.ClientConnectorError):
            await store.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_network_error_propagates_to_client_layer(self):
        """aiohttp.ClientConnectorError propagates raw."""
        import aiohttp

        exc = aiohttp.ClientConnectorError(
            connection_key=MagicMock(), os_error=OSError("Connection refused")
        )
        mock = _make_mock_storage(upload_side_effect=exc)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock)

        with pytest.raises(aiohttp.ClientConnectorError):
            await store.upload("key", b"data")

    def test_translate_error_passes_through_substrate_sdk_exception(self):
        """SubstrateSdkException passes through translate_error unchanged."""
        store = GcpAsyncBlobStore(bucket="b", _storage=_make_mock_storage())
        original = UnknownException("already sdk")

        result = store.translate_error(original)

        assert result is original

    def test_translate_error_404_returns_resource_not_found(self):
        """ClientResponseError with 404 translates to ResourceNotFoundException."""
        store = GcpAsyncBlobStore(bucket="b", _storage=_make_mock_storage())
        exc = _make_client_response_error(404, "Not Found")

        result = store.translate_error(exc)

        assert isinstance(result, ResourceNotFoundException)

    def test_translate_error_403_returns_unauthorized(self):
        """ClientResponseError with 403 translates to UnAuthorizedException."""
        store = GcpAsyncBlobStore(bucket="b", _storage=_make_mock_storage())
        exc = _make_client_response_error(403, "Forbidden")

        result = store.translate_error(exc)

        assert isinstance(result, UnAuthorizedException)

    def test_translate_error_401_returns_unauthorized(self):
        """ClientResponseError with 401 translates to UnAuthorizedException."""
        store = GcpAsyncBlobStore(bucket="b", _storage=_make_mock_storage())
        exc = _make_client_response_error(401, "Unauthorized")

        result = store.translate_error(exc)

        assert isinstance(result, UnAuthorizedException)

    def test_translate_error_400_returns_invalid_argument(self):
        """ClientResponseError with 400 translates to InvalidArgumentException."""
        store = GcpAsyncBlobStore(bucket="b", _storage=_make_mock_storage())
        exc = _make_client_response_error(400, "Bad Request")

        result = store.translate_error(exc)

        assert isinstance(result, InvalidArgumentException)

    def test_translate_error_500_returns_unknown(self):
        """ClientResponseError with 500 translates to UnknownException."""
        store = GcpAsyncBlobStore(bucket="b", _storage=_make_mock_storage())
        exc = _make_client_response_error(500, "Internal Server Error")

        result = store.translate_error(exc)

        assert isinstance(result, UnknownException)

    def test_translate_error_generic_client_error_returns_blob_async_error(self):
        """Non-response aiohttp.ClientError translates to BlobAsyncError."""
        store = GcpAsyncBlobStore(bucket="b", _storage=_make_mock_storage())
        exc = aiohttp.ClientConnectionError("connection lost")

        result = store.translate_error(exc)

        assert isinstance(result, BlobAsyncError)
        assert "Network error" in str(result)
        assert result.cause is exc

    def test_translate_error_non_aiohttp_exception_returns_blob_async_error(self):
        """Non-aiohttp exception translates to BlobAsyncError."""
        store = GcpAsyncBlobStore(bucket="b", _storage=_make_mock_storage())
        exc = ValueError("something unexpected")

        result = store.translate_error(exc)

        assert isinstance(result, BlobAsyncError)
        assert "something unexpected" in str(result)
        assert result.cause is exc


class TestGcpAsyncClientLifecycle:
    """Client creation and cleanup."""

    @pytest.mark.asyncio
    async def test_ensure_client_creates_storage_with_service_file(self):
        """_ensure_client creates Storage with service_file kwarg."""
        store = GcpAsyncBlobStore(
            bucket="b",
            service_file="/path/to/creds.json",
        )
        assert store._storage is None

        with patch.object(
            _gcp_store_mod, "Storage", return_value=MagicMock()
        ) as mock_storage_cls:
            await store._ensure_client()

            mock_storage_cls.assert_called_once_with(service_file="/path/to/creds.json")
            assert store._storage is not None

    @pytest.mark.asyncio
    async def test_ensure_client_creates_storage_without_service_file(self):
        """_ensure_client creates Storage with no kwargs when no service_file."""
        store = GcpAsyncBlobStore(bucket="b")
        assert store._storage is None

        with patch.object(
            _gcp_store_mod, "Storage", return_value=MagicMock()
        ) as mock_storage_cls:
            await store._ensure_client()

            mock_storage_cls.assert_called_once_with()
            assert store._storage is not None

    @pytest.mark.asyncio
    async def test_close_calls_storage_close(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)
        # _owns_storage is False when _storage is injected
        await store.close()
        # Should NOT close injected storage
        mock_storage.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_releases_owned_storage(self):
        mock = _make_mock_storage(upload_response={"name": "k"})
        store = GcpAsyncBlobStore(bucket="b")
        # Manually set the storage and ownership flag
        store._storage = mock
        store._owns_storage = True

        await store.close()
        mock.close.assert_called_once()
        assert store._storage is None

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        async with store as s:
            result = await s.upload("key", b"data")
            assert result.key == "key"


class TestGcpAsyncConcurrentInit:
    """Verify _ensure_client is safe under concurrent first use."""

    @pytest.mark.asyncio
    async def test_concurrent_uploads_create_single_storage(self):
        """asyncio.gather on a fresh store must only create one Storage."""
        mock = _make_mock_storage(upload_response={"name": "k", "etag": "e"})

        with patch.object(_gcp_store_mod, "Storage", return_value=mock) as storage_cls:
            store = GcpAsyncBlobStore(bucket="b")

            import asyncio

            await asyncio.gather(*[store.upload(f"k{i}", b"x") for i in range(20)])

            storage_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_endpoint_url_forwarded_as_api_root(self):
        """endpoint_url is wired through to Storage(api_root=...)."""
        mock = _make_mock_storage(upload_response={"name": "k", "etag": "e"})

        with patch.object(_gcp_store_mod, "Storage", return_value=mock) as storage_cls:
            store = GcpAsyncBlobStore(bucket="b", endpoint_url="http://localhost:9023")
            await store.upload("key", b"data")

            storage_cls.assert_called_once_with(api_root="http://localhost:9023")


class TestGcpAsyncTags:
    """get_tags / set_tags via gcp-tag-* metadata entries."""

    @pytest.mark.asyncio
    async def test_get_tags_filters_prefix(self, mock_storage):
        mock_storage.download_metadata.return_value = {
            "metadata": {
                "gcp-tag-env": "prod",
                "gcp-tag-team": "multicloud",
                "other-meta": "keep-me",
            }
        }
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        result = await store.get_tags("my-key")

        assert result == {"env": "prod", "team": "multicloud"}
        mock_storage.download_metadata.assert_awaited_once_with("b", "my-key")

    @pytest.mark.asyncio
    async def test_get_tags_empty_when_no_metadata(self, mock_storage):
        mock_storage.download_metadata.return_value = {}
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        assert await store.get_tags("key") == {}

    @pytest.mark.asyncio
    async def test_set_tags_patches_prefixed_metadata(self, mock_storage):
        mock_storage.download_metadata.return_value = {
            "metadata": {"gcp-tag-old": "stale", "keep": "value"}
        }
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        await store.set_tags("my-key", {"env": "prod"})

        args, _ = mock_storage.patch_metadata.call_args
        bucket_arg, key_arg, body = args
        assert bucket_arg == "b"
        assert key_arg == "my-key"
        sent = body["metadata"]
        # New tag added, old tag nulled for deletion, non-tag metadata preserved.
        assert sent["gcp-tag-env"] == "prod"
        assert sent["gcp-tag-old"] is None
        assert sent["keep"] == "value"

    @pytest.mark.asyncio
    async def test_set_tags_sends_metageneration_precondition(self, mock_storage):
        mock_storage.download_metadata.return_value = {
            "metageneration": "7",
            "metadata": {},
        }
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        await store.set_tags("my-key", {"env": "prod"})

        _, kwargs = mock_storage.patch_metadata.call_args
        assert kwargs["params"] == {"ifMetagenerationMatch": "7"}

    @pytest.mark.asyncio
    async def test_set_tags_omits_precondition_when_no_metageneration(
        self, mock_storage, caplog
    ):
        mock_storage.download_metadata.return_value = {"metadata": {}}
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        with caplog.at_level(logging.WARNING):
            await store.set_tags("my-key", {"env": "prod"})

        _, kwargs = mock_storage.patch_metadata.call_args
        assert kwargs["params"] == {}
        # The lost optimistic-concurrency guard must not be silent.
        assert any(
            "without optimistic concurrency control" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_set_then_get_tags_round_trip(self, mock_storage):
        """Simulate the GCS round-trip: patch stores, download returns it back."""
        stored: dict = {"metadata": {}}

        async def _download(bucket, key, **kwargs):
            return stored

        async def _patch(bucket, key, body, **kwargs):
            stored["metadata"] = {
                k: v for k, v in body["metadata"].items() if v is not None
            }
            return stored

        mock_storage.download_metadata.side_effect = _download
        mock_storage.patch_metadata.side_effect = _patch
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        await store.set_tags("my-key", {"env": "prod", "team": "multicloud"})
        result = await store.get_tags("my-key")

        assert result == {"env": "prod", "team": "multicloud"}

    @pytest.mark.asyncio
    async def test_get_tags_propagates_native_error(self, mock_storage):
        # The provider lets native errors propagate; translation happens at
        # the client layer (see test_async_client.py).
        import aiohttp

        mock_storage.download_metadata.side_effect = _make_client_response_error(404)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        with pytest.raises(aiohttp.ClientResponseError):
            await store.get_tags("missing")

    @pytest.mark.asyncio
    async def test_set_tags_propagates_native_error(self, mock_storage):
        import aiohttp

        mock_storage.download_metadata.return_value = {"metadata": {}}
        mock_storage.patch_metadata.side_effect = _make_client_response_error(403)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        with pytest.raises(aiohttp.ClientResponseError):
            await store.set_tags("key", {"env": "prod"})

    @pytest.mark.asyncio
    async def test_set_tags_propagates_precondition_failure(self, mock_storage):
        # A concurrent metadata change makes ifMetagenerationMatch fail; GCS
        # returns 412. The provider lets it propagate (translation is the
        # client layer's job).
        import aiohttp

        mock_storage.download_metadata.return_value = {
            "metageneration": "7",
            "metadata": {},
        }
        mock_storage.patch_metadata.side_effect = _make_client_response_error(412)
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await store.set_tags("key", {"env": "prod"})
        assert exc_info.value.status == 412

    def test_translate_error_maps_not_found(self, mock_storage):
        store = GcpAsyncBlobStore(bucket="b", _storage=mock_storage)

        result = store.translate_error(_make_client_response_error(404))

        assert isinstance(result, ResourceNotFoundException)


class TestGcpAsyncRegistration:
    """Provider registry integration."""

    def test_gcp_registered_in_async_registry(self):
        assert ASYNC_BLOB_REGISTRY.get("gcp") is GcpAsyncBlobStore
