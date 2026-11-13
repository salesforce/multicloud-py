"""Unit tests for AwsAsyncBlobStore.upload() with mocked aiobotocore.

Tests validate the AWS async provider correctly:
- Creates and manages the aiobotocore client lifecycle
- Passes correct parameters to S3 put_object
- Translates AWS errors to BlobAsyncError
- Handles ETag and VersionId in responses
"""

from __future__ import annotations

from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "aiobotocore", reason="aiobotocore not installed (optional async-aws dependency)"
)

import multicloud.blob.blob_aws.async_blob.aws_async_blob_store as _aws_store_mod
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from multicloud.blob.blob_aws.async_blob.aws_async_blob_store import AwsAsyncBlobStore
from multicloud.blob.blob_client.async_blob.async_driver import (
    ASYNC_BLOB_REGISTRY,
    AsyncUploadResponse,
    BlobAsyncError,
)
from multicloud.common.exceptions import (
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceConflictException,
    ResourceExistsException,
    ResourceNotFoundException,
    UnAuthorizedException,
    UnknownException,
    UnsupportedOperationException,
)


def _make_client_error(code: str, status: int = 400, message: str = "error") -> Any:
    """Create a mock botocore ClientError."""
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "PutObject",
    )


def _make_mock_session(put_response: dict | None = None, put_side_effect: Any = None):
    """Create a mock aiobotocore session with a mock S3 client."""
    mock_client = AsyncMock()
    if put_response is not None:
        mock_client.put_object.return_value = put_response
    if put_side_effect is not None:
        mock_client.put_object.side_effect = put_side_effect

    # Mock context manager for create_client
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.create_client.return_value = mock_client_cm

    return mock_session, mock_client


@pytest.fixture
def mock_session_success():
    """Session that returns a successful put_object response."""
    return _make_mock_session(
        put_response={
            "ETag": '"abc123def456"',
            "VersionId": "ver-001",
        }
    )


@pytest.fixture
def aws_store(mock_session_success):
    """AwsAsyncBlobStore with mocked session."""
    session, _ = mock_session_success
    return AwsAsyncBlobStore(
        bucket="test-bucket",
        region="us-east-1",
        _session=session,
    )


class TestAwsAsyncUploadSuccess:
    """Successful upload scenarios."""

    @pytest.mark.asyncio
    async def test_upload_bytes_returns_response(self, aws_store):
        result = await aws_store.upload("my-key", b"hello")

        assert isinstance(result, AsyncUploadResponse)
        assert result.key == "my-key"
        assert result.e_tag == "abc123def456"  # quotes stripped
        assert result.version_id == "ver-001"

    @pytest.mark.asyncio
    async def test_upload_passes_content_type(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data", content_type="text/plain")

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["ContentType"] == "text/plain"

    @pytest.mark.asyncio
    async def test_upload_passes_metadata(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data", metadata={"env": "prod"})

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["Metadata"] == {"env": "prod"}

    @pytest.mark.asyncio
    async def test_upload_passes_content_length(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"12345")

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["ContentLength"] == 5

    @pytest.mark.asyncio
    async def test_upload_sends_to_correct_bucket_and_key(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="my-bucket", _session=session)

        await store.upload("path/to/obj.bin", b"data")

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "my-bucket"
        assert call_kwargs["Key"] == "path/to/obj.bin"

    @pytest.mark.asyncio
    async def test_upload_no_version_id_when_not_in_response(self):
        session, _ = _make_mock_session(put_response={"ETag": '"etag"'})
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        result = await store.upload("key", b"data")
        assert result.version_id is None


class TestAwsAsyncUploadStorageClass:
    """Storage class selection on upload."""

    @pytest.mark.asyncio
    async def test_storage_class_sets_storage_class_param(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data", storage_class="STANDARD_IA")

        assert mock_client.put_object.call_args[1]["StorageClass"] == "STANDARD_IA"

    @pytest.mark.asyncio
    async def test_no_storage_class_omits_param(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data")

        assert "StorageClass" not in mock_client.put_object.call_args[1]

    @pytest.mark.asyncio
    async def test_blank_storage_class_rejected_before_request(
        self, mock_session_success
    ):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(InvalidArgumentException, match="storage_class"):
            await store.upload("key", b"data", storage_class="")
        mock_client.put_object.assert_not_called()

    def test_unknown_storage_class_translates_to_invalid_argument(self):
        store = AwsAsyncBlobStore(bucket="b")
        exc = _make_client_error("InvalidStorageClass", 400)
        assert isinstance(store.translate_error(exc), InvalidArgumentException)


class TestAwsAsyncUploadKmsEncryption:
    """Server-side KMS encryption on upload."""

    @pytest.mark.asyncio
    async def test_kms_key_id_sets_sse_and_key(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data", kms_key_id="arn:kms:key/abc")

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert call_kwargs["SSEKMSKeyId"] == "arn:kms:key/abc"

    @pytest.mark.asyncio
    async def test_use_kms_managed_key_sets_sse_without_key(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data", use_kms_managed_key=True)

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert "SSEKMSKeyId" not in call_kwargs

    @pytest.mark.asyncio
    async def test_explicit_key_takes_precedence_over_managed_flag(
        self, mock_session_success
    ):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data", kms_key_id="key-1", use_kms_managed_key=True)

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert call_kwargs["SSEKMSKeyId"] == "key-1"

    @pytest.mark.asyncio
    async def test_no_kms_omits_encryption_headers(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data")

        call_kwargs = mock_client.put_object.call_args[1]
        assert "ServerSideEncryption" not in call_kwargs
        assert "SSEKMSKeyId" not in call_kwargs


class TestAwsAsyncUploadWritePreconditions:
    """Conditional writes: create_if_absent and if_match, plus KMS combined."""

    @pytest.mark.asyncio
    async def test_create_if_absent_sets_if_none_match(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data", create_if_absent=True)

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["IfNoneMatch"] == "*"
        assert "IfMatch" not in call_kwargs

    @pytest.mark.asyncio
    async def test_if_match_sets_if_match_header(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data", if_match="etag-123")

        call_kwargs = mock_client.put_object.call_args[1]
        # download() returns a quote-stripped ETag; the write path re-quotes it
        # so the natural round-trip matches S3's quoted stored ETag.
        assert call_kwargs["IfMatch"] == '"etag-123"'
        assert "IfNoneMatch" not in call_kwargs

    @pytest.mark.asyncio
    async def test_no_preconditions_omits_headers(self, mock_session_success):
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data")

        call_kwargs = mock_client.put_object.call_args[1]
        assert "IfNoneMatch" not in call_kwargs
        assert "IfMatch" not in call_kwargs

    @pytest.mark.asyncio
    async def test_if_version_match_unsupported_on_s3(self, mock_session_success):
        # S3 compare-and-swap is ETag-based (IfMatch); a version-conditioned
        # write cannot be expressed, so the provider refuses it rather than
        # silently doing an unconditional overwrite.
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(UnsupportedOperationException, match="if_version_match"):
            await store.upload("key", b"data", if_version_match="42")
        mock_client.put_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_if_match_and_if_version_match_together_uses_if_match(
        self, mock_session_success
    ):
        # A portable caller may set both tokens so each provider enforces with
        # the one it supports. On S3 that is IfMatch; if_version_match is
        # redundant here and must be ignored rather than trigger the
        # "unsupported" rejection that fires when it arrives alone.
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"data", if_match="etag-123", if_version_match="42")
        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["IfMatch"] == '"etag-123"'

    @pytest.mark.asyncio
    async def test_create_if_absent_conflict_raises_resource_exists(self):
        """A 412 on create_if_absent is re-labelled ResourceExistsException."""
        exc = _make_client_error("PreconditionFailed", 412)
        session, _ = _make_mock_session(put_side_effect=exc)
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(ResourceExistsException):
            await store.upload("key", b"data", create_if_absent=True)

    @pytest.mark.asyncio
    async def test_if_match_mismatch_propagates_for_client_translation(self):
        """A 412 on if_match (not create_if_absent) propagates raw."""
        exc = _make_client_error("PreconditionFailed", 412)
        session, _ = _make_mock_session(put_side_effect=exc)
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(ClientError):
            await store.upload("key", b"data", if_match="stale-etag")

    @pytest.mark.asyncio
    async def test_if_match_on_missing_object_raises_failed_precondition(self):
        """An if_match overwrite on a missing object (404 NoSuchKey) is
        normalized to FailedPreconditionException, not ResourceNotFoundException:
        a missing object cannot satisfy an ETag precondition."""
        exc = _make_client_error("NoSuchKey", 404)
        session, _ = _make_mock_session(put_side_effect=exc)
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(FailedPreconditionException):
            await store.upload("key", b"data", if_match="etag-abc")

    @pytest.mark.asyncio
    async def test_if_match_on_missing_bucket_raises_resource_not_found(self):
        """A missing BUCKET on an if_match write is a genuine not-found, NOT a
        precondition mismatch. The store must not relabel NoSuchBucket to
        FailedPreconditionException; it propagates raw so the client layer maps
        it to ResourceNotFoundException."""
        exc = _make_client_error("NoSuchBucket", 404)
        session, _ = _make_mock_session(put_side_effect=exc)
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(ClientError) as excinfo:
            await store.upload("key", b"data", if_match="etag-abc")
        assert not isinstance(excinfo.value, FailedPreconditionException)
        assert isinstance(
            store.translate_error(excinfo.value), ResourceNotFoundException
        )

    @pytest.mark.asyncio
    async def test_unconditional_missing_object_error_propagates_raw(self):
        """Without a precondition, a 404 is a genuine not-found and propagates
        raw for the client layer to map to ResourceNotFoundException."""
        exc = _make_client_error("NoSuchKey", 404)
        session, _ = _make_mock_session(put_side_effect=exc)
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(ClientError):
            await store.upload("key", b"data")

    def test_translate_error_maps_precondition_failed(self):
        """A 412 maps to FailedPreconditionException at the client layer."""
        session, _ = _make_mock_session()
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        result = store.translate_error(_make_client_error("PreconditionFailed", 412))

        assert isinstance(result, FailedPreconditionException)

    def test_translate_error_maps_409_conflict_to_resource_conflict(self):
        """ConditionalRequestConflict maps to the retryable
        ResourceConflictException, not FailedPreconditionException.
        Matched by error code, not HTTP 409 in general."""
        session, _ = _make_mock_session()
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        result = store.translate_error(
            _make_client_error("ConditionalRequestConflict", 409)
        )

        assert isinstance(result, ResourceConflictException)
        assert not isinstance(result, FailedPreconditionException)

    def test_translate_error_generic_409_is_not_resource_conflict(self):
        """HTTP 409 without ConditionalRequestConflict stays UnknownException."""
        session, _ = _make_mock_session()
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        result = store.translate_error(_make_client_error("LimitExceeded", 409))

        assert isinstance(result, UnknownException)
        assert not isinstance(result, ResourceConflictException)

    @pytest.mark.asyncio
    async def test_create_if_absent_combined_with_kms(self, mock_session_success):
        """create_if_absent and KMS coexist in the same put_object call."""
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload(
            "key", b"data", create_if_absent=True, kms_key_id="arn:kms:key/abc"
        )

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["IfNoneMatch"] == "*"
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert call_kwargs["SSEKMSKeyId"] == "arn:kms:key/abc"

    @pytest.mark.asyncio
    async def test_if_match_combined_with_kms(self, mock_session_success):
        """if_match (ETag CAS) and KMS coexist in the same put_object call."""
        session, mock_client = mock_session_success
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload(
            "key", b"data", if_match="etag-123", kms_key_id="arn:kms:key/abc"
        )

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["IfMatch"] == '"etag-123"'
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert call_kwargs["SSEKMSKeyId"] == "arn:kms:key/abc"


class TestAwsAsyncUploadErrors:
    """Error handling and translation.

    Provider do_upload does NOT catch/translate — exceptions propagate to
    the client layer. These tests verify raw propagation + translate_error
    mapping (called by AsyncBucketClient).
    """

    @pytest.mark.asyncio
    async def test_client_error_propagates_to_client_layer(self):
        """ClientError propagates raw — client layer calls translate_error."""
        exc = _make_client_error("NoSuchBucket", 404)
        session, _ = _make_mock_session(put_side_effect=exc)
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(ClientError):
            await store.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_access_denied_propagates_to_client_layer(self):
        """AccessDenied ClientError propagates raw."""
        exc = _make_client_error("AccessDenied", 403)
        session, _ = _make_mock_session(put_side_effect=exc)
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(ClientError):
            await store.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_invalid_argument_propagates_to_client_layer(self):
        """InvalidArgument ClientError propagates raw."""
        exc = _make_client_error("InvalidArgument", 400)
        session, _ = _make_mock_session(put_side_effect=exc)
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(ClientError):
            await store.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_generic_exception_propagates_to_client_layer(self):
        """Non-boto exceptions propagate raw to the client layer."""
        session, mock_client = _make_mock_session()
        mock_client.put_object.side_effect = RuntimeError("unexpected")
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(RuntimeError, match="unexpected"):
            await store.upload("key", b"data")

    @pytest.mark.asyncio
    async def test_no_credentials_error_propagates_to_client_layer(self):
        """NoCredentialsError propagates raw — client layer translates."""
        session, mock_client = _make_mock_session()
        mock_client.put_object.side_effect = NoCredentialsError()
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(NoCredentialsError):
            await store.upload("key", b"data")

    def test_translate_error_passes_through_substrate_sdk_exception(self):
        """SubstrateSdkException passes through translate_error unchanged."""
        session, _ = _make_mock_session()
        store = AwsAsyncBlobStore(bucket="b", _session=session)
        original = UnknownException("already translated")

        result = store.translate_error(original)

        assert result is original

    @pytest.mark.asyncio
    async def test_botocore_error_propagates_to_client_layer(self):
        """BotoCoreError propagates raw — client layer translates."""
        session, mock_client = _make_mock_session()
        mock_client.put_object.side_effect = BotoCoreError()
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(BotoCoreError):
            await store.upload("key", b"data")

    def test_unrecognized_client_error_code_returns_unknown(self):
        """ClientError with unrecognized code returns UnknownException."""
        session, _ = _make_mock_session()
        store = AwsAsyncBlobStore(bucket="b", _session=session)
        exc = _make_client_error("SomeUnknownCode", 599)

        result = store.translate_error(exc)

        assert isinstance(result, UnknownException)

    def test_generic_non_botocore_exception_returns_blob_async_error(self):
        """A non-botocore, non-ClientError exception returns BlobAsyncError."""
        session, _ = _make_mock_session()
        store = AwsAsyncBlobStore(bucket="b", _session=session)
        exc = ValueError("something completely different")

        result = store.translate_error(exc)

        assert isinstance(result, BlobAsyncError)
        assert "something completely different" in str(result)
        assert result.cause is exc


class TestAwsAsyncClientLifecycle:
    """Client creation and cleanup."""

    @pytest.mark.asyncio
    async def test_lazy_client_creation(self):
        session, _ = _make_mock_session(put_response={"ETag": '"e"'})
        store = AwsAsyncBlobStore(bucket="b", region="us-west-2", _session=session)

        # Client not created until first use
        assert store._client is None

        await store.upload("key", b"x")

        # Now client is created
        assert store._client is not None
        session.create_client.assert_called_once_with(
            "s3", region_name="us-west-2", config=ANY
        )
        # The client must be pinned to SigV4 -- KMS buckets reject SigV2.
        cfg = session.create_client.call_args.kwargs["config"]
        assert cfg.signature_version == "s3v4"

    @pytest.mark.asyncio
    async def test_close_releases_client(self):
        session, _ = _make_mock_session(put_response={"ETag": '"e"'})
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        await store.upload("key", b"x")
        assert store._client is not None

        await store.close()
        assert store._client is None

    @pytest.mark.asyncio
    async def test_context_manager_closes(self):
        session, _ = _make_mock_session(put_response={"ETag": '"e"'})
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        async with store as s:
            await s.upload("key", b"x")

        assert store._client is None

    @pytest.mark.asyncio
    async def test_endpoint_url_passed_to_client(self):
        session, _ = _make_mock_session(put_response={"ETag": '"e"'})
        store = AwsAsyncBlobStore(
            bucket="b",
            endpoint_url="http://localhost:4566",
            _session=session,
        )

        await store.upload("key", b"x")

        session.create_client.assert_called_once_with(
            "s3", endpoint_url="http://localhost:4566", config=ANY
        )


class TestAwsAsyncConcurrentInit:
    """Verify _ensure_client is safe under concurrent first use."""

    @pytest.mark.asyncio
    async def test_concurrent_uploads_create_single_client(self):
        """asyncio.gather on a fresh store must only create one client."""
        session, _ = _make_mock_session(put_response={"ETag": '"e"'})
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        import asyncio

        await asyncio.gather(*[store.upload(f"k{i}", b"x") for i in range(20)])

        session.create_client.assert_called_once()


class TestAwsStripQuotes:
    """Tests for the _strip_quotes helper."""

    def test_strip_quotes_none_returns_none(self):
        """_strip_quotes(None) returns None."""
        _strip_quotes = _aws_store_mod._strip_quotes
        assert _strip_quotes(None) is None

    def test_strip_quotes_removes_surrounding_quotes(self):
        """_strip_quotes removes surrounding double quotes."""
        _strip_quotes = _aws_store_mod._strip_quotes
        assert _strip_quotes('"abc123"') == "abc123"

    def test_strip_quotes_no_quotes_unchanged(self):
        """_strip_quotes leaves unquoted strings unchanged."""
        _strip_quotes = _aws_store_mod._strip_quotes
        assert _strip_quotes("abc123") == "abc123"


class TestAwsAsyncRegistration:
    """Provider registry integration."""

    def test_aws_registered_in_async_registry(self):
        assert ASYNC_BLOB_REGISTRY.get("aws") is AwsAsyncBlobStore


class TestAwsAsyncTags:
    """get_tags / set_tags via S3 tagging APIs."""

    @pytest.mark.asyncio
    async def test_get_tags_parses_tag_set(self):
        session, mock_client = _make_mock_session(put_response={"ETag": '"e"'})
        mock_client.get_object_tagging.return_value = {
            "TagSet": [
                {"Key": "env", "Value": "prod"},
                {"Key": "team", "Value": "multicloud"},
            ]
        }
        store = AwsAsyncBlobStore(bucket="my-bucket", _session=session)

        result = await store.get_tags("my-key")

        assert result == {"env": "prod", "team": "multicloud"}
        call_kwargs = mock_client.get_object_tagging.call_args[1]
        assert call_kwargs == {"Bucket": "my-bucket", "Key": "my-key"}

    @pytest.mark.asyncio
    async def test_get_tags_empty_when_no_tag_set(self):
        session, mock_client = _make_mock_session(put_response={"ETag": '"e"'})
        mock_client.get_object_tagging.return_value = {}
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        assert await store.get_tags("key") == {}

    @pytest.mark.asyncio
    async def test_set_tags_sends_tag_set(self):
        session, mock_client = _make_mock_session(put_response={"ETag": '"e"'})
        store = AwsAsyncBlobStore(bucket="my-bucket", _session=session)

        await store.set_tags("my-key", {"env": "prod"})

        call_kwargs = mock_client.put_object_tagging.call_args[1]
        assert call_kwargs["Bucket"] == "my-bucket"
        assert call_kwargs["Key"] == "my-key"
        assert call_kwargs["Tagging"] == {"TagSet": [{"Key": "env", "Value": "prod"}]}

    @pytest.mark.asyncio
    async def test_get_tags_propagates_native_error(self):
        # The provider lets native errors propagate; translation happens at
        # the client layer (see test_async_client.py).
        session, mock_client = _make_mock_session(put_response={"ETag": '"e"'})
        mock_client.get_object_tagging.side_effect = _make_client_error(
            "NoSuchKey", 404
        )
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(ClientError):
            await store.get_tags("missing")

    @pytest.mark.asyncio
    async def test_set_tags_propagates_native_error(self):
        session, mock_client = _make_mock_session(put_response={"ETag": '"e"'})
        mock_client.put_object_tagging.side_effect = _make_client_error(
            "AccessDenied", 403
        )
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        with pytest.raises(ClientError):
            await store.set_tags("key", {"env": "prod"})

    def test_translate_error_maps_tagging_no_such_key(self):
        session, _ = _make_mock_session()
        store = AwsAsyncBlobStore(bucket="b", _session=session)

        result = store.translate_error(_make_client_error("NoSuchKey", 404))

        assert isinstance(result, ResourceNotFoundException)
