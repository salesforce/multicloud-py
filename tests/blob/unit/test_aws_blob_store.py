"""Unit tests for :class:`AwsBlobStore` — the AWS S3 provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from multicloud.blob.blob_aws.aws_blob_store import (
    AwsBlobStore,
    _build_range_header,
    _strip_quotes,
)
from multicloud.blob.blob_client.driver import (
    BlobInfo,
    BlobMetadata,
    DownloadResponse,
    ListBlobsPageResponse,
    ObjectLockConfiguration,
    ObjectLockInfo,
    ObjectRetentionConfig,
    PresignedOperation,
    PresignedUrlResponse,
    ReplicationStatus,
    RetentionMode,
    UploadResponse,
)
from multicloud.common.exceptions import (
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceConflictException,
    ResourceExistsException,
    ResourceNotFoundException,
    SubstrateSdkException,
    UnAuthorizedException,
    UnknownException,
    UnsupportedOperationException,
)


@pytest.fixture
def mock_s3():
    """A mock boto3 S3 client."""
    return MagicMock()


@pytest.fixture
def store(mock_s3):
    """An AwsBlobStore with an injected mock S3 client."""
    return AwsBlobStore(bucket="test-bucket", region="us-east-1", s3_client=mock_s3)


class TestAwsBlobStoreConstruction:
    """Tests for AwsBlobStore initialization."""

    def test_with_injected_client(self, mock_s3):
        store = AwsBlobStore(bucket="b", region="us-west-2", s3_client=mock_s3)
        assert store.bucket == "b"
        assert store.region == "us-west-2"

    def test_empty_bucket_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Bucket name cannot be null or empty"
        ):
            AwsBlobStore(bucket="", s3_client=MagicMock())

    def test_provider_id_is_aws(self, store):
        assert store.get_provider_id() == "aws"

    @patch("multicloud.blob.blob_aws.aws_blob_store.get_session")
    def test_creates_s3_client_when_no_injection(self, mock_get_session):
        # No s3_client and no credentials_overrider → resolve a session
        # via the shared helper and build the S3 client from it.
        session = MagicMock()
        mock_get_session.return_value = session
        AwsBlobStore(bucket="b", region="eu-west-1", endpoint_url="http://local:9000")
        mock_get_session.assert_called_once_with(None, "eu-west-1")
        session.client.assert_called_once_with(
            "s3",
            region_name="eu-west-1",
            endpoint_url="http://local:9000",
            config=ANY,
        )
        # The client must be pinned to SigV4 -- KMS buckets reject SigV2.
        cfg = session.client.call_args.kwargs["config"]
        assert cfg.signature_version == "s3v4"


class TestAwsBlobStoreCredentialsOverrider:
    """Tests for credentials_overrider wiring on AwsBlobStore."""

    @patch("multicloud.blob.blob_aws.aws_blob_store.get_session")
    def test_overrider_forwarded_to_get_session(self, mock_get_session):
        from multicloud.common.credentials import (
            CredentialsOverrider,
            CredentialsType,
            StsCredentials,
        )

        session = MagicMock()
        mock_get_session.return_value = session
        overrider = CredentialsOverrider(
            credentials_type=CredentialsType.SESSION,
            session_credentials=StsCredentials(
                access_key_id="AK",
                access_key_secret="SK",
                security_token="ST",
            ),
        )
        AwsBlobStore(bucket="b", region="us-east-1", credentials_overrider=overrider)
        mock_get_session.assert_called_once_with(overrider, "us-east-1")
        session.client.assert_called_once_with(
            "s3", region_name="us-east-1", config=ANY
        )

    @patch("multicloud.blob.blob_aws.aws_blob_store.get_session")
    def test_s3_client_takes_precedence_over_overrider(self, mock_get_session):
        # If both ``s3_client`` and ``credentials_overrider`` are passed,
        # the injected client wins and the overrider is silently ignored.
        from multicloud.common.credentials import (
            CredentialsOverrider,
            CredentialsType,
            StsCredentials,
        )

        injected = MagicMock()
        overrider = CredentialsOverrider(
            credentials_type=CredentialsType.SESSION,
            session_credentials=StsCredentials(
                access_key_id="AK",
                access_key_secret="SK",
                security_token="ST",
            ),
        )
        store = AwsBlobStore(
            bucket="b",
            s3_client=injected,
            credentials_overrider=overrider,
        )
        assert store._s3 is injected
        mock_get_session.assert_not_called()


class TestAwsBlobStoreUpload:
    """Tests for do_upload."""

    def test_upload_basic(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"abc123"', "VersionId": "v1"}
        resp = store.upload("path/obj.txt", b"content")
        assert resp.key == "path/obj.txt"
        assert resp.e_tag == "abc123"  # quotes stripped
        assert resp.version_id == "v1"
        mock_s3.put_object.assert_called_once()

    def test_upload_with_content_type(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", content_type="application/json")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["ContentType"] == "application/json"

    def test_upload_with_metadata(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", metadata={"env": "staging"})
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Metadata"] == {"env": "staging"}

    def test_upload_with_tags(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", tags={"env": "prod", "team": "sdk"})
        call_kwargs = mock_s3.put_object.call_args[1]
        # S3 expects tags as a URL-encoded query string in the Tagging param.
        tagging = call_kwargs["Tagging"]
        pairs = sorted(tagging.split("&"))
        assert pairs == ["env=prod", "team=sdk"]

    def test_upload_tags_url_encodes_special_chars(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", tags={"k ey": "v=lue&more"})
        call_kwargs = mock_s3.put_object.call_args[1]
        # Spaces and reserved chars must be percent-encoded so the tag string
        # parses unambiguously on the S3 side.
        assert call_kwargs["Tagging"] == "k%20ey=v%3Dlue%26more"

    def test_upload_tags_url_encodes_non_ascii_chars(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        # CJK + emoji must round-trip via UTF-8 percent-encoding so callers
        # in non-English locales aren't blocked by the tag header format.
        store.upload("k", b"x", tags={"作者": "张三🚀"})
        call_kwargs = mock_s3.put_object.call_args[1]
        assert (
            call_kwargs["Tagging"]
            == "%E4%BD%9C%E8%80%85=%E5%BC%A0%E4%B8%89%F0%9F%9A%80"
        )

    def test_upload_without_tags_omits_tagging_kwarg(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert "Tagging" not in call_kwargs

    def test_upload_with_tags_and_metadata(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", metadata={"env": "dev"}, tags={"team": "sdk"})
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Metadata"] == {"env": "dev"}
        assert call_kwargs["Tagging"] == "team=sdk"

    def test_upload_with_retention_object_lock(self, store, mock_s3):
        from datetime import datetime, timedelta, timezone

        mock_s3.put_object.return_value = {"ETag": '"e"'}
        retain_until = datetime.now(timezone.utc) + timedelta(days=1)
        store.upload(
            "k",
            b"x",
            object_lock=ObjectLockConfiguration(
                mode=RetentionMode.COMPLIANCE, retain_until_date=retain_until
            ),
        )
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["ObjectLockMode"] == "COMPLIANCE"
        assert call_kwargs["ObjectLockRetainUntilDate"] == retain_until
        assert "ObjectLockLegalHoldStatus" not in call_kwargs

    def test_upload_with_legal_hold_object_lock(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", object_lock=ObjectLockConfiguration(legal_hold=True))
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["ObjectLockLegalHoldStatus"] == "ON"
        assert "ObjectLockMode" not in call_kwargs

    def test_upload_without_object_lock_omits_lock_kwargs(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert "ObjectLockMode" not in call_kwargs
        assert "ObjectLockRetainUntilDate" not in call_kwargs
        assert "ObjectLockLegalHoldStatus" not in call_kwargs

    def test_upload_event_based_hold_flag_ignored(self, store, mock_s3):
        # S3 has no event-based hold; use_event_based_hold must not change the
        # single legal-hold flag S3 exposes.
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload(
            "k",
            b"x",
            object_lock=ObjectLockConfiguration(
                legal_hold=True, use_event_based_hold=True
            ),
        )
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["ObjectLockLegalHoldStatus"] == "ON"

    def test_upload_event_based_hold_logs_debug(self, store, mock_s3, caplog):
        # The unsupported flag is logged at DEBUG rather than silently dropped.
        import logging

        mock_s3.put_object.return_value = {"ETag": '"e"'}
        with caplog.at_level(
            logging.DEBUG, logger="multicloud.blob.blob_aws.aws_blob_store"
        ):
            store.upload(
                "k",
                b"x",
                object_lock=ObjectLockConfiguration(use_event_based_hold=True),
            )
        assert any(
            "use_event_based_hold" in record.message for record in caplog.records
        )

    def test_upload_from_path_with_object_lock(self, store, mock_s3, tmp_path):
        from datetime import datetime, timedelta, timezone

        mock_s3.put_object.return_value = {"ETag": '"e"'}
        src = tmp_path / "f.txt"
        src.write_bytes(b"data")
        retain_until = datetime.now(timezone.utc) + timedelta(days=1)
        store.upload_from_path(
            "k",
            src,
            object_lock=ObjectLockConfiguration(
                mode=RetentionMode.GOVERNANCE,
                retain_until_date=retain_until,
                legal_hold=True,
            ),
        )
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["ObjectLockMode"] == "GOVERNANCE"
        assert call_kwargs["ObjectLockRetainUntilDate"] == retain_until
        assert call_kwargs["ObjectLockLegalHoldStatus"] == "ON"

    def test_upload_client_error_not_translated_at_store_level(self, store, mock_s3):
        # Model B: do_upload has no provider catch, so the raw boto error
        # surfaces when calling the store directly. Translation happens at
        # the BucketClient layer (see TestBucketClientErrorTranslation).
        mock_s3.put_object.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "no"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "PutObject",
        )
        with pytest.raises(ClientError):
            store.upload("k", b"data")

    def test_upload_no_credentials_error_not_translated_at_store_level(
        self, store, mock_s3
    ):
        mock_s3.put_object.side_effect = NoCredentialsError()
        with pytest.raises(NoCredentialsError):
            store.upload("k", b"data")

    def test_upload_botocore_error_not_translated_at_store_level(self, store, mock_s3):
        mock_s3.put_object.side_effect = BotoCoreError()
        with pytest.raises(BotoCoreError):
            store.upload("k", b"data")


def _precondition_failed_error(operation="PutObject"):
    """Build the ClientError S3 raises for a failed 412 precondition."""
    return ClientError(
        {
            "Error": {"Code": "PreconditionFailed", "Message": "precondition failed"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        },
        operation,
    )


def _no_such_key_error(operation="PutObject"):
    """Build the ClientError S3 raises when the target object does not exist."""
    return ClientError(
        {
            "Error": {
                "Code": "NoSuchKey",
                "Message": "The specified key does not exist.",
            },
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        operation,
    )


def _no_such_bucket_error(operation="PutObject"):
    """Build the ClientError S3 raises when the target bucket does not exist."""
    return ClientError(
        {
            "Error": {
                "Code": "NoSuchBucket",
                "Message": "The specified bucket does not exist.",
            },
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        operation,
    )


def _no_such_access_point_error(operation="PutObject"):
    """Build the ClientError S3 raises when the access-point ARN does not exist."""
    return ClientError(
        {
            "Error": {
                "Code": "NoSuchAccessPoint",
                "Message": "The specified access point does not exist.",
            },
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        operation,
    )


class TestAwsBlobStoreWritePreconditions:
    """create_if_absent / if_match conditional-write behavior on upload."""

    def test_create_if_absent_sends_if_none_match_star(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", create_if_absent=True)
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["IfNoneMatch"] == "*"
        assert "IfMatch" not in call_kwargs

    def test_if_match_sends_if_match_header(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", if_match="etag-abc")
        call_kwargs = mock_s3.put_object.call_args[1]
        # download() returns a quote-stripped ETag; the write path re-quotes it
        # so the natural round-trip matches S3's quoted stored ETag.
        assert call_kwargs["IfMatch"] == '"etag-abc"'
        assert "IfNoneMatch" not in call_kwargs

    def test_no_preconditions_omits_headers(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert "IfNoneMatch" not in call_kwargs
        assert "IfMatch" not in call_kwargs

    def test_create_if_absent_and_if_match_rejected_by_validation(self, store):
        # The driver validates before delegating to do_upload.
        with pytest.raises(InvalidArgumentException, match="mutually exclusive"):
            store.upload("k", b"x", create_if_absent=True, if_match="etag")

    def test_create_if_absent_conflict_raises_resource_exists(self, store, mock_s3):
        # A 412 on a create-if-absent write means the object already exists;
        # the provider re-labels it as ResourceExistsException locally.
        mock_s3.put_object.side_effect = _precondition_failed_error()
        with pytest.raises(ResourceExistsException):
            store.upload("k", b"x", create_if_absent=True)

    def test_if_match_precondition_failure_propagates_raw(self, store, mock_s3):
        # A 412 on an if_match write is NOT re-labelled at the store; the raw
        # boto error propagates and the client layer maps it via translate_error.
        mock_s3.put_object.side_effect = _precondition_failed_error()
        with pytest.raises(ClientError):
            store.upload("k", b"x", if_match="etag-abc")

    def test_if_match_on_missing_object_raises_failed_precondition(
        self, store, mock_s3
    ):
        # An if_match overwrite whose target does not exist returns 404
        # NoSuchKey. A missing object cannot satisfy an ETag precondition, so
        # the store normalizes it to FailedPreconditionException instead of the
        # 404 -> ResourceNotFoundException the client would otherwise produce.
        mock_s3.put_object.side_effect = _no_such_key_error()
        with pytest.raises(FailedPreconditionException):
            store.upload("k", b"x", if_match="etag-abc")

    def test_if_match_on_missing_bucket_raises_resource_not_found(self, store, mock_s3):
        # A missing BUCKET on an if_match write is a genuine not-found, NOT a
        # precondition mismatch. The store must not relabel NoSuchBucket to
        # FailedPreconditionException; it propagates raw so the client layer
        # maps it to ResourceNotFoundException.
        mock_s3.put_object.side_effect = _no_such_bucket_error()
        with pytest.raises(ClientError) as excinfo:
            store.upload("k", b"x", if_match="etag-abc")
        assert not isinstance(excinfo.value, FailedPreconditionException)
        assert isinstance(
            store.translate_error(excinfo.value), ResourceNotFoundException
        )

    def test_if_match_on_missing_access_point_raises_resource_not_found(
        self, store, mock_s3
    ):
        # An access-point ARN may be passed as the bucket; a missing one returns
        # NoSuchAccessPoint (also 404). Like a missing bucket, this is a genuine
        # not-found and must NOT be relabelled to FailedPreconditionException;
        # it propagates raw so the client layer maps it to ResourceNotFound.
        mock_s3.put_object.side_effect = _no_such_access_point_error()
        with pytest.raises(ClientError) as excinfo:
            store.upload("k", b"x", if_match="etag-abc")
        assert not isinstance(excinfo.value, FailedPreconditionException)
        assert isinstance(
            store.translate_error(excinfo.value), ResourceNotFoundException
        )

    def test_unconditional_missing_object_error_not_relabelled(self, store, mock_s3):
        # Without a conditional precondition, a 404 is a genuine not-found and
        # must NOT be re-labelled as a precondition failure; it propagates raw
        # for the client layer to map to ResourceNotFoundException.
        mock_s3.put_object.side_effect = _no_such_key_error()
        with pytest.raises(ClientError):
            store.upload("k", b"x")

    def test_upload_from_path_if_match_on_missing_object_raises_failed_precondition(
        self, store, mock_s3, tmp_path
    ):
        src = tmp_path / "f.txt"
        src.write_bytes(b"data")
        mock_s3.put_object.side_effect = _no_such_key_error()
        with pytest.raises(FailedPreconditionException):
            store.upload_from_path("k", src, if_match="etag-abc")

    def test_upload_from_path_create_if_absent(self, store, mock_s3, tmp_path):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        src = tmp_path / "f.txt"
        src.write_bytes(b"data")
        store.upload_from_path("k", src, create_if_absent=True)
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["IfNoneMatch"] == "*"

    def test_if_version_match_unsupported_on_s3(self, store, mock_s3):
        # S3 compare-and-swap is ETag-based (IfMatch); a version-conditioned
        # write cannot be expressed, so the provider refuses it rather than
        # silently doing an unconditional overwrite.
        with pytest.raises(UnsupportedOperationException, match="if_version_match"):
            store.upload("k", b"x", if_version_match="42")
        mock_s3.put_object.assert_not_called()

    def test_if_version_match_unsupported_on_upload_from_path(
        self, store, mock_s3, tmp_path
    ):
        src = tmp_path / "f.txt"
        src.write_bytes(b"data")
        with pytest.raises(UnsupportedOperationException, match="if_version_match"):
            store.upload_from_path("k", src, if_version_match="42")
        mock_s3.put_object.assert_not_called()

    def test_if_match_and_if_version_match_together_uses_if_match(self, store, mock_s3):
        # A portable caller may set both tokens so each provider enforces with
        # the one it supports. On S3 that is IfMatch; if_version_match is
        # redundant here and must be ignored rather than trigger the
        # "unsupported" rejection that fires when it arrives alone.
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", if_match="etag-abc", if_version_match="42")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["IfMatch"] == '"etag-abc"'

    def test_download_etag_roundtrips_into_if_match(self, store, mock_s3):
        # The download().metadata.e_tag -> if_match round-trip must send S3 the
        # quoted ETag it stored, even though download strips the quotes. Simulate
        # a real object whose stored ETag is '"abc123"'.
        body = MagicMock()
        body.read.return_value = b""
        mock_s3.get_object.return_value = {
            "Body": body,
            "ContentLength": 0,
            "Metadata": {},
            "ETag": '"abc123"',
        }
        downloaded = store.download("k")
        assert downloaded.metadata.e_tag == "abc123"  # quotes stripped on read

        mock_s3.put_object.return_value = {"ETag": '"def456"'}
        store.upload("k", b"x", if_match=downloaded.metadata.e_tag)
        # The bare token from download is re-quoted to match S3's stored form.
        assert mock_s3.put_object.call_args[1]["IfMatch"] == '"abc123"'

    def test_if_match_already_quoted_is_not_double_quoted(self, store, mock_s3):
        # A caller who kept S3's raw quoted ETag must not get it double-quoted.
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", if_match='"already-quoted"')
        assert mock_s3.put_object.call_args[1]["IfMatch"] == '"already-quoted"'


class TestAwsBlobStoreKmsEncryption:
    """Server-side KMS encryption on upload."""

    def test_kms_key_id_sets_sse_and_key(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", kms_key_id="arn:aws:kms:us-east-1:1:key/abc")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert call_kwargs["SSEKMSKeyId"] == "arn:aws:kms:us-east-1:1:key/abc"

    def test_use_kms_managed_key_sets_sse_without_key(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", use_kms_managed_key=True)
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert "SSEKMSKeyId" not in call_kwargs

    def test_explicit_key_takes_precedence_over_managed_flag(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", kms_key_id="key-1", use_kms_managed_key=True)
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert call_kwargs["SSEKMSKeyId"] == "key-1"

    def test_no_kms_omits_encryption_headers(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert "ServerSideEncryption" not in call_kwargs
        assert "SSEKMSKeyId" not in call_kwargs

    def test_translate_error_maps_412_to_failed_precondition(self, store):
        # A generic 412 (e.g. an if_match mismatch) maps to
        # FailedPreconditionException at the client layer.
        result = store.translate_error(_precondition_failed_error())
        assert isinstance(result, FailedPreconditionException)

    def test_translate_error_maps_409_conflict_to_resource_conflict(self, store):
        # A concurrent operation racing a conditional write returns
        # ConditionalRequestConflict (HTTP 409 on S3). Matched by error
        # code, not HTTP 409 in general. Maps to ResourceConflictException,
        # NOT FailedPreconditionException, so callers can retry.
        conflict = ClientError(
            {
                "Error": {
                    "Code": "ConditionalRequestConflict",
                    "Message": "conditional request conflict",
                },
                "ResponseMetadata": {"HTTPStatusCode": 409},
            },
            "PutObject",
        )
        result = store.translate_error(conflict)
        assert isinstance(result, ResourceConflictException)
        assert not isinstance(result, FailedPreconditionException)

    def test_translate_error_generic_409_is_not_resource_conflict(self, store):
        # HTTP 409 is used by unrelated AWS APIs (IAM LimitExceeded, S3
        # BucketAlreadyExists). Only ConditionalRequestConflict is the S3
        # retry signal; a generic 409 must not claim that mapping.
        generic_409 = ClientError(
            {
                "Error": {"Code": "LimitExceeded", "Message": "LimitExceeded"},
                "ResponseMetadata": {"HTTPStatusCode": 409},
            },
            "SomeOperation",
        )
        result = store.translate_error(generic_409)
        assert isinstance(result, UnknownException)
        assert not isinstance(result, ResourceConflictException)


class TestAwsBlobStoreStorageClass:
    """Storage class selection on upload."""

    def test_upload_sets_storage_class(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x", storage_class="STANDARD_IA")
        assert mock_s3.put_object.call_args[1]["StorageClass"] == "STANDARD_IA"

    def test_upload_without_storage_class_omits_it(self, store, mock_s3):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store.upload("k", b"x")
        assert "StorageClass" not in mock_s3.put_object.call_args[1]

    def test_upload_from_path_sets_storage_class(self, store, mock_s3, tmp_path):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        src = tmp_path / "f.txt"
        src.write_bytes(b"x")
        store.upload_from_path("k", src, storage_class="GLACIER_IR")
        assert mock_s3.put_object.call_args[1]["StorageClass"] == "GLACIER_IR"

    def test_upload_from_path_without_storage_class_omits_it(
        self, store, mock_s3, tmp_path
    ):
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        src = tmp_path / "f.txt"
        src.write_bytes(b"x")
        store.upload_from_path("k", src)
        assert "StorageClass" not in mock_s3.put_object.call_args[1]

    def test_blank_storage_class_rejected_before_request(self, store, mock_s3):
        with pytest.raises(InvalidArgumentException, match="storage_class"):
            store.upload("k", b"x", storage_class="  ")
        mock_s3.put_object.assert_not_called()

    def test_unknown_storage_class_translates_to_invalid_argument(self, store):
        exc = ClientError(
            {
                "Error": {
                    "Code": "InvalidStorageClass",
                    "Message": "The storage class you specified is not valid",
                },
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "PutObject",
        )
        assert isinstance(store.translate_error(exc), InvalidArgumentException)


class TestAwsBlobStoreReplicationStatus:
    """replication_status is populated on download and get_metadata."""

    def _download_resp(self, replication_status):
        body_mock = MagicMock()
        body_mock.read.return_value = b""
        resp = {"Body": body_mock, "ContentLength": 0, "Metadata": {}}
        if replication_status is not None:
            resp["ReplicationStatus"] = replication_status
        return resp

    def test_download_maps_completed_to_complete(self, store, mock_s3):
        # S3's terminal "COMPLETED" normalizes to the common COMPLETE.
        mock_s3.get_object.return_value = self._download_resp("COMPLETED")
        resp = store.download("k")
        assert resp.metadata.replication_status == ReplicationStatus.COMPLETE

    def test_download_maps_pending(self, store, mock_s3):
        mock_s3.get_object.return_value = self._download_resp("PENDING")
        resp = store.download("k")
        assert resp.metadata.replication_status == ReplicationStatus.PENDING

    def test_download_maps_failed(self, store, mock_s3):
        mock_s3.get_object.return_value = self._download_resp("FAILED")
        resp = store.download("k")
        assert resp.metadata.replication_status == ReplicationStatus.FAILED

    def test_download_maps_replica(self, store, mock_s3):
        mock_s3.get_object.return_value = self._download_resp("REPLICA")
        resp = store.download("k")
        assert resp.metadata.replication_status == ReplicationStatus.REPLICA

    def test_download_absent_header_is_none(self, store, mock_s3):
        mock_s3.get_object.return_value = self._download_resp(None)
        resp = store.download("k")
        assert resp.metadata.replication_status is None

    def test_get_metadata_maps_replication_status(self, store, mock_s3):
        mock_s3.head_object.return_value = {
            "ContentLength": 5,
            "Metadata": {},
            "ReplicationStatus": "COMPLETED",
        }
        meta = store.get_metadata("k")
        assert meta.replication_status == ReplicationStatus.COMPLETE

    def test_get_metadata_absent_header_is_none(self, store, mock_s3):
        mock_s3.head_object.return_value = {"ContentLength": 5, "Metadata": {}}
        meta = store.get_metadata("k")
        assert meta.replication_status is None


class TestAwsBlobStoreDownloadPreconditions:
    """if_match conditional-read header on download."""

    def _empty_resp(self):
        body_mock = MagicMock()
        body_mock.read.return_value = b""
        return {"Body": body_mock, "ContentLength": 0, "Metadata": {}}

    def test_if_match_sends_header(self, store, mock_s3):
        mock_s3.get_object.return_value = self._empty_resp()
        store.download("k", if_match="etag-abc")
        call_kwargs = mock_s3.get_object.call_args[1]
        # Conditional read re-quotes so the download().e_tag -> if_match
        # round-trip matches S3's quoted stored ETag.
        assert call_kwargs["IfMatch"] == '"etag-abc"'
        assert "IfNoneMatch" not in call_kwargs

    def test_no_preconditions_omits_headers(self, store, mock_s3):
        mock_s3.get_object.return_value = self._empty_resp()
        store.download("k")
        call_kwargs = mock_s3.get_object.call_args[1]
        assert "IfMatch" not in call_kwargs
        assert "IfNoneMatch" not in call_kwargs


class TestAwsBlobStoreDownload:
    """Tests for do_download."""

    def test_download_basic(self, store, mock_s3):
        from datetime import datetime, timezone

        last_modified = datetime(2025, 4, 5, 6, 7, 8, tzinfo=timezone.utc)
        body_mock = MagicMock()
        body_mock.read.return_value = b"file content"
        mock_s3.get_object.return_value = {
            "Body": body_mock,
            "ContentLength": 12,
            "ContentType": "text/plain",
            "ETag": '"etag"',
            "VersionId": "v2",
            "Metadata": {"x": "y"},
            "LastModified": last_modified,
        }
        resp = store.download("path/file.txt")
        assert resp.key == "path/file.txt"
        assert resp.content == b"file content"
        assert resp.metadata.content_length == 12
        assert resp.metadata.content_type == "text/plain"
        assert resp.metadata.e_tag == "etag"
        assert resp.metadata.version_id == "v2"
        assert resp.metadata.metadata == {"x": "y"}
        # download() shares _build_metadata with get_metadata(), so the same
        # get_object response surfaces LastModified on both paths (no longer
        # None on download — see PR #98 review).
        assert resp.metadata.last_modified == last_modified
        assert resp.metadata.created_time == last_modified

    def test_download_with_version_id(self, store, mock_s3):
        body_mock = MagicMock()
        body_mock.read.return_value = b""
        mock_s3.get_object.return_value = {
            "Body": body_mock,
            "ContentLength": 0,
            "Metadata": {},
        }
        store.download("k", version_id="v5")
        call_kwargs = mock_s3.get_object.call_args[1]
        assert call_kwargs["VersionId"] == "v5"

    def test_download_with_byte_range(self, store, mock_s3):
        body_mock = MagicMock()
        body_mock.read.return_value = b"partial"
        mock_s3.get_object.return_value = {
            "Body": body_mock,
            "ContentLength": 7,
            "Metadata": {},
        }
        store.download("k", start=10, end=20)
        call_kwargs = mock_s3.get_object.call_args[1]
        assert call_kwargs["Range"] == "bytes=10-20"

    def test_download_not_found_error(self, store, mock_s3):
        mock_s3.get_object.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": "not found"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObject",
        )
        with pytest.raises(ResourceNotFoundException):
            store.download("missing-key")


class TestAwsBlobStoreDelete:
    """Tests for do_delete."""

    def test_delete_calls_s3(self, store, mock_s3):
        mock_s3.delete_object.return_value = {}
        store.delete("path/to/obj")
        mock_s3.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="path/to/obj"
        )

    def test_delete_with_version_id(self, store, mock_s3):
        mock_s3.delete_object.return_value = {}
        store.delete("path/to/obj", version_id="abc123")
        mock_s3.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="path/to/obj", VersionId="abc123"
        )

    def test_delete_error_not_translated_at_store_level(self, store, mock_s3):
        # do_delete has no provider catch; raw boto error surfaces at store level.
        # Translation happens at the BucketClient layer.
        mock_s3.delete_object.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "nope"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "DeleteObject",
        )
        with pytest.raises(ClientError):
            store.delete("k")


def _no_object_lock_error(operation):
    """Build the ClientError S3 raises when a lock sub-resource is unset."""
    return ClientError(
        {
            "Error": {
                "Code": "NoSuchObjectLockConfiguration",
                "Message": "no lock",
            },
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        operation,
    )


class TestAwsBlobStoreGetObjectLock:
    """Tests for do_get_object_lock (S3 GetObjectRetention + GetObjectLegalHold).

    S3 exposes retention and legal hold as two independent sub-resources; each
    call raises NoSuchObjectLockConfiguration when that piece is unset, so the
    provider treats that error as "this piece is absent" per call.
    """

    def test_assembles_retention_and_legal_hold(self, store, mock_s3):
        from datetime import datetime

        until = datetime(2030, 1, 1, 0, 0, 0)
        mock_s3.get_object_retention.return_value = {
            "Retention": {"Mode": "COMPLIANCE", "RetainUntilDate": until}
        }
        mock_s3.get_object_legal_hold.return_value = {"LegalHold": {"Status": "ON"}}

        info = store.get_object_lock("k")

        assert isinstance(info, ObjectLockInfo)
        assert info.mode is RetentionMode.COMPLIANCE
        assert info.retain_until_date == until
        assert info.legal_hold is True
        assert info.event_based_hold is None

    def test_retention_only_legal_hold_unset(self, store, mock_s3):
        # Retention is set; the legal-hold call raises
        # NoSuchObjectLockConfiguration because that piece is unset. The
        # provider maps the missing legal hold to False.
        from datetime import datetime

        until = datetime(2030, 1, 1)
        mock_s3.get_object_retention.return_value = {
            "Retention": {"Mode": "GOVERNANCE", "RetainUntilDate": until}
        }
        mock_s3.get_object_legal_hold.side_effect = _no_object_lock_error(
            "GetObjectLegalHold"
        )

        info = store.get_object_lock("k")

        assert info.mode is RetentionMode.GOVERNANCE
        assert info.retain_until_date == until
        assert info.legal_hold is False

    def test_legal_hold_only_retention_unset(self, store, mock_s3):
        # Legal hold is ON; the retention call raises
        # NoSuchObjectLockConfiguration because that piece is unset. The
        # provider maps the missing retention to mode=None.
        mock_s3.get_object_retention.side_effect = _no_object_lock_error(
            "GetObjectRetention"
        )
        mock_s3.get_object_legal_hold.return_value = {"LegalHold": {"Status": "ON"}}

        info = store.get_object_lock("k")

        assert info.mode is None
        assert info.retain_until_date is None
        assert info.legal_hold is True

    def test_no_lock_configured_returns_none(self, store, mock_s3):
        # Neither piece is set: both calls raise NoSuchObjectLockConfiguration.
        # The provider returns None rather than propagating.
        mock_s3.get_object_retention.side_effect = _no_object_lock_error(
            "GetObjectRetention"
        )
        mock_s3.get_object_legal_hold.side_effect = _no_object_lock_error(
            "GetObjectLegalHold"
        )

        assert store.get_object_lock("k") is None

    def test_passes_version_id(self, store, mock_s3):
        mock_s3.get_object_retention.return_value = {
            "Retention": {"Mode": "GOVERNANCE"}
        }
        mock_s3.get_object_legal_hold.return_value = {"LegalHold": {"Status": "OFF"}}

        info = store.get_object_lock("k", version_id="v9")

        mock_s3.get_object_retention.assert_called_once_with(
            Bucket="test-bucket", Key="k", VersionId="v9"
        )
        mock_s3.get_object_legal_hold.assert_called_once_with(
            Bucket="test-bucket", Key="k", VersionId="v9"
        )
        # Complete the round-trip: the versioned read must still parse into a
        # populated result, not silently return None.
        assert info is not None
        assert info.mode is RetentionMode.GOVERNANCE
        assert info.legal_hold is False

    def test_missing_object_propagates(self, store, mock_s3):
        # A missing object surfaces as NoSuchKey. This is not a
        # NoSuchObjectLockConfiguration error, so it propagates for the client
        # layer to translate.
        mock_s3.get_object_retention.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": "missing"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObjectRetention",
        )
        with pytest.raises(ClientError):
            store.get_object_lock("missing")

    def test_bucket_without_object_lock_propagates(self, store, mock_s3):
        # On an ordinary bucket that never enabled Object Lock, S3 raises
        # InvalidRequest (HTTP 400) "Bucket is missing Object Lock
        # Configuration". This is not a NoSuchObjectLockConfiguration error, so
        # it propagates for the client layer to translate.
        mock_s3.get_object_retention.side_effect = ClientError(
            {
                "Error": {
                    "Code": "InvalidRequest",
                    "Message": "Bucket is missing Object Lock Configuration",
                },
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "GetObjectRetention",
        )
        with pytest.raises(ClientError):
            store.get_object_lock("k")


class TestAwsBlobStoreUpdateLegalHold:
    def test_enable_sends_status_on(self, store, mock_s3):
        store.update_legal_hold("k", True)
        mock_s3.put_object_legal_hold.assert_called_once_with(
            Bucket="test-bucket",
            Key="k",
            LegalHold={"Status": "ON"},
        )

    def test_disable_sends_status_off(self, store, mock_s3):
        store.update_legal_hold("k", False)
        mock_s3.put_object_legal_hold.assert_called_once_with(
            Bucket="test-bucket",
            Key="k",
            LegalHold={"Status": "OFF"},
        )

    def test_forwards_version_id(self, store, mock_s3):
        store.update_legal_hold("k", True, version_id="v9")
        mock_s3.put_object_legal_hold.assert_called_once_with(
            Bucket="test-bucket",
            Key="k",
            LegalHold={"Status": "ON"},
            VersionId="v9",
        )

    def test_missing_object_error_not_translated_at_store_level(self, store, mock_s3):
        mock_s3.put_object_legal_hold.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": "missing"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "PutObjectLegalHold",
        )
        with pytest.raises(ClientError):
            store.update_legal_hold("missing", True)


class TestAwsBlobStoreDeleteObjects:
    """Tests for do_delete_objects (batch delete)."""

    def test_batch_delete_calls_s3(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import BlobIdentifier

        mock_s3.delete_objects.return_value = {"Deleted": [{"Key": "a"}, {"Key": "b"}]}
        objects = [BlobIdentifier(key="a"), BlobIdentifier(key="b", version_id="v1")]
        resp = store.delete_objects(objects)

        mock_s3.delete_objects.assert_called_once_with(
            Bucket="test-bucket",
            Delete={
                "Objects": [
                    {"Key": "a"},
                    {"Key": "b", "VersionId": "v1"},
                ],
                "Quiet": False,
            },
        )
        assert resp.errors == []

    def test_batch_delete_collects_errors(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import BlobIdentifier

        mock_s3.delete_objects.return_value = {
            "Deleted": [{"Key": "a"}],
            "Errors": [
                {"Key": "b", "Code": "AccessDenied", "Message": "no access"},
            ],
        }
        objects = [BlobIdentifier(key="a"), BlobIdentifier(key="b")]
        resp = store.delete_objects(objects)

        assert len(resp.errors) == 1
        assert resp.errors[0].key == "b"
        assert resp.errors[0].code == "AccessDenied"
        assert resp.errors[0].message == "no access"

    def test_batch_delete_no_chunking(self, store, mock_s3):
        """All objects sent in a single S3 request (no internal chunking)."""
        from multicloud.blob.blob_client.driver import BlobIdentifier

        mock_s3.delete_objects.return_value = {"Deleted": [], "Errors": []}
        objects = [BlobIdentifier(key=f"key-{i}") for i in range(1500)]
        resp = store.delete_objects(objects)

        # Single call with all 1500 objects — caller responsible for chunking
        assert mock_s3.delete_objects.call_count == 1
        call_kwargs = mock_s3.delete_objects.call_args[1]
        assert len(call_kwargs["Delete"]["Objects"]) == 1500
        assert resp.errors == []


class TestAwsBlobStoreCopy:
    """Tests for do_copy (copy to another bucket)."""

    def test_copy_calls_s3(self, store, mock_s3):
        from datetime import datetime

        ts = datetime(2024, 6, 15, 12, 0, 0)
        mock_s3.copy_object.return_value = {
            "CopyObjectResult": {"ETag": '"abc123"', "LastModified": ts},
            "VersionId": "new-v1",
        }
        resp = store.copy("src.txt", dest_bucket="other-bucket", dest_key="dst.txt")

        mock_s3.copy_object.assert_called_once_with(
            CopySource={"Bucket": "test-bucket", "Key": "src.txt"},
            Bucket="other-bucket",
            Key="dst.txt",
        )
        assert resp.key == "dst.txt"
        assert resp.version_id == "new-v1"
        assert resp.e_tag == "abc123"
        assert resp.last_modified == ts

    def test_copy_with_version_id(self, store, mock_s3):
        mock_s3.copy_object.return_value = {
            "CopyObjectResult": {"ETag": '"e"', "LastModified": None},
        }
        store.copy(
            "src.txt",
            dest_bucket="dest",
            dest_key="dst.txt",
            src_version_id="v5",
        )
        call_kwargs = mock_s3.copy_object.call_args[1]
        assert call_kwargs["CopySource"]["VersionId"] == "v5"


class TestAwsBlobStoreCopyFrom:
    """Tests for do_copy_from (copy from source bucket into this bucket)."""

    def test_copy_from_calls_s3(self, store, mock_s3):
        from datetime import datetime

        ts = datetime(2024, 6, 15, 12, 0, 0)
        mock_s3.copy_object.return_value = {
            "CopyObjectResult": {"ETag": '"xyz"', "LastModified": ts},
            "VersionId": "local-v1",
        }
        resp = store.copy_from("source-bucket", "remote.txt", dest_key="local.txt")

        mock_s3.copy_object.assert_called_once_with(
            CopySource={"Bucket": "source-bucket", "Key": "remote.txt"},
            Bucket="test-bucket",
            Key="local.txt",
        )
        assert resp.key == "local.txt"
        assert resp.version_id == "local-v1"
        assert resp.e_tag == "xyz"
        assert resp.last_modified == ts

    def test_copy_from_with_version_id(self, store, mock_s3):
        mock_s3.copy_object.return_value = {
            "CopyObjectResult": {"ETag": '"e"', "LastModified": None},
        }
        store.copy_from(
            "source-bucket",
            "remote.txt",
            dest_key="local.txt",
            src_version_id="v9",
        )
        call_kwargs = mock_s3.copy_object.call_args[1]
        assert call_kwargs["CopySource"]["VersionId"] == "v9"


class TestAwsBlobStoreTranslateError:
    """Tests for error translation logic."""

    def test_already_translated_passthrough(self, store):
        exc = InvalidArgumentException("already")
        result = store.translate_error(exc)
        assert result is exc

    def test_no_credentials_error(self, store):
        exc = NoCredentialsError()
        result = store.translate_error(exc)
        assert isinstance(result, UnAuthorizedException)

    def test_client_error_404(self, store):
        exc = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": ""},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObject",
        )
        result = store.translate_error(exc)
        assert isinstance(result, ResourceNotFoundException)

    def test_client_error_403(self, store):
        exc = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": ""},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "GetObject",
        )
        result = store.translate_error(exc)
        assert isinstance(result, UnAuthorizedException)

    def test_client_error_400(self, store):
        exc = ClientError(
            {
                "Error": {"Code": "InvalidArgument", "Message": ""},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "PutObject",
        )
        result = store.translate_error(exc)
        assert isinstance(result, InvalidArgumentException)

    def test_client_error_unknown_code(self, store):
        exc = ClientError(
            {
                "Error": {"Code": "InternalError", "Message": ""},
                "ResponseMetadata": {"HTTPStatusCode": 500},
            },
            "PutObject",
        )
        result = store.translate_error(exc)
        assert isinstance(result, UnknownException)

    def test_botocore_error(self, store):
        exc = BotoCoreError()
        result = store.translate_error(exc)
        assert isinstance(result, InvalidArgumentException)

    def test_unknown_exception_type(self, store):
        exc = RuntimeError("unexpected")
        result = store.translate_error(exc)
        assert isinstance(result, UnknownException)

    def test_no_such_upload_maps_to_resource_not_found(self, store):
        """Multipart-specific: NoSuchUpload (e.g. completed/aborted upload_id)
        should surface as ResourceNotFoundException, not UnknownException."""
        exc = ClientError(
            {
                "Error": {"Code": "NoSuchUpload", "Message": "no such upload"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "CompleteMultipartUpload",
        )
        result = store.translate_error(exc)
        assert isinstance(result, ResourceNotFoundException)

    def test_entity_too_small_maps_to_invalid_argument(self, store):
        """Multipart-specific: parts smaller than the provider minimum
        should surface as InvalidArgumentException so callers can fix
        their part-size policy, not UnknownException."""
        exc = ClientError(
            {
                "Error": {"Code": "EntityTooSmall", "Message": "part too small"},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "CompleteMultipartUpload",
        )
        result = store.translate_error(exc)
        assert isinstance(result, InvalidArgumentException)


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_strip_quotes_with_quotes(self):
        assert _strip_quotes('"abc"') == "abc"

    def test_strip_quotes_without_quotes(self):
        assert _strip_quotes("abc") == "abc"

    def test_strip_quotes_none(self):
        assert _strip_quotes(None) is None

    def test_build_range_header_both(self):
        assert _build_range_header(0, 100) == "bytes=0-100"

    def test_build_range_header_start_only(self):
        assert _build_range_header(50, None) == "bytes=50-"

    def test_build_range_header_end_only(self):
        assert _build_range_header(None, 99) == "bytes=-99"

    def test_build_range_header_none(self):
        assert _build_range_header(None, None) is None


class TestAwsBlobStoreLifecycle:
    """Tests for close() behavior."""

    def test_close_calls_underlying_client(self, store, mock_s3):
        mock_s3.close = MagicMock()
        store.close()
        mock_s3.close.assert_called_once()

    def test_close_noop_when_no_close_method(self):
        # Some mock clients might not have close
        s3 = MagicMock(spec=[])  # no attributes
        store = AwsBlobStore(bucket="b", s3_client=s3)
        store.close()  # should not raise


class TestAwsBlobStoreListPage:
    """Tests for do_list_page."""

    def test_list_page_basic(self, store, mock_s3):
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "file1.txt",
                    "Size": 100,
                    "ETag": '"abc"',
                    "LastModified": "2024-01-01T00:00:00Z",
                    "StorageClass": "STANDARD",
                },
                {
                    "Key": "file2.txt",
                    "Size": 200,
                    "ETag": '"def"',
                    "LastModified": "2024-01-02T00:00:00Z",
                    "StorageClass": "STANDARD",
                },
            ],
            "IsTruncated": False,
        }
        resp = store.list_page(prefix="")
        assert len(resp.blobs) == 2
        assert resp.blobs[0].key == "file1.txt"
        assert resp.blobs[0].size == 100
        assert resp.blobs[0].e_tag == "abc"
        assert resp.blobs[1].key == "file2.txt"
        assert resp.is_truncated is False
        assert resp.next_pagination_token is None

    def test_list_page_with_pagination(self, store, mock_s3):
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "a.txt", "Size": 10, "ETag": '"x"'}],
            "IsTruncated": True,
            "NextContinuationToken": "token123",
        }
        resp = store.list_page(max_results=1)
        assert resp.is_truncated is True
        assert resp.next_pagination_token == "token123"
        assert len(resp.blobs) == 1

    def test_list_page_with_delimiter(self, store, mock_s3):
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "dir/file.txt", "Size": 5, "ETag": '"e"'}],
            "CommonPrefixes": [{"Prefix": "dir/sub1/"}, {"Prefix": "dir/sub2/"}],
            "IsTruncated": False,
        }
        resp = store.list_page(prefix="dir/", delimiter="/")
        assert len(resp.blobs) == 1
        assert resp.common_prefixes == ["dir/sub1/", "dir/sub2/"]

    def test_list_page_empty_bucket(self, store, mock_s3):
        mock_s3.list_objects_v2.return_value = {"IsTruncated": False}
        resp = store.list_page()
        assert resp.blobs == []
        assert resp.common_prefixes == []
        assert resp.is_truncated is False

    def test_list_page_passes_continuation_token(self, store, mock_s3):
        mock_s3.list_objects_v2.return_value = {
            "Contents": [],
            "IsTruncated": False,
        }
        store.list_page(pagination_token="tok")
        call_kwargs = mock_s3.list_objects_v2.call_args[1]
        assert call_kwargs["ContinuationToken"] == "tok"

    def test_list_page_client_error_not_translated_at_store_level(self, store, mock_s3):
        # do_list_page has no provider catch; raw boto error surfaces at store level.
        mock_s3.list_objects_v2.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchBucket", "Message": ""},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "ListObjectsV2",
        )
        with pytest.raises(ClientError):
            store.list_page()

    def test_list_page_invalid_max_results_raises(self, store):
        with pytest.raises(InvalidArgumentException, match="max_results"):
            store.list_page(max_results=0)

    def test_list_page_max_results_over_1000_forwarded(self, store, mock_s3):
        # No cloud-agnostic upper cap: values above 1000 are forwarded to S3
        # as MaxKeys (S3 silently caps the page to 1000 server-side).
        mock_s3.list_objects_v2.return_value = {"Contents": [], "IsTruncated": False}
        store.list_page(max_results=5000)
        call_kwargs = mock_s3.list_objects_v2.call_args[1]
        assert call_kwargs["MaxKeys"] == 5000


class TestAwsBlobStoreDoesObjectExist:
    """Tests for do_does_object_exist."""

    def test_object_exists_returns_true(self, store, mock_s3):
        mock_s3.head_object.return_value = {"ContentLength": 42}
        assert store.does_object_exist("existing.txt") is True
        mock_s3.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="existing.txt"
        )

    def test_object_not_found_returns_false(self, store, mock_s3):
        mock_s3.head_object.side_effect = ClientError(
            {
                "Error": {"Code": "404", "Message": "Not Found"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "HeadObject",
        )
        assert store.does_object_exist("missing.txt") is False

    def test_object_not_found_nosuchkey_returns_false(self, store, mock_s3):
        mock_s3.head_object.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": ""},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "HeadObject",
        )
        assert store.does_object_exist("missing.txt") is False

    def test_object_exist_with_version_id(self, store, mock_s3):
        mock_s3.head_object.return_value = {"ContentLength": 10}
        assert store.does_object_exist("key.txt", version_id="v1") is True
        call_kwargs = mock_s3.head_object.call_args[1]
        assert call_kwargs["VersionId"] == "v1"

    def test_object_exist_access_denied_raises(self, store, mock_s3):
        mock_s3.head_object.side_effect = ClientError(
            {
                "Error": {"Code": "403", "Message": "Forbidden"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "HeadObject",
        )
        with pytest.raises(UnAuthorizedException):
            store.does_object_exist("forbidden.txt")

    def test_object_exist_empty_key_raises(self, store):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.does_object_exist("")

    def test_object_exist_no_credentials_not_translated_at_store_level(
        self, store, mock_s3
    ):
        # do_does_object_exist catches ClientError but not NoCredentialsError.
        # NoCredentialsError propagates raw at the store level.
        mock_s3.head_object.side_effect = NoCredentialsError()
        with pytest.raises(NoCredentialsError):
            store.does_object_exist("key.txt")


class TestAwsBlobStoreDoeBucketExist:
    """Tests for do_does_bucket_exist."""

    def test_bucket_exists_returns_true(self, store, mock_s3):
        mock_s3.head_bucket.return_value = {}
        assert store.does_bucket_exist() is True
        mock_s3.head_bucket.assert_called_once_with(Bucket="test-bucket")

    def test_bucket_not_found_returns_false(self, store, mock_s3):
        mock_s3.head_bucket.side_effect = ClientError(
            {
                "Error": {"Code": "404", "Message": "Not Found"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "HeadBucket",
        )
        assert store.does_bucket_exist() is False

    def test_bucket_not_found_nosuchbucket_returns_false(self, store, mock_s3):
        mock_s3.head_bucket.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchBucket", "Message": ""},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "HeadBucket",
        )
        assert store.does_bucket_exist() is False

    def test_bucket_access_denied_raises(self, store, mock_s3):
        mock_s3.head_bucket.side_effect = ClientError(
            {
                "Error": {"Code": "403", "Message": "Forbidden"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "HeadBucket",
        )
        with pytest.raises(UnAuthorizedException):
            store.does_bucket_exist()

    def test_bucket_no_credentials_not_translated_at_store_level(self, store, mock_s3):
        # do_does_bucket_exist catches ClientError but not NoCredentialsError.
        mock_s3.head_bucket.side_effect = NoCredentialsError()
        with pytest.raises(NoCredentialsError):
            store.does_bucket_exist()


class TestAwsBlobStoreListBlobVersions:
    """Tests for do_list_blob_versions."""

    def test_list_versions_single_page(self, store, mock_s3):
        mock_s3.list_object_versions.return_value = {
            "Versions": [
                {
                    "Key": "doc.txt",
                    "VersionId": "v2",
                    "IsLatest": True,
                    "Size": 200,
                    "ETag": '"e2"',
                    "LastModified": "2024-02-01",
                },
                {
                    "Key": "doc.txt",
                    "VersionId": "v1",
                    "IsLatest": False,
                    "Size": 100,
                    "ETag": '"e1"',
                    "LastModified": "2024-01-01",
                },
            ],
            "IsTruncated": False,
        }
        versions = list(store.list_blob_versions("doc.txt"))
        assert len(versions) == 2
        assert versions[0].key == "doc.txt"
        assert versions[0].version_id == "v2"
        assert versions[0].is_latest is True
        assert versions[0].size == 200
        assert versions[0].e_tag == "e2"  # quotes stripped
        assert versions[0].last_modified == "2024-02-01"
        assert versions[1].version_id == "v1"
        assert versions[1].is_latest is False
        assert versions[1].e_tag == "e1"

    def test_list_versions_filters_prefix_matches(self, store, mock_s3):
        mock_s3.list_object_versions.return_value = {
            "Versions": [
                {
                    "Key": "doc.txt",
                    "VersionId": "v1",
                    "IsLatest": True,
                    "Size": 10,
                    "ETag": '"a"',
                },
                {
                    "Key": "doc.txt.bak",
                    "VersionId": "v1",
                    "IsLatest": True,
                    "Size": 5,
                    "ETag": '"b"',
                },
            ],
            "IsTruncated": False,
        }
        versions = list(store.list_blob_versions("doc.txt"))
        assert len(versions) == 1
        assert versions[0].key == "doc.txt"

    def test_list_versions_pagination(self, store, mock_s3):
        mock_s3.list_object_versions.side_effect = [
            {
                "Versions": [
                    {
                        "Key": "k",
                        "VersionId": "v3",
                        "IsLatest": True,
                        "Size": 30,
                        "ETag": '"c"',
                    },
                ],
                "IsTruncated": True,
                "NextKeyMarker": "k",
                "NextVersionIdMarker": "v3",
            },
            {
                "Versions": [
                    {
                        "Key": "k",
                        "VersionId": "v2",
                        "IsLatest": False,
                        "Size": 20,
                        "ETag": '"b"',
                    },
                    {
                        "Key": "k",
                        "VersionId": "v1",
                        "IsLatest": False,
                        "Size": 10,
                        "ETag": '"a"',
                    },
                ],
                "IsTruncated": False,
            },
        ]
        versions = list(store.list_blob_versions("k"))
        assert len(versions) == 3
        assert [v.version_id for v in versions] == ["v3", "v2", "v1"]

    def test_list_versions_empty(self, store, mock_s3):
        mock_s3.list_object_versions.return_value = {"IsTruncated": False}
        versions = list(store.list_blob_versions("nonexistent.txt"))
        assert versions == []

    def test_list_versions_client_error_raises(self, store, mock_s3):
        mock_s3.list_object_versions.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchBucket", "Message": ""},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "ListObjectVersions",
        )
        with pytest.raises(ResourceNotFoundException):
            list(store.list_blob_versions("key.txt"))

    def test_list_versions_empty_key_raises(self, store):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            list(store.list_blob_versions(""))


class TestAwsBlobStoreGetTags:
    """Tests for do_get_tags."""

    def test_get_tags_returns_mapping(self, store, mock_s3):
        mock_s3.get_object_tagging.return_value = {
            "TagSet": [
                {"Key": "env", "Value": "prod"},
                {"Key": "team", "Value": "sdk"},
            ]
        }
        result = store.get_tags("path/obj.txt")
        assert result == {"env": "prod", "team": "sdk"}
        mock_s3.get_object_tagging.assert_called_once_with(
            Bucket="test-bucket", Key="path/obj.txt"
        )

    def test_get_tags_empty_tag_set(self, store, mock_s3):
        mock_s3.get_object_tagging.return_value = {"TagSet": []}
        result = store.get_tags("key.txt")
        assert result == {}

    def test_get_tags_nonexistent_key_not_translated_at_store_level(
        self, store, mock_s3
    ):
        # do_get_tags has no provider catch; raw boto error surfaces at store level.
        mock_s3.get_object_tagging.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": "not found"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObjectTagging",
        )
        with pytest.raises(ClientError):
            store.get_tags("missing.txt")

    def test_get_tags_access_denied_not_translated_at_store_level(self, store, mock_s3):
        mock_s3.get_object_tagging.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "GetObjectTagging",
        )
        with pytest.raises(ClientError):
            store.get_tags("secret.txt")

    def test_get_tags_no_credentials_not_translated_at_store_level(
        self, store, mock_s3
    ):
        mock_s3.get_object_tagging.side_effect = NoCredentialsError()
        with pytest.raises(NoCredentialsError):
            store.get_tags("key.txt")


class TestAwsBlobStoreSetTags:
    """Tests for do_set_tags."""

    def test_set_tags_calls_put_object_tagging(self, store, mock_s3):
        mock_s3.put_object_tagging.return_value = {}
        store.set_tags("path/obj.txt", {"env": "prod", "team": "sdk"})
        mock_s3.put_object_tagging.assert_called_once()
        call_kwargs = mock_s3.put_object_tagging.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "path/obj.txt"
        tag_set = call_kwargs["Tagging"]["TagSet"]
        # Convert to a set of tuples for order-independent comparison
        tags_as_set = {(t["Key"], t["Value"]) for t in tag_set}
        assert tags_as_set == {("env", "prod"), ("team", "sdk")}

    def test_set_tags_nonexistent_key_not_translated_at_store_level(
        self, store, mock_s3
    ):
        # do_set_tags has no provider catch; raw boto error surfaces at store level.
        mock_s3.put_object_tagging.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": "not found"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "PutObjectTagging",
        )
        with pytest.raises(ClientError):
            store.set_tags("missing.txt", {"env": "test"})

    def test_set_tags_access_denied_not_translated_at_store_level(self, store, mock_s3):
        mock_s3.put_object_tagging.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "PutObjectTagging",
        )
        with pytest.raises(ClientError):
            store.set_tags("secret.txt", {"env": "test"})

    def test_set_tags_no_credentials_not_translated_at_store_level(
        self, store, mock_s3
    ):
        mock_s3.put_object_tagging.side_effect = NoCredentialsError()
        with pytest.raises(NoCredentialsError):
            store.set_tags("key.txt", {"env": "test"})


class TestAwsBlobStoreSetObjectRetention:
    """Tests for do_update_object_retention."""

    @staticmethod
    def _seed_current_retention(mock_s3, mode, retain_until):
        """Make get_object_retention report an existing retention on the object."""
        mock_s3.get_object_retention.return_value = {
            "Retention": {"Mode": mode, "RetainUntilDate": retain_until}
        }

    def test_calls_put_object_retention_with_mode_and_date(self, store, mock_s3):
        from datetime import datetime, timezone

        # Object already has COMPLIANCE retention; extending the date is allowed.
        self._seed_current_retention(
            mock_s3, "COMPLIANCE", datetime(2029, 1, 1, tzinfo=timezone.utc)
        )
        mock_s3.put_object_retention.return_value = {}
        until = datetime(2030, 1, 1, tzinfo=timezone.utc)
        store.update_object_retention(
            "path/obj.txt",
            ObjectRetentionConfig(
                mode=RetentionMode.COMPLIANCE, retain_until_date=until
            ),
        )
        mock_s3.put_object_retention.assert_called_once()
        call_kwargs = mock_s3.put_object_retention.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "path/obj.txt"
        assert call_kwargs["Retention"]["Mode"] == "COMPLIANCE"
        assert call_kwargs["Retention"]["RetainUntilDate"] == until
        # No governance bypass and no version by default.
        assert "BypassGovernanceRetention" not in call_kwargs
        assert "VersionId" not in call_kwargs

    def test_none_mode_preserves_current_mode(self, store, mock_s3):
        from datetime import datetime, timezone

        # config.mode=None -> the resolved mode is whatever is currently set.
        self._seed_current_retention(
            mock_s3, "GOVERNANCE", datetime(2029, 1, 1, tzinfo=timezone.utc)
        )
        mock_s3.put_object_retention.return_value = {}
        store.update_object_retention(
            "obj.txt",
            ObjectRetentionConfig(
                mode=None,
                retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
            ),
        )
        call_kwargs = mock_s3.put_object_retention.call_args[1]
        assert call_kwargs["Retention"]["Mode"] == "GOVERNANCE"

    def test_no_current_retention_raises_failed_precondition(self, store, mock_s3):
        from datetime import datetime, timezone

        from multicloud.common.exceptions import FailedPreconditionException

        # get_object_retention reports no lock configured -> update is rejected.
        mock_s3.get_object_retention.side_effect = ClientError(
            {
                "Error": {
                    "Code": "NoSuchObjectLockConfiguration",
                    "Message": "no lock",
                },
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObjectRetention",
        )
        with pytest.raises(FailedPreconditionException):
            store.update_object_retention(
                "obj.txt",
                ObjectRetentionConfig(
                    mode=RetentionMode.GOVERNANCE,
                    retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
                ),
            )
        mock_s3.put_object_retention.assert_not_called()

    def test_bypass_governance_and_version_id_forwarded(self, store, mock_s3):
        from datetime import datetime, timezone

        self._seed_current_retention(
            mock_s3, "GOVERNANCE", datetime(2029, 1, 1, tzinfo=timezone.utc)
        )
        mock_s3.put_object_retention.return_value = {}
        store.update_object_retention(
            "obj.txt",
            ObjectRetentionConfig(
                mode=RetentionMode.GOVERNANCE,
                retain_until_date=datetime(2030, 6, 1, tzinfo=timezone.utc),
                bypass_governance=True,
            ),
            version_id="v42",
        )
        call_kwargs = mock_s3.put_object_retention.call_args[1]
        assert call_kwargs["Retention"]["Mode"] == "GOVERNANCE"
        assert call_kwargs["BypassGovernanceRetention"] is True
        assert call_kwargs["VersionId"] == "v42"

    def test_error_not_translated_at_store_level(self, store, mock_s3):
        from datetime import datetime, timezone

        # do_update_object_retention has no provider catch; raw boto error surfaces.
        self._seed_current_retention(
            mock_s3, "GOVERNANCE", datetime(2029, 1, 1, tzinfo=timezone.utc)
        )
        mock_s3.put_object_retention.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": "not found"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "PutObjectRetention",
        )
        with pytest.raises(ClientError):
            store.update_object_retention(
                "missing.txt",
                ObjectRetentionConfig(
                    mode=RetentionMode.GOVERNANCE,
                    retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
                ),
            )


class TestAwsPresignedUrl:
    """Tests for presigned URL generation."""

    def test_upload_basic(self, store, mock_s3):
        """Generate a basic presigned upload URL."""
        mock_s3.generate_presigned_url.return_value = (
            "https://s3.amazonaws.com/presigned"
        )
        resp = store.generate_presigned_url(
            "my/key.txt", PresignedOperation.UPLOAD, duration_seconds=3600
        )
        assert resp.url == "https://s3.amazonaws.com/presigned"
        assert resp.http_method == "PUT"
        assert resp.required_headers == {}
        mock_s3.generate_presigned_url.assert_called_once_with(
            ClientMethod="put_object",
            Params={"Bucket": "test-bucket", "Key": "my/key.txt"},
            ExpiresIn=3600,
        )

    def test_upload_with_content_type(self, store, mock_s3):
        """Content-type is passed as param and returned as required header."""
        mock_s3.generate_presigned_url.return_value = "https://url"
        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            content_type="image/png",
        )
        assert resp.required_headers["Content-Type"] == "image/png"
        call_params = mock_s3.generate_presigned_url.call_args[1]["Params"]
        assert call_params["ContentType"] == "image/png"

    def test_upload_with_metadata(self, store, mock_s3):
        """Metadata maps to x-amz-meta-* required headers."""
        mock_s3.generate_presigned_url.return_value = "https://url"
        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            metadata={"author": "test", "version": "1"},
        )
        assert resp.required_headers["x-amz-meta-author"] == "test"
        assert resp.required_headers["x-amz-meta-version"] == "1"
        call_params = mock_s3.generate_presigned_url.call_args[1]["Params"]
        assert call_params["Metadata"] == {"author": "test", "version": "1"}

    def test_upload_with_tags(self, store, mock_s3):
        """Tags are URL-encoded as x-amz-tagging header."""
        mock_s3.generate_presigned_url.return_value = "https://url"
        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            tags={"env": "dev", "team": "sdk"},
        )
        tagging = resp.required_headers["x-amz-tagging"]
        assert "env=dev" in tagging
        assert "team=sdk" in tagging
        call_params = mock_s3.generate_presigned_url.call_args[1]["Params"]
        assert call_params["Tagging"] == tagging

    def test_upload_with_tags_url_encodes_special_chars(self, store, mock_s3):
        """Tag keys/values with special characters are URL-encoded."""
        mock_s3.generate_presigned_url.return_value = "https://url"
        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            tags={"key with spaces": "value&special=chars"},
        )
        tagging = resp.required_headers["x-amz-tagging"]
        assert "key%20with%20spaces" in tagging
        assert "value%26special%3Dchars" in tagging

    def test_upload_with_kms_key(self, store, mock_s3):
        """KMS key adds SSE headers."""
        mock_s3.generate_presigned_url.return_value = "https://url"
        kms_arn = "arn:aws:kms:us-east-1:123456:key/abc-def"
        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            kms_key_id=kms_arn,
        )
        assert resp.required_headers["x-amz-server-side-encryption"] == "aws:kms"
        assert (
            resp.required_headers["x-amz-server-side-encryption-aws-kms-key-id"]
            == kms_arn
        )
        call_params = mock_s3.generate_presigned_url.call_args[1]["Params"]
        assert call_params["ServerSideEncryption"] == "aws:kms"
        assert call_params["SSEKMSKeyId"] == kms_arn

    def test_upload_with_content_disposition(self, store, mock_s3):
        """Content-Disposition is passed as param and required header."""
        mock_s3.generate_presigned_url.return_value = "https://url"
        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            content_disposition="attachment; filename=report.pdf",
        )
        assert (
            resp.required_headers["Content-Disposition"]
            == "attachment; filename=report.pdf"
        )
        call_params = mock_s3.generate_presigned_url.call_args[1]["Params"]
        assert call_params["ContentDisposition"] == "attachment; filename=report.pdf"

    def test_upload_with_all_options(self, store, mock_s3):
        """All options combined in one call."""
        mock_s3.generate_presigned_url.return_value = "https://full-url"
        resp = store.generate_presigned_url(
            "path/file.bin",
            PresignedOperation.UPLOAD,
            duration_seconds=7200,
            content_type="application/octet-stream",
            metadata={"origin": "ci"},
            tags={"build": "123"},
            kms_key_id="arn:key",
            content_disposition="inline",
        )
        assert resp.http_method == "PUT"
        assert resp.required_headers["Content-Type"] == "application/octet-stream"
        assert resp.required_headers["x-amz-meta-origin"] == "ci"
        assert "build=123" in resp.required_headers["x-amz-tagging"]
        assert resp.required_headers["x-amz-server-side-encryption"] == "aws:kms"
        assert resp.required_headers["Content-Disposition"] == "inline"

    def test_download_basic(self, store, mock_s3):
        """Generate a basic presigned download URL."""
        mock_s3.generate_presigned_url.return_value = "https://s3.amazonaws.com/get"
        resp = store.generate_presigned_url(
            "my/key.txt", PresignedOperation.DOWNLOAD, duration_seconds=1800
        )
        assert resp.url == "https://s3.amazonaws.com/get"
        assert resp.http_method == "GET"
        assert resp.required_headers == {}
        mock_s3.generate_presigned_url.assert_called_once_with(
            ClientMethod="get_object",
            Params={"Bucket": "test-bucket", "Key": "my/key.txt"},
            ExpiresIn=1800,
        )

    def test_download_with_content_disposition(self, store, mock_s3):
        """Download URL with content-disposition response override."""
        mock_s3.generate_presigned_url.return_value = "https://url"
        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.DOWNLOAD,
            duration_seconds=600,
            content_disposition="attachment; filename=data.csv",
        )
        assert resp.required_headers == {}  # download has no required headers
        call_params = mock_s3.generate_presigned_url.call_args[1]["Params"]
        assert (
            call_params["ResponseContentDisposition"] == "attachment; filename=data.csv"
        )

    def test_upload_client_error_translates(self, store, mock_s3):
        """ClientError during URL generation propagates raw."""
        mock_s3.generate_presigned_url.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "GeneratePresignedUrl",
        )
        with pytest.raises(ClientError):
            store.generate_presigned_url(
                "k", PresignedOperation.UPLOAD, duration_seconds=600
            )

    def test_download_no_credentials_translates(self, store, mock_s3):
        """NoCredentialsError during URL generation propagates raw."""
        mock_s3.generate_presigned_url.side_effect = NoCredentialsError()
        with pytest.raises(NoCredentialsError):
            store.generate_presigned_url(
                "k", PresignedOperation.DOWNLOAD, duration_seconds=600
            )


class TestAwsMultipartUpload:
    """Tests for AWS multipart upload operations."""

    def test_initiate_multipart_upload(self, store, mock_s3):
        mock_s3.create_multipart_upload.return_value = {"UploadId": "upload-123"}
        mpu = store.initiate_multipart_upload(
            "key.bin", metadata={"env": "test"}, content_type="application/octet-stream"
        )
        assert mpu.upload_id == "upload-123"
        assert mpu.key == "key.bin"
        assert mpu.bucket == "test-bucket"
        mock_s3.create_multipart_upload.assert_called_once_with(
            Bucket="test-bucket",
            Key="key.bin",
            Metadata={"env": "test"},
            ContentType="application/octet-stream",
        )

    def test_initiate_multipart_upload_no_metadata(self, store, mock_s3):
        mock_s3.create_multipart_upload.return_value = {"UploadId": "u-1"}
        store.initiate_multipart_upload("k")
        call_kwargs = mock_s3.create_multipart_upload.call_args[1]
        assert "Metadata" not in call_kwargs

    def test_initiate_multipart_upload_with_tags(self, store, mock_s3):
        mock_s3.create_multipart_upload.return_value = {"UploadId": "u-1"}
        mpu = store.initiate_multipart_upload("k", tags={"env": "prod", "team": "core"})
        # S3 takes tags as a URL-encoded query string on CreateMultipartUpload.
        call_kwargs = mock_s3.create_multipart_upload.call_args[1]
        assert call_kwargs["Tagging"] == "env=prod&team=core"
        # Tags are carried on the handle for cross-provider parity.
        assert mpu.tags == {"env": "prod", "team": "core"}

    def test_initiate_multipart_upload_tags_url_encoded(self, store, mock_s3):
        mock_s3.create_multipart_upload.return_value = {"UploadId": "u-1"}
        store.initiate_multipart_upload("k", tags={"a b": "c/d", "e=f": "g&h"})
        call_kwargs = mock_s3.create_multipart_upload.call_args[1]
        assert call_kwargs["Tagging"] == "a%20b=c%2Fd&e%3Df=g%26h"

    def test_initiate_multipart_upload_tags_unicode_encoded(self, store, mock_s3):
        mock_s3.create_multipart_upload.return_value = {"UploadId": "u-1"}
        store.initiate_multipart_upload("k", tags={"lang": "中文", "emoji": "🚀"})
        call_kwargs = mock_s3.create_multipart_upload.call_args[1]
        assert call_kwargs["Tagging"] == "lang=%E4%B8%AD%E6%96%87&emoji=%F0%9F%9A%80"

    def test_initiate_multipart_upload_no_tags(self, store, mock_s3):
        mock_s3.create_multipart_upload.return_value = {"UploadId": "u-1"}
        store.initiate_multipart_upload("k")
        call_kwargs = mock_s3.create_multipart_upload.call_args[1]
        assert "Tagging" not in call_kwargs

    def test_initiate_multipart_upload_with_object_lock(self, store, mock_s3):
        from datetime import datetime, timedelta, timezone

        # CreateMultipartUpload natively accepts the same object-lock params as
        # PutObject; the lock is stamped on the object when the upload completes.
        mock_s3.create_multipart_upload.return_value = {"UploadId": "u-1"}
        retain_until = datetime.now(timezone.utc) + timedelta(days=1)
        mpu = store.initiate_multipart_upload(
            "k",
            object_lock=ObjectLockConfiguration(
                mode=RetentionMode.GOVERNANCE,
                retain_until_date=retain_until,
                legal_hold=True,
            ),
        )
        call_kwargs = mock_s3.create_multipart_upload.call_args[1]
        assert call_kwargs["ObjectLockMode"] == "GOVERNANCE"
        assert call_kwargs["ObjectLockRetainUntilDate"] == retain_until
        assert call_kwargs["ObjectLockLegalHoldStatus"] == "ON"
        # The config is carried on the handle for cross-provider parity.
        assert mpu.object_lock is not None
        assert mpu.object_lock.mode is RetentionMode.GOVERNANCE

    def test_initiate_multipart_upload_no_object_lock_omits_kwargs(
        self, store, mock_s3
    ):
        mock_s3.create_multipart_upload.return_value = {"UploadId": "u-1"}
        mpu = store.initiate_multipart_upload("k")
        call_kwargs = mock_s3.create_multipart_upload.call_args[1]
        assert "ObjectLockMode" not in call_kwargs
        assert "ObjectLockRetainUntilDate" not in call_kwargs
        assert "ObjectLockLegalHoldStatus" not in call_kwargs
        assert mpu.object_lock is None

    def test_initiate_multipart_upload_client_error(self, store, mock_s3):
        mock_s3.create_multipart_upload.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}},
            "CreateMultipartUpload",
        )
        with pytest.raises(ClientError):
            store.initiate_multipart_upload("k")

    def test_upload_part(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        mock_s3.upload_part.return_value = {"ETag": '"abc123"'}
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        part = MultipartPart(part_number=2, content=b"x" * 100)
        resp = store.upload_part(mpu, part)
        assert resp.part_number == 2
        assert resp.e_tag == "abc123"
        assert resp.size_in_bytes == 100
        mock_s3.upload_part.assert_called_once_with(
            Bucket="test-bucket",
            Key="k",
            UploadId="u-1",
            PartNumber=2,
            Body=b"x" * 100,
            ContentLength=100,
        )

    def test_upload_part_client_error(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        mock_s3.upload_part.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "UploadPart"
        )
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        part = MultipartPart(part_number=1, content=b"data")
        with pytest.raises(ClientError):
            store.upload_part(mpu, part)

    def test_complete_multipart_upload(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        mock_s3.complete_multipart_upload.return_value = {"ETag": '"final-etag"'}
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        parts = [
            UploadPartResponse(part_number=2, e_tag="e2", size_in_bytes=100),
            UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100),
        ]
        resp = store.complete_multipart_upload(mpu, parts)
        assert resp.e_tag == "final-etag"
        call_kwargs = mock_s3.complete_multipart_upload.call_args[1]
        submitted_parts = call_kwargs["MultipartUpload"]["Parts"]
        assert submitted_parts[0]["PartNumber"] == 1
        assert submitted_parts[1]["PartNumber"] == 2

    def test_complete_multipart_upload_client_error(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_s3.complete_multipart_upload.side_effect = ClientError(
            {
                "Error": {"Code": "EntityTooSmall", "Message": "too small"},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "CompleteMultipartUpload",
        )
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        with pytest.raises(ClientError):
            store.complete_multipart_upload(mpu, [])

    def test_list_parts(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_s3.list_parts.return_value = {
            "Parts": [
                {"PartNumber": 2, "ETag": '"e2"', "Size": 200},
                {"PartNumber": 1, "ETag": '"e1"', "Size": 100},
            ]
        }
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        results = store.list_parts(mpu)
        assert len(results) == 2
        assert results[0].part_number == 1
        assert results[1].part_number == 2
        assert results[0].size_in_bytes == 100

    def test_list_parts_empty(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_s3.list_parts.return_value = {"Parts": []}
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        results = store.list_parts(mpu)
        assert results == []

    def test_list_parts_pagination(self, store, mock_s3):
        """list_parts paginates when IsTruncated is True."""
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_s3.list_parts.side_effect = [
            {
                "Parts": [{"PartNumber": 1, "ETag": '"e1"', "Size": 100}],
                "IsTruncated": True,
                "NextPartNumberMarker": 1,
            },
            {
                "Parts": [{"PartNumber": 2, "ETag": '"e2"', "Size": 200}],
                "IsTruncated": False,
            },
        ]
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        results = store.list_parts(mpu)
        assert len(results) == 2
        assert results[0].part_number == 1
        assert results[1].part_number == 2
        assert mock_s3.list_parts.call_count == 2

    def test_list_parts_client_error(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_s3.list_parts.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "not found"}}, "ListParts"
        )
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        with pytest.raises(ClientError):
            store.list_parts(mpu)

    def test_abort_multipart_upload(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        store.abort_multipart_upload(mpu)
        mock_s3.abort_multipart_upload.assert_called_once_with(
            Bucket="test-bucket", Key="k", UploadId="u-1"
        )

    def test_abort_multipart_upload_client_error(self, store, mock_s3):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_s3.abort_multipart_upload.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "not found"}},
            "AbortMultipartUpload",
        )
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="u-1")
        with pytest.raises(ClientError):
            store.abort_multipart_upload(mpu)
