"""Unit tests for :class:`GcpBlobStore` — the Google Cloud Storage provider."""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock, call, patch

import google.auth.exceptions
import pytest
from google.api_core import exceptions as gcs_exceptions
from multicloud.blob.blob_client.driver import (
    ChecksumAlgorithm,
    ObjectLockConfiguration,
    ObjectLockInfo,
    ObjectRetentionConfig,
    PresignedOperation,
    PresignedUrlResponse,
    RetentionMode,
    UploadResponse,
)
from multicloud.blob.blob_gcp.gcp_blob_store import GcpBlobStore
from multicloud.common.exceptions import (
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceExistsException,
    ResourceNotFoundException,
    SubstrateSdkException,
    UnAuthorizedException,
    UnknownException,
    UnsupportedOperationException,
)


@pytest.fixture
def mock_credentials():
    """Mock google.auth.default() credentials."""
    creds = MagicMock()
    creds.valid = True
    creds.token = "fake-access-token"
    creds.service_account_email = "test@project.iam.gserviceaccount.com"
    return creds


@pytest.fixture
def mock_storage_client(mock_credentials):
    """A mock google.cloud.storage.Client with google.auth.default patched."""
    with (
        patch("google.auth.default") as mock_auth,
        patch("multicloud.blob.blob_gcp.gcp_blob_store.storage.Client") as mock_cls,
    ):
        mock_auth.return_value = (mock_credentials, "test-project")
        client_instance = MagicMock()
        client_instance._credentials = mock_credentials
        mock_cls.return_value = client_instance
        yield client_instance


@pytest.fixture
def store(mock_storage_client):
    """A GcpBlobStore with a mocked storage client."""
    return GcpBlobStore(bucket="test-bucket", region="us-central1")


class TestGcpBlobStoreConstruction:
    """Tests for GcpBlobStore initialization."""

    def test_basic_construction(self, mock_storage_client):
        s = GcpBlobStore(bucket="my-bucket", region="eu-west1")
        assert s.bucket == "my-bucket"
        assert s.region == "eu-west1"

    def test_provider_id_is_gcp(self, store):
        assert store.get_provider_id() == "gcp"

    def test_empty_bucket_raises(self, mock_storage_client):
        with pytest.raises(
            InvalidArgumentException, match="Bucket name cannot be null or empty"
        ):
            GcpBlobStore(bucket="")

    def test_storage_client_injection(self):
        injected = MagicMock()
        s = GcpBlobStore(bucket="b", storage_client=injected)
        assert s._client is injected
        assert s._credentials is None

    @patch("multicloud.blob.blob_gcp.gcp_blob_store.get_credentials")
    def test_storage_client_takes_precedence_over_overrider(self, mock_get_credentials):
        from multicloud.common.credentials import (
            CredentialsOverrider,
            CredentialsType,
            StsCredentials,
        )

        injected = MagicMock()
        overrider = CredentialsOverrider(
            credentials_type=CredentialsType.SESSION,
            session_credentials=StsCredentials(
                access_key_id="",
                access_key_secret="",
                security_token="TOKEN",
            ),
        )
        s = GcpBlobStore(
            bucket="b",
            storage_client=injected,
            credentials_overrider=overrider,
        )
        assert s._client is injected
        assert s._credentials is None
        mock_get_credentials.assert_not_called()


class TestGcpBlobStoreCredentialsOverrider:
    """Tests for credentials_overrider wiring on GcpBlobStore."""

    @patch("multicloud.blob.blob_gcp.gcp_blob_store.storage.Client")
    @patch("multicloud.blob.blob_gcp.gcp_blob_store.get_credentials")
    def test_overrider_forwarded_to_get_credentials(
        self, mock_get_credentials, mock_storage_cls
    ):
        from multicloud.common.credentials import (
            CredentialsOverrider,
            CredentialsType,
            StsCredentials,
        )

        resolved_creds = MagicMock()
        mock_get_credentials.return_value = resolved_creds
        mock_storage_cls.return_value = MagicMock()
        overrider = CredentialsOverrider(
            credentials_type=CredentialsType.SESSION,
            session_credentials=StsCredentials(
                access_key_id="",
                access_key_secret="",
                security_token="TOKEN",
            ),
        )
        s = GcpBlobStore(bucket="b", credentials_overrider=overrider)
        mock_get_credentials.assert_called_once_with(
            overrider,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        mock_storage_cls.assert_called_once_with(credentials=resolved_creds)
        assert s._credentials is resolved_creds

    @patch("google.auth.default")
    @patch("multicloud.blob.blob_gcp.gcp_blob_store.storage.Client")
    @patch("multicloud.blob.blob_gcp.gcp_blob_store.get_credentials")
    def test_overrider_returns_none_falls_through_to_adc(
        self, mock_get_credentials, mock_storage_cls, mock_auth_default
    ):
        # get_credentials may return None for unsupported override types
        # (e.g. WEB_IDENTITY on GCP) — fall through to ADC.
        from multicloud.common.credentials import (
            CredentialsOverrider,
            CredentialsType,
        )

        mock_get_credentials.return_value = None
        default_creds = MagicMock()
        mock_auth_default.return_value = (default_creds, "default-project")
        mock_storage_cls.return_value = MagicMock()
        overrider = CredentialsOverrider(
            credentials_type=CredentialsType.ASSUME_ROLE_WEB_IDENTITY,
            role="some-role",
            web_identity_token_supplier=lambda: "token",
        )
        s = GcpBlobStore(bucket="b", credentials_overrider=overrider)
        mock_auth_default.assert_called_once_with(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        mock_storage_cls.assert_called_once_with(
            credentials=default_creds, project="default-project"
        )
        assert s._credentials is default_creds


class TestGcpBlobStoreUpload:
    """Tests for do_upload."""

    def test_upload_basic(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "gcp-etag"
        blob_mock.generation = 12345
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.upload("key.txt", b"data", content_type="text/plain")
        assert resp.key == "key.txt"
        assert resp.e_tag == "gcp-etag"
        assert resp.version_id == "12345"
        blob_mock.upload_from_string.assert_called_once_with(
            b"data", content_type="text/plain"
        )

    def test_upload_default_content_type(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.upload("key", b"x")
        blob_mock.upload_from_string.assert_called_once_with(
            b"x", content_type="application/octet-stream"
        )
        assert resp.version_id is None

    def test_upload_with_metadata(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.upload("key", b"x", metadata={"env": "prod"})
        assert blob_mock.metadata == {"env": "prod"}

    def test_upload_with_tags_encoded_as_metadata(self, store, mock_storage_client):
        # GCS has no native tagging API, so tags round-trip through the
        # ``gcp-tag-`` metadata prefix that get_tags / set_tags also use.
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.upload("key", b"x", tags={"env": "prod", "owner": "team"})
        assert blob_mock.metadata == {
            "gcp-tag-env": "prod",
            "gcp-tag-owner": "team",
        }

    def test_upload_with_tags_and_metadata_merged(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.upload("key", b"x", metadata={"author": "alice"}, tags={"env": "dev"})
        assert blob_mock.metadata == {
            "author": "alice",
            "gcp-tag-env": "dev",
        }

    def test_upload_with_crc32c_checksum(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        def _server_response(*_args, **_kwargs):
            # The upload response replaces the blob's properties with the
            # server's view, including its computed checksum.
            assert blob_mock.crc32c == "AAECAw=="
            blob_mock.crc32c = "server-crc=="

        blob_mock.upload_from_string.side_effect = _server_response
        resp = store.upload(
            "key",
            b"x",
            checksum_value="AAECAw==",
            checksum_algorithm=ChecksumAlgorithm.CRC32C,
        )
        # checksum=None stops the library from replacing the caller's value
        # with one it computes itself.
        blob_mock.upload_from_string.assert_called_once_with(
            b"x", content_type="application/octet-stream", checksum=None
        )
        assert resp.checksum_algorithm is ChecksumAlgorithm.CRC32C
        assert resp.checksum_value == "server-crc=="

    def test_upload_with_md5_checksum(self, store, mock_storage_client):
        md5 = base64.b64encode(hashlib.md5(b"x").digest()).decode()
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.upload(
            "key", b"x", checksum_value=md5, checksum_algorithm=ChecksumAlgorithm.MD5
        )
        assert blob_mock.md5_hash == md5
        blob_mock.upload_from_string.assert_called_once_with(
            b"x", content_type="application/octet-stream", checksum=None
        )
        assert resp.checksum_algorithm is ChecksumAlgorithm.MD5
        assert resp.checksum_value == md5

    def test_upload_without_checksum_reports_none(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        blob_mock.crc32c = "server-crc=="
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.upload("key", b"x")
        assert resp.checksum_algorithm is None
        assert resp.checksum_value is None

    def test_upload_checksum_mismatch_raises_invalid_argument(
        self, store, mock_storage_client
    ):
        blob_mock = MagicMock()
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        blob_mock.upload_from_string.side_effect = gcs_exceptions.BadRequest(
            'Provided CRC32C "AAAAAA==" doesn\'t match calculated CRC32C "yZRlqg==".'
        )
        with pytest.raises(
            InvalidArgumentException,
            match="upload: CRC32C checksum mismatch for key 'key'; the object was "
            "not written",
        ) as exc_info:
            store.upload(
                "key",
                b"x",
                checksum_value="AAAAAA==",
                checksum_algorithm=ChecksumAlgorithm.CRC32C,
            )
        assert isinstance(exc_info.value.__cause__, gcs_exceptions.BadRequest)

    def test_upload_checksum_other_bad_request_propagates(
        self, store, mock_storage_client
    ):
        # Only the checksum-mismatch 400 is relabelled; other bad requests
        # reach the client layer untouched.
        blob_mock = MagicMock()
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        blob_mock.upload_from_string.side_effect = gcs_exceptions.BadRequest(
            "Invalid storage class"
        )
        with pytest.raises(gcs_exceptions.BadRequest):
            store.upload(
                "key",
                b"x",
                checksum_value="AAAAAA==",
                checksum_algorithm=ChecksumAlgorithm.CRC32C,
            )

    def test_upload_with_compliance_retention(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        retain_until = datetime.now(timezone.utc) + timedelta(days=1)
        store.upload(
            "key",
            b"x",
            object_lock=ObjectLockConfiguration(
                mode=RetentionMode.COMPLIANCE, retain_until_date=retain_until
            ),
        )
        # COMPLIANCE maps to a Locked retention. With legal_hold defaulting to
        # False, no hold is written and no post-upload metadata patch is issued.
        assert blob_mock.retention.mode == "Locked"
        assert blob_mock.retention.retain_until_time == retain_until
        blob_mock.patch.assert_not_called()

    def test_upload_with_governance_retention(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        retain_until = datetime.now(timezone.utc) + timedelta(days=1)
        store.upload(
            "key",
            b"x",
            object_lock=ObjectLockConfiguration(
                mode=RetentionMode.GOVERNANCE, retain_until_date=retain_until
            ),
        )
        assert blob_mock.retention.mode == "Unlocked"
        assert blob_mock.retention.retain_until_time == retain_until

    def test_upload_legal_hold_defaults_to_temporary_hold(
        self, store, mock_storage_client
    ):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.upload("key", b"x", object_lock=ObjectLockConfiguration(legal_hold=True))
        assert blob_mock.temporary_hold is True
        # GCS rejects holds on the upload request, so the hold must be written
        # by a metadata patch issued after the object is created.
        blob_mock.patch.assert_called_once()
        # Order matters: patch must run after upload_from_string, otherwise the
        # object does not exist yet and the hold write fails.
        assert blob_mock.method_calls.index(
            call.upload_from_string(ANY, content_type=ANY)
        ) < blob_mock.method_calls.index(call.patch())

    def test_upload_legal_hold_event_based(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.upload(
            "key",
            b"x",
            object_lock=ObjectLockConfiguration(
                legal_hold=True, use_event_based_hold=True
            ),
        )
        assert blob_mock.event_based_hold is True
        blob_mock.patch.assert_called_once()

    def test_upload_hold_patch_failure_propagates(self, store, mock_storage_client):
        # The hold is written by a post-upload patch; if it fails the object is
        # left uploaded without the hold and the raw error must propagate.
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        blob_mock.patch.side_effect = gcs_exceptions.ServiceUnavailable("503")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with pytest.raises(gcs_exceptions.ServiceUnavailable):
            store.do_upload(
                "key",
                b"x",
                content_type=None,
                content_length=None,
                metadata={},
                tags={},
                object_lock=ObjectLockConfiguration(legal_hold=True),
            )
        blob_mock.upload_from_string.assert_called_once()

    def test_upload_from_path_with_object_lock(
        self, store, mock_storage_client, tmp_path
    ):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        src = tmp_path / "f.txt"
        src.write_bytes(b"data")

        retain_until = datetime.now(timezone.utc) + timedelta(days=1)
        store.upload_from_path(
            "key",
            src,
            object_lock=ObjectLockConfiguration(
                mode=RetentionMode.COMPLIANCE,
                retain_until_date=retain_until,
                legal_hold=True,
            ),
        )
        assert blob_mock.retention.mode == "Locked"
        assert blob_mock.retention.retain_until_time == retain_until
        assert blob_mock.temporary_hold is True
        blob_mock.upload_from_filename.assert_called_once()
        # Hold is written by a post-upload patch, after the file is uploaded.
        blob_mock.patch.assert_called_once()
        assert blob_mock.method_calls.index(
            call.upload_from_filename(ANY, content_type=ANY)
        ) < blob_mock.method_calls.index(call.patch())

    def test_upload_metadata_with_tag_prefix_passes(self, store, mock_storage_client):
        # The ``gcp-tag-*`` prefix is SDK-reserved; callers who set
        # ``gcp-tag-*`` keys via ``metadata=`` aren't rejected, but those
        # entries may be overwritten by subsequent tag operations.
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.upload(
            "key",
            b"x",
            metadata={"gcp-tag-legacy": "kept"},
            tags={"env": "prod"},
        )
        assert blob_mock.metadata == {
            "gcp-tag-legacy": "kept",
            "gcp-tag-env": "prod",
        }

    def test_upload_without_tags_or_metadata_does_not_set_metadata(
        self, store, mock_storage_client
    ):
        # Invariant under test: when neither metadata nor tags is supplied,
        # the SDK must not assign blob.metadata at all — the GCS client
        # library decides the default. We assert this directly by spying on
        # __setattr__ rather than by checking a sentinel value, because a
        # value-based assertion would still pass under buggy refactors like
        # ``blob.metadata = None``.
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        metadata_writes: list[object] = []
        original_setattr = type(blob_mock).__setattr__

        def spy_setattr(obj, name, value):
            if name == "metadata":
                metadata_writes.append(value)
            original_setattr(obj, name, value)

        with patch.object(type(blob_mock), "__setattr__", spy_setattr):
            store.upload("key", b"x")

        assert metadata_writes == [], (
            f"blob.metadata should not be assigned when no metadata/tags are "
            f"provided, but got writes: {metadata_writes!r}"
        )
        # Sanity: the actual upload still happened.
        blob_mock.upload_from_string.assert_called_once()

    def test_upload_api_error_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        # Model B: do_upload has no provider catch, so raw GCP errors surface at
        # the store level. Translation happens at the BucketClient layer.
        blob_mock = MagicMock()
        blob_mock.upload_from_string.side_effect = gcs_exceptions.Forbidden("denied")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with pytest.raises(gcs_exceptions.Forbidden):
            store.upload("key", b"data")


class TestGcpBlobStoreDownload:
    """Tests for do_download."""

    def test_download_basic(self, store, mock_storage_client):
        updated = datetime(2025, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
        created = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        blob_mock = MagicMock()
        blob_mock.download_as_bytes.return_value = b"content"
        blob_mock.size = 7
        blob_mock.content_type = "text/plain"
        blob_mock.etag = "e"
        blob_mock.generation = 999
        blob_mock.metadata = {"key": "val"}
        blob_mock.updated = updated
        blob_mock.time_created = created
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.download("file.txt")
        assert resp.key == "file.txt"
        assert resp.content == b"content"
        assert resp.metadata.content_length == 7
        assert resp.metadata.version_id == "999"
        # download() shares _build_metadata with get_metadata(), so the
        # reloaded blob's timestamps surface on the download path too
        # (last_modified <- blob.updated, created_time <- blob.time_created).
        assert resp.metadata.last_modified == updated
        assert resp.metadata.created_time == created

    def test_download_with_range(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.download_as_bytes.return_value = b"partial"
        blob_mock.size = 7
        blob_mock.content_type = None
        blob_mock.etag = None
        blob_mock.generation = None
        blob_mock.metadata = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.download("k", start=5, end=10)
        blob_mock.download_as_bytes.assert_called_once_with(start=5, end=10)

    def test_download_with_version_id(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.download_as_bytes.return_value = b""
        blob_mock.size = 0
        blob_mock.content_type = None
        blob_mock.etag = None
        blob_mock.generation = 42
        blob_mock.metadata = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.download("k", version_id="42")
        mock_storage_client.bucket.return_value.blob.assert_called_with(
            "k", generation=42
        )

    def test_download_invalid_version_id_raises(self, store, mock_storage_client):
        with pytest.raises(InvalidArgumentException, match="integer generation"):
            store.download("k", version_id="not-a-number")

    def test_download_not_found(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.download_as_bytes.side_effect = gcs_exceptions.NotFound("gone")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with pytest.raises(ResourceNotFoundException):
            store.download("missing")


class TestGcpBlobStoreWritePreconditions:
    """create_if_absent / if_match conditional-write behavior on upload."""

    def _blob(self, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        return blob_mock

    def test_create_if_absent_sends_if_generation_match_zero(
        self, store, mock_storage_client
    ):
        # GCS create-if-absent maps to if_generation_match=0 (no live object).
        blob_mock = self._blob(mock_storage_client)
        store.upload("k", b"x", create_if_absent=True)
        blob_mock.upload_from_string.assert_called_once_with(
            b"x", content_type="application/octet-stream", if_generation_match=0
        )

    def test_no_create_if_absent_omits_generation_match(
        self, store, mock_storage_client
    ):
        blob_mock = self._blob(mock_storage_client)
        store.upload("k", b"x")
        _, call_kwargs = blob_mock.upload_from_string.call_args
        assert "if_generation_match" not in call_kwargs

    def test_if_match_on_upload_raises_unsupported(self, store, mock_storage_client):
        # GCS has no ETag-based conditional overwrite.
        self._blob(mock_storage_client)
        with pytest.raises(UnsupportedOperationException, match="if_match"):
            store.upload("k", b"x", if_match="etag-abc")

    def test_if_version_match_maps_to_if_generation_match(
        self, store, mock_storage_client
    ):
        # The version id GCS returns is the object generation, so a
        # version-conditioned overwrite maps to if_generation_match=<generation>.
        blob_mock = self._blob(mock_storage_client)
        store.upload("k", b"x", if_version_match="42")
        blob_mock.upload_from_string.assert_called_once_with(
            b"x", content_type="application/octet-stream", if_generation_match=42
        )

    def test_if_version_match_non_numeric_rejected(self, store, mock_storage_client):
        # if_version_match must be a numeric generation; a non-numeric token is
        # a caller mistake, surfaced as InvalidArgumentException.
        self._blob(mock_storage_client)
        with pytest.raises(InvalidArgumentException, match="numeric GCS generation"):
            store.upload("k", b"x", if_version_match="not-a-number")

    def test_if_match_and_if_version_match_together_uses_generation(
        self, store, mock_storage_client
    ):
        # A portable caller may set both tokens so each provider enforces with
        # the one it supports. On GCS that is the generation (if_version_match);
        # the redundant if_match must be ignored rather than trigger the
        # "unsupported" rejection that fires when if_match arrives alone.
        blob_mock = self._blob(mock_storage_client)
        store.upload("k", b"x", if_match="etag-abc", if_version_match="42")
        blob_mock.upload_from_string.assert_called_once_with(
            b"x", content_type="application/octet-stream", if_generation_match=42
        )

    def test_upload_from_path_if_version_match(
        self, store, mock_storage_client, tmp_path
    ):
        blob_mock = self._blob(mock_storage_client)
        src = tmp_path / "f.txt"
        src.write_bytes(b"data")
        store.upload_from_path("k", src, if_version_match="7")
        _, call_kwargs = blob_mock.upload_from_filename.call_args
        assert call_kwargs["if_generation_match"] == 7

    def test_create_if_absent_conflict_relabelled_to_resource_exists(
        self, store, mock_storage_client
    ):
        # create-if-absent maps to if_generation_match=0; a live object at the
        # key returns 412. The store re-labels that conflict locally to
        # ResourceExistsException so callers get a semantically precise error
        # (matching the async GCP and both AWS providers).
        blob_mock = self._blob(mock_storage_client)
        blob_mock.upload_from_string.side_effect = gcs_exceptions.PreconditionFailed(
            "exists"
        )
        with pytest.raises(ResourceExistsException):
            store.upload("k", b"x", create_if_absent=True)

    def test_precondition_failure_without_create_if_absent_propagates_raw(
        self, store, mock_storage_client
    ):
        # Without create_if_absent, a 412 is not a create conflict, so it is not
        # re-labelled: the raw PreconditionFailed propagates for the client layer
        # to translate to FailedPreconditionException.
        blob_mock = self._blob(mock_storage_client)
        blob_mock.upload_from_string.side_effect = gcs_exceptions.PreconditionFailed(
            "boom"
        )
        with pytest.raises(gcs_exceptions.PreconditionFailed):
            store.upload("k", b"x")

    def test_upload_from_path_create_if_absent(
        self, store, mock_storage_client, tmp_path
    ):
        blob_mock = self._blob(mock_storage_client)
        src = tmp_path / "f.txt"
        src.write_bytes(b"data")
        store.upload_from_path("k", src, create_if_absent=True)
        _, call_kwargs = blob_mock.upload_from_filename.call_args
        assert call_kwargs["if_generation_match"] == 0

    def test_if_version_match_on_missing_object_maps_to_failed_precondition(
        self, store, mock_storage_client
    ):
        # Parity with the AWS if_match-on-missing case: a version-conditioned
        # overwrite whose target does not exist yields the precondition-failed
        # outcome. GCS returns 412 PreconditionFailed (not 404), so the store
        # propagates it raw and the client layer translates it to
        # FailedPreconditionException.
        blob_mock = self._blob(mock_storage_client)
        blob_mock.upload_from_string.side_effect = gcs_exceptions.PreconditionFailed(
            "no live object at the requested generation"
        )
        with pytest.raises(gcs_exceptions.PreconditionFailed) as excinfo:
            store.upload("k", b"x", if_version_match="7")
        assert isinstance(
            store.translate_error(excinfo.value), FailedPreconditionException
        )


class TestGcpBlobStoreKmsEncryption:
    """Server-side KMS encryption on upload."""

    def _blob(self, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.etag = "e"
        blob_mock.generation = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        return blob_mock

    def test_kms_key_id_sets_kms_key_name(self, store, mock_storage_client):
        blob_mock = self._blob(mock_storage_client)
        key = "projects/p/locations/l/keyRings/r/cryptoKeys/k"
        store.upload("k", b"x", kms_key_id=key)
        assert blob_mock.kms_key_name == key

    def test_use_kms_managed_key_alone_is_noop(self, store, mock_storage_client):
        # GCS always encrypts at rest with Google-managed keys; requesting a
        # managed key without an explicit id must not assign a customer key
        # name on the blob (it stays the unassigned MagicMock attribute).
        blob_mock = self._blob(mock_storage_client)
        sentinel = blob_mock.kms_key_name  # capture the pre-upload attribute
        store.upload("k", b"x", use_kms_managed_key=True)
        # Attribute is untouched — no explicit key was written.
        assert blob_mock.kms_key_name is sentinel
        blob_mock.upload_from_string.assert_called_once_with(
            b"x", content_type="application/octet-stream"
        )

    def test_translate_error_maps_precondition_failed(self, store):
        result = store.translate_error(gcs_exceptions.PreconditionFailed("412"))
        assert isinstance(result, FailedPreconditionException)


class TestGcpBlobStoreDownloadPreconditions:
    """if_match conditional-read behavior on download."""

    def _blob(self, mock_storage_client, body=b""):
        blob_mock = MagicMock()
        blob_mock.download_as_bytes.return_value = body
        blob_mock.size = len(body)
        blob_mock.content_type = None
        blob_mock.etag = None
        blob_mock.generation = None
        blob_mock.metadata = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        return blob_mock

    def test_if_match_maps_to_if_etag_match(self, store, mock_storage_client):
        blob_mock = self._blob(mock_storage_client)
        store.download("k", if_match="etag-abc")
        blob_mock.download_as_bytes.assert_called_once_with(if_etag_match="etag-abc")

    def test_no_preconditions_omits_conditions(self, store, mock_storage_client):
        blob_mock = self._blob(mock_storage_client)
        store.download("k")
        blob_mock.download_as_bytes.assert_called_once_with()


class TestGcpBlobStoreReplicationStatus:
    """GCS does not expose per-object replication status — always None."""

    def test_download_replication_status_is_none(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.download_as_bytes.return_value = b"content"
        blob_mock.size = 7
        blob_mock.content_type = None
        blob_mock.etag = None
        blob_mock.generation = None
        blob_mock.metadata = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.download("k")
        assert resp.metadata.replication_status is None


class TestGcpBlobStoreDelete:
    """Tests for do_delete."""

    def test_delete_calls_blob_delete(self, store, mock_storage_client):
        blob_mock = MagicMock()
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        store.delete("obj.txt")
        blob_mock.delete.assert_called_once()

    def test_delete_not_found_is_idempotent(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.delete.side_effect = gcs_exceptions.NotFound("already gone")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        # Should NOT raise
        store.delete("already-gone.txt")

    def test_delete_other_error_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        # do_delete only catches NotFound for idempotency. Other errors (like
        # Forbidden) propagate raw at the store level.
        blob_mock = MagicMock()
        blob_mock.delete.side_effect = gcs_exceptions.Forbidden("no access")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        with pytest.raises(gcs_exceptions.Forbidden):
            store.delete("protected.txt")


class TestGcpBlobStoreTranslateError:
    """Tests for error translation logic."""

    def test_already_translated(self, store):
        exc = InvalidArgumentException("already")
        assert store.translate_error(exc) is exc

    def test_not_found(self, store):
        exc = gcs_exceptions.NotFound("gone")
        result = store.translate_error(exc)
        assert isinstance(result, ResourceNotFoundException)

    def test_unauthorized(self, store):
        exc = gcs_exceptions.Unauthorized("no auth")
        result = store.translate_error(exc)
        assert isinstance(result, UnAuthorizedException)

    def test_forbidden(self, store):
        exc = gcs_exceptions.Forbidden("denied")
        result = store.translate_error(exc)
        assert isinstance(result, UnAuthorizedException)

    def test_bad_request(self, store):
        exc = gcs_exceptions.BadRequest("invalid")
        result = store.translate_error(exc)
        assert isinstance(result, InvalidArgumentException)

    def test_refresh_error_returns_unauthorized(self, store):
        # A rejected or expired token surfaces from the auth layer as
        # RefreshError, which the storage API never sees; it must still be
        # classified as an authorization failure, not UnknownException.
        exc = google.auth.exceptions.RefreshError("invalid_grant: token expired")
        result = store.translate_error(exc)
        assert isinstance(result, UnAuthorizedException)

    def test_unknown_error(self, store):
        exc = RuntimeError("boom")
        result = store.translate_error(exc)
        assert isinstance(result, UnknownException)


class TestGcpBlobStoreLifecycle:
    """Tests for close()."""

    def test_close_calls_underlying_client(self, store, mock_storage_client):
        mock_storage_client.close = MagicMock()
        store.close()
        mock_storage_client.close.assert_called_once()


class TestGcpBlobStoreListPage:
    """Tests for do_list_page."""

    def test_list_page_basic(self, store, mock_storage_client):
        blob1 = MagicMock()
        blob1.name = "file1.txt"
        blob1.size = 100
        blob1.etag = "etag1"
        blob1.updated = "2024-01-01T00:00:00Z"
        blob1.storage_class = "STANDARD"

        blob2 = MagicMock()
        blob2.name = "file2.txt"
        blob2.size = 200
        blob2.etag = "etag2"
        blob2.updated = "2024-01-02T00:00:00Z"
        blob2.storage_class = "NEARLINE"

        # Mock the iterator and pages
        page_mock = MagicMock()
        page_mock.__iter__ = MagicMock(return_value=iter([blob1, blob2]))

        iterator_mock = MagicMock()
        iterator_mock.pages = iter([page_mock])
        iterator_mock.next_page_token = None
        iterator_mock.prefixes = None
        mock_storage_client.list_blobs.return_value = iterator_mock

        resp = store.list_page(prefix="")
        assert len(resp.blobs) == 2
        assert resp.blobs[0].key == "file1.txt"
        assert resp.blobs[0].size == 100
        assert resp.blobs[0].storage_class == "STANDARD"
        assert resp.blobs[1].key == "file2.txt"
        assert resp.blobs[1].storage_class == "NEARLINE"
        assert resp.is_truncated is False
        assert resp.next_pagination_token is None

    def test_list_page_with_pagination(self, store, mock_storage_client):
        blob1 = MagicMock()
        blob1.name = "a.txt"
        blob1.size = 10
        blob1.etag = "e"
        blob1.updated = None
        blob1.storage_class = "STANDARD"

        page_mock = MagicMock()
        page_mock.__iter__ = MagicMock(return_value=iter([blob1]))

        iterator_mock = MagicMock()
        iterator_mock.pages = iter([page_mock])
        iterator_mock.next_page_token = "next-token-abc"
        iterator_mock.prefixes = None
        mock_storage_client.list_blobs.return_value = iterator_mock

        resp = store.list_page(max_results=1)
        assert resp.is_truncated is True
        assert resp.next_pagination_token == "next-token-abc"

    def test_list_page_with_delimiter(self, store, mock_storage_client):
        blob1 = MagicMock()
        blob1.name = "dir/file.txt"
        blob1.size = 5
        blob1.etag = "e"
        blob1.updated = None
        blob1.storage_class = "STANDARD"

        page_mock = MagicMock()
        page_mock.__iter__ = MagicMock(return_value=iter([blob1]))

        iterator_mock = MagicMock()
        iterator_mock.pages = iter([page_mock])
        iterator_mock.next_page_token = None
        iterator_mock.prefixes = ("dir/sub1/", "dir/sub2/")
        mock_storage_client.list_blobs.return_value = iterator_mock

        resp = store.list_page(prefix="dir/", delimiter="/")
        assert len(resp.blobs) == 1
        assert resp.common_prefixes == ["dir/sub1/", "dir/sub2/"]

    def test_list_page_empty_result(self, store, mock_storage_client):
        page_mock = MagicMock()
        page_mock.__iter__ = MagicMock(return_value=iter([]))

        iterator_mock = MagicMock()
        iterator_mock.pages = iter([page_mock])
        iterator_mock.next_page_token = None
        iterator_mock.prefixes = None
        mock_storage_client.list_blobs.return_value = iterator_mock

        resp = store.list_page()
        assert resp.blobs == []
        assert resp.is_truncated is False

    def test_list_page_api_error_raises(self, store, mock_storage_client):
        mock_storage_client.list_blobs.side_effect = gcs_exceptions.NotFound(
            "bucket not found"
        )
        with pytest.raises(ResourceNotFoundException):
            store.list_page()

    def test_list_page_invalid_max_results(self, store):
        with pytest.raises(InvalidArgumentException, match="max_results"):
            store.list_page(max_results=0)


class TestGcpBlobStoreDoesObjectExist:
    """Tests for do_does_object_exist."""

    def test_object_exists_returns_true(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.exists.return_value = True
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        assert store.does_object_exist("existing.txt") is True

    def test_object_not_exists_returns_false(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.exists.return_value = False
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        assert store.does_object_exist("missing.txt") is False

    def test_object_exist_with_version_id(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.exists.return_value = True
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        assert store.does_object_exist("key.txt", version_id="12345") is True
        mock_storage_client.bucket.return_value.blob.assert_called_with(
            "key.txt", generation=12345
        )

    def test_object_exist_invalid_version_id_raises(self, store, mock_storage_client):
        with pytest.raises(InvalidArgumentException, match="integer generation"):
            store.does_object_exist("key.txt", version_id="not-int")

    def test_object_exist_forbidden_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        # do_does_object_exist has no provider catch (blob.exists() handles 404
        # but raises other errors raw). Translation happens at the client layer.
        blob_mock = MagicMock()
        blob_mock.exists.side_effect = gcs_exceptions.Forbidden("denied")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with pytest.raises(gcs_exceptions.Forbidden):
            store.does_object_exist("protected.txt")

    def test_object_exist_empty_key_raises(self, store):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.does_object_exist("")


class TestGcpBlobStoreDoesBucketExist:
    """Tests for do_does_bucket_exist."""

    def test_bucket_exists_returns_true(self, store, mock_storage_client):
        mock_storage_client.bucket.return_value.exists.return_value = True
        assert store.does_bucket_exist() is True

    def test_bucket_not_exists_returns_false(self, store, mock_storage_client):
        # bucket.exists() handles 404 internally and returns False (it does
        # not raise NotFound), so a missing bucket surfaces as a False return.
        mock_storage_client.bucket.return_value.exists.return_value = False
        assert store.does_bucket_exist() is False

    def test_bucket_forbidden_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        mock_storage_client.bucket.return_value.exists.side_effect = (
            gcs_exceptions.Forbidden("denied")
        )
        with pytest.raises(gcs_exceptions.Forbidden):
            store.does_bucket_exist()


class TestGcpBlobStoreListBlobVersions:
    """Tests for do_list_blob_versions."""

    def test_list_versions_basic(self, store, mock_storage_client):
        blob1 = MagicMock()
        blob1.name = "doc.txt"
        blob1.size = 200
        blob1.etag = "e2"
        blob1.updated = "2024-02-01"
        blob1.storage_class = "STANDARD"
        blob1.generation = 2000
        blob1.time_deleted = None

        blob2 = MagicMock()
        blob2.name = "doc.txt"
        blob2.size = 100
        blob2.etag = "e1"
        blob2.updated = "2024-01-01"
        blob2.storage_class = "NEARLINE"
        blob2.generation = 1000
        blob2.time_deleted = "2024-02-01"

        mock_storage_client.list_blobs.return_value = iter([blob1, blob2])

        versions = list(store.list_blob_versions("doc.txt"))
        assert len(versions) == 2
        assert versions[0].key == "doc.txt"
        assert versions[0].version_id == "2000"
        assert versions[0].is_latest is True
        assert versions[0].size == 200
        assert versions[0].e_tag == "e2"
        assert versions[0].storage_class == "STANDARD"
        assert versions[1].version_id == "1000"
        assert versions[1].is_latest is False
        assert versions[1].storage_class == "NEARLINE"

    def test_list_versions_filters_prefix_matches(self, store, mock_storage_client):
        blob1 = MagicMock()
        blob1.name = "doc.txt"
        blob1.size = 10
        blob1.etag = "e"
        blob1.updated = None
        blob1.storage_class = "STANDARD"
        blob1.generation = 1
        blob1.time_deleted = None

        blob2 = MagicMock()
        blob2.name = "doc.txt.bak"
        blob2.size = 5
        blob2.etag = "f"
        blob2.updated = None
        blob2.storage_class = "STANDARD"
        blob2.generation = 2
        blob2.time_deleted = None

        mock_storage_client.list_blobs.return_value = iter([blob1, blob2])

        versions = list(store.list_blob_versions("doc.txt"))
        assert len(versions) == 1
        assert versions[0].key == "doc.txt"

    def test_list_versions_empty(self, store, mock_storage_client):
        mock_storage_client.list_blobs.return_value = iter([])
        versions = list(store.list_blob_versions("nonexistent.txt"))
        assert versions == []

    def test_list_versions_api_error_raises(self, store, mock_storage_client):
        mock_storage_client.list_blobs.side_effect = gcs_exceptions.Forbidden("denied")
        with pytest.raises(UnAuthorizedException):
            list(store.list_blob_versions("key.txt"))

    def test_list_versions_empty_key_raises(self, store):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            list(store.list_blob_versions(""))


class TestGcpBlobStoreGetTags:
    """Tests for do_get_tags."""

    def test_get_tags_returns_filtered_metadata(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.metadata = {
            "gcp-tag-env": "prod",
            "gcp-tag-team": "sdk",
            "custom-header": "value",
        }
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        result = store.get_tags("path/obj.txt")
        assert result == {"env": "prod", "team": "sdk"}
        blob_mock.reload.assert_called_once()

    def test_get_tags_no_tags_returns_empty(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.metadata = {"custom-header": "value"}
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        result = store.get_tags("key.txt")
        assert result == {}

    def test_get_tags_null_metadata_returns_empty(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.metadata = None
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        result = store.get_tags("key.txt")
        assert result == {}

    def test_get_tags_filters_null_values(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.metadata = {
            "gcp-tag-active": "yes",
            "gcp-tag-deleted": None,
        }
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        result = store.get_tags("key.txt")
        assert result == {"active": "yes"}

    def test_get_tags_nonexistent_key_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        # do_get_tags has no provider catch; raw GCP errors surface at store level.
        blob_mock = MagicMock()
        blob_mock.reload.side_effect = gcs_exceptions.NotFound("not found")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        with pytest.raises(gcs_exceptions.NotFound):
            store.get_tags("missing.txt")

    def test_get_tags_access_denied_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        blob_mock = MagicMock()
        blob_mock.reload.side_effect = gcs_exceptions.Forbidden("forbidden")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        with pytest.raises(gcs_exceptions.Forbidden):
            store.get_tags("secret.txt")


class TestGcpBlobStoreGetObjectLock:
    """Tests for do_get_object_lock (maps GCS retention + holds to ObjectLockInfo)."""

    def test_locked_retention_maps_to_compliance(self, store, mock_storage_client):
        from datetime import datetime

        until = datetime(2030, 1, 1)
        retention = MagicMock()
        retention.mode = "Locked"
        retention.retain_until_time = until
        blob_mock = MagicMock()
        blob_mock.retention = retention
        blob_mock.event_based_hold = False
        blob_mock.temporary_hold = False
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        info = store.get_object_lock("k")

        assert isinstance(info, ObjectLockInfo)
        assert info.mode is RetentionMode.COMPLIANCE
        assert info.retain_until_date == until
        assert info.legal_hold is False

    def test_unlocked_retention_maps_to_governance(self, store, mock_storage_client):
        retention = MagicMock()
        retention.mode = "Unlocked"
        retention.retain_until_time = None
        blob_mock = MagicMock()
        blob_mock.retention = retention
        blob_mock.event_based_hold = False
        blob_mock.temporary_hold = False
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        info = store.get_object_lock("k")

        assert info.mode is RetentionMode.GOVERNANCE

    def test_event_based_hold_reported_as_legal_hold(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.retention = None
        blob_mock.event_based_hold = True
        blob_mock.temporary_hold = False
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        info = store.get_object_lock("k")

        assert info.legal_hold is True
        assert info.event_based_hold is True
        assert info.mode is None

    def test_temporary_hold_reported_as_legal_hold(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.retention = None
        blob_mock.event_based_hold = False
        blob_mock.temporary_hold = True
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        info = store.get_object_lock("k")

        assert info.legal_hold is True
        assert info.event_based_hold is False

    def test_retention_present_without_mode_maps_to_governance(
        self, store, mock_storage_client
    ):
        # Retention with a retain-until but mode is None counts as a lock and
        # maps to GOVERNANCE (only an explicit Locked mode is COMPLIANCE).
        from datetime import datetime

        retention = MagicMock()
        retention.mode = None
        retention.retain_until_time = datetime(2030, 1, 1)
        blob_mock = MagicMock()
        blob_mock.retention = retention
        blob_mock.event_based_hold = False
        blob_mock.temporary_hold = False
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        info = store.get_object_lock("k")

        assert info is not None
        assert info.mode is RetentionMode.GOVERNANCE
        assert info.legal_hold is False

    def test_empty_retention_shell_returns_none(self, store, mock_storage_client):
        # The google-cloud-storage client always exposes blob.retention as a
        # (possibly empty) Retention object rather than None. An object with no
        # configured retention therefore has a retention object whose mode and
        # retain_until_time are both None; it must still be treated as "no lock"
        # and return None, matching AWS and the cross-cloud contract.
        empty_retention = MagicMock()
        empty_retention.mode = None
        empty_retention.retain_until_time = None
        blob_mock = MagicMock()
        blob_mock.retention = empty_retention
        blob_mock.event_based_hold = False
        blob_mock.temporary_hold = False
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        assert store.get_object_lock("k") is None

    def test_no_lock_state_returns_none(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.retention = None
        blob_mock.event_based_hold = False
        blob_mock.temporary_hold = False
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        assert store.get_object_lock("k") is None

    def test_missing_object_propagates(self, store, mock_storage_client):
        # do_get_object_lock reads properties via blob.reload(), which raises
        # NotFound when the object does not exist. This is not caught here; it
        # propagates for the client layer to translate.
        blob_mock = MagicMock()
        blob_mock.reload.side_effect = gcs_exceptions.NotFound("gone")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with pytest.raises(gcs_exceptions.NotFound):
            store.get_object_lock("missing")

    def test_version_id_parsed_to_generation(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.retention = None
        blob_mock.event_based_hold = False
        blob_mock.temporary_hold = False
        bucket_handle = mock_storage_client.bucket.return_value
        bucket_handle.blob.return_value = blob_mock

        store.get_object_lock("k", version_id="12345")

        # version_id is parsed to an int generation and passed to blob().
        _, kwargs = bucket_handle.blob.call_args
        assert kwargs.get("generation") == 12345


class TestGcpBlobStoreUpdateLegalHold:
    """Tests for do_update_legal_hold (maps to a GCS object hold flag)."""

    def test_enable_sets_temporary_hold_and_patches(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.event_based_hold = False
        blob_mock.metageneration = 7
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.update_legal_hold("k", True)

        blob_mock.reload.assert_called_once()
        assert blob_mock.temporary_hold is True
        # The patch must carry the metageneration read after reload() so a
        # concurrent modification is rejected instead of silently clobbered.
        blob_mock.patch.assert_called_once_with(if_metageneration_match=7)

    def test_disable_clears_both_holds_and_patches(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.event_based_hold = False
        blob_mock.temporary_hold = True
        blob_mock.metageneration = 7
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.update_legal_hold("k", False)

        # Release clears both flags: either one left set would keep the object
        # held and contradict get_object_lock's "held if either flag" read.
        blob_mock.reload.assert_called_once()
        assert blob_mock.temporary_hold is False
        assert blob_mock.event_based_hold is False
        # The patch carries the metageneration read after reload() so a
        # concurrent modification is rejected instead of silently clobbered.
        blob_mock.patch.assert_called_once_with(if_metageneration_match=7)

    def test_enable_updates_active_event_based_hold_instead_of_temporary(
        self, store, mock_storage_client
    ):
        # When placing a hold on an object that already carries an event-based
        # hold, reuse that flag rather than adding a second (temporary) one.
        blob_mock = MagicMock()
        blob_mock.event_based_hold = True
        # Sentinel: the temporary-hold flag must be left untouched on enable.
        blob_mock.temporary_hold = "untouched"
        blob_mock.metageneration = 2
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.update_legal_hold("k", True)

        assert blob_mock.event_based_hold is True
        assert blob_mock.temporary_hold == "untouched"
        blob_mock.patch.assert_called_once_with(if_metageneration_match=2)

    def test_enable_when_temporary_hold_already_active_is_idempotent(
        self, store, mock_storage_client
    ):
        # Placing a hold on an object that already carries only a temporary
        # hold re-asserts the temporary flag (event-based is not active), so
        # a repeated enable is a no-op change that still patches.
        blob_mock = MagicMock()
        blob_mock.event_based_hold = False
        blob_mock.temporary_hold = True
        blob_mock.metageneration = 4
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.update_legal_hold("k", True)

        assert blob_mock.temporary_hold is True
        blob_mock.patch.assert_called_once_with(if_metageneration_match=4)

    def test_disable_clears_both_holds_when_both_active(
        self, store, mock_storage_client
    ):
        # Regression: an object can carry both holds at once (e.g. a bucket
        # default event-based hold plus a temporary hold from a prior call).
        # Releasing must clear both -- clearing only one would report success
        # while the object stays held by the other flag, and would disagree
        # with get_object_lock's "held if either flag" read semantics.
        blob_mock = MagicMock()
        blob_mock.event_based_hold = True
        blob_mock.temporary_hold = True
        blob_mock.metageneration = 3
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.update_legal_hold("k", False)

        blob_mock.reload.assert_called_once()
        assert blob_mock.event_based_hold is False
        assert blob_mock.temporary_hold is False
        blob_mock.patch.assert_called_once_with(if_metageneration_match=3)

    def test_version_id_parsed_to_generation(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.event_based_hold = False
        blob_mock.metageneration = 1
        bucket_handle = mock_storage_client.bucket.return_value
        bucket_handle.blob.return_value = blob_mock

        store.update_legal_hold("k", True, version_id="678")

        _, kwargs = bucket_handle.blob.call_args
        assert kwargs.get("generation") == 678

    def test_missing_object_propagates(self, store, mock_storage_client):
        # do_update_legal_hold reads properties via blob.reload(), which raises
        # NotFound when the object does not exist. This is not caught here; it
        # propagates for the client layer to translate.
        blob_mock = MagicMock()
        blob_mock.reload.side_effect = gcs_exceptions.NotFound("gone")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with pytest.raises(gcs_exceptions.NotFound):
            store.update_legal_hold("missing", True)


class TestGcpBlobStoreSetTags:
    """Tests for do_set_tags."""

    def test_set_tags_preserves_non_tag_metadata(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.metadata = {"custom-header": "keep-me", "gcp-tag-old": "remove"}
        blob_mock.metageneration = 3
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.set_tags("key.txt", {"env": "prod"})

        # Verify metadata was set with old tags nulled + new tags added
        set_metadata = blob_mock.metadata
        assert set_metadata["custom-header"] == "keep-me"
        assert set_metadata["gcp-tag-old"] is None
        assert set_metadata["gcp-tag-env"] == "prod"
        blob_mock.patch.assert_called_once_with(if_metageneration_match=3)

    def test_set_tags_replaces_existing_tags(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.metadata = {"gcp-tag-env": "dev", "gcp-tag-team": "old"}
        blob_mock.metageneration = 5
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.set_tags("key.txt", {"env": "prod"})

        set_metadata = blob_mock.metadata
        assert set_metadata["gcp-tag-env"] == "prod"
        assert set_metadata["gcp-tag-team"] is None
        blob_mock.patch.assert_called_once_with(if_metageneration_match=5)

    def test_set_tags_on_object_with_no_metadata(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.metadata = None
        blob_mock.metageneration = 1
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.set_tags("key.txt", {"env": "test"})

        set_metadata = blob_mock.metadata
        assert set_metadata["gcp-tag-env"] == "test"
        blob_mock.patch.assert_called_once_with(if_metageneration_match=1)

    def test_set_tags_nonexistent_key_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        # do_set_tags has no provider catch; raw GCP errors surface at store level.
        blob_mock = MagicMock()
        blob_mock.reload.side_effect = gcs_exceptions.NotFound("not found")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        with pytest.raises(gcs_exceptions.NotFound):
            store.set_tags("missing.txt", {"env": "test"})

    def test_set_tags_patch_error_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        blob_mock = MagicMock()
        blob_mock.metadata = {}
        blob_mock.metageneration = 1
        blob_mock.patch.side_effect = gcs_exceptions.Forbidden("denied")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        with pytest.raises(gcs_exceptions.Forbidden):
            store.set_tags("key.txt", {"env": "test"})

    def test_set_tags_patch_not_found_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        """Object deleted between reload() and patch() — raw NotFound at store level."""
        blob_mock = MagicMock()
        blob_mock.metadata = {}
        blob_mock.metageneration = 1
        blob_mock.patch.side_effect = gcs_exceptions.NotFound("object deleted")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        with pytest.raises(gcs_exceptions.NotFound):
            store.set_tags("deleted-between-calls.txt", {"env": "test"})

    def test_set_tags_concurrent_modification_not_translated_at_store_level(
        self, store, mock_storage_client
    ):
        """Concurrent metadata change — raw PreconditionFailed at store level."""
        blob_mock = MagicMock()
        blob_mock.metadata = {"gcp-tag-env": "dev"}
        blob_mock.metageneration = 7
        blob_mock.patch.side_effect = gcs_exceptions.PreconditionFailed(
            "412 Precondition Failed: metageneration mismatch"
        )
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        with pytest.raises(gcs_exceptions.PreconditionFailed):
            store.set_tags("key.txt", {"env": "prod"})


class TestGcpBlobStoreSetObjectRetention:
    """Tests for do_update_object_retention."""

    @staticmethod
    def _blob_with_current(gcs_mode, retain_until):
        """A blob mock reporting an existing retention on reload()."""
        blob_mock = MagicMock()
        blob_mock.retention.mode = gcs_mode
        blob_mock.retention.retain_until_time = retain_until
        return blob_mock

    def test_compliance_maps_to_locked(self, store, mock_storage_client):
        # Already Locked (COMPLIANCE); extending the date is allowed.
        blob_mock = self._blob_with_current(
            "Locked", datetime(2029, 1, 1, tzinfo=timezone.utc)
        )
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        until = datetime(2030, 1, 1, tzinfo=timezone.utc)

        store.update_object_retention(
            "key.txt",
            ObjectRetentionConfig(
                mode=RetentionMode.COMPLIANCE, retain_until_date=until
            ),
        )

        assert blob_mock.retention.mode == "Locked"
        assert blob_mock.retention.retain_until_time == until
        blob_mock.patch.assert_called_once_with(override_unlocked_retention=False)

    def test_governance_maps_to_unlocked(self, store, mock_storage_client):
        blob_mock = self._blob_with_current(
            "Unlocked", datetime(2029, 1, 1, tzinfo=timezone.utc)
        )
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.update_object_retention(
            "key.txt",
            ObjectRetentionConfig(
                mode=RetentionMode.GOVERNANCE,
                retain_until_date=datetime(2030, 6, 1, tzinfo=timezone.utc),
            ),
        )

        assert blob_mock.retention.mode == "Unlocked"
        blob_mock.patch.assert_called_once_with(override_unlocked_retention=False)

    def test_none_mode_preserves_current_mode(self, store, mock_storage_client):
        # config.mode=None -> keep the object's current mode (Locked here).
        blob_mock = self._blob_with_current(
            "Locked", datetime(2029, 1, 1, tzinfo=timezone.utc)
        )
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.update_object_retention(
            "key.txt",
            ObjectRetentionConfig(
                mode=None,
                retain_until_date=datetime(2030, 6, 1, tzinfo=timezone.utc),
            ),
        )

        assert blob_mock.retention.mode == "Locked"

    def test_no_current_retention_raises_failed_precondition(
        self, store, mock_storage_client
    ):
        from multicloud.common.exceptions import FailedPreconditionException

        blob_mock = MagicMock()
        blob_mock.retention = None  # object has no retention configured
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with pytest.raises(FailedPreconditionException):
            store.update_object_retention(
                "key.txt",
                ObjectRetentionConfig(
                    mode=RetentionMode.GOVERNANCE,
                    retain_until_date=datetime(2030, 6, 1, tzinfo=timezone.utc),
                ),
            )
        blob_mock.patch.assert_not_called()

    def test_bypass_governance_sets_override(self, store, mock_storage_client):
        blob_mock = self._blob_with_current(
            "Unlocked", datetime(2029, 1, 1, tzinfo=timezone.utc)
        )
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.update_object_retention(
            "key.txt",
            ObjectRetentionConfig(
                mode=RetentionMode.GOVERNANCE,
                retain_until_date=datetime(2030, 6, 1, tzinfo=timezone.utc),
                bypass_governance=True,
            ),
        )

        blob_mock.patch.assert_called_once_with(override_unlocked_retention=True)

    def test_version_id_selects_generation(self, store, mock_storage_client):
        blob_mock = self._blob_with_current(
            "Unlocked", datetime(2029, 1, 1, tzinfo=timezone.utc)
        )
        blob_factory = mock_storage_client.bucket.return_value.blob
        blob_factory.return_value = blob_mock

        store.update_object_retention(
            "key.txt",
            ObjectRetentionConfig(
                mode=RetentionMode.GOVERNANCE,
                retain_until_date=datetime(2030, 6, 1, tzinfo=timezone.utc),
            ),
            version_id="12345",
        )

        blob_factory.assert_called_once_with("key.txt", generation=12345)

    def test_invalid_version_id_raises(self, store, mock_storage_client):
        with pytest.raises(InvalidArgumentException, match="integer generation"):
            store.update_object_retention(
                "key.txt",
                ObjectRetentionConfig(
                    mode=RetentionMode.GOVERNANCE,
                    retain_until_date=datetime(2030, 6, 1, tzinfo=timezone.utc),
                ),
                version_id="not-a-number",
            )

    def test_error_not_translated_at_store_level(self, store, mock_storage_client):
        blob_mock = MagicMock()
        blob_mock.reload.side_effect = gcs_exceptions.NotFound("not found")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        with pytest.raises(gcs_exceptions.NotFound):
            store.update_object_retention(
                "missing.txt",
                ObjectRetentionConfig(
                    mode=RetentionMode.GOVERNANCE,
                    retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
                ),
            )


class TestGcpBlobStoreTranslateErrorSignBlob:
    """Tests for signBlob permission error translation."""

    def test_permission_denied_in_message(self, store):
        exc = RuntimeError("iam.serviceAccounts.signBlob PERMISSION_DENIED for SA")
        result = store.translate_error(exc)
        assert isinstance(result, UnAuthorizedException)

    def test_sign_bytes_in_message(self, store):
        exc = RuntimeError("Failed to sign_bytes: insufficient permissions")
        result = store.translate_error(exc)
        assert isinstance(result, UnAuthorizedException)

    def test_generic_runtime_error_not_sign_related(self, store):
        exc = RuntimeError("connection reset by peer")
        result = store.translate_error(exc)
        assert isinstance(result, UnknownException)


class TestGcpPresignedUrl:
    """Tests for presigned URL generation via GCS V4 signed URLs."""

    def test_upload_basic(self, store, mock_storage_client):
        """Generate a basic signed upload URL."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = (
            "https://storage.googleapis.com/signed"
        )
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.generate_presigned_url(
            "my/key.txt", PresignedOperation.UPLOAD, duration_seconds=3600
        )
        assert resp.url == "https://storage.googleapis.com/signed"
        assert resp.http_method == "PUT"
        assert resp.required_headers == {}
        blob_mock.generate_signed_url.assert_called_once()
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        assert call_kwargs["version"] == "v4"
        assert call_kwargs["method"] == "PUT"
        assert call_kwargs["expiration"] == timedelta(seconds=3600)

    def test_upload_with_content_type(self, store, mock_storage_client):
        """Content-type is passed and returned as required header."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            content_type="image/png",
        )
        assert resp.required_headers["Content-Type"] == "image/png"
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        assert call_kwargs["content_type"] == "image/png"

    def test_upload_with_metadata(self, store, mock_storage_client):
        """Metadata maps to x-goog-meta-* required headers."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            metadata={"author": "test", "version": "1"},
        )
        assert resp.required_headers["x-goog-meta-author"] == "test"
        assert resp.required_headers["x-goog-meta-version"] == "1"
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        headers = call_kwargs["headers"]
        assert headers["x-goog-meta-author"] == "test"
        assert headers["x-goog-meta-version"] == "1"

    def test_upload_with_tags(self, store, mock_storage_client):
        """Tags are mapped to x-goog-meta-gcp-tag-* headers."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            tags={"env": "dev", "team": "sdk"},
        )
        assert resp.required_headers["x-goog-meta-gcp-tag-env"] == "dev"
        assert resp.required_headers["x-goog-meta-gcp-tag-team"] == "sdk"
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        headers = call_kwargs["headers"]
        assert headers["x-goog-meta-gcp-tag-env"] == "dev"
        assert headers["x-goog-meta-gcp-tag-team"] == "sdk"

    def test_upload_with_content_disposition(self, store, mock_storage_client):
        """Content-Disposition is passed as header."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

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
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        headers = call_kwargs["headers"]
        assert headers["Content-Disposition"] == "attachment; filename=report.pdf"

    def test_upload_kms_key_silently_ignored(self, store, mock_storage_client):
        """KMS key is silently ignored for GCS signed URLs."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            kms_key_id="projects/proj/locations/global/keyRings/kr/cryptoKeys/key",
        )
        # URL generated successfully - no KMS-related headers
        assert resp.http_method == "PUT"
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        # No KMS-related keyword args passed
        assert "kms_key_id" not in call_kwargs

    def test_upload_with_all_options(self, store, mock_storage_client):
        """All options combined."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://full-url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.generate_presigned_url(
            "path/file.bin",
            PresignedOperation.UPLOAD,
            duration_seconds=7200,
            content_type="application/octet-stream",
            metadata={"origin": "ci"},
            tags={"build": "123"},
            kms_key_id="ignored-key",
            content_disposition="inline",
        )
        assert resp.http_method == "PUT"
        assert resp.required_headers["Content-Type"] == "application/octet-stream"
        assert resp.required_headers["x-goog-meta-origin"] == "ci"
        assert resp.required_headers["x-goog-meta-gcp-tag-build"] == "123"
        assert resp.required_headers["Content-Disposition"] == "inline"

    def test_download_basic(self, store, mock_storage_client):
        """Generate a basic signed download URL."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = (
            "https://storage.googleapis.com/get"
        )
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.generate_presigned_url(
            "my/key.txt", PresignedOperation.DOWNLOAD, duration_seconds=1800
        )
        assert resp.url == "https://storage.googleapis.com/get"
        assert resp.http_method == "GET"
        assert resp.required_headers == {}
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        assert call_kwargs["version"] == "v4"
        assert call_kwargs["method"] == "GET"
        assert call_kwargs["expiration"] == timedelta(seconds=1800)

    def test_download_with_content_disposition(self, store, mock_storage_client):
        """Download URL with response_disposition."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        resp = store.generate_presigned_url(
            "k",
            PresignedOperation.DOWNLOAD,
            duration_seconds=600,
            content_disposition="attachment; filename=data.csv",
        )
        assert resp.required_headers == {}  # download has no required headers
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        assert call_kwargs["response_disposition"] == "attachment; filename=data.csv"

    def test_credentials_refreshed_when_expired(self, store, mock_storage_client):
        """Credentials are refreshed when not valid."""
        creds = mock_storage_client._credentials
        creds.valid = False

        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with patch("google.auth.transport.requests.Request") as mock_request_cls:
            store.generate_presigned_url(
                "k", PresignedOperation.UPLOAD, duration_seconds=60
            )
            creds.refresh.assert_called_once_with(mock_request_cls())

    def test_access_token_passed_when_no_local_signer(self, store, mock_storage_client):
        """access_token and service_account_email passed when creds lack local signer."""
        creds = mock_storage_client._credentials
        # Remove signer attribute to simulate no local private key
        del creds.signer

        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.generate_presigned_url(
            "k", PresignedOperation.UPLOAD, duration_seconds=60
        )
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        assert call_kwargs["access_token"] == "fake-access-token"
        assert (
            call_kwargs["service_account_email"]
            == "test@project.iam.gserviceaccount.com"
        )

    def test_access_token_passed_when_iam_backed_signer(
        self, store, mock_storage_client
    ):
        """access_token passed when signer exists but key_id is None (IAM-backed).

        This covers the impersonated service account case where creds.signer
        exists but uses IAM signBlob under the hood (no local private key).
        """
        creds = mock_storage_client._credentials
        # Simulate IAM-backed signer: signer attribute exists with sign method,
        # but key_id is None (no local private key).
        creds.signer = MagicMock()
        creds.signer.sign = MagicMock()
        creds.signer.key_id = None

        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.generate_presigned_url(
            "k", PresignedOperation.UPLOAD, duration_seconds=60
        )
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        assert call_kwargs["access_token"] == "fake-access-token"
        assert (
            call_kwargs["service_account_email"]
            == "test@project.iam.gserviceaccount.com"
        )

    def test_local_signer_used_when_key_id_present(self, store, mock_storage_client):
        """Local signer used directly when creds have a real key_id (SA key file).

        This ensures we don't force IAM signBlob when the credential has a
        working local private key — no extra round-trip or permission needed.
        """
        creds = mock_storage_client._credentials
        # Simulate real service account key: signer has sign method AND key_id
        creds.signer = MagicMock()
        creds.signer.sign = MagicMock()
        creds.signer.key_id = "real-private-key-id"

        blob_mock = MagicMock()
        blob_mock.generate_signed_url.return_value = "https://url"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        store.generate_presigned_url(
            "k", PresignedOperation.UPLOAD, duration_seconds=60
        )
        call_kwargs = blob_mock.generate_signed_url.call_args[1]
        # Should NOT pass access_token/service_account_email — uses local signer
        assert "access_token" not in call_kwargs
        assert "service_account_email" not in call_kwargs

    def test_sign_error_translated(self, store, mock_storage_client):
        """Exception during signing propagates raw."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.side_effect = gcs_exceptions.Forbidden("denied")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with pytest.raises(gcs_exceptions.Forbidden):
            store.generate_presigned_url(
                "k", PresignedOperation.UPLOAD, duration_seconds=600
            )

    def test_sign_blob_permission_error_translated(self, store, mock_storage_client):
        """signBlob permission error propagates raw."""
        blob_mock = MagicMock()
        blob_mock.generate_signed_url.side_effect = RuntimeError(
            "PERMISSION_DENIED: caller does not have iam.serviceAccounts.signBlob"
        )
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        with pytest.raises(RuntimeError):
            store.generate_presigned_url(
                "k", PresignedOperation.DOWNLOAD, duration_seconds=600
            )


class TestGcpMultipartUpload:
    """Tests for GCP multipart upload (compose-based simulation)."""

    def test_initiate_multipart_upload(self, store, mock_storage_client):
        mpu = store.initiate_multipart_upload(
            "key.bin", metadata={"env": "test"}, content_type="image/png"
        )
        assert mpu.key == "key.bin"
        assert mpu.bucket == "test-bucket"
        assert mpu.metadata == {"env": "test"}
        assert mpu.content_type == "image/png"
        assert len(mpu.upload_id) == 32  # uuid4().hex

    def test_initiate_multipart_upload_carries_tags(self, store, mock_storage_client):
        # GCS has no native MPU tagging; tags ride the in-memory handle and are
        # applied at complete time.
        mpu = store.initiate_multipart_upload("k", tags={"team": "storage"})
        assert mpu.tags == {"team": "storage"}

    def test_initiate_multipart_upload_carries_object_lock(
        self, store, mock_storage_client
    ):
        # The composed object does not exist until complete, so the lock rides
        # the in-memory handle and is applied when the parts are composed.
        retain_until = datetime.now(timezone.utc) + timedelta(days=1)
        lock = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=retain_until,
            legal_hold=True,
        )
        mpu = store.initiate_multipart_upload("k", object_lock=lock)
        assert mpu.object_lock is lock
        # Nothing is written at initiate — the destination object is created by
        # compose at completion.
        mock_storage_client.bucket.return_value.blob.return_value.patch.assert_not_called()

    def test_upload_part_bytes(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        blob_mock = MagicMock()
        blob_mock.etag = "etag-1"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        part = MultipartPart(part_number=1, content=b"hello")
        resp = store.upload_part(mpu, part)
        assert resp.part_number == 1
        assert resp.e_tag == "etag-1"
        assert resp.size_in_bytes == 5
        blob_mock.upload_from_string.assert_called_once_with(
            b"hello", content_type="application/octet-stream"
        )

    def test_upload_part_binary_io(self, store, mock_storage_client):
        import io

        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        blob_mock = MagicMock()
        blob_mock.etag = "etag-stream"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        stream = io.BytesIO(b"streamed")
        part = MultipartPart(part_number=2, content=stream, content_length=8)
        resp = store.upload_part(mpu, part)
        assert resp.part_number == 2
        assert resp.size_in_bytes == 8
        blob_mock.upload_from_file.assert_called_once_with(
            stream, content_type="application/octet-stream", size=8
        )

    def test_upload_part_api_error(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        blob_mock = MagicMock()
        blob_mock.upload_from_string.side_effect = gcs_exceptions.Forbidden("denied")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        part = MultipartPart(part_number=1, content=b"data")
        with pytest.raises(gcs_exceptions.Forbidden):
            store.upload_part(mpu, part)

    def test_complete_multipart_upload(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        dest_blob = MagicMock()
        dest_blob.etag = "final-etag"
        bucket_mock = mock_storage_client.bucket.return_value
        bucket_mock.blob.return_value = dest_blob

        mpu = MultipartUpload(
            bucket="test-bucket",
            key="k",
            upload_id="abc123",
            content_type="text/plain",
            metadata={"a": "b"},
        )
        parts = [
            UploadPartResponse(part_number=2, e_tag="e2", size_in_bytes=100),
            UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100),
        ]
        resp = store.complete_multipart_upload(mpu, parts)
        assert resp.e_tag == "final-etag"
        dest_blob.compose.assert_called_once()

    def test_complete_multipart_upload_applies_tags_as_metadata(
        self, store, mock_storage_client
    ):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        dest_blob = MagicMock()
        dest_blob.etag = "final-etag"
        mock_storage_client.bucket.return_value.blob.return_value = dest_blob

        mpu = MultipartUpload(
            bucket="test-bucket",
            key="k",
            upload_id="abc123",
            metadata={"env": "test"},
            tags={"team": "storage"},
        )
        parts = [UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100)]
        store.complete_multipart_upload(mpu, parts)
        # Tags land as gcp-tag-* metadata alongside user metadata, matching the
        # encoding get_tags / set_tags round-trip on.
        assert dest_blob.metadata == {
            "env": "test",
            "gcp-tag-team": "storage",
        }

    def test_complete_multipart_upload_tags_only(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        dest_blob = MagicMock()
        dest_blob.etag = "final-etag"
        mock_storage_client.bucket.return_value.blob.return_value = dest_blob

        mpu = MultipartUpload(
            bucket="test-bucket",
            key="k",
            upload_id="abc123",
            tags={"a": "1", "b": "2"},
        )
        parts = [UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100)]
        store.complete_multipart_upload(mpu, parts)
        assert dest_blob.metadata == {"gcp-tag-a": "1", "gcp-tag-b": "2"}

    def test_complete_multipart_upload_applies_retention_before_compose(
        self, store, mock_storage_client
    ):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        dest_blob = MagicMock()
        dest_blob.etag = "final-etag"
        mock_storage_client.bucket.return_value.blob.return_value = dest_blob

        # Retention is populated via attribute assignment, not a method call, so
        # method_calls ordering (used by the legal-hold test) can't see it.
        # Snapshot it inside compose's side_effect instead, proving retention was
        # already set at the moment the request is serialized. Moving
        # _apply_object_lock after compose leaves these captured values unset and
        # fails the assertions.
        captured = {}

        def snapshot_retention_on_compose(*args, **kwargs):
            captured["mode"] = dest_blob.retention.mode
            captured["retain_until_time"] = dest_blob.retention.retain_until_time

        dest_blob.compose.side_effect = snapshot_retention_on_compose

        retain_until = datetime.now(timezone.utc) + timedelta(days=1)
        mpu = MultipartUpload(
            bucket="test-bucket",
            key="k",
            upload_id="abc123",
            object_lock=ObjectLockConfiguration(
                mode=RetentionMode.GOVERNANCE, retain_until_date=retain_until
            ),
        )
        parts = [UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100)]
        store.complete_multipart_upload(mpu, parts)
        # Retention was set on the destination *before* compose ran, so it
        # serializes onto the composed object.
        assert captured["mode"] == "Unlocked"
        assert captured["retain_until_time"] == retain_until
        # Retention-only (no legal hold) writes no post-compose patch.
        dest_blob.patch.assert_not_called()

    def test_complete_multipart_upload_applies_legal_hold_after_compose(
        self, store, mock_storage_client
    ):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        dest_blob = MagicMock()
        dest_blob.etag = "final-etag"
        mock_storage_client.bucket.return_value.blob.return_value = dest_blob

        mpu = MultipartUpload(
            bucket="test-bucket",
            key="k",
            upload_id="abc123",
            object_lock=ObjectLockConfiguration(legal_hold=True),
        )
        parts = [UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100)]
        store.complete_multipart_upload(mpu, parts)
        # A hold cannot ride the compose request, so it is written by a patch
        # after the composed object exists.
        assert dest_blob.temporary_hold is True
        dest_blob.patch.assert_called_once()
        # Order matters: the hold patch must run after compose, otherwise the
        # composed object does not yet exist.
        assert dest_blob.method_calls.index(
            call.compose(ANY)
        ) < dest_blob.method_calls.index(call.patch())

    def test_complete_multipart_upload_event_based_hold_after_compose(
        self, store, mock_storage_client
    ):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        dest_blob = MagicMock()
        dest_blob.etag = "final-etag"
        mock_storage_client.bucket.return_value.blob.return_value = dest_blob

        mpu = MultipartUpload(
            bucket="test-bucket",
            key="k",
            upload_id="abc123",
            object_lock=ObjectLockConfiguration(
                legal_hold=True, use_event_based_hold=True
            ),
        )
        parts = [UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100)]
        store.complete_multipart_upload(mpu, parts)
        assert dest_blob.event_based_hold is True
        dest_blob.patch.assert_called_once()

    def test_complete_multipart_upload_api_error(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import MultipartUpload

        dest_blob = MagicMock()
        dest_blob.compose.side_effect = gcs_exceptions.NotFound("not found")
        mock_storage_client.bucket.return_value.blob.return_value = dest_blob

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        with pytest.raises(gcs_exceptions.NotFound):
            store.complete_multipart_upload(mpu, [])

    def test_list_parts(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import MultipartUpload

        blob1 = MagicMock()
        blob1.name = "k__multipart__abc123__1"
        blob1.etag = "e1"
        blob1.size = 100
        blob2 = MagicMock()
        blob2.name = "k__multipart__abc123__2"
        blob2.etag = "e2"
        blob2.size = 200
        mock_storage_client.list_blobs.return_value = [blob2, blob1]

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        results = store.list_parts(mpu)
        assert len(results) == 2
        assert results[0].part_number == 1
        assert results[1].part_number == 2
        assert results[0].size_in_bytes == 100

    def test_list_parts_empty(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_storage_client.list_blobs.return_value = []
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        results = store.list_parts(mpu)
        assert results == []

    def test_abort_multipart_upload(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import MultipartUpload

        blob1 = MagicMock()
        blob1.name = "k__multipart__abc123__1"
        blob2 = MagicMock()
        blob2.name = "k__multipart__abc123__2"
        mock_storage_client.list_blobs.return_value = [blob1, blob2]

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        store.abort_multipart_upload(mpu)
        blob1.delete.assert_called_once()
        blob2.delete.assert_called_once()

    def test_abort_multipart_upload_api_error(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_storage_client.list_blobs.side_effect = gcs_exceptions.Forbidden("denied")
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        with pytest.raises(gcs_exceptions.Forbidden):
            store.abort_multipart_upload(mpu)

    def test_abort_multipart_upload_individual_delete_failure_continues(
        self, store, mock_storage_client
    ):
        """If one blob.delete() fails, remaining blobs are still cleaned up."""
        from multicloud.blob.blob_client.driver import MultipartUpload

        blob1 = MagicMock()
        blob1.name = "k__multipart__abc123__1"
        blob1.delete.side_effect = gcs_exceptions.ServiceUnavailable("503")
        blob2 = MagicMock()
        blob2.name = "k__multipart__abc123__2"
        mock_storage_client.list_blobs.return_value = [blob1, blob2]

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        store.abort_multipart_upload(mpu)
        blob1.delete.assert_called_once()
        blob2.delete.assert_called_once()

    def test_abort_multipart_upload_not_found_ignored(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import MultipartUpload

        blob1 = MagicMock()
        blob1.name = "k__multipart__abc123__1"
        blob1.delete.side_effect = gcs_exceptions.NotFound("gone")
        mock_storage_client.list_blobs.return_value = [blob1]

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        store.abort_multipart_upload(mpu)

    def test_complete_multipart_upload_iterative_compose(
        self, store, mock_storage_client
    ):
        """Complete with >32 parts uses iterative composition."""
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        dest_blob = MagicMock()
        dest_blob.etag = "composed-etag"
        bucket_mock = mock_storage_client.bucket.return_value
        bucket_mock.blob.return_value = dest_blob

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        parts = [
            UploadPartResponse(part_number=i, e_tag=f"e{i}", size_in_bytes=100)
            for i in range(1, 35)
        ]
        resp = store.complete_multipart_upload(mpu, parts)
        assert resp.e_tag == "composed-etag"
        assert dest_blob.compose.call_count >= 2

    def test_complete_multipart_upload_reload_error(self, store, mock_storage_client):
        """reload() failure after compose propagates raw."""
        from multicloud.blob.blob_client.driver import MultipartUpload

        dest_blob = MagicMock()
        dest_blob.reload.side_effect = gcs_exceptions.ServiceUnavailable("503")
        bucket_mock = mock_storage_client.bucket.return_value
        bucket_mock.blob.return_value = dest_blob

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        with pytest.raises(gcs_exceptions.ServiceUnavailable):
            store.complete_multipart_upload(mpu, [])

    def test_complete_multipart_upload_cleanup_not_found_ignored(
        self, store, mock_storage_client
    ):
        """NotFound during temp blob cleanup after compose is silently ignored."""
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        dest_blob = MagicMock()
        dest_blob.etag = "etag"
        dest_blob.delete.side_effect = gcs_exceptions.NotFound("gone")
        bucket_mock = mock_storage_client.bucket.return_value
        bucket_mock.blob.return_value = dest_blob

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        parts = [UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100)]
        resp = store.complete_multipart_upload(mpu, parts)
        assert resp.e_tag == "etag"

    def test_complete_multipart_upload_cleanup_runs_on_compose_failure(
        self, store, mock_storage_client
    ):
        """Compose failure must still trigger cleanup of temp parts (no orphans)."""
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        # bucket.blob() is called once per source part, then once for dest.
        source_blob_1 = MagicMock(name="source-1")
        source_blob_2 = MagicMock(name="source-2")
        dest_blob = MagicMock(name="dest")
        dest_blob.compose.side_effect = gcs_exceptions.ServiceUnavailable("503")
        mock_storage_client.bucket.return_value.blob.side_effect = [
            source_blob_1,
            source_blob_2,
            dest_blob,
        ]

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        parts = [
            UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100),
            UploadPartResponse(part_number=2, e_tag="e2", size_in_bytes=100),
        ]
        with pytest.raises(gcs_exceptions.ServiceUnavailable):
            store.complete_multipart_upload(mpu, parts)
        # Both temp parts must have been cleaned up despite the compose failure.
        source_blob_1.delete.assert_called_once()
        source_blob_2.delete.assert_called_once()

    def test_complete_multipart_upload_cleanup_failure_swallowed_on_success(
        self, store, mock_storage_client
    ):
        """When compose succeeds, a failure deleting a temp part must not fail the call."""
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        source_blob = MagicMock(name="source-1")
        source_blob.delete.side_effect = gcs_exceptions.ServiceUnavailable("503")
        dest_blob = MagicMock(name="dest")
        dest_blob.etag = "final-etag"
        mock_storage_client.bucket.return_value.blob.side_effect = [
            source_blob,
            dest_blob,
        ]

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        parts = [UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100)]
        resp = store.complete_multipart_upload(mpu, parts)
        assert resp.e_tag == "final-etag"
        source_blob.delete.assert_called_once()

    def test_iterative_compose_cleanup_runs_on_intermediate_failure(
        self, store, mock_storage_client
    ):
        """Mid-loop compose failure must still clean up already-created intermediates."""
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        # 65 parts → two intermediate batches needed (32 + 32 + 1).
        source_blobs = [MagicMock(name=f"source-{i}") for i in range(1, 66)]
        intermediate_0 = MagicMock(name="intermediate-0")
        intermediate_1 = MagicMock(name="intermediate-1")
        intermediate_1.compose.side_effect = gcs_exceptions.ServiceUnavailable("503")
        dest_blob = MagicMock(name="dest")

        # Order: 65 source parts, dest, intermediate_0, intermediate_1.
        mock_storage_client.bucket.return_value.blob.side_effect = source_blobs + [
            dest_blob,
            intermediate_0,
            intermediate_1,
        ]

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        parts = [
            UploadPartResponse(part_number=i, e_tag=f"e{i}", size_in_bytes=100)
            for i in range(1, 66)
        ]
        with pytest.raises(gcs_exceptions.ServiceUnavailable):
            store.complete_multipart_upload(mpu, parts)
        # The first intermediate was created before the second compose failed,
        # so it must have been cleaned up.
        intermediate_0.delete.assert_called_once()

    def test_list_parts_skips_non_numeric_suffix(self, store, mock_storage_client):
        """Blobs with non-numeric suffixes are skipped in list_parts."""
        from multicloud.blob.blob_client.driver import MultipartUpload

        blob_valid = MagicMock()
        blob_valid.name = "k__multipart__abc123__1"
        blob_valid.etag = "e1"
        blob_valid.size = 100
        blob_invalid = MagicMock()
        blob_invalid.name = "k__multipart__abc123__intermediate_0"
        mock_storage_client.list_blobs.return_value = [blob_valid, blob_invalid]

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        results = store.list_parts(mpu)
        assert len(results) == 1
        assert results[0].part_number == 1

    def test_list_parts_api_error(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_storage_client.list_blobs.side_effect = gcs_exceptions.Forbidden("denied")
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        with pytest.raises(gcs_exceptions.Forbidden):
            store.list_parts(mpu)

    def test_upload_part_rejects_part_number_above_compose_limit(
        self, store, mock_storage_client
    ):
        """Part numbers above the 1024 GCS compose ceiling are rejected pre-flight."""
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        part = MultipartPart(part_number=1025, content=b"x")
        with pytest.raises(InvalidArgumentException, match="at most 1024 parts"):
            store.upload_part(mpu, part)
        # The blob handle must not have been touched — we fail before any I/O.
        mock_storage_client.bucket.return_value.blob.assert_not_called()

    def test_upload_part_accepts_part_number_at_compose_limit(
        self, store, mock_storage_client
    ):
        """part_number=1024 (the ceiling) is still accepted."""
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        blob_mock = MagicMock()
        blob_mock.etag = "etag-1024"
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        part = MultipartPart(part_number=1024, content=b"x")
        resp = store.upload_part(mpu, part)
        assert resp.part_number == 1024
        assert resp.e_tag == "etag-1024"

    def test_complete_rejects_too_many_parts(self, store, mock_storage_client):
        """Complete with >1024 parts is rejected pre-flight, before any compose call."""
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="abc123")
        parts = [
            UploadPartResponse(part_number=i, e_tag=f"e{i}", size_in_bytes=1)
            for i in range(1, 1026)
        ]
        with pytest.raises(InvalidArgumentException, match="at most 1024 parts"):
            store.complete_multipart_upload(mpu, parts)
        # No blob handle should be requested if we fail the pre-flight check.
        mock_storage_client.bucket.return_value.blob.assert_not_called()


class TestGcpBlobStoreDeleteWithVersion:
    """Tests for do_delete with version_id."""

    def test_delete_with_version_id(self, store, mock_storage_client):
        blob_mock = MagicMock()
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        store.delete("obj.txt", version_id="12345")
        # Should use generation= on blob handle (version selector), not if_generation_match
        mock_storage_client.bucket.return_value.blob.assert_called_with(
            "obj.txt", generation=12345
        )
        blob_mock.delete.assert_called_once_with()

    def test_delete_without_version_id(self, store, mock_storage_client):
        blob_mock = MagicMock()
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        store.delete("obj.txt")
        mock_storage_client.bucket.return_value.blob.assert_called_with(
            "obj.txt", generation=None
        )
        blob_mock.delete.assert_called_once_with()

    def test_delete_versioned_not_found_raises(self, store, mock_storage_client):
        """Versioned delete should re-raise NotFound."""
        blob_mock = MagicMock()
        blob_mock.delete.side_effect = gcs_exceptions.NotFound("gone")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        with pytest.raises(gcs_exceptions.NotFound):
            store.delete("obj.txt", version_id="12345")

    def test_delete_unversioned_not_found_is_idempotent(
        self, store, mock_storage_client
    ):
        """Unversioned delete should swallow NotFound."""
        blob_mock = MagicMock()
        blob_mock.delete.side_effect = gcs_exceptions.NotFound("gone")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock
        # Should not raise
        store.delete("obj.txt")


class TestGcpBlobStoreDeleteObjects:
    """Tests for do_delete_objects (batch delete)."""

    def test_batch_delete_success(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import BlobIdentifier

        blob_mock = MagicMock()
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        objects = [BlobIdentifier(key="a.txt"), BlobIdentifier(key="b.txt")]
        resp = store.delete_objects(objects)

        assert resp.errors == []
        assert blob_mock.delete.call_count == 2

    def test_batch_delete_with_version_id(self, store, mock_storage_client):
        """Versioned batch delete uses generation= on blob handle."""
        from multicloud.blob.blob_client.driver import BlobIdentifier

        blob_mock = MagicMock()
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        objects = [BlobIdentifier(key="a.txt", version_id="999")]
        resp = store.delete_objects(objects)

        mock_storage_client.bucket.return_value.blob.assert_called_with(
            "a.txt", generation=999
        )
        assert resp.errors == []

    def test_batch_delete_not_found_is_idempotent(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import BlobIdentifier

        blob_mock = MagicMock()
        blob_mock.delete.side_effect = gcs_exceptions.NotFound("gone")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        objects = [BlobIdentifier(key="missing.txt")]
        resp = store.delete_objects(objects)

        # NotFound should not be reported as an error (idempotent)
        assert resp.errors == []

    def test_batch_delete_collects_errors(self, store, mock_storage_client):
        from multicloud.blob.blob_client.driver import BlobIdentifier

        blob_mock = MagicMock()
        blob_mock.delete.side_effect = gcs_exceptions.Forbidden("no access")
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        objects = [BlobIdentifier(key="protected.txt")]
        resp = store.delete_objects(objects)

        assert len(resp.errors) == 1
        assert resp.errors[0].key == "protected.txt"
        assert "Forbidden" in resp.errors[0].code

    def test_batch_delete_no_chunking(self, store, mock_storage_client):
        """All objects are processed in a single pass (no internal chunking)."""
        from multicloud.blob.blob_client.driver import BlobIdentifier

        blob_mock = MagicMock()
        mock_storage_client.bucket.return_value.blob.return_value = blob_mock

        objects = [BlobIdentifier(key=f"key-{i}") for i in range(200)]
        resp = store.delete_objects(objects)

        # All 200 objects processed directly without chunking
        assert blob_mock.delete.call_count == 200
        assert resp.errors == []


class TestGcpBlobStoreCopy:
    """Tests for do_copy (copy to another bucket)."""

    def test_copy_calls_gcs(self, store, mock_storage_client):
        new_blob_mock = MagicMock()
        new_blob_mock.generation = 789
        new_blob_mock.etag = "new-etag"
        new_blob_mock.updated = None
        mock_storage_client.bucket.return_value.copy_blob.return_value = new_blob_mock

        resp = store.copy("src.txt", dest_bucket="dest-bucket", dest_key="dst.txt")

        assert resp.key == "dst.txt"
        assert resp.version_id == "789"
        assert resp.e_tag == "new-etag"

    def test_copy_with_version_id(self, store, mock_storage_client):
        new_blob_mock = MagicMock()
        new_blob_mock.generation = 100
        new_blob_mock.etag = "e"
        new_blob_mock.updated = None
        bucket_mock = mock_storage_client.bucket.return_value
        bucket_mock.copy_blob.return_value = new_blob_mock
        bucket_mock.blob.return_value = MagicMock()

        store.copy(
            "src.txt",
            dest_bucket="dest-bucket",
            dest_key="dst.txt",
            src_version_id="555",
        )
        # copy_blob must receive source_generation=555; setting
        # generation on the Blob object does not carry into the request.
        _, kwargs = bucket_mock.copy_blob.call_args
        assert kwargs.get("source_generation") == 555


class TestGcpBlobStoreCopyFrom:
    """Tests for do_copy_from (copy from source bucket into current bucket)."""

    def test_copy_from_calls_gcs(self, store, mock_storage_client):
        new_blob_mock = MagicMock()
        new_blob_mock.generation = 456
        new_blob_mock.etag = "from-etag"
        new_blob_mock.updated = None

        src_bucket_mock = MagicMock()
        src_bucket_mock.copy_blob.return_value = new_blob_mock
        src_bucket_mock.blob.return_value = MagicMock()

        # client.bucket() returns different mocks based on call
        mock_storage_client.bucket.return_value = src_bucket_mock

        resp = store.copy_from("src-bucket", "remote.txt", dest_key="local.txt")

        assert resp.key == "local.txt"
        assert resp.version_id == "456"
        assert resp.e_tag == "from-etag"
