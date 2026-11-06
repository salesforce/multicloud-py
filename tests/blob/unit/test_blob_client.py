"""Unit tests for :class:`BucketClient` — the user-facing blob client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from multicloud.blob.blob_aws import AwsBlobStore
from multicloud.blob.blob_client import (
    BucketClient,
    PresignedOperation,
    PresignedUrlResponse,
)
from multicloud.blob.blob_gcp import GcpBlobStore
from multicloud.common.exceptions import (
    InvalidArgumentException,
    ResourceNotFoundException,
    SubstrateSdkException,
    UnAuthorizedException,
    UnknownException,
)


class TestBucketClientConstruction:
    """Tests for ``BucketClient(provider=..., bucket=..., ...)``."""

    def test_aws_creates_aws_blob_store_internally(self):
        client = BucketClient(provider="aws", bucket="b", region="us-west-2")
        assert client.provider == "aws"
        assert client.bucket == "b"
        # The provider class is hidden but reachable for verification.
        assert isinstance(client._store, AwsBlobStore)
        assert client._store.region == "us-west-2"

    def test_gcp_creates_gcp_blob_store_internally(self, monkeypatch):
        # Mock google.auth.default and storage.Client to avoid needing real credentials
        mock_creds = MagicMock()
        monkeypatch.setattr(
            "google.auth.default",
            lambda **kwargs: (mock_creds, "test-project"),
        )
        monkeypatch.setattr(
            "multicloud.blob.blob_gcp.gcp_blob_store.storage.Client",
            MagicMock(),
        )
        client = BucketClient(
            provider="gcp",
            bucket="b",
        )
        assert client.provider == "gcp"
        assert client.bucket == "b"
        assert isinstance(client._store, GcpBlobStore)

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            BucketClient(provider="azure", bucket="b")
        assert "azure" in str(exc_info.value)

    def test_provider_is_keyword_only(self):
        # `provider` and `bucket` are keyword-only (PEP 3102) — positional
        # args are rejected so the call site stays self-documenting.
        with pytest.raises(TypeError):
            BucketClient("aws", "b")  # type: ignore[misc]

    def test_missing_bucket_raises(self):
        with pytest.raises((InvalidArgumentException, TypeError)):
            BucketClient(provider="aws", bucket="")

    def test_endpoint_forwarded_to_provider(self):
        client = BucketClient(
            provider="aws",
            bucket="b",
            region="us-west-2",
            endpoint="http://localhost:9000",
        )
        # AwsBlobStore accepts ``endpoint_url`` — verify it was forwarded
        # via the ``endpoint`` -> ``endpoint_url`` mapping in BucketClient.
        s3_client = client._store._s3
        # boto3 stores the endpoint on the client; it surfaces via meta.
        assert "localhost:9000" in s3_client.meta.endpoint_url

    def test_region_forwarded_to_provider(self):
        # Region passes through to the provider constructor.
        client = BucketClient(
            provider="aws",
            bucket="b",
            region="eu-central-1",
        )
        assert client._store.region == "eu-central-1"

    def test_credentials_overrider_forwarded_to_provider(self):
        from unittest.mock import MagicMock, patch

        from multicloud.common.credentials import (
            CredentialsOverrider,
            CredentialsType,
            StsCredentials,
        )

        overrider = CredentialsOverrider(
            credentials_type=CredentialsType.SESSION,
            session_credentials=StsCredentials(
                access_key_id="AK",
                access_key_secret="SK",
                security_token="ST",
            ),
        )
        with patch(
            "multicloud.blob.blob_aws.aws_blob_store.get_session"
        ) as mock_get_session:
            mock_get_session.return_value = MagicMock()
            client = BucketClient(
                provider="aws",
                bucket="b",
                region="us-east-1",
                credentials_overrider=overrider,
            )
            # The overrider reaches the driver layer and the shared session helper.
            assert client._store._credentials_overrider is overrider
            mock_get_session.assert_called_once_with(overrider, "us-east-1")


class TestBucketClientLifecycle:
    """Tests for ``close()`` and the context-manager protocol."""

    def test_context_manager(self):
        with BucketClient(provider="aws", bucket="b", region="us-west-2") as c:
            assert c.provider == "aws"

    def test_close_is_idempotent(self):
        client = BucketClient(provider="aws", bucket="b", region="us-west-2")
        client.close()
        client.close()  # second call should not raise


class TestBucketClientListingAPIs:
    """Tests that BucketClient forwards listing/metadata APIs to the store."""

    def test_list_page_forwards(self):
        client = BucketClient(provider="aws", bucket="b", region="us-west-2")
        mock_store = MagicMock()
        from multicloud.blob.blob_client.driver import ListBlobsPageResponse

        mock_store.list_page.return_value = ListBlobsPageResponse()
        client._store = mock_store

        resp = client.list_page(prefix="data/", delimiter="/", max_results=10)
        mock_store.list_page.assert_called_once_with(
            prefix="data/", delimiter="/", max_results=10, pagination_token=None
        )

    def test_does_object_exist_forwards(self):
        client = BucketClient(provider="aws", bucket="b", region="us-west-2")
        mock_store = MagicMock()
        mock_store.does_object_exist.return_value = True
        client._store = mock_store

        result = client.does_object_exist("key.txt", version_id="v1")
        assert result is True
        mock_store.does_object_exist.assert_called_once_with("key.txt", version_id="v1")

    def test_does_bucket_exist_forwards(self):
        client = BucketClient(provider="aws", bucket="b", region="us-west-2")
        mock_store = MagicMock()
        mock_store.does_bucket_exist.return_value = True
        client._store = mock_store

        result = client.does_bucket_exist()
        assert result is True
        mock_store.does_bucket_exist.assert_called_once()

    def test_list_blob_versions_forwards(self):
        client = BucketClient(provider="aws", bucket="b", region="us-west-2")
        mock_store = MagicMock()
        from multicloud.blob.blob_client.driver import BlobInfo

        expected = [BlobInfo(key="doc.txt", version_id="v1")]
        mock_store.list_blob_versions.return_value = iter(expected)
        client._store = mock_store

        result = list(client.list_blob_versions("doc.txt"))
        mock_store.list_blob_versions.assert_called_once_with("doc.txt")
        assert len(result) == 1
        assert result[0].key == "doc.txt"
        assert result[0].version_id == "v1"


class TestBucketClientForwardingMethods:
    """Tests for forwarding methods that delegate to the underlying store."""

    def _make_client_with_mock_store(self):
        client = BucketClient(provider="aws", bucket="b", region="us-west-2")
        mock_store = MagicMock()
        client._store = mock_store
        return client, mock_store

    def test_upload_forwards(self):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import UploadResponse

        mock_store.upload.return_value = UploadResponse(
            key="k", e_tag="etag", version_id="v1"
        )
        resp = client.upload(
            "k", b"data", content_type="text/plain", metadata={"a": "b"}
        )
        assert resp.key == "k"
        assert resp.e_tag == "etag"
        mock_store.upload.assert_called_once_with(
            "k",
            b"data",
            content_type="text/plain",
            content_length=None,
            metadata={"a": "b"},
            tags=None,
            object_lock=None,
            create_if_absent=False,
            if_match=None,
            if_version_match=None,
            kms_key_id=None,
            use_kms_managed_key=False,
            storage_class=None,
        )

    def test_upload_forwards_tags(self):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import UploadResponse

        mock_store.upload.return_value = UploadResponse(key="k")
        client.upload("k", b"data", tags={"env": "prod", "owner": "team"})
        mock_store.upload.assert_called_once_with(
            "k",
            b"data",
            content_type=None,
            content_length=None,
            metadata=None,
            tags={"env": "prod", "owner": "team"},
            object_lock=None,
            create_if_absent=False,
            if_match=None,
            if_version_match=None,
            kms_key_id=None,
            use_kms_managed_key=False,
            storage_class=None,
        )

    def test_upload_forwards_object_lock(self):
        from datetime import datetime, timedelta, timezone

        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import (
            ObjectLockConfiguration,
            RetentionMode,
            UploadResponse,
        )

        mock_store.upload.return_value = UploadResponse(key="k")
        lock = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=datetime.now(timezone.utc) + timedelta(days=1),
            legal_hold=True,
        )
        client.upload("k", b"data", object_lock=lock)
        mock_store.upload.assert_called_once_with(
            "k",
            b"data",
            content_type=None,
            content_length=None,
            metadata=None,
            tags=None,
            object_lock=lock,
            create_if_absent=False,
            if_match=None,
            if_version_match=None,
            kms_key_id=None,
            use_kms_managed_key=False,
            storage_class=None,
        )

    def test_upload_forwards_preconditions_and_kms(self):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import UploadResponse

        mock_store.upload.return_value = UploadResponse(key="k")
        client.upload(
            "k",
            b"data",
            create_if_absent=True,
            kms_key_id="arn:key/abc",
            use_kms_managed_key=True,
        )
        mock_store.upload.assert_called_once_with(
            "k",
            b"data",
            content_type=None,
            content_length=None,
            metadata=None,
            tags=None,
            object_lock=None,
            create_if_absent=True,
            if_match=None,
            if_version_match=None,
            kms_key_id="arn:key/abc",
            use_kms_managed_key=True,
            storage_class=None,
        )

    def test_upload_forwards_storage_class(self):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import UploadResponse

        mock_store.upload.return_value = UploadResponse(key="k")
        client.upload("k", b"data", storage_class="NEARLINE")
        assert mock_store.upload.call_args.kwargs["storage_class"] == "NEARLINE"

    def test_upload_from_path_forwards_storage_class(self, tmp_path):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import UploadResponse

        mock_store.upload_from_path.return_value = UploadResponse(key="k")
        src = tmp_path / "f.txt"
        src.write_bytes(b"data")
        client.upload_from_path("k", src, storage_class="STANDARD_IA")
        assert (
            mock_store.upload_from_path.call_args.kwargs["storage_class"]
            == "STANDARD_IA"
        )

    def test_upload_rejected_storage_class_is_translated(self):
        # A provider rejecting an unknown storage class surfaces through the
        # client's translate_error, chained to the native failure.
        client, mock_store = self._make_client_with_mock_store()
        native = RuntimeError("invalid storage class")
        mock_store.upload.side_effect = native
        mock_store.translate_error.return_value = InvalidArgumentException("bad")
        with pytest.raises(InvalidArgumentException) as excinfo:
            client.upload("k", b"data", storage_class="NOT_A_CLASS")
        mock_store.translate_error.assert_called_once_with(native)
        assert excinfo.value.__cause__ is native

    def test_upload_forwards_if_match(self):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import UploadResponse

        mock_store.upload.return_value = UploadResponse(key="k")
        client.upload("k", b"data", if_match="etag-abc")
        mock_store.upload.assert_called_once_with(
            "k",
            b"data",
            content_type=None,
            content_length=None,
            metadata=None,
            tags=None,
            object_lock=None,
            create_if_absent=False,
            if_match="etag-abc",
            if_version_match=None,
            kms_key_id=None,
            use_kms_managed_key=False,
            storage_class=None,
        )

    def test_upload_forwards_if_version_match(self):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import UploadResponse

        mock_store.upload.return_value = UploadResponse(key="k")
        client.upload("k", b"data", if_version_match="42")
        mock_store.upload.assert_called_once_with(
            "k",
            b"data",
            content_type=None,
            content_length=None,
            metadata=None,
            tags=None,
            object_lock=None,
            create_if_absent=False,
            if_match=None,
            if_version_match="42",
            kms_key_id=None,
            use_kms_managed_key=False,
            storage_class=None,
        )

    def test_download_forwards(self):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import BlobMetadata, DownloadResponse

        mock_store.download.return_value = DownloadResponse(
            key="k", content=b"hi", metadata=BlobMetadata(key="k", content_length=2)
        )
        resp = client.download("k", version_id="v1", start=0, end=10, if_match="etag-x")
        assert resp.content == b"hi"
        mock_store.download.assert_called_once_with(
            "k",
            version_id="v1",
            start=0,
            end=10,
            if_match="etag-x",
        )

    def test_get_metadata_forwards(self):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import BlobMetadata

        expected = BlobMetadata(
            key="k", content_length=12, content_type="text/plain", e_tag="etag"
        )
        mock_store.get_metadata.return_value = expected
        result = client.get_metadata("k")
        # The store's result is returned unchanged.
        assert result is expected
        mock_store.get_metadata.assert_called_once_with("k", version_id=None)

    def test_get_metadata_forwards_version_id(self):
        client, mock_store = self._make_client_with_mock_store()
        from multicloud.blob.blob_client.driver import BlobMetadata

        mock_store.get_metadata.return_value = BlobMetadata(key="k", version_id="v9")
        result = client.get_metadata("k", version_id="v9")
        assert result.version_id == "v9"
        mock_store.get_metadata.assert_called_once_with("k", version_id="v9")

    def test_delete_forwards(self):
        client, mock_store = self._make_client_with_mock_store()
        client.delete("k")
        mock_store.delete.assert_called_once_with("k", version_id=None)

    def test_get_tags_forwards(self):
        client, mock_store = self._make_client_with_mock_store()
        mock_store.get_tags.return_value = {"env": "prod"}
        result = client.get_tags("k")
        assert result == {"env": "prod"}
        mock_store.get_tags.assert_called_once_with("k")

    def test_set_tags_forwards(self):
        client, mock_store = self._make_client_with_mock_store()
        client.set_tags("k", {"env": "dev"})
        mock_store.set_tags.assert_called_once_with("k", {"env": "dev"})

    def test_get_object_lock_forwards(self):
        from multicloud.blob.blob_client.driver import ObjectLockInfo, RetentionMode

        client, mock_store = self._make_client_with_mock_store()
        mock_store.get_object_lock.return_value = ObjectLockInfo(
            mode=RetentionMode.COMPLIANCE, legal_hold=True
        )
        result = client.get_object_lock("k", version_id="v1")
        assert result.mode is RetentionMode.COMPLIANCE
        assert result.legal_hold is True
        mock_store.get_object_lock.assert_called_once_with("k", version_id="v1")

    def test_update_legal_hold_forwards_enable(self):
        client, mock_store = self._make_client_with_mock_store()
        client.update_legal_hold("k", True, version_id="v1")
        mock_store.update_legal_hold.assert_called_once_with("k", True, version_id="v1")

    def test_update_legal_hold_forwards_disable(self):
        client, mock_store = self._make_client_with_mock_store()
        client.update_legal_hold("k", False)
        mock_store.update_legal_hold.assert_called_once_with(
            "k", False, version_id=None
        )

    def test_update_legal_hold_translates_error(self):
        client, mock_store = self._make_client_with_mock_store()
        native = RuntimeError("boom")
        mock_store.update_legal_hold.side_effect = native
        mock_store.translate_error.return_value = ResourceNotFoundException("nope")
        with pytest.raises(ResourceNotFoundException):
            client.update_legal_hold("k", True)
        mock_store.translate_error.assert_called_once_with(native)

    def test_update_object_retention_forwards(self):
        from datetime import datetime, timezone

        from multicloud.blob.blob_client.driver import (
            ObjectRetentionConfig,
            RetentionMode,
        )

        client, mock_store = self._make_client_with_mock_store()
        retention = ObjectRetentionConfig(
            mode=RetentionMode.COMPLIANCE,
            retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        client.update_object_retention("k", retention, version_id="v1")
        mock_store.update_object_retention.assert_called_once_with(
            "k", retention, version_id="v1"
        )

    def test_generate_presigned_url_forwards(self):
        client, mock_store = self._make_client_with_mock_store()
        mock_store.generate_presigned_url.return_value = PresignedUrlResponse(
            url="https://signed", http_method="PUT", required_headers={}
        )
        resp = client.generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            content_type="text/plain",
            metadata={"a": "b"},
            tags={"env": "test"},
            kms_key_id="key-arn",
            content_disposition="inline",
        )
        assert resp.url == "https://signed"
        mock_store.generate_presigned_url.assert_called_once_with(
            "k",
            PresignedOperation.UPLOAD,
            duration_seconds=600,
            content_type="text/plain",
            metadata={"a": "b"},
            tags={"env": "test"},
            kms_key_id="key-arn",
            content_disposition="inline",
        )

    def test_initiate_multipart_upload_forwards(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        client, mock_store = self._make_client_with_mock_store()
        mock_store.initiate_multipart_upload.return_value = MultipartUpload(
            bucket="b", key="k", upload_id="id-1"
        )
        result = client.initiate_multipart_upload(
            "k", metadata={"a": "b"}, content_type="text/plain"
        )
        assert result.upload_id == "id-1"
        mock_store.initiate_multipart_upload.assert_called_once_with(
            "k",
            metadata={"a": "b"},
            content_type="text/plain",
            tags=None,
            object_lock=None,
        )

    def test_initiate_multipart_upload_forwards_tags(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        client, mock_store = self._make_client_with_mock_store()
        mock_store.initiate_multipart_upload.return_value = MultipartUpload(
            bucket="b", key="k", upload_id="id-1", tags={"team": "storage"}
        )
        result = client.initiate_multipart_upload("k", tags={"team": "storage"})
        assert result.tags == {"team": "storage"}
        mock_store.initiate_multipart_upload.assert_called_once_with(
            "k",
            metadata=None,
            content_type=None,
            tags={"team": "storage"},
            object_lock=None,
        )

    def test_initiate_multipart_upload_forwards_object_lock(self):
        from datetime import datetime, timezone

        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            ObjectLockConfiguration,
            RetentionMode,
        )

        client, mock_store = self._make_client_with_mock_store()
        lock = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        mock_store.initiate_multipart_upload.return_value = MultipartUpload(
            bucket="b", key="k", upload_id="id-1", object_lock=lock
        )
        result = client.initiate_multipart_upload("k", object_lock=lock)
        assert result.object_lock is lock
        mock_store.initiate_multipart_upload.assert_called_once_with(
            "k", metadata=None, content_type=None, tags=None, object_lock=lock
        )

    def test_upload_part_forwards(self):
        from multicloud.blob.blob_client.driver import (
            MultipartPart,
            MultipartUpload,
            UploadPartResponse,
        )

        client, mock_store = self._make_client_with_mock_store()
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        part = MultipartPart(part_number=1, content=b"data")
        mock_store.upload_part.return_value = UploadPartResponse(
            part_number=1, e_tag="etag", size_in_bytes=4
        )
        result = client.upload_part(mpu, part)
        assert result.e_tag == "etag"
        mock_store.upload_part.assert_called_once_with(mpu, part)

    def test_complete_multipart_upload_forwards(self):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            MultipartUploadResponse,
            UploadPartResponse,
        )

        client, mock_store = self._make_client_with_mock_store()
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        parts = [UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100)]
        mock_store.complete_multipart_upload.return_value = MultipartUploadResponse(
            e_tag="final"
        )
        result = client.complete_multipart_upload(mpu, parts)
        assert result.e_tag == "final"
        mock_store.complete_multipart_upload.assert_called_once_with(mpu, parts)

    def test_list_parts_forwards(self):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        client, mock_store = self._make_client_with_mock_store()
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        mock_store.list_parts.return_value = [
            UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100)
        ]
        result = client.list_parts(mpu)
        assert len(result) == 1
        assert result[0].part_number == 1
        mock_store.list_parts.assert_called_once_with(mpu)

    def test_abort_multipart_upload_forwards(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        client, mock_store = self._make_client_with_mock_store()
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        client.abort_multipart_upload(mpu)
        mock_store.abort_multipart_upload.assert_called_once_with(mpu)

    def test_delete_objects_forwards(self):
        from multicloud.blob.blob_client.driver import (
            BlobIdentifier,
            DeleteObjectsResponse,
        )

        client, mock_store = self._make_client_with_mock_store()
        mock_store.delete_objects.return_value = DeleteObjectsResponse(errors=[])
        objects = [BlobIdentifier(key="a"), BlobIdentifier(key="b", version_id="v1")]
        resp = client.delete_objects(objects)
        assert resp.errors == []
        mock_store.delete_objects.assert_called_once_with(objects)

    def test_copy_forwards(self):
        from multicloud.blob.blob_client.driver import CopyResponse

        client, mock_store = self._make_client_with_mock_store()
        mock_store.copy.return_value = CopyResponse(
            key="dst.txt", version_id="v2", e_tag="etag", last_modified=None
        )
        resp = client.copy(
            "src.txt", dest_bucket="other", dest_key="dst.txt", src_version_id="v1"
        )
        assert resp.key == "dst.txt"
        assert resp.version_id == "v2"
        mock_store.copy.assert_called_once_with(
            "src.txt", dest_bucket="other", dest_key="dst.txt", src_version_id="v1"
        )

    def test_copy_from_forwards(self):
        from multicloud.blob.blob_client.driver import CopyResponse

        client, mock_store = self._make_client_with_mock_store()
        mock_store.copy_from.return_value = CopyResponse(
            key="dst.txt", version_id="v3", e_tag="etag2", last_modified=None
        )
        resp = client.copy_from(
            "source-bucket", "src.txt", dest_key="dst.txt", src_version_id="v1"
        )
        assert resp.key == "dst.txt"
        assert resp.version_id == "v3"
        mock_store.copy_from.assert_called_once_with(
            "source-bucket", "src.txt", dest_key="dst.txt", src_version_id="v1"
        )


def _make_aws_client_with_mock_s3(mock_s3):
    """Return a BucketClient backed by a real AwsBlobStore with a mock S3 client."""
    store = AwsBlobStore(bucket="test-bucket", region="us-east-1", s3_client=mock_s3)
    client = BucketClient(provider="aws", bucket="test-bucket", region="us-east-1")
    client._store = store
    return client


def _forbidden_client_error():
    return ClientError(
        {
            "Error": {"Code": "AccessDenied", "Message": "denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "Op",
    )


def _not_found_client_error():
    return ClientError(
        {
            "Error": {"Code": "NoSuchKey", "Message": "not found"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        "Op",
    )


class TestBucketClientErrorTranslation:
    """Model B: verify provider errors are translated at the client layer.

    Each test constructs a BucketClient backed by a real AwsBlobStore with a
    mock S3 client.  When the mock raises a raw boto error, the client's
    centralized try/except calls store.translate_error() and surfaces a
    cloud-agnostic SubstrateSdkException subclass.
    """

    def test_upload_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.upload("k", b"data")

    def test_upload_no_credentials_translated(self):
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = NoCredentialsError()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.upload("k", b"data")

    def test_delete_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.delete_object.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.delete("k")

    def test_list_page_not_found_translated(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchBucket", "Message": ""},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "ListObjectsV2",
        )
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(ResourceNotFoundException):
            client.list_page()

    def test_get_tags_not_found_translated(self):
        mock_s3 = MagicMock()
        mock_s3.get_object_tagging.side_effect = _not_found_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(ResourceNotFoundException):
            client.get_tags("missing.txt")

    def test_get_tags_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.get_object_tagging.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.get_tags("secret.txt")

    def test_get_tags_no_credentials_translated(self):
        mock_s3 = MagicMock()
        mock_s3.get_object_tagging.side_effect = NoCredentialsError()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.get_tags("key.txt")

    def test_get_metadata_not_found_translated(self):
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = _not_found_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(ResourceNotFoundException):
            client.get_metadata("missing.txt")

    def test_get_metadata_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.get_metadata("secret.txt")

    def test_get_metadata_no_credentials_translated(self):
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = NoCredentialsError()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.get_metadata("key.txt")

    def test_set_tags_not_found_translated(self):
        mock_s3 = MagicMock()
        mock_s3.put_object_tagging.side_effect = _not_found_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(ResourceNotFoundException):
            client.set_tags("missing.txt", {"env": "test"})

    def test_set_tags_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.put_object_tagging.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.set_tags("secret.txt", {"env": "test"})

    def test_set_tags_no_credentials_translated(self):
        mock_s3 = MagicMock()
        mock_s3.put_object_tagging.side_effect = NoCredentialsError()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.set_tags("key.txt", {"env": "test"})

    def test_update_object_retention_not_found_translated(self):
        from datetime import datetime, timezone

        from multicloud.blob.blob_client.driver import (
            ObjectRetentionConfig,
            RetentionMode,
        )

        mock_s3 = MagicMock()
        # Object already has retention so the update passes the rules table;
        # the write step is what surfaces the native not-found error.
        mock_s3.get_object_retention.return_value = {
            "Retention": {
                "Mode": "COMPLIANCE",
                "RetainUntilDate": datetime(2029, 1, 1, tzinfo=timezone.utc),
            }
        }
        mock_s3.put_object_retention.side_effect = _not_found_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        retention = ObjectRetentionConfig(
            mode=RetentionMode.COMPLIANCE,
            retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        with pytest.raises(ResourceNotFoundException):
            client.update_object_retention("missing.txt", retention)

    def test_does_object_exist_no_credentials_translated(self):
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = NoCredentialsError()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.does_object_exist("key.txt")

    def test_does_bucket_exist_no_credentials_translated(self):
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = NoCredentialsError()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.does_bucket_exist()

    def test_validation_errors_pass_through_unchanged(self):
        # InvalidArgumentException is already a SubstrateSdkException — the
        # no-double-wrap guard returns it as-is without re-wrapping.
        client = BucketClient(provider="aws", bucket="b", region="us-east-1")
        mock_store = MagicMock()
        mock_store.upload.side_effect = InvalidArgumentException("bad key")
        mock_store.translate_error.side_effect = lambda exc: exc
        client._store = mock_store
        with pytest.raises(InvalidArgumentException):
            client.upload("k", b"data")

    def test_unknown_exception_becomes_unknown_exception(self):
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = RuntimeError("boom")
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnknownException):
            client.upload("k", b"data")

    def test_delete_objects_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.delete_objects.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        from multicloud.blob.blob_client.driver import BlobIdentifier

        with pytest.raises(UnAuthorizedException):
            client.delete_objects([BlobIdentifier(key="k")])

    def test_copy_not_found_translated(self):
        mock_s3 = MagicMock()
        mock_s3.copy_object.side_effect = _not_found_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(ResourceNotFoundException):
            client.copy("src.txt", dest_bucket="b2", dest_key="dst.txt")

    def test_copy_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.copy_object.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.copy("src.txt", dest_bucket="b2", dest_key="dst.txt")

    def test_copy_from_not_found_translated(self):
        mock_s3 = MagicMock()
        mock_s3.copy_object.side_effect = _not_found_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(ResourceNotFoundException):
            client.copy_from("src-bucket", "src.txt", dest_key="dst.txt")

    def test_copy_from_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.copy_object.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.copy_from("src-bucket", "src.txt", dest_key="dst.txt")

    def test_list_blob_versions_validation_error_translated(self):
        # Validation error before the iterator is obtained.
        client = BucketClient(provider="aws", bucket="b", region="us-east-1")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            client.list_blob_versions("")

    def test_generate_presigned_url_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.generate_presigned_url("k", PresignedOperation.UPLOAD)

    def test_generate_presigned_url_no_credentials_translated(self):
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.side_effect = NoCredentialsError()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.generate_presigned_url("k", PresignedOperation.UPLOAD)

    def test_initiate_multipart_upload_access_denied_translated(self):
        mock_s3 = MagicMock()
        mock_s3.create_multipart_upload.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        with pytest.raises(UnAuthorizedException):
            client.initiate_multipart_upload("k")

    def test_initiate_multipart_upload_invalid_object_lock_raises(self):
        from multicloud.blob.blob_client.driver import (
            ObjectLockConfiguration,
            RetentionMode,
        )

        mock_s3 = MagicMock()
        client = _make_aws_client_with_mock_s3(mock_s3)
        # A retention mode without a retain_until_date is rejected during
        # validation, before any provider call is made.
        with pytest.raises(InvalidArgumentException):
            client.initiate_multipart_upload(
                "k", object_lock=ObjectLockConfiguration(mode=RetentionMode.GOVERNANCE)
            )
        mock_s3.create_multipart_upload.assert_not_called()

    def test_upload_part_not_found_translated(self):
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        mock_s3 = MagicMock()
        mock_s3.upload_part.side_effect = _not_found_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="uid")
        part = MultipartPart(part_number=1, content=b"data")
        with pytest.raises(ResourceNotFoundException):
            client.upload_part(mpu, part)

    def test_complete_multipart_upload_access_denied_translated(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_s3 = MagicMock()
        mock_s3.complete_multipart_upload.side_effect = _forbidden_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="uid")
        with pytest.raises(UnAuthorizedException):
            client.complete_multipart_upload(mpu, [])

    def test_list_parts_not_found_translated(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_s3 = MagicMock()
        mock_s3.list_parts.side_effect = _not_found_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="uid")
        with pytest.raises(ResourceNotFoundException):
            client.list_parts(mpu)

    def test_abort_multipart_upload_not_found_translated(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mock_s3 = MagicMock()
        mock_s3.abort_multipart_upload.side_effect = _not_found_client_error()
        client = _make_aws_client_with_mock_s3(mock_s3)
        mpu = MultipartUpload(bucket="test-bucket", key="k", upload_id="uid")
        with pytest.raises(ResourceNotFoundException):
            client.abort_multipart_upload(mpu)
