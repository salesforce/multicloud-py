"""Unit tests for upload_from_path — the file-path convenience upload API.

Tests cover:
- BlobStoreValidator.validate_upload_from_path()
- BaseBlobStore.upload_from_path() (driver layer)
- BucketClient.upload_from_path() (client layer)
- _coerce_to_bytes() with Path payloads
- Content-type auto-detection from file extension
- Content-length auto-calculation from file size
- Error cases (missing file, directory instead of file, empty key)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator, Mapping, Optional
from unittest.mock import MagicMock, patch

import pytest
from multicloud.blob.blob_client import BucketClient
from multicloud.blob.blob_client.driver import (
    BaseBlobStore,
    BlobInfo,
    BlobMetadata,
    BlobStoreValidator,
    DownloadResponse,
    ListBlobsPageResponse,
    MultipartPart,
    MultipartUpload,
    MultipartUploadResponse,
    ObjectLockConfiguration,
    PresignedOperation,
    PresignedUrlResponse,
    StreamingDownloadResponse,
    UploadPartResponse,
)
from multicloud.blob.blob_client.driver import UploadResponse as DriverUploadResponse
from multicloud.blob.blob_client.driver import (
    _coerce_to_bytes,
)
from multicloud.common.exceptions import (
    InvalidArgumentException,
    SubstrateSdkException,
)


class _FakeBlobStore(BaseBlobStore):
    """Minimal concrete implementation for testing upload_from_path."""

    provider_id = "fake"

    def __init__(self, bucket: str, region: Optional[str] = None, **kwargs):
        super().__init__(bucket=bucket, region=region)
        self.upload_calls = []

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        if isinstance(exc, SubstrateSdkException):
            return exc
        return SubstrateSdkException(str(exc))

    def do_upload(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: Optional[str],
        content_length: Optional[int],
        metadata: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration] = None,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        storage_class: Optional[str] = None,
    ) -> DriverUploadResponse:
        self.upload_calls.append(
            {
                "key": key,
                "payload": payload,
                "content_type": content_type,
                "content_length": content_length,
                "metadata": metadata,
                "object_lock": object_lock,
                "create_if_absent": create_if_absent,
                "if_match": if_match,
                "if_version_match": if_version_match,
                "kms_key_id": kms_key_id,
                "use_kms_managed_key": use_kms_managed_key,
                "storage_class": storage_class,
            }
        )
        return DriverUploadResponse(key=key, e_tag="fake-etag", version_id="v1")

    def do_upload_from_path(
        self,
        key: str,
        path: Path,
        *,
        content_type: Optional[str],
        content_length: Optional[int],
        metadata: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration] = None,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        storage_class: Optional[str] = None,
    ) -> DriverUploadResponse:
        # Read file and delegate to do_upload for test tracking
        payload = path.read_bytes()
        return self.do_upload(
            key,
            payload,
            content_type=content_type,
            content_length=content_length,
            metadata=metadata,
            object_lock=object_lock,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
            kms_key_id=kms_key_id,
            use_kms_managed_key=use_kms_managed_key,
            storage_class=storage_class,
        )

    def do_download(self, key, *, version_id, start, end, if_match=None):
        meta = BlobMetadata(key=key)
        return DownloadResponse(key=key, content=b"", metadata=meta)

    def do_download_to_path(self, key, destination, *, version_id, start, end):
        meta = BlobMetadata(key=key)
        return DownloadResponse(key=key, content=b"", metadata=meta)

    def do_download_to_file(self, key, file_obj, *, version_id, start, end):
        meta = BlobMetadata(key=key)
        return DownloadResponse(key=key, content=b"", metadata=meta)

    def do_download_stream(self, key, *, version_id, start, end):
        meta = BlobMetadata(key=key)
        return StreamingDownloadResponse(key=key, metadata=meta, stream=iter([]))

    def do_delete(self, key, *, version_id=None):
        pass

    def do_delete_objects(self, objects):
        from multicloud.blob.blob_client.driver import DeleteObjectsResponse

        return DeleteObjectsResponse(errors=[])

    def do_copy(self, src_key, *, dest_bucket, dest_key, src_version_id):
        from multicloud.blob.blob_client.driver import CopyResponse

        return CopyResponse(key=dest_key)

    def do_copy_from(self, src_bucket, src_key, *, dest_key, src_version_id):
        from multicloud.blob.blob_client.driver import CopyResponse

        return CopyResponse(key=dest_key)

    def do_get_object_lock(self, key, *, version_id=None):
        from multicloud.blob.blob_client.driver import ObjectLockInfo

        return ObjectLockInfo()

    def do_update_legal_hold(self, key, enabled, *, version_id=None):
        pass

    def do_get_tags(self, key):
        return {}

    def do_set_tags(self, key, tags):
        pass

    def do_update_object_retention(self, key, retention, *, version_id):
        pass

    def do_generate_presigned_url(
        self,
        key,
        operation,
        *,
        duration_seconds,
        content_type,
        metadata,
        tags,
        kms_key_id,
        content_disposition,
    ):
        return PresignedUrlResponse(url="https://fake", http_method="PUT")

    def do_list_page(self, *, prefix, delimiter, max_results, pagination_token):
        return ListBlobsPageResponse()

    def do_does_object_exist(self, key, *, version_id):
        return True

    def do_get_metadata(self, key, *, version_id=None):
        return BlobMetadata(key=key)

    def do_does_bucket_exist(self):
        return True

    def do_initiate_multipart_upload(self, key, *, metadata, content_type, tags):
        return MultipartUpload(bucket=self.bucket, key=key, upload_id="fake-upload-id")

    def do_upload_part(self, mpu, part):
        return UploadPartResponse(
            part_number=part.part_number, e_tag="fake-etag", size_in_bytes=0
        )

    def do_complete_multipart_upload(self, mpu, parts):
        return MultipartUploadResponse(e_tag="fake-final-etag")

    def do_list_parts(self, mpu):
        return []

    def do_abort_multipart_upload(self, mpu):
        pass


class TestValidateUploadFromPath:
    """Tests for BlobStoreValidator.validate_upload_from_path()."""

    def setup_method(self):
        self.validator = BlobStoreValidator()

    def test_empty_key_raises(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_upload_from_path("", f)

    def test_nonexistent_file_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.txt"
        with pytest.raises(InvalidArgumentException, match="file not found"):
            self.validator.validate_upload_from_path("key", missing)

    def test_directory_instead_of_file_raises(self, tmp_path):
        with pytest.raises(InvalidArgumentException, match="not a file"):
            self.validator.validate_upload_from_path("key", tmp_path)

    def test_valid_file_passes(self, tmp_path):
        f = tmp_path / "valid.txt"
        f.write_text("content")
        # Should not raise
        self.validator.validate_upload_from_path("key", f)


class TestBaseBlobStoreUploadFromPath:
    """Tests for BaseBlobStore.upload_from_path()."""

    def test_uploads_file_content(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_bytes(b"Hello, World!")
        store = _FakeBlobStore(bucket="test-bucket")

        resp = store.upload_from_path("docs/hello.txt", f)

        assert resp.key == "docs/hello.txt"
        assert resp.e_tag == "fake-etag"
        assert len(store.upload_calls) == 1
        assert store.upload_calls[0]["payload"] == b"Hello, World!"

    def test_auto_detects_content_type_txt(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_bytes(b"text content")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f)

        assert store.upload_calls[0]["content_type"] == "text/plain"

    def test_auto_detects_content_type_json(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_bytes(b'{"key": "value"}')
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f)

        assert store.upload_calls[0]["content_type"] == "application/json"

    def test_auto_detects_content_type_html(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_bytes(b"<html></html>")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f)

        assert store.upload_calls[0]["content_type"] == "text/html"

    def test_auto_detects_content_type_png(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f)

        assert store.upload_calls[0]["content_type"] == "image/png"

    def test_auto_detects_content_type_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b,c\n1,2,3")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f)

        assert store.upload_calls[0]["content_type"] == "text/csv"

    def test_unknown_extension_defaults_to_octet_stream(self, tmp_path):
        f = tmp_path / "data.xyz123"
        f.write_bytes(b"binary data")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f)

        assert store.upload_calls[0]["content_type"] == "application/octet-stream"

    def test_explicit_content_type_overrides_detection(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_bytes(b"content")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f, content_type="application/octet-stream")

        assert store.upload_calls[0]["content_type"] == "application/octet-stream"

    def test_auto_calculates_content_length(self, tmp_path):
        content = b"A" * 1024
        f = tmp_path / "big.bin"
        f.write_bytes(content)
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f)

        assert store.upload_calls[0]["content_length"] == 1024

    def test_explicit_content_length_overrides_auto(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"12345")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f, content_length=999)

        assert store.upload_calls[0]["content_length"] == 999

    def test_empty_file_uploads_zero_bytes(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        store = _FakeBlobStore(bucket="test-bucket")

        resp = store.upload_from_path("key", f)

        assert resp.key == "key"
        assert store.upload_calls[0]["payload"] == b""
        assert store.upload_calls[0]["content_length"] == 0

    def test_metadata_forwarded(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f, metadata={"env": "prod", "team": "sdk"})

        assert store.upload_calls[0]["metadata"] == {"env": "prod", "team": "sdk"}

    def test_kms_params_forwarded(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path(
            "key",
            f,
            kms_key_id="key-arn-123",
            use_kms_managed_key=True,
        )

        assert store.upload_calls[0]["kms_key_id"] == "key-arn-123"
        assert store.upload_calls[0]["use_kms_managed_key"] is True

    def test_storage_class_forwarded(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f, storage_class="COLDLINE")

        assert store.upload_calls[0]["storage_class"] == "COLDLINE"

    def test_blank_storage_class_rejected_before_delegating(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")
        store = _FakeBlobStore(bucket="test-bucket")

        with pytest.raises(InvalidArgumentException, match="storage_class"):
            store.upload_from_path("key", f, storage_class="")

        assert store.upload_calls == []

    def test_conditional_write_params_forwarded(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f, if_match="etag-xyz", if_version_match="42")

        assert store.upload_calls[0]["if_match"] == "etag-xyz"
        assert store.upload_calls[0]["if_version_match"] == "42"

    def test_create_if_absent_forwarded(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f, create_if_absent=True)

        assert store.upload_calls[0]["create_if_absent"] is True

    def test_kms_params_default_to_unset(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f)

        assert store.upload_calls[0]["kms_key_id"] is None
        assert store.upload_calls[0]["use_kms_managed_key"] is False

    def test_metadata_defaults_to_empty_dict(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")
        store = _FakeBlobStore(bucket="test-bucket")

        store.upload_from_path("key", f)

        assert store.upload_calls[0]["metadata"] == {}

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "string_path.txt"
        f.write_bytes(b"via string")
        store = _FakeBlobStore(bucket="test-bucket")

        resp = store.upload_from_path("key", str(f))

        assert resp.key == "key"
        assert store.upload_calls[0]["payload"] == b"via string"

    def test_accepts_os_fspath(self, tmp_path):
        f = tmp_path / "fspath.txt"
        f.write_bytes(b"via fspath")
        store = _FakeBlobStore(bucket="test-bucket")

        # os.fspath returns str; the method should handle it
        resp = store.upload_from_path("key", os.fspath(f))

        assert resp.key == "key"
        assert store.upload_calls[0]["payload"] == b"via fspath"

    def test_nonexistent_file_raises_invalid_argument(self, tmp_path):
        store = _FakeBlobStore(bucket="test-bucket")
        missing = tmp_path / "ghost.txt"

        with pytest.raises(InvalidArgumentException, match="file not found"):
            store.upload_from_path("key", missing)

    def test_directory_path_raises_invalid_argument(self, tmp_path):
        store = _FakeBlobStore(bucket="test-bucket")

        with pytest.raises(InvalidArgumentException, match="not a file"):
            store.upload_from_path("key", tmp_path)

    def test_empty_key_raises_invalid_argument(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")
        store = _FakeBlobStore(bucket="test-bucket")

        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.upload_from_path("", f)


class TestBucketClientUploadFromPath:
    """Tests for BucketClient.upload_from_path()."""

    @pytest.fixture
    def client_with_mock_store(self):
        """Create a BucketClient with its internal store replaced by a mock."""
        client = BucketClient(provider="aws", bucket="b", region="us-west-2")
        mock_store = MagicMock()
        client._store = mock_store
        return client, mock_store

    def test_forwards_all_params(self, client_with_mock_store, tmp_path):
        client, mock_store = client_with_mock_store
        mock_store.upload_from_path.return_value = DriverUploadResponse(
            key="k", e_tag="etag", version_id="v1"
        )
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")

        # create_if_absent is mutually exclusive with the compare-and-swap
        # preconditions, so a realistic "forward everything" call pairs if_match
        # with if_version_match (the valid portable combo) and leaves
        # create_if_absent at its default.
        resp = client.upload_from_path(
            "k",
            f,
            content_type="text/plain",
            content_length=4,
            metadata={"env": "dev"},
            if_match="etag-xyz",
            if_version_match="42",
            kms_key_id="key-arn-123",
            use_kms_managed_key=True,
        )

        mock_store.upload_from_path.assert_called_once_with(
            "k",
            f,
            content_type="text/plain",
            content_length=4,
            metadata={"env": "dev"},
            object_lock=None,
            create_if_absent=False,
            if_match="etag-xyz",
            if_version_match="42",
            kms_key_id="key-arn-123",
            use_kms_managed_key=True,
            storage_class=None,
        )
        assert resp.key == "k"
        assert resp.e_tag == "etag"

    def test_defaults_optional_params(self, client_with_mock_store, tmp_path):
        client, mock_store = client_with_mock_store
        mock_store.upload_from_path.return_value = DriverUploadResponse(
            key="k", e_tag="etag"
        )
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")

        client.upload_from_path("k", f)

        mock_store.upload_from_path.assert_called_once_with(
            "k",
            f,
            content_type=None,
            content_length=None,
            metadata=None,
            object_lock=None,
            create_if_absent=False,
            if_match=None,
            if_version_match=None,
            kms_key_id=None,
            use_kms_managed_key=False,
            storage_class=None,
        )

    def test_accepts_string_path(self, client_with_mock_store, tmp_path):
        client, mock_store = client_with_mock_store
        mock_store.upload_from_path.return_value = DriverUploadResponse(key="k")
        f = tmp_path / "str_path.txt"
        f.write_bytes(b"data")

        client.upload_from_path("k", str(f))

        # Should forward the string path as-is
        call_args = mock_store.upload_from_path.call_args
        assert call_args[0][1] == str(f)

    def test_error_translation(self, client_with_mock_store, tmp_path):
        client, mock_store = client_with_mock_store
        mock_store.upload_from_path.side_effect = InvalidArgumentException("bad")
        mock_store.translate_error.side_effect = lambda exc: exc
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")

        with pytest.raises(InvalidArgumentException):
            client.upload_from_path("k", f)

    def test_runtime_error_translated(self, client_with_mock_store, tmp_path):
        client, mock_store = client_with_mock_store
        mock_store.upload_from_path.side_effect = RuntimeError("boom")
        mock_store.translate_error.return_value = SubstrateSdkException("translated")
        f = tmp_path / "file.txt"
        f.write_bytes(b"data")

        with pytest.raises(SubstrateSdkException, match="translated"):
            client.upload_from_path("k", f)


class TestCoerceToBytesWithPath:
    """Tests for _coerce_to_bytes — Path is NOT supported (use upload_from_path)."""

    def test_path_reads_file_bytes(self, tmp_path):
        """Path payload raises InvalidArgumentException (use upload_from_path)."""
        from multicloud.common.exceptions import InvalidArgumentException

        f = tmp_path / "coerce.bin"
        f.write_bytes(b"\x00\x01\x02\x03")

        with pytest.raises(InvalidArgumentException):
            _coerce_to_bytes(f)

    def test_path_empty_file(self, tmp_path):
        """Empty Path also raises InvalidArgumentException."""
        from multicloud.common.exceptions import InvalidArgumentException

        f = tmp_path / "empty.bin"
        f.write_bytes(b"")

        with pytest.raises(InvalidArgumentException):
            _coerce_to_bytes(f)

    def test_path_large_content(self, tmp_path):
        """Large Path raises InvalidArgumentException."""
        from multicloud.common.exceptions import InvalidArgumentException

        f = tmp_path / "large.bin"
        f.write_bytes(b"X" * 100)

        with pytest.raises(InvalidArgumentException):
            _coerce_to_bytes(f)


class TestUploadFromPathIntegration:
    """End-to-end tests using _FakeBlobStore without mocks."""

    def test_full_upload_cycle(self, tmp_path):
        """Verify the complete path: client -> driver -> provider hook."""
        content = b"Integration test content!"
        f = tmp_path / "integration.json"
        f.write_bytes(content)

        store = _FakeBlobStore(bucket="test-bucket")
        resp = store.upload_from_path(
            "data/integration.json",
            f,
            metadata={"source": "test"},
        )

        assert resp.key == "data/integration.json"
        assert resp.e_tag == "fake-etag"
        call = store.upload_calls[0]
        assert call["payload"] == content
        assert call["content_type"] == "application/json"
        assert call["content_length"] == len(content)
        assert call["metadata"] == {"source": "test"}

    def test_upload_via_upload_with_path_payload(self, tmp_path):
        """Verify that upload() rejects Path — users must use upload_from_path()."""
        from multicloud.common.exceptions import InvalidArgumentException

        f = tmp_path / "via_upload.txt"
        f.write_bytes(b"path payload")
        store = _FakeBlobStore(bucket="test-bucket")

        with pytest.raises(InvalidArgumentException):
            store.upload("key", f)

    def test_symlink_to_file_works(self, tmp_path):
        """Symlinks to regular files should work."""
        real_file = tmp_path / "real.txt"
        real_file.write_bytes(b"symlink content")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)

        store = _FakeBlobStore(bucket="test-bucket")
        resp = store.upload_from_path("key", link)

        assert store.upload_calls[0]["payload"] == b"symlink content"

    def test_binary_content_preserved(self, tmp_path):
        """Ensure binary content with all byte values is preserved."""
        content = bytes(range(256))
        f = tmp_path / "binary.bin"
        f.write_bytes(content)

        store = _FakeBlobStore(bucket="test-bucket")
        store.upload_from_path("key", f)

        assert store.upload_calls[0]["payload"] == content


class TestAwsUploadFromPath:
    """Tests for AwsBlobStore.do_upload_from_path using put_object."""

    def _make_store(self, mock_s3):
        from multicloud.blob.blob_aws import AwsBlobStore

        return AwsBlobStore(bucket="test-bucket", region="us-east-1", s3_client=mock_s3)

    def test_calls_put_object_with_correct_args(self, tmp_path):
        """Verify put_object is called with Bucket, Key, Body, ContentType, ContentLength."""
        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {
            "ETag": '"abc123"',
            "VersionId": "v1",
        }
        store = self._make_store(mock_s3)

        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")

        resp = store.do_upload_from_path(
            "docs/test.txt",
            f,
            content_type="text/plain",
            content_length=11,
            metadata={"author": "test"},
            object_lock=None,
        )

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "docs/test.txt"
        assert call_kwargs["ContentType"] == "text/plain"
        assert call_kwargs["ContentLength"] == 11
        assert call_kwargs["Metadata"] == {"author": "test"}
        assert "Body" in call_kwargs
        assert resp.key == "docs/test.txt"
        assert resp.e_tag == "abc123"
        assert resp.version_id == "v1"

    def test_defaults_content_type_to_octet_stream(self, tmp_path):
        """Verify ContentType defaults to application/octet-stream when None."""
        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {"ETag": '"etag"'}
        store = self._make_store(mock_s3)

        f = tmp_path / "file.bin"
        f.write_bytes(b"\x00\x01")

        store.do_upload_from_path(
            "key",
            f,
            content_type=None,
            content_length=2,
            metadata={},
            object_lock=None,
        )

        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["ContentType"] == "application/octet-stream"
        assert call_kwargs["ContentLength"] == 2
        # No Metadata key when metadata is empty
        assert "Metadata" not in call_kwargs

    def test_put_object_error_propagates(self, tmp_path):
        """Verify boto3 errors during put_object propagate (client layer translates)."""
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "PutObject",
        )
        store = self._make_store(mock_s3)

        f = tmp_path / "denied.txt"
        f.write_bytes(b"data")

        with pytest.raises(ClientError):
            store.do_upload_from_path(
                "key",
                f,
                content_type="text/plain",
                content_length=4,
                metadata={},
                object_lock=None,
            )

    def test_no_content_length_omits_key(self, tmp_path):
        """Verify ContentLength is omitted when content_length is None."""
        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store = self._make_store(mock_s3)

        f = tmp_path / "file.dat"
        f.write_bytes(b"abc")

        store.do_upload_from_path(
            "key",
            f,
            content_type="text/plain",
            content_length=None,
            metadata={},
            object_lock=None,
        )

        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert "ContentLength" not in call_kwargs

    def test_content_type_only_no_metadata(self, tmp_path):
        """Verify only ContentType is set when metadata is empty."""
        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {"ETag": '"e"'}
        store = self._make_store(mock_s3)

        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")

        store.do_upload_from_path(
            "img.png",
            f,
            content_type="image/png",
            content_length=4,
            metadata={},
            object_lock=None,
        )

        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["ContentType"] == "image/png"
        assert call_kwargs["ContentLength"] == 4
        assert "Metadata" not in call_kwargs

    def test_end_to_end_via_upload_from_path(self, tmp_path):
        """Full end-to-end test: upload_from_path -> validate -> do_upload_from_path."""
        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {
            "ETag": '"full-etag"',
            "VersionId": "gen-42",
        }
        store = self._make_store(mock_s3)

        f = tmp_path / "document.json"
        f.write_bytes(b'{"key": "value"}')

        # Call the public method (validates + auto-detects content type)
        resp = store.upload_from_path("data/doc.json", f, metadata={"env": "test"})

        assert resp.key == "data/doc.json"
        assert resp.e_tag == "full-etag"
        assert resp.version_id == "gen-42"
        # Verify put_object was called (not upload_file)
        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Key"] == "data/doc.json"
        assert call_kwargs["ContentType"] == "application/json"


class TestGcpUploadFromPath:
    """Tests for GcpBlobStore.do_upload_from_path using GCS upload_from_filename."""

    def _make_store(self, monkeypatch):
        """Create a GcpBlobStore with mocked GCP auth and storage client."""
        mock_creds = MagicMock()
        monkeypatch.setattr(
            "google.auth.default",
            lambda **kwargs: (mock_creds, "test-project"),
        )
        mock_storage_client = MagicMock()
        monkeypatch.setattr(
            "multicloud.blob.blob_gcp.gcp_blob_store.storage.Client",
            lambda **kwargs: mock_storage_client,
        )

        from multicloud.blob.blob_gcp import GcpBlobStore

        store = GcpBlobStore(bucket="test-bucket")
        return store, mock_storage_client

    def test_calls_upload_from_filename(self, monkeypatch, tmp_path):
        """Verify GCS upload_from_filename is called with correct args."""
        store, mock_client = self._make_store(monkeypatch)
        mock_blob = MagicMock()
        mock_blob.etag = "gcs-etag"
        mock_blob.generation = 12345
        store._bucket_handle.blob.return_value = mock_blob

        f = tmp_path / "test.txt"
        f.write_bytes(b"hello gcs")

        resp = store.do_upload_from_path(
            "docs/test.txt",
            f,
            content_type="text/plain",
            content_length=9,
            metadata={"env": "dev"},
            object_lock=None,
        )

        store._bucket_handle.blob.assert_called_once_with("docs/test.txt")
        mock_blob.upload_from_filename.assert_called_once_with(
            str(f),
            content_type="text/plain",
        )
        assert mock_blob.metadata == {"env": "dev"}
        assert resp.key == "docs/test.txt"
        assert resp.e_tag == "gcs-etag"
        assert resp.version_id == "12345"

    def test_no_metadata_does_not_set_blob_metadata(self, monkeypatch, tmp_path):
        """Verify blob.metadata is not set when metadata dict is empty."""
        store, _ = self._make_store(monkeypatch)
        mock_blob = MagicMock()
        mock_blob.etag = "e"
        mock_blob.generation = 1
        mock_blob.metadata = None  # initial state
        store._bucket_handle.blob.return_value = mock_blob

        f = tmp_path / "no_meta.bin"
        f.write_bytes(b"\x00")

        store.do_upload_from_path(
            "key",
            f,
            content_type="application/octet-stream",
            content_length=1,
            metadata={},
            object_lock=None,
        )

        # metadata should not have been set (empty dict is falsy)
        mock_blob.upload_from_filename.assert_called_once()
        # Check blob.metadata was not changed from None
        assert mock_blob.metadata is None

    def test_default_content_type_when_none(self, monkeypatch, tmp_path):
        """Verify fallback to application/octet-stream when content_type is None."""
        store, _ = self._make_store(monkeypatch)
        mock_blob = MagicMock()
        mock_blob.etag = "e"
        mock_blob.generation = 1
        store._bucket_handle.blob.return_value = mock_blob

        f = tmp_path / "noext"
        f.write_bytes(b"data")

        store.do_upload_from_path(
            "key",
            f,
            content_type=None,
            content_length=4,
            metadata={},
            object_lock=None,
        )

        mock_blob.upload_from_filename.assert_called_once_with(
            str(f),
            content_type="application/octet-stream",
        )

    def test_gcs_error_propagates(self, monkeypatch, tmp_path):
        """Verify GCS API errors propagate raw (client layer translates)."""
        from google.api_core import exceptions as gcs_exceptions

        store, _ = self._make_store(monkeypatch)
        mock_blob = MagicMock()
        mock_blob.upload_from_filename.side_effect = gcs_exceptions.Forbidden("denied")
        store._bucket_handle.blob.return_value = mock_blob

        f = tmp_path / "denied.txt"
        f.write_bytes(b"data")

        with pytest.raises(gcs_exceptions.Forbidden):
            store.do_upload_from_path(
                "key",
                f,
                content_type="text/plain",
                content_length=4,
                metadata={},
                object_lock=None,
            )

    def test_end_to_end_via_upload_from_path(self, monkeypatch, tmp_path):
        """Full end-to-end: upload_from_path -> validate -> do_upload_from_path."""
        store, _ = self._make_store(monkeypatch)
        mock_blob = MagicMock()
        mock_blob.etag = "full-gcs-etag"
        mock_blob.generation = 99
        store._bucket_handle.blob.return_value = mock_blob

        f = tmp_path / "report.csv"
        f.write_bytes(b"a,b,c\n1,2,3\n")

        resp = store.upload_from_path("data/report.csv", f, metadata={"team": "sdk"})

        assert resp.key == "data/report.csv"
        assert resp.e_tag == "full-gcs-etag"
        assert resp.version_id == "99"
        # Verify upload_from_filename was called (not upload_from_string)
        mock_blob.upload_from_filename.assert_called_once()
        call_args = mock_blob.upload_from_filename.call_args
        assert str(f) in call_args.args or str(f) == call_args.args[0]
