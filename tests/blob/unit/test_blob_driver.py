"""Unit tests for the blob driver layer (BaseBlobStore, validators, helpers)."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Optional
from unittest.mock import MagicMock

import pytest
from multicloud.blob.blob_client.driver import (
    BLOB_REGISTRY,
    BaseBlobStore,
    BlobIdentifier,
    BlobInfo,
    BlobMetadata,
    BlobStoreValidator,
    ChecksumAlgorithm,
    CopyResponse,
    DeleteObjectsResponse,
    DownloadResponse,
    ListBlobsPageResponse,
    ObjectLockConfiguration,
    ObjectLockInfo,
    ObjectRetentionConfig,
    PresignedOperation,
    PresignedUrlResponse,
    RetentionMode,
    StreamingDownloadResponse,
    UploadResponse,
    _coerce_to_bytes,
    load_from_path,
)
from multicloud.common.exceptions import (
    InvalidArgumentException,
    SubstrateSdkException,
    UnsupportedOperationException,
)


class _FakeBlobStore(BaseBlobStore):
    """Minimal concrete implementation for testing BaseBlobStore."""

    provider_id = "fake"

    def __init__(self, bucket: str, region: Optional[str] = None, **kwargs):
        super().__init__(bucket=bucket, region=region)
        self.upload_calls = []
        self.download_calls = []
        self.delete_calls = []
        self.get_tags_calls = []
        self.set_tags_calls = []
        self.get_object_lock_calls = []
        self.update_legal_hold_calls = []
        self.update_object_retention_calls = []
        self._tags: dict[str, Mapping[str, str]] = {}

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
        tags: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration] = None,
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        checksum_value: Optional[str] = None,
        checksum_algorithm: Optional[ChecksumAlgorithm] = None,
    ) -> UploadResponse:
        self.upload_calls.append(
            {
                "key": key,
                "payload": payload,
                "content_type": content_type,
                "content_length": content_length,
                "metadata": metadata,
                "tags": tags,
                "object_lock": object_lock,
                "create_if_absent": create_if_absent,
                "if_match": if_match,
                "if_version_match": if_version_match,
                "checksum_value": checksum_value,
                "checksum_algorithm": checksum_algorithm,
            }
        )
        return UploadResponse(key=key, e_tag="fake-etag", version_id="v1")

    def do_upload_from_path(
        self,
        key,
        path,
        *,
        content_type,
        content_length,
        metadata,
        object_lock=None,
        create_if_absent=False,
        if_match=None,
        if_version_match=None,
        kms_key_id=None,
        use_kms_managed_key=False,
    ):
        payload = path.read_bytes()
        return self.do_upload(
            key,
            payload,
            content_type=content_type,
            content_length=content_length,
            metadata=metadata,
            tags={},
            object_lock=object_lock,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
            kms_key_id=kms_key_id,
            use_kms_managed_key=use_kms_managed_key,
        )

    def do_download(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
        if_match: Optional[str] = None,
    ) -> DownloadResponse:
        self.download_calls.append(
            {"key": key, "version_id": version_id, "start": start, "end": end}
        )
        meta = BlobMetadata(key=key, content_length=5, content_type="text/plain")
        return DownloadResponse(key=key, content=b"hello", metadata=meta)

    def do_download_to_path(
        self,
        key: str,
        destination,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> DownloadResponse:
        destination.write_bytes(b"hello")
        meta = BlobMetadata(key=key, content_length=5, content_type="text/plain")
        return DownloadResponse(key=key, content=b"", metadata=meta)

    def do_download_to_file(
        self,
        key: str,
        file_obj,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> DownloadResponse:
        file_obj.write(b"hello")
        meta = BlobMetadata(key=key, content_length=5, content_type="text/plain")
        return DownloadResponse(key=key, content=b"", metadata=meta)

    def do_download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> StreamingDownloadResponse:
        meta = BlobMetadata(key=key, content_length=5, content_type="text/plain")
        return StreamingDownloadResponse(
            key=key,
            metadata=meta,
            stream=iter([b"hello"]),
        )

    def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        self.delete_calls.append({"key": key, "version_id": version_id})

    def do_delete_objects(self, objects):
        return DeleteObjectsResponse(errors=[])

    def do_copy(self, src_key, *, dest_bucket, dest_key, src_version_id):
        return CopyResponse(key=dest_key, e_tag="fake-etag")

    def do_copy_from(self, src_bucket, src_key, *, dest_key, src_version_id):
        return CopyResponse(key=dest_key, e_tag="fake-etag")

    def do_get_object_lock(self, key, *, version_id=None):
        self.get_object_lock_calls.append({"key": key, "version_id": version_id})
        return ObjectLockInfo()

    def do_update_legal_hold(self, key, enabled, *, version_id=None):
        self.update_legal_hold_calls.append(
            {"key": key, "enabled": enabled, "version_id": version_id}
        )

    def do_get_tags(self, key: str) -> Mapping[str, str]:
        self.get_tags_calls.append(key)
        return self._tags.get(key, {})

    def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        self.set_tags_calls.append({"key": key, "tags": tags})
        self._tags[key] = tags

    def do_update_object_retention(self, key, retention, *, version_id):
        self.update_object_retention_calls.append(
            {"key": key, "retention": retention, "version_id": version_id}
        )

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
        return PresignedUrlResponse(
            url=f"https://fake.example.com/{key}",
            http_method="PUT" if operation == PresignedOperation.UPLOAD else "GET",
            required_headers={},
        )

    def do_list_page(self, *, prefix, delimiter, max_results, pagination_token):
        from multicloud.blob.blob_client.driver import ListBlobsPageResponse

        return ListBlobsPageResponse()

    def do_does_object_exist(self, key, *, version_id):
        return True

    def do_get_metadata(self, key, *, version_id=None):
        return BlobMetadata(key=key)

    def do_does_bucket_exist(self):
        return True

    def do_initiate_multipart_upload(
        self, key, *, metadata, content_type, tags, object_lock=None
    ):
        from multicloud.blob.blob_client.driver import MultipartUpload

        return MultipartUpload(
            bucket="fake",
            key=key,
            upload_id="fake-id",
            metadata=metadata,
            content_type=content_type,
            tags=tags,
            object_lock=object_lock,
        )

    def do_upload_part(self, mpu, part):
        from multicloud.blob.blob_client.driver import UploadPartResponse

        return UploadPartResponse(
            part_number=part.part_number,
            e_tag="fake-etag",
            size_in_bytes=part.content_length,
        )

    def do_complete_multipart_upload(self, mpu, parts):
        from multicloud.blob.blob_client.driver import MultipartUploadResponse

        return MultipartUploadResponse(e_tag="fake-complete-etag")

    def do_list_parts(self, mpu):
        return []

    def do_abort_multipart_upload(self, mpu):
        pass


class TestBlobStoreValidator:
    """Tests for input validation."""

    def setup_method(self):
        self.validator = BlobStoreValidator()

    def test_validate_upload_empty_key_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_upload("")

    def test_validate_upload_valid_key_passes(self):
        self.validator.validate_upload("some/key.txt")  # no exception

    def test_validate_download_empty_key_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_download("")

    def test_validate_download_valid_key_passes(self):
        self.validator.validate_download("path/to/file")  # no exception

    def test_validate_delete_key_empty_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_delete_key("")

    def test_validate_delete_key_valid_passes(self):
        self.validator.validate_delete_key("obj")  # no exception

    def test_validate_delete_empty_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_delete("")

    def test_validate_delete_valid_passes(self):
        self.validator.validate_delete("obj")  # no exception

    def test_validate_delete_key_is_alias_for_validate_delete(self):
        # Backward-compat alias must behave identically to validate_delete.
        with pytest.raises(InvalidArgumentException):
            self.validator.validate_delete_key("")

    def test_validate_get_tags_empty_key_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_get_tags("")

    def test_validate_get_tags_valid_key_passes(self):
        self.validator.validate_get_tags("some/key.txt")  # no exception

    def test_validate_set_tags_empty_key_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_set_tags("", {"env": "test"})

    def test_validate_set_tags_empty_tags_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Map of tags cannot be null or empty"
        ):
            self.validator.validate_set_tags("key.txt", {})

    def test_validate_set_tags_none_tags_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Map of tags cannot be null or empty"
        ):
            self.validator.validate_set_tags("key.txt", None)  # type: ignore[arg-type]

    def test_validate_set_tags_valid_passes(self):
        self.validator.validate_set_tags("key.txt", {"env": "prod"})  # no exception

    def test_validate_object_lock_none_passes(self):
        self.validator.validate_object_lock(None)  # no exception

    def test_validate_object_lock_defaults_pass(self):
        # An all-default config (no retention, no hold) is a valid no-op.
        self.validator.validate_object_lock(ObjectLockConfiguration())

    def test_validate_object_lock_legal_hold_only_passes(self):
        self.validator.validate_object_lock(ObjectLockConfiguration(legal_hold=True))

    def test_validate_object_lock_mode_without_date_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="retain_until_date is required"
        ):
            self.validator.validate_object_lock(
                ObjectLockConfiguration(mode=RetentionMode.GOVERNANCE)
            )

    def test_validate_object_lock_date_without_mode_raises(self):
        future = datetime.now(timezone.utc) + timedelta(days=1)
        with pytest.raises(InvalidArgumentException, match="mode is required"):
            self.validator.validate_object_lock(
                ObjectLockConfiguration(retain_until_date=future)
            )

    def test_validate_object_lock_naive_date_raises(self):
        naive = datetime.now() + timedelta(days=1)
        with pytest.raises(InvalidArgumentException, match="timezone-aware"):
            self.validator.validate_object_lock(
                ObjectLockConfiguration(
                    mode=RetentionMode.COMPLIANCE, retain_until_date=naive
                )
            )

    def test_validate_object_lock_past_date_passes(self):
        # No client-side future-date check — a past date defers to the server.
        past = datetime.now(timezone.utc) - timedelta(days=1)
        self.validator.validate_object_lock(
            ObjectLockConfiguration(
                mode=RetentionMode.COMPLIANCE, retain_until_date=past
            )
        )

    def test_validate_object_lock_valid_passes(self):
        future = datetime.now(timezone.utc) + timedelta(days=1)
        self.validator.validate_object_lock(
            ObjectLockConfiguration(
                mode=RetentionMode.GOVERNANCE, retain_until_date=future
            )
        )

    def test_validate_update_object_retention_valid_passes(self):
        config = ObjectRetentionConfig(
            mode=RetentionMode.COMPLIANCE,
            retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        self.validator.validate_update_object_retention(
            "key.txt", config
        )  # no exception

    def test_validate_update_object_retention_empty_key_raises(self):
        config = ObjectRetentionConfig(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_update_object_retention("", config)

    def test_validate_update_object_retention_none_config_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Retention config cannot be null"
        ):
            self.validator.validate_update_object_retention("key.txt", None)

    def test_validate_update_object_retention_none_mode_passes(self):
        # mode is an optional sentinel (None = preserve current mode), so it is
        # not rejected by stateless validation.
        config = ObjectRetentionConfig(
            mode=None,
            retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        self.validator.validate_update_object_retention(
            "key.txt", config
        )  # no exception

    def test_validate_update_object_retention_missing_retain_until_raises(self):
        config = ObjectRetentionConfig(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=None,  # type: ignore[arg-type]
        )
        with pytest.raises(
            InvalidArgumentException,
            match="Retention retain_until_date cannot be null",
        ):
            self.validator.validate_update_object_retention("key.txt", config)

    def test_validate_update_object_retention_naive_retain_until_raises(self):
        # A naive (timezone-less) retain_until_date cannot be compared against
        # the timezone-aware retain-until providers read back, so it is rejected
        # rather than silently assuming a zone.
        config = ObjectRetentionConfig(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=datetime(2030, 1, 1),
        )
        with pytest.raises(
            InvalidArgumentException,
            match="must be timezone-aware",
        ):
            self.validator.validate_update_object_retention("key.txt", config)


class TestBaseBlobStore:
    """Tests for the BaseBlobStore abstract base + validation wiring."""

    def test_init_empty_bucket_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Bucket name cannot be null or empty"
        ):
            _FakeBlobStore(bucket="")

    def test_init_stores_bucket_and_region(self):
        store = _FakeBlobStore(bucket="my-bucket", region="us-west-2")
        assert store.bucket == "my-bucket"
        assert store.region == "us-west-2"

    def test_init_region_defaults_to_none(self):
        store = _FakeBlobStore(bucket="b")
        assert store.region is None

    def test_upload_validates_then_delegates(self):
        store = _FakeBlobStore(bucket="b")
        resp = store.upload("k", b"data")
        assert resp.key == "k"
        assert resp.e_tag == "fake-etag"
        assert len(store.upload_calls) == 1

    def test_upload_empty_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException):
            store.upload("", b"data")

    def test_upload_coerces_file_like_to_bytes(self):
        store = _FakeBlobStore(bucket="b")
        payload = io.BytesIO(b"binary content")
        store.upload("key.bin", payload)
        assert store.upload_calls[0]["payload"] == b"binary content"

    def test_upload_with_metadata(self):
        store = _FakeBlobStore(bucket="b")
        store.upload("key", b"x", metadata={"env": "prod"})
        assert store.upload_calls[0]["metadata"] == {"env": "prod"}

    def test_upload_none_metadata_becomes_empty_dict(self):
        store = _FakeBlobStore(bucket="b")
        store.upload("key", b"x")
        assert store.upload_calls[0]["metadata"] == {}

    def test_upload_with_tags(self):
        store = _FakeBlobStore(bucket="b")
        store.upload("key", b"x", tags={"env": "prod", "owner": "team-sdk"})
        assert store.upload_calls[0]["tags"] == {"env": "prod", "owner": "team-sdk"}

    def test_upload_none_tags_becomes_empty_dict(self):
        store = _FakeBlobStore(bucket="b")
        store.upload("key", b"x")
        assert store.upload_calls[0]["tags"] == {}

    def test_download_validates_then_delegates(self):
        store = _FakeBlobStore(bucket="b")
        resp = store.download("k")
        assert resp.key == "k"
        assert resp.content == b"hello"
        assert len(store.download_calls) == 1

    def test_download_empty_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException):
            store.download("")

    def test_download_with_range_params(self):
        store = _FakeBlobStore(bucket="b")
        store.download("k", start=10, end=20, version_id="v2")
        call = store.download_calls[0]
        assert call["start"] == 10
        assert call["end"] == 20
        assert call["version_id"] == "v2"

    def test_download_negative_start_rejected_before_do_download(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read boundaries cannot be negative",
        ):
            store.download("k", start=-1, end=10)
        assert store.download_calls == []  # do_download never reached

    def test_download_end_before_start_rejected_before_do_download(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read start cannot be larger than end",
        ):
            store.download("k", start=100, end=10)
        assert store.download_calls == []

    def test_download_to_path_invalid_range_rejected(self, tmp_path):
        store = _FakeBlobStore(bucket="b")
        dest = tmp_path / "out.bin"
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read start cannot be larger than end",
        ):
            store.download_to_path("k", dest, start=50, end=10)
        assert not dest.exists()  # provider never wrote anything

    def test_download_to_file_invalid_range_rejected(self):
        store = _FakeBlobStore(bucket="b")
        buf = io.BytesIO()
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read boundaries cannot be negative",
        ):
            store.download_to_file("k", buf, start=-5)
        assert buf.getvalue() == b""

    def test_download_stream_invalid_range_rejected(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read boundaries cannot be negative",
        ):
            store.download_stream("k", start=0, end=-1)

    def test_upload_negative_content_length_rejected_before_do_upload(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="content_length must be a non-negative"
        ):
            store.upload("k", b"data", content_length=-1)
        assert store.upload_calls == []

    def test_upload_oversize_key_rejected_before_do_upload(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot exceed 1024 bytes"
        ):
            store.upload("a" * 1025, b"data")
        assert store.upload_calls == []

    def test_upload_forwards_checksum_to_do_upload(self):
        store = _FakeBlobStore(bucket="b")
        value = base64.b64encode(hashlib.md5(b"data").digest()).decode()
        store.upload(
            "k",
            b"data",
            checksum_value=value,
            checksum_algorithm=ChecksumAlgorithm.MD5,
        )
        assert store.upload_calls[0]["checksum_value"] == value
        assert store.upload_calls[0]["checksum_algorithm"] is ChecksumAlgorithm.MD5

    def test_upload_without_checksum_forwards_none(self):
        store = _FakeBlobStore(bucket="b")
        store.upload("k", b"data")
        assert store.upload_calls[0]["checksum_value"] is None
        assert store.upload_calls[0]["checksum_algorithm"] is None

    @pytest.mark.parametrize("name", ["CRC32C", "crc32c"])
    def test_upload_normalizes_checksum_algorithm_string(self, name):
        store = _FakeBlobStore(bucket="b")
        store.upload(
            "k",
            b"data",
            checksum_value=base64.b64encode(b"\x00\x01\x02\x03").decode(),
            checksum_algorithm=name,
        )
        assert store.upload_calls[0]["checksum_algorithm"] is ChecksumAlgorithm.CRC32C

    def test_upload_unsupported_checksum_algorithm_rejected_before_do_upload(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException,
            match="checksum_algorithm must be one of CRC32C, MD5. value='SHA256'",
        ):
            store.upload(
                "k", b"data", checksum_value="AAAA", checksum_algorithm="SHA256"
            )
        assert store.upload_calls == []

    def test_upload_checksum_value_without_algorithm_rejected_before_do_upload(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="checksum_value requires checksum_algorithm"
        ):
            store.upload("k", b"data", checksum_value="AAAAAA==")
        assert store.upload_calls == []

    def test_generate_presigned_download_rejects_upload_only_params(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="metadata is only valid for UPLOAD"
        ):
            store.generate_presigned_url(
                "k", PresignedOperation.DOWNLOAD, metadata={"env": "prod"}
            )

    def test_delete_validates_then_delegates(self):
        store = _FakeBlobStore(bucket="b")
        store.delete("k")
        assert store.delete_calls == [{"key": "k", "version_id": None}]

    def test_delete_empty_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException):
            store.delete("")

    def test_get_tags_validates_then_delegates(self):
        store = _FakeBlobStore(bucket="b")
        store._tags["k"] = {"env": "prod", "team": "sdk"}
        result = store.get_tags("k")
        assert result == {"env": "prod", "team": "sdk"}
        assert store.get_tags_calls == ["k"]

    def test_get_tags_empty_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.get_tags("")

    def test_get_tags_returns_empty_dict_when_no_tags(self):
        store = _FakeBlobStore(bucket="b")
        result = store.get_tags("k")
        assert result == {}

    def test_set_tags_validates_then_delegates(self):
        store = _FakeBlobStore(bucket="b")
        store.set_tags("k", {"env": "test"})
        assert store.set_tags_calls == [{"key": "k", "tags": {"env": "test"}}]
        assert store._tags["k"] == {"env": "test"}

    def test_set_tags_empty_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.set_tags("", {"env": "test"})

    def test_set_tags_empty_tags_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="Map of tags cannot be null or empty"
        ):
            store.set_tags("k", {})

    def test_update_object_retention_validates_then_delegates(self):
        store = _FakeBlobStore(bucket="b")
        until = datetime(2030, 1, 1, tzinfo=timezone.utc)
        config = ObjectRetentionConfig(
            mode=RetentionMode.COMPLIANCE, retain_until_date=until
        )
        store.update_object_retention("k", config)
        assert store.update_object_retention_calls == [
            {"key": "k", "retention": config, "version_id": None}
        ]

    def test_update_object_retention_forwards_version_id(self):
        store = _FakeBlobStore(bucket="b")
        config = ObjectRetentionConfig(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        store.update_object_retention("k", config, version_id="v9")
        assert store.update_object_retention_calls[0]["version_id"] == "v9"

    def test_update_object_retention_empty_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        config = ObjectRetentionConfig(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.update_object_retention("", config)
        assert store.update_object_retention_calls == []

    def test_update_object_retention_none_config_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="Retention config cannot be null"
        ):
            store.update_object_retention("k", None)
        assert store.update_object_retention_calls == []

    def test_close_is_noop_by_default(self):
        store = _FakeBlobStore(bucket="b")
        store.close()  # should not raise

    def test_context_manager_protocol(self):
        with _FakeBlobStore(bucket="b") as store:
            assert store.bucket == "b"


class TestCoerceToBytes:
    """Tests for the _coerce_to_bytes helper."""

    def test_bytes_passthrough(self):
        assert _coerce_to_bytes(b"hello") == b"hello"

    def test_bytearray_converted(self):
        assert _coerce_to_bytes(bytearray(b"hi")) == b"hi"

    def test_memoryview_converted(self):
        assert _coerce_to_bytes(memoryview(b"mv")) == b"mv"

    def test_binary_io_read(self):
        bio = io.BytesIO(b"stream data")
        assert _coerce_to_bytes(bio) == b"stream data"

    def test_text_stream_raises(self):
        sio = io.StringIO("text data")
        with pytest.raises(InvalidArgumentException, match="binary"):
            _coerce_to_bytes(sio)

    def test_unsupported_type_raises(self):
        with pytest.raises(InvalidArgumentException, match="Unsupported"):
            _coerce_to_bytes(12345)  # type: ignore[arg-type]


class TestLoadFromPath:
    """Tests for the load_from_path convenience helper."""

    def test_reads_binary_file(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"file content")
            f.flush()
            path = f.name
        try:
            assert load_from_path(path) == b"file content"
        finally:
            os.unlink(path)

    def test_reads_path_object(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"pathlib")
            f.flush()
            path = Path(f.name)
        try:
            assert load_from_path(path) == b"pathlib"
        finally:
            os.unlink(path)


class TestDataclasses:
    """Tests for frozen dataclasses used in the driver layer."""

    def test_upload_response_fields(self):
        r = UploadResponse(key="a/b.txt", e_tag="etag123", version_id="v1")
        assert r.key == "a/b.txt"
        assert r.e_tag == "etag123"
        assert r.version_id == "v1"

    def test_upload_response_defaults(self):
        r = UploadResponse(key="k")
        assert r.e_tag is None
        assert r.version_id is None
        assert r.checksum_algorithm is None
        assert r.checksum_value is None

    def test_upload_response_frozen(self):
        r = UploadResponse(key="k")
        with pytest.raises(AttributeError):
            r.key = "other"  # type: ignore[misc]

    def test_blob_metadata_fields(self):
        m = BlobMetadata(
            key="k",
            content_length=100,
            content_type="text/plain",
            e_tag="etag",
            version_id="v2",
            metadata={"foo": "bar"},
        )
        assert m.key == "k"
        assert m.content_length == 100
        assert m.metadata == {"foo": "bar"}

    def test_blob_metadata_defaults(self):
        m = BlobMetadata(key="k")
        assert m.content_length is None
        assert m.content_type is None
        assert m.metadata == {}

    def test_download_response_fields(self):
        meta = BlobMetadata(key="k")
        r = DownloadResponse(key="k", content=b"data", metadata=meta)
        assert r.content == b"data"
        assert r.metadata is meta

    def test_blob_info_fields(self):
        from datetime import datetime

        dt = datetime(2024, 1, 15, 12, 0, 0)
        info = BlobInfo(
            key="file.txt",
            size=1024,
            e_tag="etag",
            last_modified=dt,
            storage_class="STANDARD",
            version_id="v1",
            is_latest=True,
        )
        assert info.key == "file.txt"
        assert info.size == 1024
        assert info.e_tag == "etag"
        assert info.last_modified == dt
        assert info.storage_class == "STANDARD"
        assert info.version_id == "v1"
        assert info.is_latest is True

    def test_blob_info_defaults(self):
        info = BlobInfo(key="k")
        assert info.size is None
        assert info.e_tag is None
        assert info.last_modified is None
        assert info.storage_class is None
        assert info.version_id is None
        assert info.is_latest is None

    def test_blob_info_frozen(self):
        info = BlobInfo(key="k")
        with pytest.raises(AttributeError):
            info.key = "other"  # type: ignore[misc]

    def test_list_blobs_page_response_fields(self):
        blob = BlobInfo(key="a.txt", size=10)
        r = ListBlobsPageResponse(
            blobs=[blob],
            common_prefixes=["dir/"],
            is_truncated=True,
            next_pagination_token="tok",
        )
        assert len(r.blobs) == 1
        assert r.blobs[0].key == "a.txt"
        assert r.common_prefixes == ["dir/"]
        assert r.is_truncated is True
        assert r.next_pagination_token == "tok"

    def test_list_blobs_page_response_defaults(self):
        r = ListBlobsPageResponse()
        assert r.blobs == []
        assert r.common_prefixes == []
        assert r.is_truncated is False
        assert r.next_pagination_token is None

    def test_list_blobs_page_response_frozen(self):
        r = ListBlobsPageResponse()
        with pytest.raises(AttributeError):
            r.is_truncated = True  # type: ignore[misc]


class TestBlobStoreValidatorListingAPIs:
    """Tests for validator methods added for listing/metadata APIs."""

    def setup_method(self):
        self.validator = BlobStoreValidator()

    def test_validate_key_empty_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_key("")

    def test_validate_key_valid_passes(self):
        self.validator.validate_key("some/key.txt")  # no exception

    def test_validate_list_page_zero_raises(self):
        with pytest.raises(InvalidArgumentException, match="max_results"):
            self.validator.validate_list_page(0)

    def test_validate_list_page_negative_raises(self):
        with pytest.raises(InvalidArgumentException, match="max_results"):
            self.validator.validate_list_page(-1)

    def test_validate_list_page_over_1000_allowed(self):
        # No cloud-agnostic upper bound: large page sizes are accepted here
        # and left for each provider to cap natively, since providers differ
        # in whether their per-page limit is a hard cap or a recommendation.
        self.validator.validate_list_page(1001)  # no exception
        self.validator.validate_list_page(10000)  # no exception

    def test_validate_list_page_valid_passes(self):
        self.validator.validate_list_page(1)  # no exception
        self.validator.validate_list_page(500)  # no exception
        self.validator.validate_list_page(1000)  # no exception


class TestBaseBlobStoreListingAPIs:
    """Tests that BaseBlobStore validates and delegates for listing APIs."""

    def test_list_page_validates_max_results(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException, match="max_results"):
            store.list_page(max_results=0)

    def test_list_page_delegates_with_correct_args(self):
        """Verify list_page passes all parameters through to do_list_page."""
        store = _FakeBlobStore(bucket="b")
        store.list_page_calls = []

        original_do = store.do_list_page

        def tracking_do(*, prefix, delimiter, max_results, pagination_token):
            store.list_page_calls.append(
                {
                    "prefix": prefix,
                    "delimiter": delimiter,
                    "max_results": max_results,
                    "pagination_token": pagination_token,
                }
            )
            return original_do(
                prefix=prefix,
                delimiter=delimiter,
                max_results=max_results,
                pagination_token=pagination_token,
            )

        store.do_list_page = tracking_do
        store.list_page(
            prefix="data/", delimiter="/", max_results=50, pagination_token="tok"
        )
        assert len(store.list_page_calls) == 1
        assert store.list_page_calls[0] == {
            "prefix": "data/",
            "delimiter": "/",
            "max_results": 50,
            "pagination_token": "tok",
        }

    def test_does_object_exist_validates_key(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.does_object_exist("")

    def test_does_object_exist_delegates_with_version_id(self):
        """Verify does_object_exist forwards version_id to do_does_object_exist."""
        store = _FakeBlobStore(bucket="b")
        store.exist_calls = []

        def tracking_exist(key, *, version_id):
            store.exist_calls.append({"key": key, "version_id": version_id})
            return key == "found.txt"

        store.do_does_object_exist = tracking_exist
        assert store.does_object_exist("found.txt", version_id="v2") is True
        assert store.does_object_exist("missing.txt") is False
        assert store.exist_calls == [
            {"key": "found.txt", "version_id": "v2"},
            {"key": "missing.txt", "version_id": None},
        ]

    def test_does_bucket_exist_delegates(self):
        """Verify does_bucket_exist calls do_does_bucket_exist and returns its result."""
        store = _FakeBlobStore(bucket="b")
        store.do_does_bucket_exist = lambda: False
        assert store.does_bucket_exist() is False

        store.do_does_bucket_exist = lambda: True
        assert store.does_bucket_exist() is True

    def test_list_blob_versions_validates_key(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.list_blob_versions("")

    def test_list_blob_versions_default_raises_unsupported(self):
        """Test that the default do_list_blob_versions raises UnsupportedOperationException."""

        # Use a store that doesn't override do_list_blob_versions
        class _MinimalStore(_FakeBlobStore):
            def do_list_blob_versions(self, key):
                # Call super to test the default implementation
                return super(_FakeBlobStore, self).do_list_blob_versions(key)

        store = _MinimalStore(bucket="b")
        with pytest.raises(UnsupportedOperationException):
            list(store.list_blob_versions("key"))


class _NoMultipartStore(BaseBlobStore):
    """A BaseBlobStore subclass that implements every required hook EXCEPT
    the 5 multipart hooks. Verifies that B1 (the abstractmethod fix) holds:
    such a subclass must remain instantiable, and calling any multipart
    method must raise UnsupportedOperationException rather than be blocked
    at class-construction time."""

    def translate_error(self, exc):
        return SubstrateSdkException(str(exc))

    def do_upload(
        self,
        key,
        payload,
        *,
        content_type,
        content_length,
        metadata,
        tags,
        object_lock=None,
        create_if_absent=False,
        if_match=None,
        if_version_match=None,
        kms_key_id=None,
        use_kms_managed_key=False,
        checksum_value=None,
        checksum_algorithm=None,
    ):
        return UploadResponse(key=key)

    def do_upload_from_path(
        self,
        key,
        path,
        *,
        content_type,
        content_length,
        metadata,
        object_lock=None,
        create_if_absent=False,
        if_match=None,
        if_version_match=None,
        kms_key_id=None,
        use_kms_managed_key=False,
    ):
        return UploadResponse(key=key)

    def do_download(self, key, *, version_id, start, end, if_match=None):
        return DownloadResponse(key=key, content=b"", metadata=BlobMetadata(key=key))

    def do_download_to_path(self, key, destination, *, version_id, start, end):
        return DownloadResponse(key=key, content=b"", metadata=BlobMetadata(key=key))

    def do_download_to_file(self, key, file_obj, *, version_id, start, end):
        return DownloadResponse(key=key, content=b"", metadata=BlobMetadata(key=key))

    def do_download_stream(self, key, *, version_id, start, end):
        return StreamingDownloadResponse(
            key=key, metadata=BlobMetadata(key=key), stream=iter([])
        )

    def do_delete(self, key, *, version_id=None):
        pass

    def do_delete_objects(self, objects):
        return DeleteObjectsResponse(errors=[])

    def do_copy(self, src_key, *, dest_bucket, dest_key, src_version_id=None):
        return CopyResponse(key=dest_key, e_tag="fake-etag")

    def do_copy_from(self, src_bucket, src_key, *, dest_key, src_version_id=None):
        return CopyResponse(key=dest_key, e_tag="fake-etag")

    def do_get_object_lock(self, key, *, version_id=None):
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
        return PresignedUrlResponse(url="https://fake/", http_method="GET")

    def do_list_page(self, *, prefix, delimiter, max_results, pagination_token):
        from multicloud.blob.blob_client.driver import ListBlobsPageResponse

        return ListBlobsPageResponse()

    def do_does_object_exist(self, key, *, version_id):
        return False

    def do_get_metadata(self, key, *, version_id=None):
        return BlobMetadata(key=key)

    def do_does_bucket_exist(self):
        return False


class TestB1NoMultipartSubclass:
    """B1 regression: see Recommendation #5 (separate blocker) in PR #38 review."""

    def test_subclass_without_multipart_is_instantiable(self):
        """A subclass that does not implement do_*_multipart_* must construct
        successfully; otherwise future providers / in-memory test stores are
        un-instantiable."""
        # If the 5 multipart hooks were @abstractmethod with no default body,
        # this line would raise TypeError("Can't instantiate abstract class ...").
        store = _NoMultipartStore(bucket="b")
        assert store.bucket == "b"

    def test_initiate_multipart_default_raises_unsupported(self):
        store = _NoMultipartStore(bucket="b")
        with pytest.raises(UnsupportedOperationException):
            store.initiate_multipart_upload("k")

    def test_upload_part_default_raises_unsupported(self):
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        store = _NoMultipartStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="x")
        with pytest.raises(UnsupportedOperationException):
            store.upload_part(mpu, MultipartPart(part_number=1, content=b"x"))

    def test_complete_multipart_default_raises_unsupported(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        store = _NoMultipartStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="x")
        with pytest.raises(UnsupportedOperationException):
            store.complete_multipart_upload(mpu, [])

    def test_list_parts_default_raises_unsupported(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        store = _NoMultipartStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="x")
        with pytest.raises(UnsupportedOperationException):
            store.list_parts(mpu)

    def test_abort_multipart_default_raises_unsupported(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        store = _NoMultipartStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="x")
        with pytest.raises(UnsupportedOperationException):
            store.abort_multipart_upload(mpu)


class TestPresignedUrlValidation:
    """Tests for presigned URL validation on BaseBlobStore."""

    def setup_method(self):
        self.validator = BlobStoreValidator()
        self.store = _FakeBlobStore(bucket="b")

    def test_validate_empty_key_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_generate_presigned_url(
                "", PresignedOperation.UPLOAD, 3600
            )

    def test_validate_none_operation_raises(self):
        with pytest.raises(
            InvalidArgumentException,
            match="presigned url operation type must be non-null",
        ):
            self.validator.validate_generate_presigned_url("key", None, 3600)

    def test_validate_invalid_operation_type_raises(self):
        with pytest.raises(
            InvalidArgumentException, match="must be a PresignedOperation"
        ):
            self.validator.validate_generate_presigned_url("key", "UPLOAD", 3600)

    def test_validate_zero_duration_raises(self):
        with pytest.raises(InvalidArgumentException, match="non-zero positive value"):
            self.validator.validate_generate_presigned_url(
                "key", PresignedOperation.UPLOAD, 0
            )

    def test_validate_negative_duration_raises(self):
        with pytest.raises(InvalidArgumentException, match="non-zero positive value"):
            self.validator.validate_generate_presigned_url(
                "key", PresignedOperation.DOWNLOAD, -100
            )

    def test_validate_exceeds_max_duration_raises(self):
        with pytest.raises(InvalidArgumentException, match="cannot exceed 604800"):
            self.validator.validate_generate_presigned_url(
                "key", PresignedOperation.UPLOAD, 604801
            )

    def test_validate_max_duration_passes(self):
        # Exactly 7 days should be fine
        self.validator.validate_generate_presigned_url(
            "key", PresignedOperation.UPLOAD, 604800
        )

    def test_validate_valid_upload_passes(self):
        self.validator.validate_generate_presigned_url(
            "my/key.txt", PresignedOperation.UPLOAD, 3600
        )

    def test_validate_valid_download_passes(self):
        self.validator.validate_generate_presigned_url(
            "my/key.txt", PresignedOperation.DOWNLOAD, 7200
        )


class TestPresignedUrlTemplateMethod:
    """Tests for the generate_presigned_url template method on BaseBlobStore."""

    def test_upload_delegates_to_do_generate(self):
        store = _FakeBlobStore(bucket="b")
        resp = store.generate_presigned_url(
            "file.txt", PresignedOperation.UPLOAD, duration_seconds=600
        )
        assert resp.url == "https://fake.example.com/file.txt"
        assert resp.http_method == "PUT"
        assert resp.required_headers == {}

    def test_download_delegates_to_do_generate(self):
        store = _FakeBlobStore(bucket="b")
        resp = store.generate_presigned_url(
            "file.txt", PresignedOperation.DOWNLOAD, duration_seconds=600
        )
        assert resp.url == "https://fake.example.com/file.txt"
        assert resp.http_method == "GET"
        assert resp.required_headers == {}

    def test_with_all_optional_params(self):
        store = _FakeBlobStore(bucket="b")
        resp = store.generate_presigned_url(
            "file.txt",
            PresignedOperation.UPLOAD,
            duration_seconds=3600,
            content_type="text/plain",
            metadata={"author": "test"},
            tags={"env": "dev"},
            kms_key_id="arn:aws:kms:us-east-1:123:key/abc",
            content_disposition="attachment; filename=test.txt",
        )
        assert resp.url == "https://fake.example.com/file.txt"
        assert resp.http_method == "PUT"

    def test_empty_key_raises_before_do(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.generate_presigned_url(
                "", PresignedOperation.UPLOAD, duration_seconds=3600
            )

    def test_none_metadata_tags_default_to_empty(self):
        """Ensure None metadata/tags are normalized to empty dicts."""
        store = _FakeBlobStore(bucket="b")
        # Should not raise — None gets converted to {}
        resp = store.generate_presigned_url(
            "key",
            PresignedOperation.UPLOAD,
            duration_seconds=3600,
            metadata=None,
            tags=None,
        )
        assert resp.url == "https://fake.example.com/key"


class TestMultipartDataclasses:
    """Tests for multipart upload value objects."""

    def test_multipart_part_bytes_auto_content_length(self):
        from multicloud.blob.blob_client.driver import MultipartPart

        part = MultipartPart(part_number=1, content=b"x" * 500)
        assert part.content_length == 500

    def test_multipart_part_explicit_content_length_honored(self):
        from multicloud.blob.blob_client.driver import MultipartPart

        part = MultipartPart(part_number=1, content=b"abc", content_length=3)
        assert part.content_length == 3

    def test_multipart_part_binary_io_requires_content_length(self):
        from multicloud.blob.blob_client.driver import MultipartPart

        stream = io.BytesIO(b"stream data")
        with pytest.raises(ValueError, match="content_length is required"):
            MultipartPart(part_number=1, content=stream)

    def test_multipart_part_binary_io_with_content_length(self):
        from multicloud.blob.blob_client.driver import MultipartPart

        stream = io.BytesIO(b"stream data")
        part = MultipartPart(part_number=1, content=stream, content_length=11)
        assert part.content_length == 11
        assert part.content.read() == b"stream data"

    def test_multipart_upload_frozen(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mpu = MultipartUpload(bucket="b", key="k", upload_id="id-1")
        with pytest.raises(AttributeError):
            mpu.upload_id = "other"  # type: ignore[misc]

    def test_multipart_upload_metadata_defaults_empty(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mpu = MultipartUpload(bucket="b", key="k", upload_id="id-1")
        assert mpu.metadata == {}
        assert mpu.content_type is None

    def test_upload_part_response_fields(self):
        from multicloud.blob.blob_client.driver import UploadPartResponse

        r = UploadPartResponse(part_number=3, e_tag="abc123", size_in_bytes=1024)
        assert r.part_number == 3
        assert r.e_tag == "abc123"
        assert r.size_in_bytes == 1024

    def test_multipart_upload_response_defaults(self):
        from multicloud.blob.blob_client.driver import MultipartUploadResponse

        r = MultipartUploadResponse()
        assert r.e_tag is None


class TestMultipartValidator:
    """Tests for multipart-specific validation."""

    def setup_method(self):
        self.validator = BlobStoreValidator()

    def test_validate_multipart_upload_bucket_mismatch_raises(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mpu = MultipartUpload(bucket="other-bucket", key="k", upload_id="id")
        with pytest.raises(
            InvalidArgumentException, match="Bucket names must be equal"
        ):
            self.validator.validate_multipart_upload(mpu, "my-bucket")

    def test_validate_multipart_upload_case_insensitive_match(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mpu = MultipartUpload(bucket="My-Bucket", key="k", upload_id="id")
        self.validator.validate_multipart_upload(mpu, "my-bucket")  # no exception

    def test_validate_multipart_upload_same_bucket_passes(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        self.validator.validate_multipart_upload(mpu, "b")  # no exception

    def test_validate_multipart_upload_empty_key_raises(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        mpu = MultipartUpload(bucket="b", key="", upload_id="id")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            self.validator.validate_multipart_upload(mpu, "b")


class TestBaseBlobStoreMultipart:
    """Tests that multipart methods validate input and delegate correctly."""

    def test_initiate_multipart_upload_empty_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(
            InvalidArgumentException, match="Object name cannot be null or empty"
        ):
            store.initiate_multipart_upload("")

    def test_initiate_multipart_upload_returns_handle(self):
        store = _FakeBlobStore(bucket="fake")
        mpu = store.initiate_multipart_upload("obj/key.bin", content_type="image/png")
        assert mpu.key == "obj/key.bin"
        assert mpu.upload_id == "fake-id"
        assert mpu.bucket == "fake"

    def test_initiate_multipart_upload_passes_metadata(self):
        store = _FakeBlobStore(bucket="fake")
        mpu = store.initiate_multipart_upload("k", metadata={"env": "test"})
        assert mpu.metadata == {"env": "test"}

    def test_initiate_multipart_upload_passes_tags(self):
        store = _FakeBlobStore(bucket="fake")
        mpu = store.initiate_multipart_upload("k", tags={"team": "storage"})
        assert mpu.tags == {"team": "storage"}

    def test_initiate_multipart_upload_none_tags_becomes_empty_dict(self):
        """Ensure None tags is normalized to {} before reaching do_*."""
        store = _FakeBlobStore(bucket="fake")
        mpu = store.initiate_multipart_upload("k", tags=None)
        assert mpu.tags == {}

    def test_initiate_multipart_upload_validates_object_lock(self):
        """An invalid lock config is rejected before reaching do_*."""
        store = _FakeBlobStore(bucket="fake")
        with pytest.raises(
            InvalidArgumentException, match="retain_until_date is required"
        ):
            store.initiate_multipart_upload(
                "k", object_lock=ObjectLockConfiguration(mode=RetentionMode.GOVERNANCE)
            )

    def test_initiate_multipart_upload_passes_object_lock(self):
        store = _FakeBlobStore(bucket="fake")
        lock = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        mpu = store.initiate_multipart_upload("k", object_lock=lock)
        assert mpu.object_lock is lock

    def test_upload_part_bucket_mismatch_raises(self):
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        store = _FakeBlobStore(bucket="my-bucket")
        mpu = MultipartUpload(bucket="wrong-bucket", key="k", upload_id="id")
        part = MultipartPart(part_number=1, content=b"data")
        with pytest.raises(
            InvalidArgumentException, match="Bucket names must be equal"
        ):
            store.upload_part(mpu, part)

    def test_upload_part_delegates_and_returns_response(self):
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        store = _FakeBlobStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        part = MultipartPart(part_number=2, content=b"x" * 100)
        resp = store.upload_part(mpu, part)
        assert resp.part_number == 2
        assert resp.e_tag == "fake-etag"
        assert resp.size_in_bytes == 100

    def test_upload_part_with_binary_io(self):
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        store = _FakeBlobStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        stream = io.BytesIO(b"streamed")
        part = MultipartPart(part_number=1, content=stream, content_length=8)
        resp = store.upload_part(mpu, part)
        assert resp.part_number == 1

    def test_complete_multipart_upload_bucket_mismatch_raises(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        store = _FakeBlobStore(bucket="my-bucket")
        mpu = MultipartUpload(bucket="wrong", key="k", upload_id="id")
        with pytest.raises(
            InvalidArgumentException, match="Bucket names must be equal"
        ):
            store.complete_multipart_upload(mpu, [])

    def test_complete_multipart_upload_returns_response(self):
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        store = _FakeBlobStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        parts = [
            UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100),
            UploadPartResponse(part_number=2, e_tag="e2", size_in_bytes=200),
        ]
        result = store.complete_multipart_upload(mpu, parts)
        assert result.e_tag == "fake-complete-etag"

    def test_list_parts_bucket_mismatch_raises(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        store = _FakeBlobStore(bucket="my-bucket")
        mpu = MultipartUpload(bucket="wrong", key="k", upload_id="id")
        with pytest.raises(
            InvalidArgumentException, match="Bucket names must be equal"
        ):
            store.list_parts(mpu)

    def test_list_parts_delegates(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        store = _FakeBlobStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        result = store.list_parts(mpu)
        assert result == []

    def test_abort_multipart_upload_bucket_mismatch_raises(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        store = _FakeBlobStore(bucket="my-bucket")
        mpu = MultipartUpload(bucket="wrong", key="k", upload_id="id")
        with pytest.raises(
            InvalidArgumentException, match="Bucket names must be equal"
        ):
            store.abort_multipart_upload(mpu)

    def test_abort_multipart_upload_delegates(self):
        from multicloud.blob.blob_client.driver import MultipartUpload

        store = _FakeBlobStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        store.abort_multipart_upload(mpu)  # no exception, void return

    def test_initiate_multipart_upload_none_metadata_becomes_empty_dict(self):
        """Ensure None metadata is normalized to {} before reaching do_*."""
        store = _FakeBlobStore(bucket="fake")
        mpu = store.initiate_multipart_upload("k", metadata=None)
        assert mpu.metadata == {}

    def test_complete_multipart_upload_empty_parts_list(self):
        """Providers must handle empty parts (would fail on some providers, but driver passes through)."""
        from multicloud.blob.blob_client.driver import MultipartUpload

        store = _FakeBlobStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        result = store.complete_multipart_upload(mpu, [])
        assert result.e_tag == "fake-complete-etag"

    def test_multipart_part_zero_length_bytes(self):
        """Empty bytes is allowed at the driver layer (some providers reject on the wire)."""
        from multicloud.blob.blob_client.driver import MultipartPart

        part = MultipartPart(part_number=1, content=b"")
        assert part.content_length == 0

    def test_multipart_part_content_length_mismatch_rejected(self):
        """Mismatched content_length raises ValueError at construction time."""
        from multicloud.blob.blob_client.driver import MultipartPart

        with pytest.raises(ValueError, match="does not match"):
            MultipartPart(part_number=1, content=b"abc", content_length=999)

    def test_multipart_part_content_length_explicit_match_accepted(self):
        """Matching explicit content_length on bytes is accepted."""
        from multicloud.blob.blob_client.driver import MultipartPart

        part = MultipartPart(part_number=1, content=b"abc", content_length=3)
        assert part.content_length == 3

    def test_upload_part_large_part_number(self):
        """Part numbers up to 10000 are valid per common provider limits; driver doesn't cap."""
        from multicloud.blob.blob_client.driver import MultipartPart, MultipartUpload

        store = _FakeBlobStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        part = MultipartPart(part_number=10000, content=b"x")
        resp = store.upload_part(mpu, part)
        assert resp.part_number == 10000

    def test_complete_with_out_of_order_parts_passed_through(self):
        """Driver does not sort; sorting is the provider's responsibility."""
        from multicloud.blob.blob_client.driver import (
            MultipartUpload,
            UploadPartResponse,
        )

        store = _FakeBlobStore(bucket="b")
        mpu = MultipartUpload(bucket="b", key="k", upload_id="id")
        parts = [
            UploadPartResponse(part_number=3, e_tag="e3", size_in_bytes=100),
            UploadPartResponse(part_number=1, e_tag="e1", size_in_bytes=100),
            UploadPartResponse(part_number=2, e_tag="e2", size_in_bytes=100),
        ]
        result = store.complete_multipart_upload(mpu, parts)
        assert result.e_tag == "fake-complete-etag"


class TestBaseBlobStoreDeleteWithVersion:
    """Tests for delete() with version_id parameter."""

    def test_delete_with_version_id_delegates(self):
        store = _FakeBlobStore(bucket="b")
        store.delete("obj.txt", version_id="v42")
        assert store.delete_calls == [{"key": "obj.txt", "version_id": "v42"}]

    def test_delete_without_version_id_passes_none(self):
        store = _FakeBlobStore(bucket="b")
        store.delete("obj.txt")
        assert store.delete_calls == [{"key": "obj.txt", "version_id": None}]


class TestBaseBlobStoreDeleteObjects:
    """Tests for delete_objects() validation and delegation."""

    def test_delete_objects_empty_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException, match="cannot be empty"):
            store.delete_objects([])

    def test_delete_objects_empty_key_in_collection_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException, match="empty key"):
            store.delete_objects([BlobIdentifier(key="")])

    def test_delete_objects_valid_delegates(self):
        store = _FakeBlobStore(bucket="b")
        objects = [
            BlobIdentifier(key="a.txt"),
            BlobIdentifier(key="b.txt", version_id="v1"),
        ]
        resp = store.delete_objects(objects)
        assert resp.errors == []


class TestBaseBlobStoreCopy:
    """Tests for copy() validation and delegation."""

    def test_copy_empty_src_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException, match="src_key is required"):
            store.copy("", dest_bucket="dest", dest_key="d.txt")

    def test_copy_empty_dest_bucket_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException, match="dest_bucket is required"):
            store.copy("s.txt", dest_bucket="", dest_key="d.txt")

    def test_copy_empty_dest_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException, match="dest_key is required"):
            store.copy("s.txt", dest_bucket="dest", dest_key="")

    def test_copy_valid_delegates(self):
        store = _FakeBlobStore(bucket="b")
        resp = store.copy("src.txt", dest_bucket="other", dest_key="dst.txt")
        assert resp.key == "dst.txt"
        assert resp.e_tag == "fake-etag"


class TestBaseBlobStoreCopyFrom:
    """Tests for copy_from() validation and delegation."""

    def test_copy_from_empty_src_bucket_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException, match="src_bucket is required"):
            store.copy_from("", "key", dest_key="d.txt")

    def test_copy_from_empty_src_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException, match="src_key is required"):
            store.copy_from("src-bucket", "", dest_key="d.txt")

    def test_copy_from_empty_dest_key_raises(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException, match="dest_key is required"):
            store.copy_from("src-bucket", "src.txt", dest_key="")

    def test_copy_from_valid_delegates(self):
        store = _FakeBlobStore(bucket="b")
        resp = store.copy_from("src-bucket", "src.txt", dest_key="local.txt")
        assert resp.key == "local.txt"
        assert resp.e_tag == "fake-etag"


class TestBlobIdentifier:
    """Tests for BlobIdentifier dataclass."""

    def test_key_only(self):
        b = BlobIdentifier(key="my-file.txt")
        assert b.key == "my-file.txt"
        assert b.version_id is None

    def test_key_and_version(self):
        b = BlobIdentifier(key="my-file.txt", version_id="v3")
        assert b.key == "my-file.txt"
        assert b.version_id == "v3"

    def test_frozen(self):
        b = BlobIdentifier(key="x")
        with pytest.raises(AttributeError):
            b.key = "y"  # type: ignore[misc]


class TestCopyResponse:
    """Tests for CopyResponse dataclass."""

    def test_all_fields(self):
        from datetime import datetime

        ts = datetime(2024, 1, 15, 10, 30, 0)
        r = CopyResponse(key="dst.txt", version_id="v1", e_tag="abc", last_modified=ts)
        assert r.key == "dst.txt"
        assert r.version_id == "v1"
        assert r.e_tag == "abc"
        assert r.last_modified == ts

    def test_defaults(self):
        r = CopyResponse(key="k")
        assert r.version_id is None
        assert r.e_tag is None
        assert r.last_modified is None


class TestRetentionMode:
    """Tests for the RetentionMode enum."""

    def test_values(self):
        assert RetentionMode.GOVERNANCE.value == "GOVERNANCE"
        assert RetentionMode.COMPLIANCE.value == "COMPLIANCE"

    def test_members(self):
        assert set(RetentionMode) == {
            RetentionMode.GOVERNANCE,
            RetentionMode.COMPLIANCE,
        }


class TestObjectLockInfo:
    """Tests for the ObjectLockInfo dataclass."""

    def test_all_fields(self):
        ts = datetime(2030, 1, 1, 0, 0, 0)
        info = ObjectLockInfo(
            mode=RetentionMode.COMPLIANCE,
            retain_until_date=ts,
            legal_hold=True,
            event_based_hold=True,
        )
        assert info.mode is RetentionMode.COMPLIANCE
        assert info.retain_until_date == ts
        assert info.legal_hold is True
        assert info.event_based_hold is True

    def test_defaults(self):
        info = ObjectLockInfo()
        assert info.mode is None
        assert info.retain_until_date is None
        assert info.legal_hold is False
        assert info.event_based_hold is None

    def test_frozen(self):
        info = ObjectLockInfo()
        with pytest.raises(AttributeError):
            info.legal_hold = True  # type: ignore[misc]


class TestObjectRetentionConfig:
    """Tests for the ObjectRetentionConfig dataclass."""

    def test_all_fields(self):
        ts = datetime(2030, 6, 1, 12, 0, 0)
        cfg = ObjectRetentionConfig(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=ts,
            bypass_governance=True,
        )
        assert cfg.mode is RetentionMode.GOVERNANCE
        assert cfg.retain_until_date == ts
        assert cfg.bypass_governance is True

    def test_bypass_governance_defaults_false(self):
        cfg = ObjectRetentionConfig(
            mode=RetentionMode.COMPLIANCE,
            retain_until_date=datetime(2030, 6, 1),
        )
        assert cfg.bypass_governance is False

    def test_frozen(self):
        cfg = ObjectRetentionConfig(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=datetime(2030, 6, 1),
        )
        with pytest.raises(AttributeError):
            cfg.bypass_governance = True  # type: ignore[misc]


class TestBaseBlobStoreGetObjectLock:
    """Driver-layer get_object_lock: validation + delegation to do_*."""

    def test_validates_empty_key(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException):
            store.get_object_lock("")

    def test_delegates_to_do_get_object_lock(self):
        store = _FakeBlobStore(bucket="b")
        result = store.get_object_lock("k")
        assert isinstance(result, ObjectLockInfo)
        assert store.get_object_lock_calls == [{"key": "k", "version_id": None}]

    def test_forwards_version_id(self):
        store = _FakeBlobStore(bucket="b")
        store.get_object_lock("k", version_id="v7")
        assert store.get_object_lock_calls == [{"key": "k", "version_id": "v7"}]


class TestBaseBlobStoreUpdateLegalHold:
    """Driver-layer update_legal_hold: validation + delegation to do_*."""

    def test_validates_empty_key(self):
        store = _FakeBlobStore(bucket="b")
        with pytest.raises(InvalidArgumentException):
            store.update_legal_hold("", True)

    def test_delegates_enable_to_do_update_legal_hold(self):
        store = _FakeBlobStore(bucket="b")
        store.update_legal_hold("k", True)
        assert store.update_legal_hold_calls == [
            {"key": "k", "enabled": True, "version_id": None}
        ]

    def test_delegates_disable_to_do_update_legal_hold(self):
        store = _FakeBlobStore(bucket="b")
        store.update_legal_hold("k", False)
        assert store.update_legal_hold_calls == [
            {"key": "k", "enabled": False, "version_id": None}
        ]

    def test_forwards_version_id(self):
        store = _FakeBlobStore(bucket="b")
        store.update_legal_hold("k", True, version_id="v7")
        assert store.update_legal_hold_calls == [
            {"key": "k", "enabled": True, "version_id": "v7"}
        ]
