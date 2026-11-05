"""Tests for the standalone BlobStoreValidator helpers.

Operation-level end-to-end coverage stays in test_blob_driver.py;
this file focuses on the generic helpers and field-level validators.
"""

from __future__ import annotations

import base64
import hashlib

import pytest
from multicloud.blob.blob_client.driver import ChecksumAlgorithm, PresignedOperation
from multicloud.blob.blob_client.validators import (
    CHECKSUM_ALGORITHM_WITHOUT_VALUE_MSG,
    CHECKSUM_VALUE_WITHOUT_ALGORITHM_MSG,
    CREATE_IF_ABSENT_WITH_IF_MATCH_MSG,
    CREATE_IF_ABSENT_WITH_IF_VERSION_MATCH_MSG,
    EMPTY_IF_MATCH_MSG,
    EMPTY_IF_VERSION_MATCH_MSG,
    INVALID_BUCKET_NAME_MSG,
    INVALID_OBJECT_NAME_MSG,
    INVALID_RANGED_READ_BOUNDARIES_MSG,
    INVALID_RANGED_READ_NEGATIVE_BOUNDARIES_MSG,
    INVALID_TAGS_COLLECTION_MSG,
    MAX_OBJECT_NAME_LENGTH_BYTES,
    MISMATCHED_BUCKET_NAME_MSG,
    BlobStoreValidator,
)
from multicloud.common.exceptions import InvalidArgumentException


@pytest.fixture
def v() -> BlobStoreValidator:
    return BlobStoreValidator()


class TestRequireNotBlank:
    def test_accepts_non_blank(self, v):
        v.require_not_blank("hello", "msg")  # no raise

    def test_rejects_none(self, v):
        with pytest.raises(InvalidArgumentException, match="msg"):
            v.require_not_blank(None, "msg")

    def test_rejects_empty(self, v):
        with pytest.raises(InvalidArgumentException, match="msg"):
            v.require_not_blank("", "msg")

    def test_rejects_whitespace(self, v):
        with pytest.raises(InvalidArgumentException, match="msg"):
            v.require_not_blank("   ", "msg")


class TestRequireNotEmpty:
    def test_accepts_non_empty_list(self, v):
        v.require_not_empty(["a"], "msg")

    def test_accepts_non_empty_dict(self, v):
        v.require_not_empty({"k": "v"}, "msg")

    def test_rejects_none(self, v):
        with pytest.raises(InvalidArgumentException, match="msg"):
            v.require_not_empty(None, "msg")

    def test_rejects_empty_list(self, v):
        with pytest.raises(InvalidArgumentException, match="msg"):
            v.require_not_empty([], "msg")

    def test_rejects_empty_dict(self, v):
        with pytest.raises(InvalidArgumentException, match="msg"):
            v.require_not_empty({}, "msg")


class TestRequireEqualsIgnoreCase:
    def test_accepts_equal(self, v):
        v.require_equals_ignore_case("Bucket", "bucket", "msg")

    def test_accepts_both_none(self, v):
        v.require_equals_ignore_case(None, None, "msg")

    def test_rejects_different(self, v):
        with pytest.raises(InvalidArgumentException, match=MISMATCHED_BUCKET_NAME_MSG):
            v.require_equals_ignore_case("a", "b", MISMATCHED_BUCKET_NAME_MSG)

    def test_rejects_one_none(self, v):
        with pytest.raises(InvalidArgumentException, match="msg"):
            v.require_equals_ignore_case("a", None, "msg")


class TestValidateKey:
    def test_accepts_valid_key(self, v):
        v.validate_key("some/key.txt")

    def test_rejects_empty(self, v):
        with pytest.raises(InvalidArgumentException, match=INVALID_OBJECT_NAME_MSG):
            v.validate_key("")

    def test_accepts_max_length_key(self, v):
        v.validate_key("a" * MAX_OBJECT_NAME_LENGTH_BYTES)

    def test_rejects_key_one_byte_over_limit(self, v):
        with pytest.raises(
            InvalidArgumentException,
            match=f"Object name cannot exceed {MAX_OBJECT_NAME_LENGTH_BYTES} bytes",
        ):
            v.validate_key("a" * (MAX_OBJECT_NAME_LENGTH_BYTES + 1))

    def test_length_measured_in_utf8_bytes_not_codepoints(self, v):
        # Each emoji is 4 UTF-8 bytes — 257 emoji = 1028 bytes, just over.
        too_long = "🚀" * 257
        assert len(too_long) == 257  # 257 code points
        assert len(too_long.encode("utf-8")) == 1028  # but 1028 bytes
        with pytest.raises(
            InvalidArgumentException,
            match="Object name cannot exceed 1024 bytes",
        ):
            v.validate_key(too_long)


class TestValidateContentLength:
    def test_accepts_none(self, v):
        v.validate_content_length(None)

    def test_accepts_zero(self, v):
        # Zero-byte uploads are a legal real-world case (empty marker files).
        v.validate_content_length(0)

    def test_accepts_positive(self, v):
        v.validate_content_length(1024)

    def test_rejects_negative(self, v):
        with pytest.raises(
            InvalidArgumentException, match="content_length must be a non-negative"
        ):
            v.validate_content_length(-1)


class TestValidateUploadWithContentLength:
    def test_accepts_no_content_length(self, v):
        v.validate_upload("k")

    def test_accepts_zero(self, v):
        v.validate_upload("k", 0)

    def test_rejects_negative_content_length(self, v):
        with pytest.raises(
            InvalidArgumentException, match="content_length must be a non-negative"
        ):
            v.validate_upload("k", -5)

    def test_rejects_empty_key_before_content_length(self, v):
        # Key validation runs first.
        with pytest.raises(InvalidArgumentException, match=INVALID_OBJECT_NAME_MSG):
            v.validate_upload("", -5)


_MD5_B64 = base64.b64encode(hashlib.md5(b"data").digest()).decode()
_CRC32C_B64 = base64.b64encode(b"\x00\x01\x02\x03").decode()


class TestValidateChecksum:
    """checksum_value / checksum_algorithm pairing and digest shape."""

    def test_accepts_neither(self, v):
        v.validate_checksum(None, None)

    def test_accepts_md5_digest(self, v):
        v.validate_checksum(_MD5_B64, ChecksumAlgorithm.MD5)

    def test_accepts_crc32c_digest(self, v):
        v.validate_checksum(_CRC32C_B64, ChecksumAlgorithm.CRC32C)

    def test_rejects_value_without_algorithm(self, v):
        with pytest.raises(
            InvalidArgumentException, match=CHECKSUM_VALUE_WITHOUT_ALGORITHM_MSG
        ):
            v.validate_checksum(_MD5_B64, None)

    def test_rejects_algorithm_without_value(self, v):
        with pytest.raises(
            InvalidArgumentException, match=CHECKSUM_ALGORITHM_WITHOUT_VALUE_MSG
        ):
            v.validate_checksum(None, ChecksumAlgorithm.MD5)

    def test_rejects_non_base64_value(self, v):
        with pytest.raises(
            InvalidArgumentException, match="base64-encoded 16-byte MD5 digest"
        ):
            v.validate_checksum("not base64!", ChecksumAlgorithm.MD5)

    def test_rejects_hex_encoded_md5(self, v):
        # A hex digest is valid base64 text but decodes to 24 bytes, not 16.
        with pytest.raises(
            InvalidArgumentException, match="base64-encoded 16-byte MD5 digest"
        ):
            v.validate_checksum(hashlib.md5(b"data").hexdigest(), ChecksumAlgorithm.MD5)

    def test_rejects_wrong_length_for_algorithm(self, v):
        # A valid MD5 digest is the wrong size for CRC32C.
        with pytest.raises(
            InvalidArgumentException, match="base64-encoded 4-byte CRC32C digest"
        ):
            v.validate_checksum(_MD5_B64, ChecksumAlgorithm.CRC32C)


class TestValidateWritePreconditions:
    """Mutual-exclusion rules for create_if_absent / if_match on writes."""

    def test_accepts_neither(self, v):
        v.validate_write_preconditions(False, None)  # no raise

    def test_accepts_create_if_absent_alone(self, v):
        v.validate_write_preconditions(True, None)  # no raise

    def test_accepts_if_match_alone(self, v):
        v.validate_write_preconditions(False, "etag-123")  # no raise

    def test_rejects_create_if_absent_with_if_match(self, v):
        with pytest.raises(
            InvalidArgumentException, match=CREATE_IF_ABSENT_WITH_IF_MATCH_MSG
        ):
            v.validate_write_preconditions(True, "etag-123")

    def test_validate_upload_rejects_conflicting_preconditions(self, v):
        with pytest.raises(
            InvalidArgumentException, match=CREATE_IF_ABSENT_WITH_IF_MATCH_MSG
        ):
            v.validate_upload("k", create_if_absent=True, if_match="etag")

    def test_validate_upload_accepts_create_if_absent(self, v):
        v.validate_upload("k", create_if_absent=True)  # no raise

    def test_validate_upload_from_path_rejects_conflict(self, v, tmp_path):
        src = tmp_path / "f.txt"
        src.write_bytes(b"data")
        with pytest.raises(
            InvalidArgumentException, match=CREATE_IF_ABSENT_WITH_IF_MATCH_MSG
        ):
            v.validate_upload_from_path(
                "k", src, create_if_absent=True, if_match="etag"
            )

    def test_accepts_if_version_match_alone(self, v):
        v.validate_write_preconditions(False, None, "42")  # no raise

    def test_accepts_if_match_and_if_version_match_together(self, v):
        # Portable callers may set both so each provider enforces with the
        # token it supports; combining them is allowed.
        v.validate_write_preconditions(False, "etag-123", "42")  # no raise

    def test_rejects_create_if_absent_with_if_version_match(self, v):
        with pytest.raises(
            InvalidArgumentException,
            match=CREATE_IF_ABSENT_WITH_IF_VERSION_MATCH_MSG,
        ):
            v.validate_write_preconditions(True, None, "42")

    def test_rejects_empty_if_version_match(self, v):
        with pytest.raises(InvalidArgumentException, match=EMPTY_IF_VERSION_MATCH_MSG):
            v.validate_write_preconditions(False, None, "")

    def test_rejects_whitespace_if_version_match(self, v):
        with pytest.raises(InvalidArgumentException, match=EMPTY_IF_VERSION_MATCH_MSG):
            v.validate_write_preconditions(False, None, "   ")

    def test_rejects_empty_if_match(self, v):
        with pytest.raises(InvalidArgumentException, match=EMPTY_IF_MATCH_MSG):
            v.validate_write_preconditions(False, "")

    def test_rejects_whitespace_if_match(self, v):
        with pytest.raises(InvalidArgumentException, match=EMPTY_IF_MATCH_MSG):
            v.validate_write_preconditions(False, "   ")

    def test_validate_upload_rejects_empty_if_match(self, v):
        with pytest.raises(InvalidArgumentException, match=EMPTY_IF_MATCH_MSG):
            v.validate_upload("k", if_match="")

    def test_validate_upload_rejects_create_if_absent_with_if_version_match(self, v):
        with pytest.raises(
            InvalidArgumentException,
            match=CREATE_IF_ABSENT_WITH_IF_VERSION_MATCH_MSG,
        ):
            v.validate_upload("k", create_if_absent=True, if_version_match="42")


class TestValidateDownloadPreconditions:
    """Conditional-read precondition on download."""

    def test_accepts_if_match_alone(self, v):
        v.validate_download("k", if_match="etag-123")  # no raise

    def test_accepts_none_if_match(self, v):
        v.validate_download("k", if_match=None)  # no raise — precondition skipped

    def test_rejects_empty_if_match(self, v):
        with pytest.raises(InvalidArgumentException, match=EMPTY_IF_MATCH_MSG):
            v.validate_download("k", if_match="")

    def test_rejects_whitespace_if_match(self, v):
        with pytest.raises(InvalidArgumentException, match=EMPTY_IF_MATCH_MSG):
            v.validate_download("k", if_match="   ")

    def test_rejects_empty_key_before_precondition_check(self, v):
        # Key validation runs first, even when a precondition is supplied.
        with pytest.raises(InvalidArgumentException, match=INVALID_OBJECT_NAME_MSG):
            v.validate_download("", if_match="a")


class TestValidatePresignedDownloadRejectsUploadOnlyParams:
    def test_download_with_no_upload_only_params_accepts(self, v):
        v.validate_generate_presigned_url("k", PresignedOperation.DOWNLOAD, 3600)

    def test_download_rejects_metadata(self, v):
        with pytest.raises(
            InvalidArgumentException, match="metadata is only valid for UPLOAD"
        ):
            v.validate_generate_presigned_url(
                "k", PresignedOperation.DOWNLOAD, 3600, metadata={"env": "prod"}
            )

    def test_download_rejects_tags(self, v):
        with pytest.raises(
            InvalidArgumentException, match="tags is only valid for UPLOAD"
        ):
            v.validate_generate_presigned_url(
                "k", PresignedOperation.DOWNLOAD, 3600, tags={"owner": "rose"}
            )

    def test_download_rejects_kms_key_id(self, v):
        with pytest.raises(
            InvalidArgumentException, match="kms_key_id is only valid for UPLOAD"
        ):
            v.validate_generate_presigned_url(
                "k", PresignedOperation.DOWNLOAD, 3600, kms_key_id="alias/my-key"
            )

    def test_download_allows_empty_dicts_and_none(self, v):
        # Empty dicts and None should NOT be rejected — they mean "not set".
        v.validate_generate_presigned_url(
            "k",
            PresignedOperation.DOWNLOAD,
            3600,
            metadata={},
            tags={},
            kms_key_id=None,
        )

    def test_upload_allows_all_params(self, v):
        v.validate_generate_presigned_url(
            "k",
            PresignedOperation.UPLOAD,
            3600,
            metadata={"env": "prod"},
            tags={"owner": "rose"},
            kms_key_id="alias/my-key",
        )


class TestValidateBucket:
    def test_accepts_valid_bucket(self, v):
        v.validate_bucket("my-bucket")

    def test_rejects_empty(self, v):
        with pytest.raises(InvalidArgumentException, match=INVALID_BUCKET_NAME_MSG):
            v.validate_bucket("")


class TestValidateKeys:
    def test_accepts_non_empty(self, v):
        v.validate_keys(["a", "b"])

    def test_rejects_empty_collection(self, v):
        with pytest.raises(InvalidArgumentException):
            v.validate_keys([])

    def test_rejects_blank_key_in_collection(self, v):
        with pytest.raises(InvalidArgumentException, match=INVALID_OBJECT_NAME_MSG):
            v.validate_keys(["a", ""])


class TestValidateTags:
    def test_accepts_non_empty(self, v):
        v.validate_tags({"k": "v"})

    def test_rejects_empty(self, v):
        with pytest.raises(InvalidArgumentException, match=INVALID_TAGS_COLLECTION_MSG):
            v.validate_tags({})


class TestValidateDuration:
    def test_accepts_positive(self, v):
        v.validate_duration(3600)

    def test_rejects_zero(self, v):
        with pytest.raises(InvalidArgumentException, match="non-zero positive value"):
            v.validate_duration(0)

    def test_rejects_negative(self, v):
        with pytest.raises(InvalidArgumentException, match="non-zero positive value"):
            v.validate_duration(-100)

    def test_rejects_none(self, v):
        with pytest.raises(InvalidArgumentException, match="non-zero positive value"):
            v.validate_duration(None)


class TestValidateRange:
    def test_accepts_none_none(self, v):
        v.validate_range(None, None)  # no raise

    def test_accepts_start_only(self, v):
        v.validate_range(0, None)
        v.validate_range(100, None)

    def test_accepts_end_only(self, v):
        v.validate_range(None, 500)

    def test_accepts_start_equals_end(self, v):
        v.validate_range(10, 10)

    def test_accepts_start_less_than_end(self, v):
        v.validate_range(0, 100)

    def test_rejects_negative_start(self, v):
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read boundaries cannot be negative",
        ):
            v.validate_range(-1, 10)

    def test_rejects_negative_end(self, v):
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read boundaries cannot be negative",
        ):
            v.validate_range(0, -1)

    def test_rejects_end_less_than_start(self, v):
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read start cannot be larger than end",
        ):
            v.validate_range(100, 50)

    def test_message_includes_values(self, v):
        with pytest.raises(InvalidArgumentException) as exc_info:
            v.validate_range(200, 100)
        assert "start=200" in str(exc_info.value)
        assert "end=100" in str(exc_info.value)


class TestValidateDelete:
    def test_accepts_valid_key(self, v):
        v.validate_delete("some/key.txt")  # no raise

    def test_rejects_empty(self, v):
        with pytest.raises(InvalidArgumentException, match=INVALID_OBJECT_NAME_MSG):
            v.validate_delete("")

    def test_rejects_none(self, v):
        with pytest.raises(InvalidArgumentException, match=INVALID_OBJECT_NAME_MSG):
            v.validate_delete(None)  # type: ignore[arg-type]


class TestValidateDownloadWithRange:
    def test_accepts_key_only(self, v):
        v.validate_download("k")

    def test_accepts_key_with_valid_range(self, v):
        v.validate_download("k", 0, 100)

    def test_accepts_key_with_open_ended_range(self, v):
        v.validate_download("k", 100, None)
        v.validate_download("k", None, 200)

    def test_rejects_empty_key_before_range_check(self, v):
        # Key validation runs first; even a "valid" range cannot rescue an empty key.
        with pytest.raises(InvalidArgumentException, match=INVALID_OBJECT_NAME_MSG):
            v.validate_download("", 0, 100)

    def test_rejects_negative_start(self, v):
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read boundaries cannot be negative",
        ):
            v.validate_download("k", -1, 10)

    def test_rejects_end_less_than_start(self, v):
        with pytest.raises(
            InvalidArgumentException,
            match="Ranged read start cannot be larger than end",
        ):
            v.validate_download("k", 100, 10)


class TestValidateEndpoint:
    def test_accepts_https_no_port(self, v):
        v.validate_endpoint("https://example.com", require_port=False)

    def test_accepts_http_with_port(self, v):
        v.validate_endpoint("http://example.com:8080", require_port=True)

    def test_rejects_none(self, v):
        with pytest.raises(InvalidArgumentException, match="Endpoint cannot be null"):
            v.validate_endpoint(None, require_port=False)

    def test_rejects_unsupported_scheme(self, v):
        with pytest.raises(InvalidArgumentException, match="protocol of http or https"):
            v.validate_endpoint("ftp://example.com", require_port=False)

    def test_rejects_no_scheme(self, v):
        with pytest.raises(InvalidArgumentException, match="protocol of http or https"):
            v.validate_endpoint("example.com", require_port=False)

    def test_rejects_empty_hostname(self, v):
        with pytest.raises(InvalidArgumentException, match="non-empty hostname"):
            v.validate_endpoint("https://", require_port=False)

    def test_rejects_missing_port_when_required(self, v):
        with pytest.raises(
            InvalidArgumentException, match="non-negative value for the port"
        ):
            v.validate_endpoint("https://example.com", require_port=True)

    def test_rejects_path(self, v):
        with pytest.raises(InvalidArgumentException, match="must not have a path"):
            v.validate_endpoint("https://example.com/api", require_port=False)

    def test_rejects_query_params(self, v):
        with pytest.raises(
            InvalidArgumentException, match="must not have query params"
        ):
            v.validate_endpoint("https://example.com?x=1", require_port=False)

    def test_accepts_root_path(self, v):
        # A single "/" path is allowed; anything else is not.
        v.validate_endpoint("https://example.com/", require_port=False)

    def test_rejects_double_slash_path(self, v):
        # "//" is not exactly "/" and must be rejected — verifying we do
        # not collapse leading slashes before the comparison.
        with pytest.raises(InvalidArgumentException, match="must not have a path"):
            v.validate_endpoint("https://example.com//", require_port=False)

    def test_accepts_port_zero(self, v):
        # urllib returns port=0 for ":0" and port=None when omitted;
        # only the missing case should be rejected when require_port is True.
        v.validate_endpoint("http://example.com:0", require_port=True)

    def test_rejects_non_numeric_port(self, v):
        with pytest.raises(InvalidArgumentException, match="port") as exc_info:
            v.validate_endpoint("http://example.com:abc", require_port=True)
        assert "abc" in str(exc_info.value)

    def test_rejects_out_of_range_port(self, v):
        with pytest.raises(InvalidArgumentException, match="port") as exc_info:
            v.validate_endpoint("http://example.com:99999", require_port=True)
        assert "99999" in str(exc_info.value)

    def test_non_numeric_port_ignored_when_not_required(self, v):
        v.validate_endpoint("http://example.com:abc", require_port=False)
