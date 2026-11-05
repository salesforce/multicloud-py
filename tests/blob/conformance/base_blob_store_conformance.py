"""Reusable blob conformance test suite.

Defines the conformance tests that every blob provider must pass.
Provider-specific test classes extend ``BaseBlobStoreConformance``
and supply a ``client`` fixture that returns a :class:`BucketClient`
configured for the provider under test.

Class naming notes:
  - ``BaseBlobStoreConformance`` ends in ``Conformance`` (not
    ``Tests``) so pytest's default ``python_classes = Test*`` collection
    rule doesn't pick it up. Concrete subclasses must start with ``Test``
    to be collected.
"""

from __future__ import annotations

import base64
import hashlib
import io
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import parse_qs, urlparse

import pytest
import requests
from multicloud.blob.blob_client import (
    BlobIdentifier,
    BucketClient,
    ChecksumAlgorithm,
    MultipartPart,
    MultipartUpload,
    ObjectLockConfiguration,
    ObjectRetentionConfig,
    PresignedOperation,
    RetentionMode,
    UploadPartResponse,
)
from multicloud.common.exceptions import (
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceExistsException,
    ResourceNotFoundException,
    SubstrateSdkException,
    UnAuthorizedException,
)

from tests.blob.conformance.vcr_filters import captured_legal_hold_writes


def _crc32c_b64(data: bytes) -> str:
    """Return the base64 big-endian CRC32C (Castagnoli) of *data*.

    Computed here rather than with a provider library so the shared suite stays
    independent of any provider's dependencies.
    """
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 & -(crc & 1))
    return base64.b64encode((crc ^ 0xFFFFFFFF).to_bytes(4, "big")).decode()


class Harness(ABC):
    """Test harness interface that providers implement.

    The harness owns the lifecycle of a :class:`BucketClient` instance
    for a single test session. Providers wire up authentication, bucket
    name, region, etc. inside the harness so tests stay cloud-agnostic.
    """

    @abstractmethod
    def create_client(self) -> BucketClient:
        """Return a BucketClient configured for the test bucket."""
        ...

    @abstractmethod
    def get_provider_id(self) -> str:
        """Return the provider id (e.g. ``"aws"``, ``"gcp"``)."""
        ...

    @abstractmethod
    def get_bucket(self) -> str:
        """Return the bucket name the client is configured against.

        Tests that exercise cross-bucket operations (``copy``,
        ``copy_from``) need a cloud-agnostic way to reference the
        current bucket without reaching into the client internals.
        """
        ...

    @abstractmethod
    def create_client_for_absent_bucket(self) -> BucketClient:
        """Return a BucketClient bound to a bucket that does not exist.

        ``does_bucket_exist`` takes no arguments -- it reports on the bucket the
        client was constructed with -- so the negative case can only be reached
        through a separately configured client. Required rather than optional:
        every provider can name a bucket that does not exist, and letting a
        provider opt out would silently drop the only test that can fail an
        implementation hard-coded to return True.

        The name must be stable across runs, because it appears in the request
        URI and the cassette matcher compares host and path.
        """

    def create_versioned_client(self) -> Optional[BucketClient]:
        """Return a BucketClient bound to a versioning-enabled bucket, or None.

        Providers that have provisioned a separate versioning-enabled
        bucket override this to return a client for it. Providers that
        haven't (or can't) return ``None`` and the versioning tests are
        skipped.
        """
        return None

    def get_versioned_bucket(self) -> Optional[str]:
        """Return the versioning-enabled bucket name, or None.

        Mirrors :meth:`create_versioned_client` for tests that need the
        bucket name (e.g. cross-bucket copy destinations) without
        reaching into the client internals.
        """
        return None

    def validates_part_etags(self) -> bool:
        """Whether the provider validates caller-supplied part ETags.

        Providers that assemble a multipart upload by re-reading the bytes
        already stored for each part number (rather than checking the ETag
        the caller passes to complete) cannot detect a wrong ETag and
        complete successfully. Such providers override this to return
        ``False``; the default assumes server-side validation.
        """
        return True

    def create_object_lock_client(self) -> Optional[BucketClient]:
        """Return a BucketClient bound to an object-lock-enabled bucket, or None.

        Placing a legal hold requires the bucket to have object lock (or the
        provider equivalent) enabled. Providers that have provisioned such a
        bucket override this to return a client for it; providers that haven't
        return ``None`` and the object-lock tests are skipped.
        """
        return None

    def supports_write_if_match(self) -> bool:
        """Whether the provider honors an ETag precondition on writes.

        Raises by default so a provider must declare its stance rather than
        silently disable the ETag-conditioned write tests.
        """
        raise NotImplementedError(
            "Harness must declare whether writes honor an ETag precondition."
        )

    def supports_write_if_version_match(self) -> bool:
        """Whether the provider honors a version precondition on writes.

        Raises by default so a provider must declare its stance rather than
        silently disable the version-conditioned write tests.
        """
        raise NotImplementedError(
            "Harness must declare whether writes honor a version precondition."
        )

    def get_kms_key_id(self) -> str:
        """Return a provider key id for server-side encryption.

        Raises by default so a provider must supply a key rather than silently
        disable the encrypted-upload tests.
        """
        raise NotImplementedError(
            "Harness must supply a server-side-encryption key id."
        )

    @abstractmethod
    def create_nonexistent_bucket_client(self) -> BucketClient:
        """Return a BucketClient bound to a bucket that does not exist.

        Used by the error-path tests to prove that every network operation
        surfaces a translated exception when the target bucket is absent. Every
        provider can construct one (no infrastructure needed), so this is
        required rather than optional — a provider that forgets it fails loudly
        rather than silently skipping the coverage.
        """
        ...

    @abstractmethod
    def create_invalid_credentials_client(self) -> BucketClient:
        """Return a BucketClient for the real bucket but with invalid credentials.

        Used by the error-path tests to prove that every network operation
        surfaces a translated exception when the caller is not authorized.
        Providers supply deliberately bogus credentials via a credentials
        override; construction must not itself reach the network so the tests
        replay without real credentials.
        """
        ...

    def delete_retained_object(self, client: BucketClient, key: str) -> None:
        """Delete an object that still carries active retention.

        Deleting a retained object is provider-specific and cannot be
        expressed through the cloud-agnostic API: some clouds bypass
        governance on the delete itself (retention stays set), others reject
        both a past retain-until instant and a plain delete and require
        removing the retention entirely first. Neither path is a plain
        ``delete``. Providers that supply an object-lock client override
        this; the default fails loudly so a provider cannot leave retained
        objects behind silently.
        """
        raise NotImplementedError(
            "Harness supplied an object-lock client but no retention teardown."
        )

    def close(self) -> None:  # pragma: no cover - default no-op
        """Clean up resources. Override if needed."""


class BaseBlobStoreConformance:
    """Abstract conformance test suite for blob providers.

    Conformance tests are written ONCE here and executed by each
    provider. Provider test classes inherit from this class and supply a
    ``client`` fixture that returns the provider-specific
    :class:`BucketClient`.
    """

    TEST_PREFIX = "conformance-test/"

    # Minimum size (bytes) every non-final multipart part must reach on some
    # providers. Non-final parts are filled to exactly this size so a
    # multi-part upload completes without a too-small-part rejection; the
    # request body is truncated in the cassette (see vcr_filters) so this size
    # never bloats committed fixtures.
    MPU_PART_SIZE = 5 * 1024 * 1024

    @pytest.fixture(scope="module")
    def versioned_client(self, harness):
        """Provide a BucketClient bound to a versioning-enabled bucket.

        Skips the test when the provider harness has not supplied one.
        The base :class:`Harness` returns ``None`` by default, so harnesses
        that have not provisioned a versioning-enabled bucket (e.g. a
        placeholder provider or a CI environment without the shared
        versioned bucket) auto-skip these tests instead of erroring.
        """
        versioned = harness.create_versioned_client()
        if versioned is None:
            pytest.skip("Provider harness did not supply a versioning-enabled bucket.")
        return versioned

    @pytest.fixture(scope="module")
    def object_lock_client(self, harness):
        """Provide a BucketClient bound to an object-lock-enabled bucket.

        Skips the test when the provider harness has not supplied one. The
        base :class:`Harness` returns ``None`` by default, so harnesses that
        have not provisioned an object-lock-enabled bucket auto-skip these
        tests instead of erroring.
        """
        locked = harness.create_object_lock_client()
        if locked is None:
            pytest.skip(
                "Provider harness did not supply an object-lock-enabled bucket."
            )
        return locked

    @pytest.fixture(scope="module")
    def nonexistent_bucket_client(self, harness):
        """Provide a BucketClient bound to a bucket that does not exist."""
        client = harness.create_nonexistent_bucket_client()
        yield client
        client.close()

    @pytest.fixture(scope="module")
    def invalid_credentials_client(self, harness):
        """Provide a BucketClient for the real bucket with invalid credentials."""
        client = harness.create_invalid_credentials_client()
        yield client
        client.close()

    @pytest.fixture(scope="module")
    def kms_key_id(self, harness):
        """Provide a server-side-encryption key id.

        The harness hook raises if a provider fails to supply one, so a missing
        override fails loudly rather than silently skipping the encrypted-upload
        tests.
        """
        return harness.get_kms_key_id()

    def _safe_delete(self, client: BucketClient, *keys: str) -> None:
        """Delete blobs ignoring errors.

        Prevents test object accumulation in real cloud buckets during
        record mode. Each test that creates objects calls this in a
        finally block.
        """
        for key in keys:
            try:
                client.delete(key)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Cleanup failed for {key}: {e}")

    def _safe_delete_version(
        self, client: BucketClient, key: str, version_id: Optional[str]
    ) -> None:
        """Delete a specific blob version ignoring errors.

        Used by versioned-delete tests so recorded buckets don't retain
        stale versions between record runs.
        """
        try:
            client.delete(key, version_id=version_id)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Cleanup failed for {key}@{version_id}: {e}")

    def _safe_delete_all_versions(self, client: BucketClient, key: str) -> None:
        """Delete every version of a key, ignoring errors.

        Versioned buckets keep prior versions after a plain ``delete``;
        this walks the version list and removes each explicitly so
        record-mode buckets don't accumulate versions.
        """
        try:
            versions = list(client.list_blob_versions(key))
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Version listing failed for {key}: {e}")
            return
        for version in versions:
            try:
                client.delete(key, version_id=version.version_id)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Version cleanup failed for {key}: {e}")

    def _safe_abort(self, client: BucketClient, mpu: MultipartUpload) -> None:
        """Abort an in-progress multipart upload, ignoring errors.

        Multipart tests call this in a finally block so an incomplete
        upload never leaves temporary parts behind in a recorded bucket.
        """
        try:
            client.abort_multipart_upload(mpu)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Abort cleanup failed for {mpu.key}: {e}")

    def test_upload_simple(self, client):
        """Upload a simple text blob, then read it back and verify the bytes."""
        key = f"{self.TEST_PREFIX}upload/simple.txt"
        payload = b"Hello, conformance test!"
        try:
            resp = client.upload(key, payload, content_type="text/plain")
            assert resp.key == key
            assert resp.e_tag is not None
            fetched = client.download(key)
            assert fetched.content == payload
            assert fetched.metadata.content_length == len(payload)
        finally:
            self._safe_delete(client, key)

    def test_upload_empty_content(self, client):
        """Upload a zero-byte blob and verify it reads back as empty."""
        key = f"{self.TEST_PREFIX}upload/empty.txt"
        try:
            resp = client.upload(key, b"", content_type="text/plain")
            assert resp.key == key
            fetched = client.download(key)
            assert fetched.content == b""
            assert fetched.metadata.content_length == 0
        finally:
            self._safe_delete(client, key)

    def test_upload_binary_content(self, client):
        """Upload binary data and verify every byte round-trips intact."""
        key = f"{self.TEST_PREFIX}upload/binary.bin"
        payload = bytes(range(256)) * 4  # 1024 bytes of binary
        try:
            resp = client.upload(key, payload, content_type="application/octet-stream")
            assert resp.key == key
            fetched = client.download(key)
            assert fetched.content == payload
            assert fetched.metadata.content_length == len(payload)
        finally:
            self._safe_delete(client, key)

    def test_upload_with_metadata(self, client):
        """Upload with user metadata, then read it back and verify it persisted."""
        key = f"{self.TEST_PREFIX}upload/with-metadata.txt"
        metadata = {"author": "conformance-test", "version": "1.0"}
        payload = b"metadata test"
        try:
            resp = client.upload(
                key, payload, content_type="text/plain", metadata=metadata
            )
            assert resp.key == key
            fetched = client.download(key)
            assert fetched.content == payload
            assert fetched.metadata.content_type == "text/plain"
            for meta_key, meta_value in metadata.items():
                assert fetched.metadata.metadata.get(meta_key) == meta_value
        finally:
            self._safe_delete(client, key)

    def test_upload_overwrites_existing(self, client):
        """Uploading to the same key overwrites the content."""
        key = f"{self.TEST_PREFIX}upload/overwrite.txt"
        try:
            client.upload(key, b"version1")
            client.upload(key, b"version2")
            resp = client.download(key)
            assert resp.content == b"version2"
        finally:
            self._safe_delete(client, key)

    # Upload checksum conformance tests
    #
    # The provider validates the caller's digest against the bytes it stores
    # and reports its own digest back; a mismatch must reject the write.

    def test_upload_with_md5_checksum_roundtrips(self, client):
        """A correct MD5 is accepted and echoed back as the server's digest."""
        key = f"{self.TEST_PREFIX}upload/checksum-md5.txt"
        payload = b"md5 checksum conformance payload"
        expected = base64.b64encode(hashlib.md5(payload).digest()).decode()
        try:
            resp = client.upload(
                key,
                payload,
                checksum_value=expected,
                checksum_algorithm=ChecksumAlgorithm.MD5,
            )
            assert resp.checksum_algorithm is ChecksumAlgorithm.MD5
            assert resp.checksum_value == expected
            assert client.download(key).content == payload
        finally:
            self._safe_delete(client, key)

    def test_upload_with_crc32c_checksum_roundtrips(self, client):
        """A correct CRC32C, named as a string, is accepted and echoed back."""
        key = f"{self.TEST_PREFIX}upload/checksum-crc32c.txt"
        payload = b"crc32c checksum conformance payload"
        expected = _crc32c_b64(payload)
        try:
            resp = client.upload(
                key, payload, checksum_value=expected, checksum_algorithm="crc32c"
            )
            assert resp.checksum_algorithm is ChecksumAlgorithm.CRC32C
            assert resp.checksum_value == expected
            assert client.download(key).content == payload
        finally:
            self._safe_delete(client, key)

    def test_upload_checksum_mismatch_rejected(self, client):
        """A well-formed digest of different bytes fails and writes nothing."""
        key = f"{self.TEST_PREFIX}upload/checksum-mismatch.txt"
        wrong = base64.b64encode(hashlib.md5(b"other bytes").digest()).decode()
        try:
            assert client.does_object_exist(key) is False
            with pytest.raises(
                InvalidArgumentException,
                match=f"upload: MD5 checksum mismatch for key '{key}'",
            ):
                client.upload(
                    key,
                    b"checksum mismatch payload",
                    checksum_value=wrong,
                    checksum_algorithm=ChecksumAlgorithm.MD5,
                )
            assert client.does_object_exist(key) is False
        finally:
            self._safe_delete(client, key)

    # Conditional-write conformance tests
    #
    # create_if_absent is supported on every provider. ETag- and
    # version-conditioned overwrites are asymmetric across clouds (one provider
    # enforces with an ETag, another with a version id), so those tests are
    # gated on a harness capability hook that each provider must declare.

    def test_create_if_absent(self, client):
        """create_if_absent writes a new key and rejects an existing one.

        The first write creates the object; a second create_if_absent against
        the now-existing key raises ResourceExistsException and leaves the
        original content intact.
        """
        key = f"{self.TEST_PREFIX}conditional/create-if-absent.txt"
        try:
            client.upload(key, b"fresh", create_if_absent=True)
            assert client.download(key).content == b"fresh"
            with pytest.raises(ResourceExistsException):
                client.upload(key, b"replacement", create_if_absent=True)
            # The failed conditional write must not have replaced the object.
            assert client.download(key).content == b"fresh"
        finally:
            self._safe_delete(client, key)

    def test_if_match_overwrite_and_stale(self, client, harness):
        """if_match overwrites on the current ETag and fails on a stale one.

        Overwriting with the object's current ETag succeeds; reusing that now
        outdated ETag against the moved-forward object raises
        FailedPreconditionException.
        """
        if not harness.supports_write_if_match():
            pytest.skip("Provider does not enforce ETag preconditions on writes.")
        key = f"{self.TEST_PREFIX}conditional/if-match.txt"
        try:
            client.upload(key, b"v1")
            first_etag = client.get_metadata(key).e_tag
            client.upload(key, b"v2", if_match=first_etag)
            assert client.download(key).content == b"v2"
            # first_etag is now stale: the object has moved on to v2.
            with pytest.raises(FailedPreconditionException):
                client.upload(key, b"v3", if_match=first_etag)
        finally:
            self._safe_delete(client, key)

    def test_if_match_on_missing_object_raises_failed_precondition(
        self, client, harness
    ):
        """if_match against a missing key raises FailedPreconditionException.

        A missing object cannot satisfy an ETag precondition; every provider
        that enforces if_match normalizes this to the same precondition-failed
        outcome rather than a not-found error.
        """
        if not harness.supports_write_if_match():
            pytest.skip("Provider does not enforce ETag preconditions on writes.")
        key = f"{self.TEST_PREFIX}conditional/if-match-missing.txt"
        with pytest.raises(FailedPreconditionException):
            client.upload(key, b"data", if_match='"nonexistent-etag"')

    def test_if_version_match_overwrite_and_stale(self, client, harness):
        """if_version_match overwrites on the current version and fails on a stale one.

        Overwriting with the object's current version id succeeds; reusing that
        now outdated version against the moved-forward object raises
        FailedPreconditionException.
        """
        if not harness.supports_write_if_version_match():
            pytest.skip("Provider does not enforce version preconditions on writes.")
        key = f"{self.TEST_PREFIX}conditional/if-version.txt"
        try:
            first_version = client.upload(key, b"v1").version_id
            client.upload(key, b"v2", if_version_match=first_version)
            assert client.download(key).content == b"v2"
            # first_version is now stale: the object has moved on to v2.
            with pytest.raises(FailedPreconditionException):
                client.upload(key, b"v3", if_version_match=first_version)
        finally:
            self._safe_delete(client, key)

    def test_if_version_match_on_missing_object_raises_failed_precondition(
        self, client, harness
    ):
        """if_version_match against a missing key raises FailedPreconditionException.

        A missing object cannot satisfy a version precondition; every provider
        that enforces if_version_match normalizes this to the same
        precondition-failed outcome rather than a not-found error.
        """
        if not harness.supports_write_if_version_match():
            pytest.skip("Provider does not enforce version preconditions on writes.")
        key = f"{self.TEST_PREFIX}conditional/if-version-missing.txt"
        with pytest.raises(FailedPreconditionException):
            client.upload(key, b"data", if_version_match="1")

    # Server-side-encryption conformance tests
    #
    # The cloud-agnostic metadata surface does not expose which key encrypted an
    # object, so the strongest portable assertion is a round-trip: a request
    # that names a key the caller cannot use, or that the provider rejects,
    # fails the write outright. The granular "this exact key/header was applied"
    # assertion lives in each provider's unit tests.

    def test_upload_with_kms_key_roundtrips(self, client, kms_key_id):
        """upload with a server-side-encryption key round-trips the content."""
        key = f"{self.TEST_PREFIX}encryption/kms-upload.txt"
        payload = b"kms-encrypted conformance payload"
        try:
            client.upload(key, payload, kms_key_id=kms_key_id)
            assert client.download(key).content == payload
        finally:
            self._safe_delete(client, key)

    def test_upload_from_path_with_kms_key_roundtrips(
        self, client, kms_key_id, tmp_path
    ):
        """upload_from_path with a server-side-encryption key round-trips."""
        key = f"{self.TEST_PREFIX}encryption/kms-upload-from-path.txt"
        payload = b"kms-encrypted from-path payload"
        source = tmp_path / "kms-source.txt"
        source.write_bytes(payload)
        try:
            client.upload_from_path(key, str(source), kms_key_id=kms_key_id)
            assert client.download(key).content == payload
        finally:
            self._safe_delete(client, key)

    def test_upload_nested_path(self, client):
        """Upload to a deeply nested path and verify the content round-trips."""
        key = f"{self.TEST_PREFIX}upload/a/b/c/d/e/deep.txt"
        payload = b"deeply nested"
        try:
            resp = client.upload(key, payload)
            assert resp.key == key
            fetched = client.download(key)
            assert fetched.content == payload
        finally:
            self._safe_delete(client, key)

    def test_upload_binary_io_stream(self, client):
        """Upload from an in-memory binary stream (BinaryIO)."""
        key = f"{self.TEST_PREFIX}upload/from-stream.bin"
        payload = b"stream-uploaded payload " * 32
        stream = io.BytesIO(payload)
        try:
            resp = client.upload(key, stream, content_type="application/octet-stream")
            assert resp.key == key
            fetched = client.download(key)
            assert fetched.content == payload
        finally:
            self._safe_delete(client, key)

    def test_upload_empty_key_rejected(self, client):
        """Uploading with an empty key raises a validation error."""
        with pytest.raises(InvalidArgumentException):
            client.upload("", b"payload")

    # upload_from_path conformance tests

    def test_upload_from_path_simple(self, client, tmp_path):
        """upload_from_path uploads a local file and content round-trips."""
        key = f"{self.TEST_PREFIX}upload-from-path/simple.txt"
        local = tmp_path / "simple.txt"
        payload = b"upload_from_path conformance test"
        local.write_bytes(payload)
        try:
            resp = client.upload_from_path(key, local)
            assert resp.key == key
            fetched = client.download(key)
            assert fetched.content == payload
        finally:
            self._safe_delete(client, key)

    def test_upload_from_path_auto_content_type(self, client, tmp_path):
        """upload_from_path infers content_type from the file extension."""
        key = f"{self.TEST_PREFIX}upload-from-path/auto-type.txt"
        local = tmp_path / "auto-type.txt"
        local.write_bytes(b"plain text body")
        try:
            client.upload_from_path(key, local)
            fetched = client.download(key)
            # ``.txt`` should map to a text/* content type across providers.
            assert (fetched.metadata.content_type or "").startswith("text/")
        finally:
            self._safe_delete(client, key)

    def test_upload_from_path_missing_file_rejected(self, client, tmp_path):
        """upload_from_path with a non-existent path raises validation error."""
        key = f"{self.TEST_PREFIX}upload-from-path/does-not-matter.txt"
        missing = tmp_path / "definitely-not-here.txt"
        with pytest.raises(InvalidArgumentException):
            client.upload_from_path(key, missing)

    # Download conformance tests

    def test_download_simple(self, client):
        """Upload then download; verify content matches."""
        key = f"{self.TEST_PREFIX}download/simple.txt"
        payload = b"download conformance test content"
        try:
            client.upload(key, payload, content_type="text/plain")
            resp = client.download(key)
            assert resp.content == payload
        finally:
            self._safe_delete(client, key)

    def test_download_metadata_populated(self, client):
        """Download response includes correct metadata."""
        key = f"{self.TEST_PREFIX}download/with-meta.txt"
        payload = b"metadata download test"
        try:
            client.upload(
                key,
                payload,
                content_type="text/plain",
                metadata={"test-key": "test-value"},
            )
            resp = client.download(key)
            assert resp.metadata is not None
            assert resp.metadata.content_type == "text/plain"
            assert resp.metadata.content_length == len(payload)
        finally:
            self._safe_delete(client, key)

    def test_download_user_metadata_round_trip(self, client):
        """User-defined metadata set on upload is returned on download."""
        key = f"{self.TEST_PREFIX}download/user-meta.txt"
        user_metadata = {"author": "conformance", "purpose": "round-trip"}
        try:
            client.upload(
                key, b"hello", content_type="text/plain", metadata=user_metadata
            )
            resp = client.download(key)
            for k, v in user_metadata.items():
                assert resp.metadata.metadata.get(k) == v
        finally:
            self._safe_delete(client, key)

    def test_download_nonexistent_key(self, client):
        """Downloading a non-existent key raises ResourceNotFoundException."""
        key = f"{self.TEST_PREFIX}download/does-not-exist-xyz123.txt"
        with pytest.raises(ResourceNotFoundException):
            client.download(key)

    def test_download_empty_blob(self, client):
        """Download a zero-byte blob."""
        key = f"{self.TEST_PREFIX}download/empty.txt"
        try:
            client.upload(key, b"")
            resp = client.download(key)
            assert resp.content == b""
        finally:
            self._safe_delete(client, key)

    def test_download_binary_content(self, client):
        """Download binary content preserves all bytes."""
        key = f"{self.TEST_PREFIX}download/binary.bin"
        payload = bytes(range(256))
        try:
            client.upload(key, payload, content_type="application/octet-stream")
            resp = client.download(key)
            assert resp.content == payload
        finally:
            self._safe_delete(client, key)

    def test_download_large_blob(self, client):
        """Download a larger blob (64KB) to verify streaming works."""
        key = f"{self.TEST_PREFIX}download/large.bin"
        payload = b"Y" * (64 * 1024)
        try:
            client.upload(key, payload, content_type="application/octet-stream")
            resp = client.download(key)
            assert len(resp.content) == 64 * 1024
            assert resp.content == payload
        finally:
            self._safe_delete(client, key)

    # get_metadata conformance tests
    #
    # get_metadata is a HEAD-style lookup: it returns BlobMetadata without
    # transferring the object body. Assertions stay cloud-agnostic — we check
    # observable properties every provider must satisfy (size, content-type,
    # user metadata, presence of an e_tag and timestamps) and avoid
    # provider-specific field formats (e.g. S3 opaque version ids vs GCS
    # numeric generations), which belong in the provider unit tests.

    def test_get_metadata_returns_object_metadata(self, client):
        """get_metadata returns size/content-type/user-metadata for an object."""
        key = f"{self.TEST_PREFIX}get-metadata/simple.txt"
        payload = b"get_metadata conformance payload"
        user_metadata = {"author": "conformance", "purpose": "get-metadata"}
        try:
            client.upload(
                key,
                payload,
                content_type="text/plain",
                metadata=user_metadata,
            )
            meta = client.get_metadata(key)
            assert meta.key == key
            assert meta.content_length == len(payload)
            assert meta.content_type == "text/plain"
            for meta_key, meta_value in user_metadata.items():
                assert meta.metadata.get(meta_key) == meta_value
            # Every provider populates an e_tag and both timestamps on a
            # successful metadata read (last_modified/created_time were unified
            # into the shared mapping — see PR #98).
            assert meta.e_tag is not None
            assert meta.last_modified is not None
            assert meta.created_time is not None
        finally:
            self._safe_delete(client, key)

    def test_get_metadata_nonexistent_key(self, client):
        """get_metadata on a missing key raises ResourceNotFoundException."""
        key = f"{self.TEST_PREFIX}get-metadata/does-not-exist-xyz123.txt"
        with pytest.raises(ResourceNotFoundException):
            client.get_metadata(key)

    # does_object_exist conformance tests
    #
    # Missing is a False return, not an exception -- unlike get_metadata above.
    # These pin that distinction and the object lifecycle.

    def test_does_object_exist_true_for_existing_object(self, client):
        """An uploaded object is reported as existing, and only that exact key.

        The two negative assertions matter as much as the positive one: a key
        that is a strict prefix of a live object is not itself an object, so an
        implementation answering existence from a prefix listing rather than a
        per-key lookup would report both of them present.
        """
        key = f"{self.TEST_PREFIX}exists/present.txt"
        try:
            # Establish the precondition so the True below is known to come
            # from this upload rather than residue from an earlier run.
            assert client.does_object_exist(key) is False
            client.upload(key, b"exists conformance payload")
            assert client.does_object_exist(key) is True
            assert client.does_object_exist(key[: -len(".txt")]) is False
            assert client.does_object_exist(f"{self.TEST_PREFIX}exists/") is False
        finally:
            self._safe_delete(client, key)

    def test_does_object_exist_false_for_nonexistent_key(self, client):
        """A key that was never written is reported as absent, not an error."""
        key = f"{self.TEST_PREFIX}exists/never-written-xyz789.txt"
        assert client.does_object_exist(key) is False

    def test_does_object_exist_tracks_object_lifecycle(self, client):
        """Existence flips False -> True -> False across upload and delete.

        Asserting the full transition in one test is what makes this
        meaningful: an implementation hard-coded to return True would pass a
        present-object check, and one hard-coded to False would pass an
        absent-object check. Neither survives the round trip.
        """
        key = f"{self.TEST_PREFIX}exists/lifecycle.txt"
        try:
            assert client.does_object_exist(key) is False
            client.upload(key, b"now it exists")
            assert client.does_object_exist(key) is True
            client.delete(key)
            assert client.does_object_exist(key) is False
        finally:
            self._safe_delete(client, key)

    def test_does_object_exist_empty_key_rejected(self, client):
        """An empty key is rejected client-side rather than queried."""
        with pytest.raises(InvalidArgumentException):
            client.does_object_exist("")

    def test_does_object_exist_by_version_id(self, versioned_client):
        """Each version of an object is individually reported as existing.

        The version id must come from the upload responses -- a literal
        cannot be used, because the provider-native format differs by cloud
        (an opaque token versus a numeric generation) and a malformed value
        is not portably a "missing" answer.
        """
        client = versioned_client
        key = f"{self.TEST_PREFIX}exists/versioned.txt"
        other_key = f"{self.TEST_PREFIX}exists/versioned-other.txt"
        try:
            first = client.upload(key, b"version one")
            second = client.upload(key, b"version two")
            assert first.version_id, "Provider must return version_id."
            assert second.version_id, "Provider must return version_id."
            assert first.version_id != second.version_id

            assert client.does_object_exist(key, version_id=first.version_id) is True
            assert client.does_object_exist(key, version_id=second.version_id) is True
            # Omitting the version resolves to the latest, which still exists.
            assert client.does_object_exist(key) is True

            # A version belonging to a different object must not resolve for
            # this key. Without this the test asserts only True values and so
            # cannot fail an implementation that ignores version_id entirely.
            other = client.upload(other_key, b"unrelated object")
            assert other.version_id, "Provider must return version_id."
            assert client.does_object_exist(key, version_id=other.version_id) is False
        finally:
            self._safe_delete_all_versions(client, key)
            self._safe_delete_all_versions(client, other_key)

    def test_does_object_exist_false_for_deleted_version(self, versioned_client):
        """A version that existed and was purged is reported as absent.

        Built by upload -> capture -> delete that version, so the id is
        guaranteed to have been well-formed and to be genuinely gone. The
        key itself survives via its newer version, which keeps this test
        about version resolution rather than key absence.
        """
        client = versioned_client
        key = f"{self.TEST_PREFIX}exists/deleted-version.txt"
        try:
            first = client.upload(key, b"doomed version")
            client.upload(key, b"surviving version")
            assert first.version_id, "Provider must return version_id."

            client.delete(key, version_id=first.version_id)

            assert client.does_object_exist(key, version_id=first.version_id) is False
            # The key is still present through its remaining version.
            assert client.does_object_exist(key) is True
        finally:
            self._safe_delete_all_versions(client, key)

    # does_bucket_exist conformance tests
    #
    # It reports on the bucket the client was built with and takes no argument,
    # so the two cases need two clients. Only the not-found path is asserted:
    # reaching permission-denied needs a bucket that exists but is invisible,
    # which this setup does not provision.

    def test_does_bucket_exist_true_for_configured_bucket(self, client):
        """The bucket the client is configured against is reported present."""
        assert client.does_bucket_exist() is True

    def test_does_bucket_exist_false_for_absent_bucket(self, harness):
        """A client bound to a nonexistent bucket reports absent, not an error.

        Paired with the test above, this is what fails an implementation
        hard-coded either way -- neither test alone can.
        """
        absent = harness.create_client_for_absent_bucket()
        try:
            assert absent.does_bucket_exist() is False
        finally:
            absent.close()

    # Delete conformance tests

    def test_delete_single(self, client):
        """Delete an existing blob; it is no longer downloadable."""
        key = f"{self.TEST_PREFIX}delete/single.txt"
        client.upload(key, b"delete me", content_type="text/plain")
        client.delete(key)
        with pytest.raises(ResourceNotFoundException):
            client.download(key)

    def test_delete_nonexistent_no_error(self, client):
        """Deleting a non-existent blob is idempotent (no error)."""
        key = f"{self.TEST_PREFIX}delete/never-existed-xyz789.txt"
        # Should return silently rather than raise.
        assert client.delete(key) is None

    def test_delete_idempotent_subsequent(self, client):
        """A second delete of the same key is a no-op (no error)."""
        key = f"{self.TEST_PREFIX}delete/idempotent.txt"
        client.upload(key, b"once")
        client.delete(key)
        # Subsequent deletes must not raise.
        assert client.delete(key) is None
        with pytest.raises(ResourceNotFoundException):
            client.download(key)

    def test_delete_objects_batch(self, client):
        """Batch delete removes every requested object; none downloadable."""
        keys = [f"{self.TEST_PREFIX}delete/batch-{i}.txt" for i in range(3)]
        try:
            for key in keys:
                client.upload(key, b"batch payload")
            resp = client.delete_objects([BlobIdentifier(key=k) for k in keys])
            assert resp.errors == []
            for key in keys:
                with pytest.raises(ResourceNotFoundException):
                    client.download(key)
        finally:
            self._safe_delete(client, *keys)

    def test_delete_objects_empty_collection_raises(self, client):
        """An empty batch is rejected with InvalidArgumentException."""
        with pytest.raises(InvalidArgumentException):
            client.delete_objects([])

    def test_delete_objects_all_nonexistent(self, client):
        """Batch delete of only never-existed keys returns no errors."""
        keys = [
            f"{self.TEST_PREFIX}delete/batch-missing-only-a.txt",
            f"{self.TEST_PREFIX}delete/batch-missing-only-b.txt",
            f"{self.TEST_PREFIX}delete/batch-missing-only-c.txt",
        ]
        resp = client.delete_objects([BlobIdentifier(key=k) for k in keys])
        assert resp.errors == []
        for key in keys:
            with pytest.raises(ResourceNotFoundException):
                client.download(key)

    def test_delete_objects_nonexistent_and_duplicates(self, client):
        """Batch delete tolerates non-existent keys and duplicate entries."""
        existing = f"{self.TEST_PREFIX}delete/batch-existing.txt"
        missing = f"{self.TEST_PREFIX}delete/batch-missing-abc.txt"
        try:
            client.upload(existing, b"present")
            identifiers = [
                BlobIdentifier(key=existing),
                BlobIdentifier(key=existing),  # duplicate
                BlobIdentifier(key=missing),  # never existed
            ]
            resp = client.delete_objects(identifiers)
            assert resp.errors == []
            with pytest.raises(ResourceNotFoundException):
                client.download(existing)
        finally:
            self._safe_delete(client, existing)

    def test_delete_specific_version_others_remain(self, client):
        """Deleting one version leaves the other versions intact."""
        key = f"{self.TEST_PREFIX}delete/versioned-specific.txt"
        try:
            first = client.upload(key, b"version-one")
            second = client.upload(key, b"version-two")
            assert first.version_id is not None
            assert second.version_id is not None
            assert first.version_id != second.version_id

            client.delete(key, version_id=first.version_id)

            remaining = {info.version_id for info in client.list_blob_versions(key)}
            assert second.version_id in remaining
            assert first.version_id not in remaining

            # The surviving version is still downloadable.
            resp = client.download(key, version_id=second.version_id)
            assert resp.content == b"version-two"
        finally:
            for info in client.list_blob_versions(key):
                self._safe_delete_version(client, key, info.version_id)

    def test_delete_all_versions(self, client):
        """Deleting every version leaves the key with no versions."""
        key = f"{self.TEST_PREFIX}delete/versioned-all.txt"
        try:
            first = client.upload(key, b"v1")
            second = client.upload(key, b"v2")
            for version_id in (first.version_id, second.version_id):
                client.delete(key, version_id=version_id)

            assert list(client.list_blob_versions(key)) == []
        finally:
            for info in client.list_blob_versions(key):
                self._safe_delete_version(client, key, info.version_id)

    def test_delete_objects_versioned_mixed_with_nonexistent(self, client):
        """Bulk versioned delete handles real versions mixed with a bad key."""
        key = f"{self.TEST_PREFIX}delete/versioned-bulk.txt"
        missing = f"{self.TEST_PREFIX}delete/versioned-bulk-missing.txt"
        try:
            first = client.upload(key, b"bulk-v1")
            second = client.upload(key, b"bulk-v2")
            identifiers = [
                BlobIdentifier(key=key, version_id=first.version_id),
                BlobIdentifier(key=key, version_id=second.version_id),
                BlobIdentifier(key=missing),  # never existed
            ]
            resp = client.delete_objects(identifiers)
            assert resp.errors == []
            assert list(client.list_blob_versions(key)) == []
        finally:
            for info in client.list_blob_versions(key):
                self._safe_delete_version(client, key, info.version_id)

    # Copy conformance tests

    def test_copy_simple(self, client, harness):
        """Copy an object within the same bucket and verify content matches."""
        src_key = f"{self.TEST_PREFIX}copy/src-simple.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-simple.txt"
        payload = b"copy conformance payload"
        try:
            client.upload(src_key, payload, content_type="text/plain")
            resp = client.copy(
                src_key,
                dest_bucket=harness.get_bucket(),
                dest_key=dest_key,
            )
            assert resp.key == dest_key
            assert resp.e_tag is not None
            downloaded = client.download(dest_key)
            assert downloaded.content == payload
        finally:
            self._safe_delete(client, src_key, dest_key)

    def test_copy_preserves_content_metadata(self, client, harness):
        """Copied object has the same content-type and content-length as source."""
        src_key = f"{self.TEST_PREFIX}copy/src-meta.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-meta.txt"
        payload = b"metadata preservation test"
        try:
            client.upload(src_key, payload, content_type="text/plain")
            client.copy(
                src_key,
                dest_bucket=harness.get_bucket(),
                dest_key=dest_key,
            )
            src_download = client.download(src_key)
            dest_download = client.download(dest_key)
            assert dest_download.metadata is not None
            assert src_download.metadata is not None
            assert (
                dest_download.metadata.content_type
                == src_download.metadata.content_type
            )
            assert (
                dest_download.metadata.content_length
                == src_download.metadata.content_length
            )
        finally:
            self._safe_delete(client, src_key, dest_key)

    def test_copy_overwrites_existing_destination(self, client, harness):
        """Copying to an existing key overwrites the destination content."""
        src_key = f"{self.TEST_PREFIX}copy/src-overwrite.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-overwrite.txt"
        src_payload = b"source content wins"
        try:
            client.upload(src_key, src_payload, content_type="text/plain")
            client.upload(dest_key, b"stale destination content")
            resp = client.copy(
                src_key,
                dest_bucket=harness.get_bucket(),
                dest_key=dest_key,
            )
            assert resp.key == dest_key
            assert client.download(dest_key).content == src_payload
        finally:
            self._safe_delete(client, src_key, dest_key)

    def test_copy_nonexistent_source_raises(self, client, harness):
        """Copying from a non-existent source key raises ResourceNotFoundException."""
        missing_key = f"{self.TEST_PREFIX}copy/does-not-exist-abc123.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-missing-src.txt"
        with pytest.raises(ResourceNotFoundException):
            client.copy(
                missing_key,
                dest_bucket=harness.get_bucket(),
                dest_key=dest_key,
            )

    def test_copy_nonexistent_destination_bucket_raises(self, client):
        """Copying to a non-existent destination bucket raises an SDK exception."""
        src_key = f"{self.TEST_PREFIX}copy/src-bad-dest-bucket.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-bad-bucket.txt"
        bogus_bucket = "multicloud-conformance-nonexistent-bucket-xyz-9f8e7d"
        try:
            client.upload(src_key, b"payload", content_type="text/plain")
            with pytest.raises(SubstrateSdkException):
                client.copy(
                    src_key,
                    dest_bucket=bogus_bucket,
                    dest_key=dest_key,
                )
        finally:
            self._safe_delete(client, src_key)

    # CopyFrom Conformance Tests

    def test_copy_from_simple(self, client, harness):
        """copy_from a source bucket into this bucket; verify content matches."""
        src_key = f"{self.TEST_PREFIX}copyfrom/src-simple.txt"
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-simple.txt"
        payload = b"copy_from conformance payload"
        try:
            client.upload(src_key, payload, content_type="text/plain")
            resp = client.copy_from(
                harness.get_bucket(),
                src_key,
                dest_key=dest_key,
            )
            assert resp.key == dest_key
            assert resp.e_tag is not None
            assert client.download(dest_key).content == payload
        finally:
            self._safe_delete(client, src_key, dest_key)

    def test_copy_from_preserves_content_metadata(self, client, harness):
        """copy_from destination has the same content-type and content-length as source."""
        src_key = f"{self.TEST_PREFIX}copyfrom/src-meta.txt"
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-meta.txt"
        payload = b"copy_from metadata preservation test"
        try:
            client.upload(src_key, payload, content_type="text/plain")
            client.copy_from(
                harness.get_bucket(),
                src_key,
                dest_key=dest_key,
            )
            src_download = client.download(src_key)
            dest_download = client.download(dest_key)
            assert dest_download.metadata is not None
            assert src_download.metadata is not None
            assert (
                dest_download.metadata.content_type
                == src_download.metadata.content_type
            )
            assert (
                dest_download.metadata.content_length
                == src_download.metadata.content_length
            )
        finally:
            self._safe_delete(client, src_key, dest_key)

    def test_copy_from_overwrites_existing_destination(self, client, harness):
        """copy_from to an existing key overwrites the destination content."""
        src_key = f"{self.TEST_PREFIX}copyfrom/src-overwrite.txt"
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-overwrite.txt"
        src_payload = b"copy_from source content wins"
        try:
            client.upload(src_key, src_payload, content_type="text/plain")
            client.upload(dest_key, b"stale destination content")
            resp = client.copy_from(
                harness.get_bucket(),
                src_key,
                dest_key=dest_key,
            )
            assert resp.key == dest_key
            assert client.download(dest_key).content == src_payload
        finally:
            self._safe_delete(client, src_key, dest_key)

    def test_copy_from_nonexistent_source_raises(self, client, harness):
        """copy_from with a non-existent source key raises ResourceNotFoundException."""
        missing_key = f"{self.TEST_PREFIX}copyfrom/does-not-exist-abc123.txt"
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-missing-src.txt"
        with pytest.raises(ResourceNotFoundException):
            client.copy_from(
                harness.get_bucket(),
                missing_key,
                dest_key=dest_key,
            )

    def test_copy_from_nonexistent_source_bucket_raises(self, client):
        """copy_from with a non-existent source bucket raises an SDK exception."""
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-bad-src-bucket.txt"
        bogus_bucket = "multicloud-conformance-nonexistent-bucket-xyz-9f8e7d"
        with pytest.raises(SubstrateSdkException):
            client.copy_from(
                bogus_bucket,
                "any-key",
                dest_key=dest_key,
            )

    # Versioned Copy Conformance Tests (require a versioning-enabled bucket)

    def test_copy_specific_version(self, versioned_client, harness):
        """Copying a specific src_version_id yields that version's content."""
        dest_bucket = harness.get_versioned_bucket()
        src_key = f"{self.TEST_PREFIX}copy-versioned/src-specific.txt"
        dest_key = f"{self.TEST_PREFIX}copy-versioned/dest-specific-v1.txt"
        v1_payload = b"version one content"
        v2_payload = b"version two content"
        try:
            v1 = versioned_client.upload(src_key, v1_payload, content_type="text/plain")
            versioned_client.upload(src_key, v2_payload, content_type="text/plain")
            assert v1.version_id is not None
            versioned_client.copy(
                src_key,
                dest_bucket=dest_bucket,
                dest_key=dest_key,
                src_version_id=v1.version_id,
            )
            assert versioned_client.download(dest_key).content == v1_payload
        finally:
            self._safe_delete_all_versions(versioned_client, src_key)
            self._safe_delete_all_versions(versioned_client, dest_key)

    def test_copy_latest_version_when_version_id_none(self, versioned_client, harness):
        """Copy without src_version_id uses the latest version of the source."""
        dest_bucket = harness.get_versioned_bucket()
        src_key = f"{self.TEST_PREFIX}copy-versioned/src-latest.txt"
        dest_key = f"{self.TEST_PREFIX}copy-versioned/dest-latest.txt"
        v1_payload = b"version one content"
        v2_payload = b"version two content"
        try:
            versioned_client.upload(src_key, v1_payload, content_type="text/plain")
            versioned_client.upload(src_key, v2_payload, content_type="text/plain")
            versioned_client.copy(
                src_key,
                dest_bucket=dest_bucket,
                dest_key=dest_key,
            )
            assert versioned_client.download(dest_key).content == v2_payload
        finally:
            self._safe_delete_all_versions(versioned_client, src_key)
            self._safe_delete_all_versions(versioned_client, dest_key)

    def test_copy_response_includes_version_id_on_versioned_bucket(
        self, versioned_client, harness
    ):
        """Copy into a versioned bucket returns a non-null version_id on the response."""
        dest_bucket = harness.get_versioned_bucket()
        src_key = f"{self.TEST_PREFIX}copy-versioned/src-response.txt"
        dest_key = f"{self.TEST_PREFIX}copy-versioned/dest-response.txt"
        try:
            versioned_client.upload(src_key, b"payload", content_type="text/plain")
            resp = versioned_client.copy(
                src_key,
                dest_bucket=dest_bucket,
                dest_key=dest_key,
            )
            assert resp.version_id is not None
        finally:
            self._safe_delete_all_versions(versioned_client, src_key)
            self._safe_delete_all_versions(versioned_client, dest_key)

    # Ranged read conformance tests
    # Range bounds are inclusive on both ends: start=0, end=0 is "first byte".

    def test_download_range_first_bytes(self, client):
        """Download the first N bytes of an object."""
        key = f"{self.TEST_PREFIX}download-range/first.txt"
        payload = b"0123456789ABCDEF"
        try:
            client.upload(key, payload)
            resp = client.download(key, start=0, end=3)
            assert resp.content == b"0123"
        finally:
            self._safe_delete(client, key)

    def test_download_range_middle(self, client):
        """Download a middle byte range."""
        key = f"{self.TEST_PREFIX}download-range/middle.txt"
        payload = b"0123456789ABCDEF"
        try:
            client.upload(key, payload)
            resp = client.download(key, start=4, end=9)
            assert resp.content == b"456789"
        finally:
            self._safe_delete(client, key)

    def test_download_range_single_byte(self, client):
        """Download exactly one byte (start == end)."""
        key = f"{self.TEST_PREFIX}download-range/single.txt"
        payload = b"0123456789ABCDEF"
        try:
            client.upload(key, payload)
            resp = client.download(key, start=7, end=7)
            assert resp.content == b"7"
        finally:
            self._safe_delete(client, key)

    def test_download_range_open_ended_start(self, client):
        """Range with only ``start`` returns bytes from ``start`` to EOF."""
        key = f"{self.TEST_PREFIX}download-range/open-start.txt"
        payload = b"0123456789ABCDEF"
        try:
            client.upload(key, payload)
            resp = client.download(key, start=10)
            assert resp.content == b"ABCDEF"
        finally:
            self._safe_delete(client, key)

    def test_download_range_invalid_boundaries_rejected(self, client):
        """A range with ``end`` < ``start`` is rejected before any I/O."""
        key = f"{self.TEST_PREFIX}download-range/invalid.txt"
        with pytest.raises(InvalidArgumentException):
            client.download(key, start=10, end=5)

    def test_download_range_negative_boundary_rejected(self, client):
        """A negative range boundary is rejected before any I/O."""
        key = f"{self.TEST_PREFIX}download-range/negative.txt"
        with pytest.raises(InvalidArgumentException):
            client.download(key, start=-1, end=5)

    # download_to_path / download_to_file / download_stream conformance tests

    def test_download_to_path_round_trip(self, client, tmp_path):
        """download_to_path writes the object to a local file."""
        key = f"{self.TEST_PREFIX}download-to-path/simple.txt"
        payload = b"download_to_path payload"
        destination = tmp_path / "out.txt"
        try:
            client.upload(key, payload, content_type="text/plain")
            resp = client.download_to_path(key, destination)
            assert destination.exists()
            assert destination.read_bytes() == payload
            assert resp.metadata.content_length == len(payload)
        finally:
            self._safe_delete(client, key)

    def test_download_to_path_creates_parent_dirs(self, client, tmp_path):
        """download_to_path creates missing parent dirs when requested."""
        key = f"{self.TEST_PREFIX}download-to-path/nested.txt"
        payload = b"nested destination payload"
        destination = tmp_path / "a" / "b" / "c" / "nested.txt"
        try:
            client.upload(key, payload)
            client.download_to_path(key, destination, create_parent_dirs=True)
            assert destination.exists()
            assert destination.read_bytes() == payload
        finally:
            self._safe_delete(client, key)

    def test_download_to_path_missing_parent_dir_without_flag(self, client, tmp_path):
        """Without ``create_parent_dirs`` a missing parent dir raises an SDK error.

        The client layer wraps every exception via ``translate_error``, so the
        raw ``FileNotFoundError`` is re-raised as a ``SubstrateSdkException``
        subclass. Assert against the cloud-agnostic base type rather than a
        specific subclass so providers stay free to classify local filesystem
        errors as they see fit.
        """
        key = f"{self.TEST_PREFIX}download-to-path/missing-parent.txt"
        destination = tmp_path / "does-not-exist" / "out.txt"
        try:
            client.upload(key, b"payload")
            with pytest.raises(SubstrateSdkException):
                client.download_to_path(key, destination)
        finally:
            self._safe_delete(client, key)

    def test_download_to_file_round_trip(self, client):
        """download_to_file writes the object into an in-memory buffer."""
        key = f"{self.TEST_PREFIX}download-to-file/simple.txt"
        payload = b"download_to_file payload"
        buffer = io.BytesIO()
        try:
            client.upload(key, payload, content_type="text/plain")
            resp = client.download_to_file(key, buffer)
            assert buffer.getvalue() == payload
            assert resp.metadata.content_length == len(payload)
        finally:
            self._safe_delete(client, key)

    def test_download_stream_iterates_chunks(self, client):
        """download_stream yields chunks that reassemble to the original."""
        key = f"{self.TEST_PREFIX}download-stream/iter.bin"
        payload = b"X" * (32 * 1024)
        try:
            client.upload(key, payload, content_type="application/octet-stream")
            with client.download_stream(key) as stream:
                collected = b"".join(chunk for chunk in stream)
            assert collected == payload
        finally:
            self._safe_delete(client, key)

    def test_download_stream_read_all(self, client):
        """download_stream.read() returns the full payload."""
        key = f"{self.TEST_PREFIX}download-stream/read.bin"
        payload = b"stream-read payload " * 128
        try:
            client.upload(key, payload, content_type="application/octet-stream")
            with client.download_stream(key) as stream:
                data = stream.read()
            assert data == payload
        finally:
            self._safe_delete(client, key)

    def test_download_stream_range(self, client):
        """download_stream honors range bounds."""
        key = f"{self.TEST_PREFIX}download-stream/range.txt"
        payload = b"0123456789ABCDEF"
        try:
            client.upload(key, payload)
            with client.download_stream(key, start=2, end=7) as stream:
                data = stream.read()
            assert data == b"234567"
        finally:
            self._safe_delete(client, key)

    # Versioned round-trip conformance tests
    # Skipped when the harness has no versioning-enabled bucket.
    #
    # These tests issue two sequential ``upload`` calls to the same key
    # and rely on the second upload producing a strictly-newer version
    # than the first. Both providers guarantee this per-key:
    #   * S3 orders versions by server-side receive time; sequential
    #     PUTs from the same client are strictly ordered.
    #   * GCS assigns a monotonically-increasing ``generation`` number
    #     per object on every overwrite, independent of wall-clock time.
    # So the ordering assumption holds by contract, not by timing.

    @pytest.fixture
    def versioned_client(self, harness):
        versioned = harness.create_versioned_client()
        if versioned is None:
            pytest.skip("Provider harness did not supply a versioning-enabled bucket.")
        return versioned

    def test_versioned_download_returns_latest_by_default(self, versioned_client):
        """Downloading without a version_id returns the most recent version."""
        client = versioned_client
        key = f"{self.TEST_PREFIX}versioned/latest.txt"
        try:
            client.upload(key, b"first")
            client.upload(key, b"second")
            resp = client.download(key)
            assert resp.content == b"second"
        finally:
            self._safe_delete_all_versions(client, key)

    def test_versioned_download_by_version_id(self, versioned_client):
        """Downloading with a specific version_id returns that version."""
        client = versioned_client
        key = f"{self.TEST_PREFIX}versioned/by-id.txt"
        try:
            first = client.upload(key, b"first")
            client.upload(key, b"second")
            assert (
                first.version_id
            ), "Provider must return version_id for versioned bucket uploads."
            resp = client.download(key, version_id=first.version_id)
            assert resp.content == b"first"
        finally:
            self._safe_delete_all_versions(client, key)

    def test_versioned_download_unknown_version_raises(self, versioned_client):
        """A well-formed but no-longer-existing version_id raises ResourceNotFound.

        The provider-native version id format differs by cloud (S3 uses an
        opaque token, GCS uses a numeric generation), so we can't invent
        one that is both well-formed *and* nonexistent. Instead we upload
        twice, capture the older version's id, permanently delete that
        version, then try to download it — guaranteeing the id was valid
        and is now gone.
        """
        client = versioned_client
        key = f"{self.TEST_PREFIX}versioned/unknown-version.txt"
        try:
            first = client.upload(key, b"first")
            client.upload(key, b"second")
            assert first.version_id, "Provider must return version_id."
            client.delete(key, version_id=first.version_id)
            with pytest.raises(ResourceNotFoundException):
                client.download(key, version_id=first.version_id)
        finally:
            self._safe_delete_all_versions(client, key)

    # Every read API takes a version_id, and each one reaches the service by a
    # different call, so covering one says nothing about the others. The two
    # versions below differ in both length and content so an implementation
    # that ignored the argument, or crossed the two versions, is visible in
    # either assertion on its own.

    _V1 = b"first version bytes"
    _V2 = b"second version payload!!"

    def test_versioned_get_metadata_by_version_id(self, versioned_client):
        """get_metadata reports the requested version, not the latest."""
        client = versioned_client
        key = f"{self.TEST_PREFIX}versioned/metadata-by-id.txt"
        try:
            first = client.upload(key, self._V1, metadata={"gen": "one"})
            second = client.upload(key, self._V2, metadata={"gen": "two"})
            assert first.version_id and second.version_id
            assert first.version_id != second.version_id

            older = client.get_metadata(key, version_id=first.version_id)
            newer = client.get_metadata(key, version_id=second.version_id)

            assert older.version_id == first.version_id
            assert older.content_length == len(self._V1)
            assert older.metadata.get("gen") == "one"
            assert newer.version_id == second.version_id
            assert newer.content_length == len(self._V2)
            assert newer.metadata.get("gen") == "two"
        finally:
            self._safe_delete_all_versions(client, key)

    def test_versioned_download_to_path_by_version_id(self, versioned_client, tmp_path):
        """download_to_path writes the requested version to disk."""
        client = versioned_client
        key = f"{self.TEST_PREFIX}versioned/to-path-by-id.txt"
        destination = tmp_path / "older.txt"
        try:
            first = client.upload(key, self._V1)
            client.upload(key, self._V2)
            assert first.version_id

            resp = client.download_to_path(
                key, destination, version_id=first.version_id
            )

            assert destination.read_bytes() == self._V1
            assert resp.metadata.content_length == len(self._V1)
        finally:
            self._safe_delete_all_versions(client, key)

    def test_versioned_download_to_file_by_version_id(self, versioned_client):
        """download_to_file writes the requested version into the buffer."""
        client = versioned_client
        key = f"{self.TEST_PREFIX}versioned/to-file-by-id.txt"
        buffer = io.BytesIO()
        try:
            first = client.upload(key, self._V1)
            client.upload(key, self._V2)
            assert first.version_id

            resp = client.download_to_file(key, buffer, version_id=first.version_id)

            assert buffer.getvalue() == self._V1
            assert resp.metadata.content_length == len(self._V1)
        finally:
            self._safe_delete_all_versions(client, key)

    def test_versioned_download_stream_by_version_id(self, versioned_client):
        """download_stream streams the requested version."""
        client = versioned_client
        key = f"{self.TEST_PREFIX}versioned/stream-by-id.txt"
        try:
            first = client.upload(key, self._V1)
            client.upload(key, self._V2)
            assert first.version_id

            with client.download_stream(key, version_id=first.version_id) as stream:
                collected = stream.read()

            assert collected == self._V1
        finally:
            self._safe_delete_all_versions(client, key)

    # list_page conformance tests
    #
    # list_page returns a single page of results. Tests use a unique
    # sub-namespace per test under TEST_PREFIX so a stray object from
    # another test can never leak into a prefix-scoped listing.

    def test_list_page_empty_prefix_returns_nothing(self, client):
        """Listing an empty, never-populated prefix yields no blobs."""
        prefix = f"{self.TEST_PREFIX}list-page/empty-ns/"
        resp = client.list_page(prefix=prefix)
        assert resp.blobs == []
        assert resp.is_truncated is False

    def test_list_page_single_object(self, client):
        """A prefix with one object returns exactly that key with its size."""
        prefix = f"{self.TEST_PREFIX}list-page/single-ns/"
        key = f"{prefix}only.txt"
        payload = b"single object listing"
        try:
            client.upload(key, payload, content_type="text/plain")
            resp = client.list_page(prefix=prefix)
            assert len(resp.blobs) == 1
            assert resp.blobs[0].key == key
            assert resp.blobs[0].size == len(payload)
        finally:
            self._safe_delete(client, key)

    def test_list_page_multiple_objects(self, client):
        """All objects under a prefix are returned."""
        prefix = f"{self.TEST_PREFIX}list-page/multi-ns/"
        keys = [f"{prefix}obj-{i}.txt" for i in range(5)]
        try:
            for key in keys:
                client.upload(key, b"multi listing payload")
            resp = client.list_page(prefix=prefix)
            assert {b.key for b in resp.blobs} == set(keys)
        finally:
            self._safe_delete(client, *keys)

    def test_list_page_max_results_truncates(self, client):
        """max_results caps the page size and flags truncation."""
        prefix = f"{self.TEST_PREFIX}list-page/truncate-ns/"
        keys = [f"{prefix}obj-{i}.txt" for i in range(4)]
        try:
            for key in keys:
                client.upload(key, b"truncation payload")
            resp = client.list_page(prefix=prefix, max_results=2)
            assert len(resp.blobs) == 2
            assert resp.is_truncated is True
        finally:
            self._safe_delete(client, *keys)

    def test_list_page_pagination_round_trip(self, client):
        """Following the pagination token walks the full result set once."""
        prefix = f"{self.TEST_PREFIX}list-page/paginate-ns/"
        keys = [f"{prefix}obj-{i}.txt" for i in range(4)]
        try:
            for key in keys:
                client.upload(key, b"pagination payload")

            first = client.list_page(prefix=prefix, max_results=2)
            assert len(first.blobs) == 2
            assert first.is_truncated is True

            second = client.list_page(
                prefix=prefix,
                max_results=2,
                pagination_token=first.next_pagination_token,
            )
            assert len(second.blobs) == 2

            first_keys = [b.key for b in first.blobs]
            second_keys = [b.key for b in second.blobs]
            # No key appears on both pages, and together they cover everything.
            assert set(first_keys).isdisjoint(second_keys)
            assert set(first_keys) | set(second_keys) == set(keys)
            assert second.is_truncated is False
        finally:
            self._safe_delete(client, *keys)

    def test_list_page_delimiter_groups_common_prefixes(self, client):
        """A delimiter collapses nested keys into common_prefixes."""
        prefix = f"{self.TEST_PREFIX}list-page/delimiter-ns/"
        top_key = f"{prefix}top.txt"
        nested_keys = [f"{prefix}sub/a.txt", f"{prefix}sub/b.txt"]
        try:
            client.upload(top_key, b"top level")
            for key in nested_keys:
                client.upload(key, b"nested")
            resp = client.list_page(prefix=prefix, delimiter="/")
            # The top-level object is listed directly; the "sub/" group is
            # collapsed into a single common prefix rather than two objects.
            assert [b.key for b in resp.blobs] == [top_key]
            assert f"{prefix}sub/" in resp.common_prefixes
        finally:
            self._safe_delete(client, top_key, *nested_keys)

    def test_list_page_prefix_filters_out_others(self, client):
        """Objects outside the requested prefix are excluded from the page."""
        prefix = f"{self.TEST_PREFIX}list-page/filter-ns/"
        included = f"{prefix}included.txt"
        excluded = f"{self.TEST_PREFIX}list-page/other-ns/excluded.txt"
        try:
            client.upload(included, b"in")
            client.upload(excluded, b"out")
            resp = client.list_page(prefix=prefix)
            keys = [b.key for b in resp.blobs]
            assert included in keys
            assert excluded not in keys
        finally:
            self._safe_delete(client, included, excluded)

    def test_list_page_invalid_max_results_rejected(self, client):
        """max_results below 1 is rejected before any I/O."""
        with pytest.raises(InvalidArgumentException):
            client.list_page(max_results=0)

    # list_blob_versions conformance tests (require a versioning-enabled bucket)

    def test_list_blob_versions_returns_all_versions(self, versioned_client):
        """Every uploaded version is listed, with exactly one marked latest."""
        client = versioned_client
        key = f"{self.TEST_PREFIX}list-versions/all.txt"
        try:
            first = client.upload(key, b"v1")
            second = client.upload(key, b"v2")
            assert first.version_id != second.version_id

            versions = list(client.list_blob_versions(key))
            assert len(versions) == 2
            listed_ids = {v.version_id for v in versions}
            assert listed_ids == {first.version_id, second.version_id}
            assert sum(1 for v in versions if v.is_latest) == 1
        finally:
            self._safe_delete_all_versions(client, key)

    def test_list_blob_versions_latest_is_most_recent(self, versioned_client):
        """The version flagged is_latest is the most recently uploaded one."""
        client = versioned_client
        key = f"{self.TEST_PREFIX}list-versions/latest.txt"
        try:
            first = client.upload(key, b"older")
            second = client.upload(key, b"newer")
            assert first.version_id != second.version_id

            latest = [v for v in client.list_blob_versions(key) if v.is_latest]
            assert len(latest) == 1
            assert latest[0].version_id == second.version_id
        finally:
            self._safe_delete_all_versions(client, key)

    def test_list_blob_versions_nonexistent_key_empty(self, versioned_client):
        """Listing versions of a never-existed key yields an empty result."""
        client = versioned_client
        key = f"{self.TEST_PREFIX}list-versions/never-existed-xyz789.txt"
        assert list(client.list_blob_versions(key)) == []

    # Tags conformance tests

    def test_set_tags_then_get_tags_round_trip(self, client):
        """set_tags followed by get_tags returns exactly the tags set."""
        key = f"{self.TEST_PREFIX}tags/roundtrip.txt"
        tags = {"env": "conformance"}
        try:
            client.upload(key, b"tag round-trip payload")
            client.set_tags(key, tags)
            assert client.get_tags(key) == tags
        finally:
            self._safe_delete(client, key)

    def test_get_tags_reflects_upload_time_tags(self, client):
        """get_tags returns the tags supplied at upload time."""
        key = f"{self.TEST_PREFIX}tags/upload-time.txt"
        tags = {"source": "upload", "team": "conformance"}
        try:
            client.upload(key, b"upload-time tags payload", tags=tags)
            assert client.get_tags(key) == tags
        finally:
            self._safe_delete(client, key)

    def test_get_tags_empty_when_no_tags(self, client):
        """get_tags on an object with no tags returns an empty mapping."""
        key = f"{self.TEST_PREFIX}tags/none.txt"
        try:
            client.upload(key, b"no tags payload")
            assert client.get_tags(key) == {}
        finally:
            self._safe_delete(client, key)

    def test_set_tags_replaces_existing_tags(self, client):
        """set_tags replaces all existing tags rather than merging them."""
        key = f"{self.TEST_PREFIX}tags/replace.txt"
        try:
            client.upload(key, b"tag replacement payload")
            client.set_tags(key, {"a": "1"})
            client.set_tags(key, {"b": "2"})
            fetched = client.get_tags(key)
            assert fetched == {"b": "2"}
            assert "a" not in fetched
        finally:
            self._safe_delete(client, key)

    def test_set_tags_multiple_pairs_round_trip(self, client):
        """set_tags with multiple key-value pairs round-trips every pair."""
        key = f"{self.TEST_PREFIX}tags/multi.txt"
        tags = {"env": "conformance", "owner": "sdk", "tier": "gold"}
        try:
            client.upload(key, b"multi-tag payload")
            client.set_tags(key, tags)
            fetched = client.get_tags(key)
            assert fetched == tags
            for tag_key, tag_value in tags.items():
                assert fetched.get(tag_key) == tag_value
        finally:
            self._safe_delete(client, key)

    def test_set_tags_empty_mapping_rejected(self, client):
        """set_tags with an empty mapping is rejected with a validation error."""
        key = f"{self.TEST_PREFIX}tags/empty.txt"
        with pytest.raises(InvalidArgumentException):
            client.set_tags(key, {})

    def test_tags_round_trip_on_object_lock_bucket(self, versioned_client):
        """Tags set/get/replace round-trips on a versioning-enabled bucket.

        A versioning-enabled bucket stores objects differently from a plain
        bucket, so the tagging API is exercised end-to-end against one to prove
        tags are unaffected by the bucket's versioning configuration.
        """
        client = versioned_client
        key = f"{self.TEST_PREFIX}tags/object-lock.txt"
        upload_tags = {"tag1": "value1"}
        replaced_tags = {"tag3": "value3"}
        try:
            client.upload(key, b"tags on versioned bucket", tags=upload_tags)
            assert client.get_tags(key) == upload_tags
            client.set_tags(key, replaced_tags)
            fetched = client.get_tags(key)
            assert fetched == replaced_tags
            assert "tag1" not in fetched
        finally:
            self._safe_delete_all_versions(client, key)

    # Presigned URL conformance tests
    #
    # Signing never contacts the object store, so what gets recorded differs by
    # provider: one signs locally and records nothing, the other signs through a
    # remote credential API whose host these tests opt back into recording.
    # Header names are provider-prefixed, so assertions match on suffix + value.

    @staticmethod
    def _signed_expiry_seconds(url: str) -> int:
        """Return the lifetime, in seconds, the URL was actually signed for.

        Providers prefix the query parameter with their own vendor token, so
        match on the suffix rather than a literal name.
        """
        for name, values in parse_qs(urlparse(url).query).items():
            if name.lower().endswith("expires"):
                return int(values[0])
        raise AssertionError(f"signed URL carries no expiry parameter: {url}")

    def test_generate_presigned_url_upload_shape(self, client):
        """An UPLOAD URL is HTTPS, carries a signature and expiry, and is a PUT."""
        key = f"{self.TEST_PREFIX}presign/upload-shape.txt"
        result = client.generate_presigned_url(
            key, PresignedOperation.UPLOAD, duration_seconds=900
        )
        assert result.url.startswith("https://")
        assert result.http_method == "PUT"
        assert "Signature" in result.url
        assert self._signed_expiry_seconds(result.url) == 900

    def test_generate_presigned_url_download_shape(self, client):
        """A DOWNLOAD URL is a GET and requires no headers of the caller."""
        key = f"{self.TEST_PREFIX}presign/download-shape.txt"
        result = client.generate_presigned_url(
            key, PresignedOperation.DOWNLOAD, duration_seconds=900
        )
        assert result.url.startswith("https://")
        assert result.http_method == "GET"
        assert dict(result.required_headers) == {}
        assert self._signed_expiry_seconds(result.url) == 900

    def test_generate_presigned_url_signs_the_requested_duration(self, client):
        """The requested lifetime reaches the signature rather than a default.

        Two values, because one cannot tell an honored argument from a constant.
        """
        key = f"{self.TEST_PREFIX}presign/duration.txt"
        for duration in (900, 4500):
            result = client.generate_presigned_url(
                key, PresignedOperation.UPLOAD, duration_seconds=duration
            )
            assert self._signed_expiry_seconds(result.url) == duration

    def test_generate_presigned_url_upload_requires_bound_content_type(self, client):
        """A bound content type comes back as a header the caller must send.

        Binding it without reporting it would leave the caller omitting the
        header, and the upload would then fail the signature check.
        """
        key = f"{self.TEST_PREFIX}presign/content-type.txt"
        result = client.generate_presigned_url(
            key,
            PresignedOperation.UPLOAD,
            duration_seconds=900,
            content_type="text/plain",
        )
        assert result.required_headers.get("Content-Type") == "text/plain"

    def test_generate_presigned_url_upload_requires_bound_metadata(self, client):
        """Bound metadata comes back under a provider-prefixed header.

        The prefix differs by provider, so match on the suffix and the value
        rather than on a literal header name.
        """
        key = f"{self.TEST_PREFIX}presign/metadata.txt"
        result = client.generate_presigned_url(
            key,
            PresignedOperation.UPLOAD,
            duration_seconds=900,
            metadata={"owner": "conformance"},
        )
        headers = dict(result.required_headers)
        matching = [k for k in headers if k.lower().endswith("owner")]
        assert matching, f"metadata not surfaced in required_headers: {headers}"
        assert headers[matching[0]] == "conformance"

    def test_generate_presigned_url_upload_without_bindings_requires_no_headers(
        self, client
    ):
        """With nothing bound there is nothing for the caller to replay."""
        key = f"{self.TEST_PREFIX}presign/bare.txt"
        result = client.generate_presigned_url(
            key, PresignedOperation.UPLOAD, duration_seconds=900
        )
        assert dict(result.required_headers) == {}

    def test_generate_presigned_url_explicit_null_kms_key_matches_omitting_it(
        self, client
    ):
        """``kms_key_id=None`` is equivalent to omitting it.

        So None is never taken as a request for default encryption. A real key
        is not covered: providers diverge, so there is no portable assertion.
        """
        key = f"{self.TEST_PREFIX}presign/no-kms.txt"
        explicit_null = client.generate_presigned_url(
            key,
            PresignedOperation.UPLOAD,
            duration_seconds=900,
            kms_key_id=None,
        )
        omitted = client.generate_presigned_url(
            key, PresignedOperation.UPLOAD, duration_seconds=900
        )

        assert explicit_null.http_method == omitted.http_method
        assert dict(explicit_null.required_headers) == dict(omitted.required_headers)
        assert not [
            header
            for header in explicit_null.required_headers
            if "encrypt" in header.lower()
        ], "no encryption header expected when kms_key_id is unset"

    # No presigned upload round-trip: the buckets have object retention, which
    # requires an integrity header that required_headers omits and
    # generate_presigned_url cannot sign. The reference SDK has the same gap.

    def test_presigned_url_is_refused_after_it_expires(self, client):
        """A URL past its lifetime is refused by the service.

        Generation only checks out-of-range values are rejected; this checks an
        accepted value is actually enforced.
        """
        key = f"{self.TEST_PREFIX}presign/expired.txt"
        payload = b"should not be reachable once expired"
        client.upload(key, payload)
        try:
            result = client.generate_presigned_url(
                key, PresignedOperation.DOWNLOAD, duration_seconds=1
            )
            time.sleep(2)

            response = requests.get(result.url)

            assert response.status_code in (400, 403), (
                f"expired URL should be refused, got {response.status_code}: "
                f"{response.text[:200]}"
            )
        finally:
            self._safe_delete(client, key)

    def test_presigned_upload_url_refuses_omitted_required_headers(self, client):
        """Omitting a reported header fails the signature check.

        Makes ``required_headers`` load-bearing rather than advisory.
        """
        key = f"{self.TEST_PREFIX}presign/missing-headers.txt"
        result = client.generate_presigned_url(
            key,
            PresignedOperation.UPLOAD,
            duration_seconds=900,
            metadata={"owner": "conformance"},
        )
        assert result.required_headers, "expected bound metadata to be reported"

        response = requests.put(result.url, data=b"no headers sent")

        assert response.status_code in (400, 403), (
            f"upload omitting the required headers should be refused, got "
            f"{response.status_code}: {response.text[:200]}"
        )

    def test_presigned_download_url_returns_the_object(self, client):
        """A presigned DOWNLOAD URL returns exactly the bytes that were stored."""
        key = f"{self.TEST_PREFIX}presign/download-roundtrip.txt"
        payload = b"fetched through a presigned url"
        client.upload(key, payload)
        try:
            result = client.generate_presigned_url(
                key, PresignedOperation.DOWNLOAD, duration_seconds=900
            )

            response = requests.get(result.url, headers=dict(result.required_headers))

            assert response.status_code == 200, response.text
            assert response.content == payload
        finally:
            self._safe_delete(client, key)

    def test_generate_presigned_url_binds_the_requested_key(self, client):
        """Different keys produce different URLs.

        Without this, an implementation signing a constant path would satisfy
        every other assertion in this section.
        """
        first = client.generate_presigned_url(
            f"{self.TEST_PREFIX}presign/first.txt",
            PresignedOperation.UPLOAD,
            duration_seconds=900,
        )
        second = client.generate_presigned_url(
            f"{self.TEST_PREFIX}presign/second.txt",
            PresignedOperation.UPLOAD,
            duration_seconds=900,
        )
        assert first.url != second.url

    def test_generate_presigned_url_zero_duration_rejected(self, client):
        """A zero lifetime is rejected rather than signed.

        Neither provider SDK validates this, so the rejection is ours.
        """
        with pytest.raises(InvalidArgumentException):
            client.generate_presigned_url(
                f"{self.TEST_PREFIX}presign/zero.txt",
                PresignedOperation.UPLOAD,
                duration_seconds=0,
            )

    def test_generate_presigned_url_negative_duration_rejected(self, client):
        """A negative lifetime is rejected rather than signed."""
        with pytest.raises(InvalidArgumentException):
            client.generate_presigned_url(
                f"{self.TEST_PREFIX}presign/negative.txt",
                PresignedOperation.UPLOAD,
                duration_seconds=-1,
            )

    def test_generate_presigned_url_duration_beyond_maximum_rejected(self, client):
        """A lifetime past the signing-key limit is rejected up front.

        Both schemes cap a signature at seven days, so a longer URL could never
        hold for its stated window; failing at generation beats returning a URL
        that expires earlier than the caller was told.
        """
        with pytest.raises(InvalidArgumentException):
            client.generate_presigned_url(
                f"{self.TEST_PREFIX}presign/too-long.txt",
                PresignedOperation.UPLOAD,
                duration_seconds=7 * 24 * 60 * 60 + 1,
            )

    def test_generate_presigned_url_empty_key_rejected(self, client):
        """An empty key is rejected rather than signed."""
        with pytest.raises(InvalidArgumentException):
            client.generate_presigned_url(
                "", PresignedOperation.UPLOAD, duration_seconds=900
            )

    def test_generate_presigned_url_download_rejects_upload_only_bindings(self, client):
        """Upload-only bindings are rejected on a DOWNLOAD URL.

        Accepting them silently would imply the URL enforces values that a
        download URL cannot.
        """
        key = f"{self.TEST_PREFIX}presign/download-reject.txt"
        for binding in (
            {"metadata": {"owner": "conformance"}},
            {"tags": {"env": "test"}},
            {"kms_key_id": "some-key"},
        ):
            with pytest.raises(InvalidArgumentException):
                client.generate_presigned_url(
                    key,
                    PresignedOperation.DOWNLOAD,
                    duration_seconds=900,
                    **binding,
                )

    def test_generate_presigned_url_upload_requires_bound_content_disposition(
        self, client
    ):
        """A bound content disposition comes back as a caller header.

        Unlike a tag set, the header is named the same by every provider, so
        the name is asserted alongside the value.
        """
        key = f"{self.TEST_PREFIX}presign/upload-disposition.txt"
        disposition = 'attachment; filename="report.csv"'
        result = client.generate_presigned_url(
            key,
            PresignedOperation.UPLOAD,
            duration_seconds=900,
            content_disposition=disposition,
        )
        assert result.required_headers.get("Content-Disposition") == disposition

    def test_generate_presigned_url_download_binds_content_disposition(self, client):
        """A content disposition on a DOWNLOAD URL is signed into the query.

        A download URL binds nothing for the caller to send, so the
        disposition has to travel in the URL itself for the response to carry
        it. Asserting the parameter, not merely that the URLs differ, is what
        distinguishes it reaching the signature from it perturbing it.
        """
        key = f"{self.TEST_PREFIX}presign/download-disposition.txt"
        disposition = 'attachment; filename="report.csv"'
        result = client.generate_presigned_url(
            key,
            PresignedOperation.DOWNLOAD,
            duration_seconds=900,
            content_disposition=disposition,
        )
        query = parse_qs(urlparse(result.url).query)
        assert query.get("response-content-disposition") == [disposition]
        assert dict(result.required_headers) == {}

    # Multipart upload conformance tests
    #
    # Non-final parts are filled to MPU_PART_SIZE because some providers reject
    # a too-small non-final part at completion time; the final (highest) part
    # is exempt and stays small. The large part bodies are truncated in the
    # cassette (see vcr_filters) so committed fixtures stay tiny. Assembled
    # multi-part objects are verified with boundary single-byte ranged
    # downloads instead of a full download so the recorded response bodies are
    # a single byte each.

    def test_multipart_single_part(self, client):
        """A one-part multipart upload assembles into the original bytes."""
        key = f"{self.TEST_PREFIX}multipart/single-part.bin"
        payload = b"single part multipart payload"
        try:
            mpu = client.initiate_multipart_upload(key)
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=payload)
            )
            assert part.part_number == 1
            assert part.e_tag
            result = client.complete_multipart_upload(mpu, [part])
            assert result is not None
            fetched = client.download(key)
            assert fetched.content == payload
        finally:
            self._safe_delete(client, key)

    def test_multipart_multiple_parts(self, client):
        """Multiple parts assemble in ascending part-number order."""
        key = f"{self.TEST_PREFIX}multipart/multiple-parts.bin"
        try:
            mpu = client.initiate_multipart_upload(key)
            p1 = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"\x01" * self.MPU_PART_SIZE)
            )
            p2 = client.upload_part(
                mpu, MultipartPart(part_number=2, content=b"\x02" * 16)
            )
            assert p1.part_number == 1
            assert p2.part_number == 2
            result = client.complete_multipart_upload(mpu, [p1, p2])
            assert result is not None
            assert client.does_object_exist(key) is True
            first = client.download(key, start=0, end=0)
            assert first.content == b"\x01"
            boundary = client.download(
                key, start=self.MPU_PART_SIZE, end=self.MPU_PART_SIZE
            )
            assert boundary.content == b"\x02"
        finally:
            self._safe_delete(client, key)

    def test_multipart_unordered_parts(self, client):
        """Parts uploaded out of order still assemble by part number."""
        key = f"{self.TEST_PREFIX}multipart/unordered-parts.bin"
        try:
            mpu = client.initiate_multipart_upload(key)
            # Upload the higher-numbered part first, then the lower one.
            p2 = client.upload_part(
                mpu, MultipartPart(part_number=2, content=b"\x02" * 16)
            )
            p1 = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"\x01" * self.MPU_PART_SIZE)
            )
            # Pass parts to complete in reverse order; assembly is by number.
            result = client.complete_multipart_upload(mpu, [p2, p1])
            assert result is not None
            first = client.download(key, start=0, end=0)
            assert first.content == b"\x01"
            boundary = client.download(
                key, start=self.MPU_PART_SIZE, end=self.MPU_PART_SIZE
            )
            assert boundary.content == b"\x02"
        finally:
            self._safe_delete(client, key)

    def test_multipart_skipping_part_numbers(self, client):
        """Non-contiguous part numbers assemble in ascending order."""
        key = f"{self.TEST_PREFIX}multipart/skipping-numbers.bin"
        try:
            mpu = client.initiate_multipart_upload(key)
            p1 = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"\x01" * self.MPU_PART_SIZE)
            )
            # Skip part number 2 entirely.
            p3 = client.upload_part(
                mpu, MultipartPart(part_number=3, content=b"\x03" * 16)
            )
            result = client.complete_multipart_upload(mpu, [p1, p3])
            assert result is not None
            first = client.download(key, start=0, end=0)
            assert first.content == b"\x01"
            boundary = client.download(
                key, start=self.MPU_PART_SIZE, end=self.MPU_PART_SIZE
            )
            assert boundary.content == b"\x03"
        finally:
            self._safe_delete(client, key)

    def test_multipart_duplicate_part_number(self, client):
        """Re-uploading the same part number keeps the most recent bytes."""
        key = f"{self.TEST_PREFIX}multipart/duplicate-part.bin"
        try:
            mpu = client.initiate_multipart_upload(key)
            client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"first write")
            )
            second = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"second write wins")
            )
            result = client.complete_multipart_upload(mpu, [second])
            assert result is not None
            fetched = client.download(key)
            assert fetched.content == b"second write wins"
        finally:
            self._safe_delete(client, key)

    def test_multipart_preserves_metadata_and_content_type(self, client):
        """Metadata and content type set at initiate survive to the object."""
        key = f"{self.TEST_PREFIX}multipart/metadata.txt"
        try:
            mpu = client.initiate_multipart_upload(
                key, metadata={"team": "conformance"}, content_type="text/plain"
            )
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"metadata payload")
            )
            client.complete_multipart_upload(mpu, [part])
            fetched = client.download(key)
            assert fetched.content == b"metadata payload"
            assert fetched.metadata.content_type == "text/plain"
            assert fetched.metadata.metadata.get("team") == "conformance"
        finally:
            self._safe_delete(client, key)

    def test_multipart_multiple_uploads_same_key(self, client):
        """Two uploads for one key get distinct ids and stay independent."""
        key = f"{self.TEST_PREFIX}multipart/same-key.bin"
        mpu_a = client.initiate_multipart_upload(key)
        mpu_b = client.initiate_multipart_upload(key)
        try:
            assert mpu_a.upload_id != mpu_b.upload_id
            a_part = client.upload_part(
                mpu_a, MultipartPart(part_number=1, content=b"from upload A")
            )
            client.upload_part(
                mpu_b, MultipartPart(part_number=1, content=b"from upload B")
            )
            # Completing A must not be affected by the still-open B.
            client.complete_multipart_upload(mpu_a, [a_part])
            fetched = client.download(key)
            assert fetched.content == b"from upload A"
        finally:
            self._safe_abort(client, mpu_b)
            self._safe_delete(client, key)

    def test_multipart_abort_discards_parts(self, client):
        """Aborting an upload discards its parts; no object is assembled."""
        key = f"{self.TEST_PREFIX}multipart/abort.bin"
        mpu = client.initiate_multipart_upload(key)
        try:
            client.upload_part(mpu, MultipartPart(part_number=1, content=b"discard me"))
            parts_before = client.list_parts(mpu)
            assert len(parts_before) == 1
            assert parts_before[0].part_number == 1

            client.abort_multipart_upload(mpu)

            # The parts must be gone after the abort. Providers signal this
            # differently: some invalidate the upload handle so listing its
            # parts raises not-found, others drop the staged parts so the
            # listing comes back empty. Either proves the abort discarded the
            # parts; a no-op abort would still return the part above and fail.
            try:
                parts_after = client.list_parts(mpu)
            except ResourceNotFoundException:
                pass
            else:
                assert parts_after == []
            # No object is ever assembled from an aborted upload.
            assert client.does_object_exist(key) is False
        finally:
            self._safe_abort(client, mpu)
            self._safe_delete(client, key)

    def test_multipart_list_parts_returns_uploaded_parts(self, client):
        """list_parts reports every uploaded part, sorted by part number.

        Parts are uploaded out of order with distinct sizes so the result
        proves both that every part is listed and that the listing is ordered
        by part number rather than by upload order. Sizes and part numbers are
        cloud-agnostic. Each listed part's ETag is asserted to equal the ETag
        the corresponding upload_part returned — a value-level check that holds
        across providers without depending on any provider's ETag format.
        """
        key = f"{self.TEST_PREFIX}multipart/list-parts.bin"
        mpu = client.initiate_multipart_upload(key)
        try:
            # Upload out of order (2, 1, 3) with distinct sizes, keeping the
            # per-part ETag each upload reports so the listing can be matched
            # against it below.
            up2 = client.upload_part(
                mpu, MultipartPart(part_number=2, content=b"\x02" * 20)
            )
            up1 = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"\x01" * 10)
            )
            up3 = client.upload_part(
                mpu, MultipartPart(part_number=3, content=b"\x03" * 30)
            )

            parts = client.list_parts(mpu)

            assert [p.part_number for p in parts] == [1, 2, 3]
            assert [p.size_in_bytes for p in parts] == [10, 20, 30]
            assert {p.part_number: p.e_tag for p in parts} == {
                1: up1.e_tag,
                2: up2.e_tag,
                3: up3.e_tag,
            }
        finally:
            self._safe_abort(client, mpu)
            self._safe_delete(client, key)

    def test_multipart_list_parts_with_gaps_in_part_numbers(self, client):
        """list_parts reports non-contiguous part numbers, sorted numerically.

        Part numbers need not be contiguous; uploading 2, 3, 10 must list
        exactly those three, in order, without inventing the missing numbers.
        The numbers deliberately span single and multiple digits so numeric
        order (2, 3, 10) diverges from lexicographic order (10, 2, 3): a
        provider that ordered the underlying keys as strings would surface 10
        first, so asserting numeric order proves the listing is sorted by part
        number rather than by key text. Sizes are decorrelated from the part
        numbers so the size assertion is an independent check, not a restatement
        of the part-number one, and each listed ETag is asserted to equal the
        value the matching upload_part returned.
        """
        key = f"{self.TEST_PREFIX}multipart/list-parts-gaps.bin"
        mpu = client.initiate_multipart_upload(key)
        try:
            up2 = client.upload_part(
                mpu, MultipartPart(part_number=2, content=b"\x02" * 15)
            )
            up3 = client.upload_part(
                mpu, MultipartPart(part_number=3, content=b"\x03" * 40)
            )
            up10 = client.upload_part(
                mpu, MultipartPart(part_number=10, content=b"\x0a" * 25)
            )

            parts = client.list_parts(mpu)

            assert [p.part_number for p in parts] == [2, 3, 10]
            assert [p.size_in_bytes for p in parts] == [15, 40, 25]
            assert {p.part_number: p.e_tag for p in parts} == {
                2: up2.e_tag,
                3: up3.e_tag,
                10: up10.e_tag,
            }
        finally:
            self._safe_abort(client, mpu)
            self._safe_delete(client, key)

    def test_multipart_list_parts_empty_then_reflects_first_part(self, client):
        """list_parts is empty on a fresh upload, then reflects the first part.

        Asserts the empty-to-one transition against the same upload handle
        rather than only the empty listing. A fresh handle listing empty proves
        little on its own: where a provider derives the listing from a key
        prefix, a brand-new handle's prefix matches nothing by construction, so
        empty is guaranteed regardless of whether the handle is live. Uploading
        one part against that handle and then seeing exactly that part appear
        proves the handle actually tracks parts.
        """
        key = f"{self.TEST_PREFIX}multipart/list-parts-empty.bin"
        mpu = client.initiate_multipart_upload(key)
        try:
            assert client.list_parts(mpu) == []

            uploaded = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"\x01" * 12)
            )

            parts = client.list_parts(mpu)
            assert [p.part_number for p in parts] == [1]
            assert [p.size_in_bytes for p in parts] == [12]
            assert parts[0].e_tag == uploaded.e_tag
        finally:
            self._safe_abort(client, mpu)
            self._safe_delete(client, key)

    def test_multipart_list_parts_with_invalid_credentials_raises(
        self, invalid_credentials_client, harness
    ):
        """list_parts under invalid credentials raises a translated error.

        The handle is bound to the real bucket so input validation passes and
        the request is actually sent; the only possible failure is then
        authorization. The base exception type is asserted because an
        unauthorized request surfaces through different native error codes
        across providers — the guarantee under test is that every one is
        translated into the common hierarchy rather than leaking a native
        exception.
        """
        key = f"{self.TEST_PREFIX}multipart/list-parts-invalid-credentials.bin"
        handle = MultipartUpload(
            bucket=harness.get_bucket(),
            key=key,
            upload_id="deadbeefdeadbeefdeadbeefdeadbeef",
            metadata={},
            content_type=None,
        )
        with pytest.raises(SubstrateSdkException):
            invalid_credentials_client.list_parts(handle)

    def test_multipart_complete_after_abort_fails(self, client):
        """Completing an already-aborted upload raises not-found."""
        key = f"{self.TEST_PREFIX}multipart/complete-after-abort.bin"
        try:
            mpu = client.initiate_multipart_upload(key)
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"gone soon")
            )
            client.abort_multipart_upload(mpu)
            with pytest.raises(ResourceNotFoundException):
                client.complete_multipart_upload(mpu, [part])
        finally:
            self._safe_delete(client, key)

    def test_multipart_invalid_handle_fails(self, client, harness):
        """Completing with an unknown upload id raises not-found."""
        key = f"{self.TEST_PREFIX}multipart/invalid-handle.bin"
        bogus = MultipartUpload(
            bucket=harness.get_bucket(),
            key=key,
            upload_id="deadbeefdeadbeefdeadbeefdeadbeef",
            metadata={},
            content_type=None,
        )
        fake_part = UploadPartResponse(part_number=1, e_tag="0" * 32, size_in_bytes=10)
        with pytest.raises(ResourceNotFoundException):
            client.complete_multipart_upload(bogus, [fake_part])

    def test_multipart_nonexistent_parts_fails(self, client):
        """Completing with a part that was never uploaded raises."""
        key = f"{self.TEST_PREFIX}multipart/nonexistent-parts.bin"
        mpu = client.initiate_multipart_upload(key)
        try:
            client.upload_part(mpu, MultipartPart(part_number=1, content=b"real part"))
            ghost = UploadPartResponse(part_number=99, e_tag="0" * 32, size_in_bytes=10)
            with pytest.raises(SubstrateSdkException):
                client.complete_multipart_upload(mpu, [ghost])
        finally:
            self._safe_abort(client, mpu)
            self._safe_delete(client, key)

    def test_multipart_bad_etag(self, client, harness):
        """A wrong part ETag is rejected where the provider validates it.

        Providers that assemble parts by re-reading stored bytes (rather
        than validating the caller-supplied ETag) instead complete
        successfully; ``harness.validates_part_etags`` selects the branch.
        """
        key = f"{self.TEST_PREFIX}multipart/bad-etag.bin"
        mpu = client.initiate_multipart_upload(key)
        try:
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"real content")
            )
            tampered = UploadPartResponse(
                part_number=1, e_tag="0" * 32, size_in_bytes=part.size_in_bytes
            )
            if harness.validates_part_etags():
                with pytest.raises(InvalidArgumentException):
                    client.complete_multipart_upload(mpu, [tampered])
            else:
                result = client.complete_multipart_upload(mpu, [tampered])
                assert result is not None
                assert client.does_object_exist(key) is True
                fetched = client.download(key)
                assert fetched.content == b"real content"
        finally:
            self._safe_abort(client, mpu)
            self._safe_delete(client, key)

    def test_multipart_with_content_type(self, client):
        """Content type set at initiate survives to the assembled object."""
        key = f"{self.TEST_PREFIX}multipart/content-type.txt"
        try:
            mpu = client.initiate_multipart_upload(key, content_type="text/plain")
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"typed payload")
            )
            client.complete_multipart_upload(mpu, [part])
            fetched = client.download(key)
            assert fetched.content == b"typed payload"
            assert fetched.metadata.content_type == "text/plain"
        finally:
            self._safe_delete(client, key)

    def test_multipart_with_tags(self, client):
        """Tags applied to a multipart-assembled object round-trip.

        Tags here are set after the upload completes, so this covers
        ``set_tags`` on an assembled object. The ``tags`` argument on
        initiate is a separate path, covered by the tests below.
        """
        key = f"{self.TEST_PREFIX}multipart/tags.bin"
        tags = {"env": "conformance", "tier": "gold"}
        try:
            mpu = client.initiate_multipart_upload(key)
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"tagged payload")
            )
            client.complete_multipart_upload(mpu, [part])
            client.set_tags(key, tags)
            assert dict(client.get_tags(key)) == tags
        finally:
            self._safe_delete(client, key)

    # Tags supplied at initiate use a different path from set_tags.

    def test_multipart_applies_tags_given_at_initiate(self, client):
        """Tags passed to initiate land on the assembled object."""
        key = f"{self.TEST_PREFIX}multipart/tags-at-initiate.bin"
        tags = {"env": "conformance", "tier": "gold"}
        try:
            mpu = client.initiate_multipart_upload(key, tags=tags)
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"tagged at initiate")
            )
            client.complete_multipart_upload(mpu, [part])
            assert dict(client.get_tags(key)) == tags
        finally:
            self._safe_delete(client, key)

    def test_multipart_tags_and_metadata_both_survive_initiate(self, client):
        """Tags and user metadata given at initiate do not displace each other.

        A provider that carries a tag set inside the same map as user
        metadata can drop one while writing the other, which neither a
        tags-only nor a metadata-only assertion would catch. The two values
        are different strings so a swap between them is visible.
        """
        key = f"{self.TEST_PREFIX}multipart/tags-and-metadata.bin"
        try:
            mpu = client.initiate_multipart_upload(
                key, metadata={"team": "multicloud"}, tags={"env": "conformance"}
            )
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"tagged and described")
            )
            client.complete_multipart_upload(mpu, [part])
            assert dict(client.get_tags(key)) == {"env": "conformance"}
            # Assert the user entry; tag encoding may share this map.
            fetched = client.get_metadata(key)
            assert fetched.metadata.get("team") == "multicloud"
        finally:
            self._safe_delete(client, key)

    def test_multipart_tags_with_reserved_characters_round_trip(self, client):
        """Tag values holding separator characters survive the round trip.

        A space and a ``+`` are legal in a tag value but carry meaning in the
        encodings used to transmit a tag set, so a value written unencoded
        comes back altered rather than rejected.
        """
        key = f"{self.TEST_PREFIX}multipart/tags-reserved-chars.bin"
        tags = {"note": "a b+c", "owner": "team/sub"}
        try:
            mpu = client.initiate_multipart_upload(key, tags=tags)
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"reserved characters")
            )
            client.complete_multipart_upload(mpu, [part])
            assert dict(client.get_tags(key)) == tags
        finally:
            self._safe_delete(client, key)

    # A fixed literal instant so record and replay agree; VCR matches on
    # method + URI (not body), so a non-deterministic date would still replay,
    # but the recorded response body carries the date these tests assert on.
    # Record-mode maintenance: object-lock APIs reject a past retain-until, so
    # once this instant is in the past, re-recording (--record-mode=rewrite)
    # requires bumping it to a future date; replay mode is unaffected.
    _OBJECT_LOCK_RETAIN_UNTIL = datetime(2026, 8, 30, tzinfo=timezone.utc)

    def test_multipart_with_object_lock(self, object_lock_client, harness):
        """An object lock set at initiate is present on the completed object.

        Runs on an object-lock-enabled bucket (skipped otherwise). The lock —
        governance-mode retention plus a legal hold — is passed to
        ``initiate_multipart_upload`` and must be readable via
        ``get_object_lock`` once the upload completes. Retention blocks delete
        and overwrite, not read, so the assembled content still round-trips.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}multipart/object-lock.bin"
        payload = b"object-locked multipart payload"
        lock_cfg = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=self._OBJECT_LOCK_RETAIN_UNTIL,
            legal_hold=True,
        )
        try:
            mpu = client.initiate_multipart_upload(key, object_lock=lock_cfg)
            part = client.upload_part(
                mpu, MultipartPart(part_number=1, content=payload)
            )
            client.complete_multipart_upload(mpu, [part])
            # Content round-trips despite the lock (retention blocks writes,
            # not reads). A single-byte range keeps the recorded body tiny.
            assert client.download(key, start=0, end=0).content == payload[:1]
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.mode is RetentionMode.GOVERNANCE
            assert lock.retain_until_date == self._OBJECT_LOCK_RETAIN_UNTIL
            assert lock.legal_hold is True
        finally:
            self._safe_release_lock_and_delete(client, harness, key)

    def _safe_release_lock_and_delete(
        self, client: BucketClient, harness: "Harness", key: str
    ) -> None:
        """Release the lock and delete the object, best-effort, to stay clean.

        A retained or held object cannot be deleted normally. Releasing a
        legal hold is uniform across providers, so it lives here; deleting the
        still-retained object is not (some clouds bypass governance on the
        delete, others must remove the retention first) and is delegated to
        the harness. Without this, record-mode buckets would accumulate
        undeletable locked objects.
        """
        try:
            client.update_legal_hold(key, False)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Hold release failed for {key}: {e}")
        try:
            harness.delete_retained_object(client, key)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Retained-object delete failed for {key}: {e}")

    def _safe_delete_retained(
        self, client: BucketClient, harness: "Harness", key: str
    ) -> None:
        """Delete a retained (but not held) object, best-effort, to stay clean.

        Omits the legal-hold release (a metageneration-guarded write that cannot
        replay once the object's metageneration has advanced); the delete itself
        is provider-specific and delegated to the harness.
        """
        try:
            harness.delete_retained_object(client, key)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Retained-object delete failed for {key}: {e}")

    # Retention instants for the update_object_retention tests. Constants, not
    # runtime-computed: replay asserts the read-back equals these, and the
    # read-back comes from the frozen cassette. To re-record, bump only
    # _RETENTION_ANCHOR to just past the recording date (the others derive from
    # it by fixed offsets) — one edit, so a mistyped year can't mint a
    # long-lived lock.
    _RETENTION_ANCHOR = datetime(2026, 9, 15, tzinfo=timezone.utc)
    _RETENTION_EARLIER = _RETENTION_ANCHOR
    _RETENTION_LATER = _RETENTION_ANCHOR + timedelta(days=15)

    # For the mode-upgrade test, which writes an immutable COMPLIANCE lock. Set
    # just past the recording date so the lock self-expires within days instead
    # of lingering for months. Bump when re-recording upgrade-mode.
    _RETENTION_UPGRADE = datetime(2026, 8, 13, tzinfo=timezone.utc)

    def test_update_object_retention_extends_date(self, object_lock_client, harness):
        """Extending the retain-until date updates it and preserves the mode.

        Runs on an object-lock-enabled bucket (skipped otherwise). Governance
        retention is set at upload, then pushed to a later instant without a
        mode change; extending never needs the bypass flag. The read-back must
        show the later instant with the original mode intact.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-retention/extend-date.txt"
        lock_cfg = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=self._RETENTION_EARLIER,
        )
        try:
            client.upload(key, b"retention-extend", object_lock=lock_cfg)
            client.update_object_retention(
                key, ObjectRetentionConfig(retain_until_date=self._RETENTION_LATER)
            )
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.mode is RetentionMode.GOVERNANCE
            assert lock.retain_until_date == self._RETENTION_LATER
        finally:
            self._safe_delete_retained(client, harness, key)

    def test_update_object_retention_shortens_with_bypass(
        self, object_lock_client, harness
    ):
        """Shortening governance retention succeeds only with bypass_governance.

        Runs on an object-lock-enabled bucket (skipped otherwise). Governance
        retention is set to a later instant at upload, then pulled back to an
        earlier one with ``bypass_governance=True`` — the flag a privileged
        caller must supply to shorten. The read-back must show the earlier
        instant.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-retention/shorten-bypass.txt"
        lock_cfg = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=self._RETENTION_LATER,
        )
        try:
            client.upload(key, b"retention-shorten", object_lock=lock_cfg)
            client.update_object_retention(
                key,
                ObjectRetentionConfig(
                    retain_until_date=self._RETENTION_EARLIER,
                    bypass_governance=True,
                ),
            )
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.mode is RetentionMode.GOVERNANCE
            assert lock.retain_until_date == self._RETENTION_EARLIER
        finally:
            self._safe_delete_retained(client, harness, key)

    def test_update_object_retention_shorten_without_bypass_raises(
        self, object_lock_client, harness
    ):
        """Shortening governance retention without bypass_governance is rejected.

        Runs on an object-lock-enabled bucket (skipped otherwise). The mirror of
        ``shortens_with_bypass``: the same pull-back to an earlier instant, but
        with ``bypass_governance`` left at its default. Without the flag the
        shorten must be rejected with FailedPreconditionException — this is what
        proves the flag is what authorizes the shorten, not merely that a
        bypass=True call happens to succeed.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-retention/shorten-no-bypass.txt"
        lock_cfg = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=self._RETENTION_LATER,
        )
        try:
            client.upload(key, b"retention-noshorten", object_lock=lock_cfg)
            with pytest.raises(FailedPreconditionException):
                client.update_object_retention(
                    key,
                    ObjectRetentionConfig(
                        retain_until_date=self._RETENTION_EARLIER,
                    ),
                )
        finally:
            self._safe_delete_retained(client, harness, key)

    def test_update_object_retention_upgrades_mode(self, object_lock_client, harness):
        """Upgrading the mode to the immutable one succeeds with bypass_governance.

        Runs on an object-lock-enabled bucket (skipped otherwise). Only the
        reducible → immutable upgrade is permitted, and providers require the
        bypass flag for it. Uses the short-lived ``_RETENTION_UPGRADE`` instant
        so the COMPLIANCE lock self-expires; the key is suffixed with that date
        so a re-record writes to a fresh, not-yet-locked key.
        """
        client = object_lock_client
        date_suffix = self._RETENTION_UPGRADE.strftime("%Y%m%d")
        key = f"{self.TEST_PREFIX}update-retention/upgrade-mode-{date_suffix}.txt"
        lock_cfg = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=self._RETENTION_UPGRADE,
        )
        try:
            client.upload(key, b"retention-upgrade", object_lock=lock_cfg)
            client.update_object_retention(
                key,
                ObjectRetentionConfig(
                    retain_until_date=self._RETENTION_UPGRADE,
                    mode=RetentionMode.COMPLIANCE,
                    bypass_governance=True,
                ),
            )
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.mode is RetentionMode.COMPLIANCE
            assert lock.retain_until_date == self._RETENTION_UPGRADE
        finally:
            self._safe_delete_retained(client, harness, key)

    def test_update_object_retention_rejects_naive_datetime(self, client):
        """A timezone-naive retain-until is rejected before any I/O.

        retain_until_date denotes an absolute instant, so a naive value is
        rejected by validation with InvalidArgumentException. The failure
        happens before any request is issued, so this needs no object-lock
        bucket and no recorded interaction.
        """
        key = f"{self.TEST_PREFIX}update-retention/naive.txt"
        naive_instant = datetime(2026, 9, 30)  # no tzinfo
        with pytest.raises(InvalidArgumentException):
            client.update_object_retention(
                key, ObjectRetentionConfig(retain_until_date=naive_instant)
            )

    def test_update_object_retention_without_existing_retention_raises(
        self, object_lock_client
    ):
        """Updating retention on an object that has none raises a precondition error.

        Runs on an object-lock-enabled bucket (skipped otherwise). update targets
        an object that already has retention; there is no first-time set here, so
        an object uploaded without retention must be rejected with
        FailedPreconditionException rather than silently gaining retention.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-retention/no-existing-retention.txt"
        try:
            client.upload(key, b"unretained")
            with pytest.raises(FailedPreconditionException):
                client.update_object_retention(
                    key,
                    ObjectRetentionConfig(retain_until_date=self._RETENTION_LATER),
                )
        finally:
            # No retention was set, so the object deletes freely; object-lock
            # buckets are versioned, so remove every version.
            self._safe_delete_all_versions(client, key)

    def _safe_release_hold(self, client: BucketClient, key: str) -> None:
        """Release a legal hold best-effort, without attempting a delete.

        For compliance-retained objects: their retention cannot be lifted
        and the object cannot be deleted until the retain-until instant
        elapses, so the only releasable state is a legal hold (if one was
        somehow set). This clears that and stops — it deliberately does not
        issue the delete the provider is guaranteed to reject with a lock
        error, which would otherwise be swallowed here and only add failure
        noise to the record-mode logs.
        """
        try:
            client.update_legal_hold(key, False)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Hold release failed for {key}: {e}")

    def test_get_object_lock_retention_governance(self, object_lock_client, harness):
        """Governance-mode retention set at upload reads back via get_object_lock.

        Runs on an object-lock-enabled bucket (skipped otherwise). Governance
        retention is removable by a privileged caller, so the object can be
        torn down after the read.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}object-lock/retention-governance.txt"
        lock_cfg = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=self._OBJECT_LOCK_RETAIN_UNTIL,
        )
        try:
            client.upload(key, b"governance-retained", object_lock=lock_cfg)
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.mode is RetentionMode.GOVERNANCE
            assert lock.retain_until_date == self._OBJECT_LOCK_RETAIN_UNTIL
            assert lock.legal_hold is False
        finally:
            self._safe_release_lock_and_delete(client, harness, key)

    def test_get_object_lock_retention_compliance(self, object_lock_client):
        """Compliance-mode retention set at upload reads back via get_object_lock.

        Runs on an object-lock-enabled bucket (skipped otherwise). Compliance
        retention cannot be shortened or removed by anyone until it expires, so
        cleanup can only release a legal hold (there is none here) — the object
        itself survives in the record-mode bucket until the retain-until instant
        elapses. That tradeoff is accepted for coverage of the compliance path.

        The key is suffixed with the retain-until date so a compliance object
        left locked by an earlier recording does not block a later one: a
        re-record after the date is bumped writes to a fresh, not-yet-locked
        key rather than colliding with the still-locked object at the old key.
        """
        client = object_lock_client
        date_suffix = self._OBJECT_LOCK_RETAIN_UNTIL.strftime("%Y%m%d")
        key = f"{self.TEST_PREFIX}object-lock/retention-compliance-{date_suffix}.txt"
        lock_cfg = ObjectLockConfiguration(
            mode=RetentionMode.COMPLIANCE,
            retain_until_date=self._OBJECT_LOCK_RETAIN_UNTIL,
        )
        try:
            client.upload(key, b"compliance-retained", object_lock=lock_cfg)
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.mode is RetentionMode.COMPLIANCE
            assert lock.retain_until_date == self._OBJECT_LOCK_RETAIN_UNTIL
            assert lock.legal_hold is False
        finally:
            # Compliance retention cannot be lifted and the object cannot be
            # deleted until retain-until elapses, so teardown only releases a
            # legal hold (none set here). No delete is attempted: the provider
            # would reject it with a lock error until the retain-until date, and
            # swallowing that only adds noise to record-mode logs. The object is
            # expected to survive in the record-mode bucket until then.
            self._safe_release_hold(client, key)

    def test_get_object_lock_legal_hold_only(self, object_lock_client, harness, vcr):
        """A legal hold with no retention reads back via get_object_lock.

        Runs on an object-lock-enabled bucket (skipped otherwise). The object is
        uploaded without retention, then a legal hold is applied and read back —
        so ``mode`` is None while ``legal_hold`` is True.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}object-lock/legal-hold-only.txt"
        try:
            client.upload(key, b"legally-held")
            client.update_legal_hold(key, True)
            writes = captured_legal_hold_writes(vcr)
            assert writes, "placing a hold must issue a hold write to the wire"
            assert writes[-1].enabled is True
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.mode is None
            # No retention was set, so retain_until_date must be None. On some
            # providers the native client always hands back a (possibly empty)
            # retention object rather than None, so "no retention" is read off
            # its mode/retain-until fields being unset rather than the object's
            # absence; this holds only while that empty object reports no mode.
            assert lock.retain_until_date is None
            assert lock.legal_hold is True
        finally:
            self._safe_release_lock_and_delete(client, harness, key)

    def test_update_legal_hold_enable_then_disable(
        self, object_lock_client, harness, vcr
    ):
        """Toggling a legal hold on then off is observable in both directions.

        The read-back alone is not enough on replay: a bodyless read matches
        positionally, so a write carrying the wrong enable value still replays
        the recorded read-back and the test passes. ``captured_legal_hold_writes``
        returns the value each write actually put on the wire, so the enable
        value reaching the request itself is asserted too.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-legal-hold/toggle.txt"
        try:
            client.upload(key, b"held-then-released")

            client.update_legal_hold(key, True)
            writes = captured_legal_hold_writes(vcr)
            assert writes, "placing a hold must issue a hold write to the wire"
            assert writes[-1].enabled is True
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.legal_hold is True
            assert lock.mode is None

            client.update_legal_hold(key, False)
            assert captured_legal_hold_writes(vcr)[-1].enabled is False
            # No retention and no hold remain, so the object has no lock state.
            assert client.get_object_lock(key) is None
        finally:
            # Release the hold before delete: if an assertion above fails with
            # the hold still on, a plain delete would be refused.
            self._safe_release_lock_and_delete(client, harness, key)

    def test_update_legal_hold_rejects_blank_key(self, client):
        """A blank key is rejected in validation before any request is issued."""
        with pytest.raises(InvalidArgumentException):
            client.update_legal_hold("   ", True)

    def test_update_legal_hold_nonexistent_key_raises(self, object_lock_client):
        """Placing a hold on a missing object raises ResourceNotFoundException."""
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-legal-hold/does-not-exist-xyz123.txt"
        with pytest.raises(ResourceNotFoundException):
            client.update_legal_hold(key, True)

    def test_update_legal_hold_preserved_across_retention_update(
        self, object_lock_client, harness
    ):
        """A legal hold survives an unrelated retention update.

        Retention is extended (no bypass, does not touch the hold); the
        read-back must still report legal_hold True — the two lock states are
        independent.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-legal-hold/preserved-across-retention.txt"
        lock_cfg = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=self._RETENTION_EARLIER,
            legal_hold=True,
        )
        try:
            client.upload(key, b"held-and-retained", object_lock=lock_cfg)
            client.update_object_retention(
                key, ObjectRetentionConfig(retain_until_date=self._RETENTION_LATER)
            )
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.legal_hold is True
            assert lock.mode is RetentionMode.GOVERNANCE
            assert lock.retain_until_date == self._RETENTION_LATER
        finally:
            self._safe_release_lock_and_delete(client, harness, key)

    def test_update_legal_hold_via_method_preserved_across_retention_update(
        self, object_lock_client, harness, vcr
    ):
        """A hold set via update_legal_hold survives a later retention update.

        Retention is set at upload (extending it later needs existing
        retention); the hold is applied afterward via the method under test,
        not the upload flag.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-legal-hold/applied-across-retention.txt"
        lock_cfg = ObjectLockConfiguration(
            mode=RetentionMode.GOVERNANCE,
            retain_until_date=self._RETENTION_EARLIER,
        )
        try:
            client.upload(key, b"retained-then-held", object_lock=lock_cfg)
            client.update_legal_hold(key, True)
            writes = captured_legal_hold_writes(vcr)
            assert writes, "placing a hold must issue a hold write to the wire"
            assert writes[-1].enabled is True
            client.update_object_retention(
                key, ObjectRetentionConfig(retain_until_date=self._RETENTION_LATER)
            )
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.legal_hold is True
            assert lock.mode is RetentionMode.GOVERNANCE
            assert lock.retain_until_date == self._RETENTION_LATER
        finally:
            self._safe_release_lock_and_delete(client, harness, key)

    def test_legal_hold_blocks_delete(self, object_lock_client, harness, vcr):
        """A legal hold refuses deletion of the held version.

        Deletes the specific version, not the key: a keyed delete on a
        versioned bucket only adds a delete marker and never reaches the hold.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-legal-hold/blocks-delete.txt"
        try:
            resp = client.upload(key, b"cannot-delete-while-held")
            assert resp.version_id is not None
            client.update_legal_hold(key, True)
            writes = captured_legal_hold_writes(vcr)
            assert writes, "placing a hold must issue a hold write to the wire"
            assert writes[-1].enabled is True
            with pytest.raises(UnAuthorizedException):
                client.delete(key, version_id=resp.version_id)
        finally:
            # Release the hold before delete: the held version is refused
            # otherwise, and the object is hold-only so it deletes freely once
            # the hold is off.
            self._safe_release_lock_and_delete(client, harness, key)

    def test_get_object_lock_none_when_unlocked(self, object_lock_client):
        """get_object_lock returns None for an object with no lock configured.

        Runs on an object-lock-enabled bucket (skipped otherwise). An object
        uploaded without retention or a hold has no lock state, so the read
        returns None rather than an empty configuration. This assumes the
        bucket carries no default retention configuration; otherwise new
        objects would be auto-retained and the read would not be None.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}object-lock/unlocked.txt"
        try:
            client.upload(key, b"no lock here")
            assert client.get_object_lock(key) is None
        finally:
            # Object-lock buckets are versioning-enabled, so a plain keyed
            # delete only inserts a delete marker; remove every version so the
            # recorded bucket doesn't accumulate objects across record runs.
            self._safe_delete_all_versions(client, key)

    def test_get_object_lock_nonexistent_key(self, object_lock_client):
        """get_object_lock on a missing key raises ResourceNotFoundException.

        Runs on the object-lock-enabled bucket (skipped otherwise) so the
        lookup reaches the missing-object check rather than a "bucket has no
        object lock" error.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}object-lock/does-not-exist-xyz123.txt"
        with pytest.raises(ResourceNotFoundException):
            client.get_object_lock(key)

    # Error-path batteries: missing bucket, invalid credentials, and the
    # local-signing operations that must keep working regardless.
    #
    # The nonexistent-bucket battery covers only operations that reach the
    # network and must fail when the target bucket is absent:
    # download, get_metadata, upload, list_page, and get_tags. It deliberately
    # excludes delete / delete_objects / initiate-multipart-upload and the
    # does_*_exist probes, because those legitimately differ across providers
    # on a missing target — some treat a not-found as a no-op success, some
    # accumulate the failure instead of raising, and existence checks are
    # defined to return a boolean rather than raise. Those shapes are asserted
    # elsewhere (delete_objects below; existence checks in
    # test_does_bucket_exist_* ) rather than forced into a single raises-check.

    @pytest.mark.parametrize(
        "invoke, expected",
        [
            pytest.param(
                lambda c, k: c.download(k),
                ResourceNotFoundException,
                id="download",
            ),
            pytest.param(
                lambda c, k: c.get_metadata(k),
                ResourceNotFoundException,
                id="get_metadata",
            ),
            pytest.param(
                lambda c, k: c.upload(k, b"probe"),
                SubstrateSdkException,
                id="upload",
            ),
            pytest.param(
                lambda c, k: c.list_page(prefix=k),
                SubstrateSdkException,
                id="list_page",
            ),
            pytest.param(
                lambda c, k: c.get_tags(k),
                SubstrateSdkException,
                id="get_tags",
            ),
        ],
    )
    def test_operation_on_nonexistent_bucket_raises(
        self, nonexistent_bucket_client, invoke, expected
    ):
        """Network operations against a missing bucket raise a translated error.

        ``expected`` is the specific cloud-agnostic type where every provider
        agrees (a missing bucket surfaces as not-found on read paths); it is
        widened to the base exception where providers may legitimately
        classify the failure differently. Because every accepted type is a
        subclass of :class:`SubstrateSdkException`, a raw ``TypeError`` /
        ``AttributeError`` from a broken test would not satisfy the check.
        """
        key = self.TEST_PREFIX + "nonexistent-bucket-probe.txt"
        with pytest.raises(expected):
            invoke(nonexistent_bucket_client, key)

    @pytest.mark.parametrize(
        "invoke",
        [
            pytest.param(lambda c, k: c.download(k), id="download"),
            pytest.param(lambda c, k: c.get_metadata(k), id="get_metadata"),
            pytest.param(lambda c, k: c.upload(k, b"probe"), id="upload"),
            pytest.param(lambda c, k: c.list_page(prefix=k), id="list_page"),
            pytest.param(lambda c, k: c.get_tags(k), id="get_tags"),
            pytest.param(lambda c, k: c.delete(k), id="delete"),
        ],
    )
    def test_operation_with_invalid_credentials_raises(
        self, invalid_credentials_client, invoke
    ):
        """Network operations with invalid credentials raise UnAuthorizedException.

        The bucket exists, so the only possible failure is authorization, and
        the guarantee under test is that every provider maps that failure to
        the same cloud-agnostic type. An unauthorized request surfaces through
        different native errors across providers — a rejected request the
        service answers with 401/403, or a credential the auth layer refuses to
        refresh before any request is sent — and each must translate to
        ``UnAuthorizedException``. ``delete`` is included here — unlike the
        missing-bucket battery — because an authorization failure is not a
        not-found and so is never swallowed by a delete's no-op-on-missing
        behavior.
        """
        key = self.TEST_PREFIX + "invalid-credentials-probe.txt"
        with pytest.raises(UnAuthorizedException):
            invoke(invalid_credentials_client, key)

    def test_delete_objects_with_invalid_credentials_reports_failure(
        self, invalid_credentials_client
    ):
        """Batch delete under invalid credentials either raises or reports errors.

        Providers differ in batch-delete shape: one issues a single batch call
        that fails wholesale (translated exception), another deletes per object
        and accumulates per-object failures into the response. Both are correct
        ways to signal that an unauthorized batch delete did not succeed, so the
        assertion accepts either shape and rejects only silent success.
        """
        key = self.TEST_PREFIX + "invalid-credentials-batch-probe.txt"
        raised = None
        response = None
        try:
            response = invalid_credentials_client.delete_objects(
                [BlobIdentifier(key=key)]
            )
        except SubstrateSdkException as exc:
            raised = exc
        if raised is not None:
            # Wholesale batch failure surfaced as a translated exception. An
            # unauthorized batch delete is an authorization failure, so the
            # translated type must be UnAuthorizedException specifically -- not
            # merely the base type, which would be true by construction here
            # (raised is only assigned in the except SubstrateSdkException arm).
            assert isinstance(raised, UnAuthorizedException)
        else:
            # Per-object accumulation: the failure must be reported, not swallowed.
            assert response is not None
            assert response.errors
            assert response.errors[0].key == key
