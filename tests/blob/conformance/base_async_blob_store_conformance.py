"""Reusable async blob conformance test suite for ``AsyncBucketClient``.

Companion to :mod:`base_blob_store_conformance` (the synchronous suite). Covers
directory operations -- ``upload_directory``/``download_directory``, which exist
only on the async client -- plus ``does_object_exist``, ``does_bucket_exist``,
and ``generate_presigned_url``. Provider subclasses supply an ``AsyncHarness``.

Notes:
  - Subclasses must start with ``Test`` (for pytest collection) and contain the
    provider id so ``conftest._provider_for`` routes cassettes correctly. Async
    cassettes are class-prefixed and must not be shared with the sync suite: the
    two stacks use different HTTP libraries and record different requests.
  - Directory and single-object tests use different key prefixes
    (``DIRECTORY_TEST_PREFIX`` / ``TEST_PREFIX``); the prefix is part of the
    matched request URI, so they cannot be unified without re-recording.
  - Directory calls pass ``max_concurrency=1`` for deterministic request order
    on replay.
"""

from __future__ import annotations

import asyncio
import copy
import io
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
import pytest
import pytest_asyncio
import vcr.request
import vcr.stubs.aiohttp_stubs
from multicloud.blob.blob_client import (
    AsyncBucketClient,
    BlobIdentifier,
    MultipartPart,
    MultipartUpload,
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

#: Object storage requires every part but the last to be at least 5 MiB.
MPU_PART_SIZE = 5 * 1024 * 1024


def _install_async_request_deepcopy_patch() -> None:
    """Let VCR deep-copy an async request whose body it cannot copy.

    VCR deep-copies every request before recording/matching. An async upload
    body is an aws-chunked streaming payload whose ``read`` is a coroutine, which
    ``copy.deepcopy`` cannot pickle, so the copy aborts with ``TypeError``. The
    body is never matched or scrubbed, so this ``__deepcopy__`` substitutes an
    empty body only when the body itself cannot be copied. Idempotent.
    """
    request_cls = vcr.request.Request
    if getattr(request_cls, "_async_deepcopy_body_safe", False):
        return

    def __deepcopy__(self, memo):  # type: ignore[no-untyped-def]
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            try:
                new.__dict__[key] = copy.deepcopy(value, memo)
            except TypeError:
                # A coroutine-backed streaming body: never matched or asserted
                # on, so drop it to an empty body rather than abort the copy.
                new.__dict__[key] = b""
        return new

    request_cls.__deepcopy__ = __deepcopy__
    request_cls._async_deepcopy_body_safe = True


def _install_async_raw_headers_patch() -> None:
    """Give VCR's replayed aiohttp response the ``raw_headers`` aiobotocore reads.

    On replay VCR sets ``_headers`` but leaves ``_raw_headers`` as ``None``.
    aiobotocore iterates ``response.raw_headers``, so a replayed response raises
    ``TypeError: 'NoneType' object is not iterable`` before any assertion runs.
    The wrapper derives ``_raw_headers`` from the headers VCR deserialized.
    Idempotent.
    """
    stubs = vcr.stubs.aiohttp_stubs
    if getattr(stubs.build_response, "_async_raw_headers_safe", False):
        return

    stock_build_response = stubs.build_response

    def build_response(vcr_request, vcr_response, history):  # type: ignore[no-untyped-def]
        response = stock_build_response(vcr_request, vcr_response, history)
        if getattr(response, "_raw_headers", None) is None:
            response._raw_headers = tuple(
                (k.encode("utf-8"), v.encode("utf-8"))
                for k, v in response.headers.items()
            )
        return response

    build_response._async_raw_headers_safe = True
    stubs.build_response = build_response


def _install_async_request_body_roundtrip_patch() -> None:
    """Make VCR's ``Request.body`` setter idempotent for file-like bodies.

    An async upload hands VCR a ``BytesIO`` body, so its ``body`` getter returns
    ``BytesIO(self._body)``. When a filter reads and assigns it back, the stock
    setter stores the ``BytesIO`` into ``_body``, and the next read raises
    ``TypeError: a bytes-like object is required, not '_io.BytesIO'``. This setter
    reads file-like values back to bytes (restoring stream position) so ``_body``
    is always bytes. Only the recorded copy is affected. Idempotent.
    """
    request_cls = vcr.request.Request
    if getattr(request_cls, "_async_body_roundtrip_safe", False):
        return

    def _set_body(self, value):  # type: ignore[no-untyped-def]
        read_fn = getattr(value, "read", None)
        if callable(read_fn):
            tell = getattr(value, "tell", None)
            seek = getattr(value, "seek", None)
            if callable(tell) and callable(seek):
                pos = value.tell()
                value = value.read()
                seek(pos)
            else:
                value = value.read()
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._body = value

    request_cls.body = property(request_cls.body.fget, _set_body)
    request_cls._async_body_roundtrip_safe = True


_install_async_request_deepcopy_patch()
_install_async_raw_headers_patch()
_install_async_request_body_roundtrip_patch()


class AsyncHarness(ABC):
    """Test harness interface that async providers implement."""

    @abstractmethod
    def create_client(self) -> AsyncBucketClient:
        """Return an AsyncBucketClient configured for the test bucket."""
        ...

    def create_directory_client(self) -> AsyncBucketClient:
        """Return an AsyncBucketClient for the directory-operation tests.

        Defaults to the standard client; override to bind a different bucket.
        """
        return self.create_client()

    @abstractmethod
    def get_provider_id(self) -> str:
        """Return the provider id (e.g. ``"aws"``, ``"gcp"``)."""
        ...

    @abstractmethod
    def get_bucket(self) -> str:
        """Return the bucket name the client is configured against."""
        ...

    @abstractmethod
    def create_client_for_absent_bucket(self) -> AsyncBucketClient:
        """Return an AsyncBucketClient bound to a bucket that does not exist.

        The name must be stable across runs, since it appears in the matched
        request URI.
        """

    def create_versioned_client(self) -> Optional[AsyncBucketClient]:
        """Return an AsyncBucketClient bound to a versioning-enabled bucket, or None.

        Providers without one return ``None`` and the versioned tests are skipped.
        """
        return None

    def get_versioned_bucket(self) -> Optional[str]:
        """Return the versioning-enabled bucket name, or None.

        Mirrors :meth:`create_versioned_client` for tests that need the
        bucket name without reaching into the client internals.
        """
        return None

    def supports_copy_source_version(self) -> bool:
        """Whether copy can pin a specific source version."""
        return True

    def create_object_lock_client(self) -> Optional[AsyncBucketClient]:
        """Return an AsyncBucketClient bound to an object-lock-enabled bucket, or None.

        Placing a legal hold requires the bucket to have object lock (or the
        provider equivalent) enabled. Providers with such a bucket override this;
        those without return ``None`` and the legal-hold tests are skipped.
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

    def get_storage_class(self) -> str:
        """Return a provider-native, non-default storage class for uploads.

        Raises by default so a provider must supply one rather than silently
        disable the storage-class upload tests.
        """
        raise NotImplementedError("Harness must supply a non-default storage class.")

    def create_invalid_credentials_client(self) -> AsyncBucketClient:
        """Return a client for the test bucket holding unusable credentials."""
        raise NotImplementedError

    def detects_unknown_upload_id(self) -> bool:
        """Whether the provider rejects an upload id it has no record of.

        True where multipart sessions live server-side. False where multipart is
        emulated with ordinary objects and there is no session to check against,
        so a part uploaded under an unknown id is simply stored.
        """
        return True


class BaseAsyncBlobStoreConformance(ABC):
    """Abstract async conformance test suite for blob providers.

    Concrete provider subclasses supply a module-level ``harness`` fixture that
    returns an :class:`AsyncBucketClient` for the provider under test, and are
    decorated with ``@pytest.mark.vcr``.
    """

    # Single-object / signing tests (does_object_exist, does_bucket_exist,
    # generate_presigned_url) run under this prefix.
    TEST_PREFIX = "async-conformance-test/"
    # Directory tests were recorded under a different prefix; kept distinct so
    # both cassette families replay unchanged.
    DIRECTORY_TEST_PREFIX = "conformance-test-async/"

    # Client fixtures are function-scoped: async HTTP backends bind their session
    # to the event loop they were created on, and pytest-asyncio gives each test a
    # fresh loop, so a shared client would fail on every test after the first.

    @pytest_asyncio.fixture
    async def async_client(self, harness):
        """Provide a fresh AsyncBucketClient for the test bucket, closed after."""
        client = harness.create_client()
        try:
            yield client
        finally:
            await client.close()

    @pytest_asyncio.fixture
    async def directory_client(self, harness):
        """Provide a fresh AsyncBucketClient for the directory tests, closed after."""
        client = harness.create_directory_client()
        try:
            yield client
        finally:
            await client.close()

    @pytest_asyncio.fixture
    async def versioned_client(self, harness):
        """Provide an AsyncBucketClient bound to a versioning-enabled bucket.

        Skips the test when the provider harness has not supplied one.
        """
        versioned = harness.create_versioned_client()
        if versioned is None:
            pytest.skip("Provider harness did not supply a versioning-enabled bucket.")
        try:
            yield versioned
        finally:
            await versioned.close()

    @pytest_asyncio.fixture
    async def object_lock_client(self, harness):
        """Provide an AsyncBucketClient bound to an object-lock-enabled bucket.

        Skips the test when the provider harness has not supplied one.
        """
        locked = harness.create_object_lock_client()
        if locked is None:
            pytest.skip(
                "Provider harness did not supply an object-lock-enabled bucket."
            )
        try:
            yield locked
        finally:
            await locked.close()

    @pytest_asyncio.fixture(loop_scope="function")
    async def invalid_credentials_async_client(self, harness):
        """A client for the real bucket whose credentials will be rejected."""
        client = harness.create_invalid_credentials_client()
        yield client
        await client.close()

    @pytest.fixture
    def kms_key_id(self, harness):
        """Provide a server-side-encryption key id.

        The harness hook raises if a provider fails to supply one, so a missing
        override fails loudly rather than silently skipping the encrypted-upload
        tests.
        """
        return harness.get_kms_key_id()

    @pytest.fixture
    def storage_class(self, harness):
        """Provide a provider-native, non-default storage class."""
        return harness.get_storage_class()

    # Helpers

    async def _safe_delete(self, client: AsyncBucketClient, *keys: str) -> None:
        """Delete blobs ignoring errors.

        Prevents test object accumulation in real cloud buckets during record
        mode. Each test that creates objects calls this in a finally block.
        """
        for key in keys:
            try:
                await client.delete(key)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Cleanup failed for {key}: {e}")

    async def _safe_delete_version(
        self, client: AsyncBucketClient, key: str, version_id: str
    ) -> None:
        """Delete a specific version of a key, ignoring errors.

        A version-less delete on a versioned bucket only archives the live
        version, and the async listing API cannot enumerate noncurrent versions,
        so cleanup must delete each captured version id explicitly.
        """
        try:
            await client.delete(key, version_id=version_id)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Cleanup failed for {key}@{version_id}: {e}")

    async def _safe_delete_all_versions(
        self, client: AsyncBucketClient, key: str
    ) -> None:
        """Delete every version of a key, ignoring errors.

        Re-lists rather than deleting only the ids a test captured, so a run
        that died part-way still gets cleaned up on the next one. Without that
        the leftovers stay in the shared versioned bucket and break the exact
        counts these tests assert when cassettes are next re-recorded.
        """
        try:
            version_ids = [
                info.version_id async for info in client.list_blob_versions(key)
            ]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Version cleanup failed to list {key}: {exc}")
            return
        for version_id in version_ids:
            if version_id:
                await self._safe_delete_version(client, key, version_id)

    async def _safe_release_lock_and_delete(
        self, client: AsyncBucketClient, key: str
    ) -> None:
        """Release the hold on every version, then delete each, best-effort.

        A held version cannot be deleted while its hold is on, and a versionless
        release only clears the hold on the latest version. So an older held
        version would be refused deletion and, because these test keys are
        stable, every interrupted record run could strand another held version
        indefinitely. Enumerate the versions and release each hold by version_id
        before deleting that version. These tests set only a legal hold (no
        retention), so releasing it is enough to allow the delete; every step is
        cloud-agnostic and best-effort so a cleanup failure warns rather than
        masking the assertion that already ran.
        """
        try:
            version_ids = [
                info.version_id async for info in client.list_blob_versions(key)
            ]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Version cleanup failed to list {key}: {exc}")
            return
        for version_id in version_ids:
            if not version_id:
                continue
            try:
                await client.update_legal_hold(key, False, version_id=version_id)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Hold release failed for {key}@{version_id}: {exc}")
            await self._safe_delete_version(client, key, version_id)

    async def _safe_delete_prefix(self, client: AsyncBucketClient, prefix: str) -> None:
        """Delete every object under a prefix, ignoring errors.

        Teardown only, so failures are warnings rather than test failures --
        a cleanup error must not mask the assertion that already ran. Uses
        list-then-batch-delete because the async client on this branch has no
        prefix-delete operation.
        """
        try:
            page = await client.list_page(prefix=prefix, max_results=1000)
            identifiers = [BlobIdentifier(key=blob.key) for blob in page.blobs]
            if identifiers:
                await client.delete_objects(identifiers)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Cleanup failed for prefix {prefix}: {exc}")

    @staticmethod
    def _write_tree(root: Path) -> Dict[str, bytes]:
        """Create a small nested directory and return {relative path: content}.

        Two files at the top level and one a directory down, so a recursive
        upload and a top-level-only upload produce different results.
        """
        files = {
            "top-one.txt": b"top level file one",
            "top-two.txt": b"top level file two",
            "nested/deep.txt": b"nested file content",
        }
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return files

    async def _keys_under(self, client: AsyncBucketClient, prefix: str) -> set:
        """Return the set of object keys currently under a prefix."""
        page = await client.list_page(prefix=prefix, max_results=1000)
        return {blob.key for blob in page.blobs}

    async def _upload_tree_for_download(
        self, client: AsyncBucketClient, source: Path, prefix: str, expected: int
    ) -> None:
        """Upload a tree as test setup, failing loudly if it did not fully land.

        The download tests only assert on what they download, so a partially
        successful arrange step would otherwise go unnoticed and quietly weaken
        them -- and in replay, an incomplete cassette shows up here first.
        """
        response = await client.upload_directory(
            str(source), prefix=prefix, include_sub_folders=True, max_concurrency=1
        )
        assert response.successful_uploads == expected, (
            f"setup upload incomplete: {response.successful_uploads}/{expected} "
            f"stored, failures={response.failed_transfers}"
        )

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

    # upload_directory conformance tests

    @pytest.mark.asyncio
    async def test_upload_directory_includes_subfolders(
        self, directory_client, tmp_path
    ):
        """A recursive upload stores every file, preserving relative paths."""
        prefix = f"{self.DIRECTORY_TEST_PREFIX}upload-recursive/"
        files = self._write_tree(tmp_path)
        try:
            response = await directory_client.upload_directory(
                str(tmp_path),
                prefix=prefix,
                include_sub_folders=True,
                max_concurrency=1,
            )
            assert response.total_files == len(files)
            assert response.successful_uploads == len(files)
            assert response.failed_transfers == []
            assert response.has_failures is False
            assert response.total_bytes_transferred == sum(
                len(content) for content in files.values()
            )

            assert await self._keys_under(directory_client, prefix) == {
                f"{prefix}{relative}" for relative in files
            }
        finally:
            await self._safe_delete_prefix(directory_client, prefix)

    @pytest.mark.asyncio
    async def test_upload_directory_excludes_subfolders_when_disabled(
        self, directory_client, tmp_path
    ):
        """With recursion off, only top-level files are stored.

        Asserting the nested key is *absent* is the point -- a provider that
        ignored the flag would still pass a count-only check if the counts
        happened to line up.
        """
        prefix = f"{self.DIRECTORY_TEST_PREFIX}upload-flat/"
        self._write_tree(tmp_path)
        try:
            response = await directory_client.upload_directory(
                str(tmp_path),
                prefix=prefix,
                include_sub_folders=False,
                max_concurrency=1,
            )
            assert response.total_files == 2
            assert response.successful_uploads == 2

            keys = await self._keys_under(directory_client, prefix)
            assert keys == {f"{prefix}top-one.txt", f"{prefix}top-two.txt"}
            assert f"{prefix}nested/deep.txt" not in keys
        finally:
            await self._safe_delete_prefix(directory_client, prefix)

    @pytest.mark.asyncio
    async def test_upload_directory_applies_tags_to_every_object(
        self, directory_client, tmp_path
    ):
        """Tags given once are applied to all uploaded objects."""
        prefix = f"{self.DIRECTORY_TEST_PREFIX}upload-tags/"
        tags = {"origin": "conformance", "kind": "directory"}
        files = self._write_tree(tmp_path)
        try:
            response = await directory_client.upload_directory(
                str(tmp_path),
                prefix=prefix,
                include_sub_folders=True,
                max_concurrency=1,
                tags=tags,
            )
            # Asserted so an incomplete recording fails here rather than
            # silently reducing what the loop below checks.
            assert response.successful_uploads == len(files), (
                f"{response.successful_uploads}/{len(files)} stored, "
                f"failures={response.failed_transfers}"
            )

            for relative in files:
                fetched = await directory_client.get_tags(f"{prefix}{relative}")
                assert fetched == tags
        finally:
            await self._safe_delete_prefix(directory_client, prefix)

    @pytest.mark.asyncio
    async def test_upload_directory_empty_directory_uploads_nothing(
        self, directory_client, tmp_path
    ):
        """An empty source directory is a successful no-op, not an error."""
        prefix = f"{self.DIRECTORY_TEST_PREFIX}upload-empty/"
        empty = tmp_path / "empty"
        empty.mkdir()
        response = await directory_client.upload_directory(
            str(empty), prefix=prefix, max_concurrency=1
        )
        assert response.total_files == 0
        assert response.successful_uploads == 0
        assert response.failed_transfers == []
        assert response.has_failures is False

    @pytest.mark.asyncio
    async def test_upload_directory_missing_source_rejected(self, directory_client):
        """A source path that does not exist is rejected before any transfer."""
        with pytest.raises(
            InvalidArgumentException, match="source directory does not exist"
        ):
            await directory_client.upload_directory(
                "/nonexistent/conformance/source/path",
                prefix=f"{self.DIRECTORY_TEST_PREFIX}never/",
            )

    @pytest.mark.asyncio
    async def test_upload_directory_source_must_be_a_directory(
        self, directory_client, tmp_path
    ):
        """A file given where a directory is expected is rejected."""
        single = tmp_path / "not-a-directory.txt"
        single.write_bytes(b"regular file")
        with pytest.raises(
            InvalidArgumentException, match="source path is not a directory"
        ):
            await directory_client.upload_directory(
                str(single), prefix=f"{self.DIRECTORY_TEST_PREFIX}never/"
            )

    @pytest.mark.asyncio
    async def test_upload_directory_rejects_non_positive_concurrency(
        self, directory_client, tmp_path
    ):
        """Concurrency must be positive, and is checked before transferring."""
        self._write_tree(tmp_path)
        with pytest.raises(
            InvalidArgumentException, match="max_concurrency must be > 0"
        ):
            await directory_client.upload_directory(
                str(tmp_path),
                prefix=f"{self.DIRECTORY_TEST_PREFIX}never/",
                max_concurrency=0,
            )

    # download_directory conformance tests

    @pytest.mark.asyncio
    async def test_download_directory_round_trip(self, directory_client, tmp_path):
        """Every uploaded file comes back with identical bytes and layout."""
        prefix = f"{self.DIRECTORY_TEST_PREFIX}download-round-trip/"
        source = tmp_path / "source"
        source.mkdir()
        destination = tmp_path / "destination"
        files = self._write_tree(source)
        try:
            await self._upload_tree_for_download(
                directory_client, source, prefix, len(files)
            )

            response = await directory_client.download_directory(
                prefix, str(destination), max_concurrency=1
            )
            assert response.total_files == len(files)
            assert response.successful_downloads == len(files)
            assert response.failed_transfers == []
            assert response.has_failures is False

            for relative, content in files.items():
                assert (destination / relative).read_bytes() == content
        finally:
            await self._safe_delete_prefix(directory_client, prefix)

    @pytest.mark.asyncio
    async def test_download_directory_honors_prefix_exclusions(
        self, directory_client, tmp_path
    ):
        """Excluded prefixes are skipped and their files are not written."""
        prefix = f"{self.DIRECTORY_TEST_PREFIX}download-exclude/"
        source = tmp_path / "source"
        source.mkdir()
        destination = tmp_path / "destination"
        uploaded = self._write_tree(source)
        try:
            await self._upload_tree_for_download(
                directory_client, source, prefix, len(uploaded)
            )

            response = await directory_client.download_directory(
                prefix,
                str(destination),
                max_concurrency=1,
                prefixes_to_exclude=[f"{prefix}nested/"],
            )
            assert response.total_files == 2
            assert response.successful_downloads == 2
            assert (destination / "top-one.txt").exists()
            assert not (destination / "nested" / "deep.txt").exists()
        finally:
            await self._safe_delete_prefix(directory_client, prefix)

    @pytest.mark.asyncio
    async def test_download_directory_empty_prefix_downloads_nothing(
        self, directory_client, tmp_path
    ):
        """A prefix with no objects yields an empty, successful result."""
        prefix = f"{self.DIRECTORY_TEST_PREFIX}download-nothing-here/"
        destination = tmp_path / "destination"
        response = await directory_client.download_directory(
            prefix, str(destination), max_concurrency=1
        )
        assert response.total_files == 0
        assert response.successful_downloads == 0
        assert response.failed_transfers == []
        assert response.has_failures is False

    @pytest.mark.asyncio
    async def test_download_directory_blank_prefix_rejected(
        self, directory_client, tmp_path
    ):
        """A blank prefix is rejected rather than treated as the whole bucket."""
        with pytest.raises(InvalidArgumentException, match="prefix is required"):
            await directory_client.download_directory("", str(tmp_path))

    @pytest.mark.asyncio
    async def test_download_directory_blank_destination_rejected(
        self, directory_client
    ):
        """A blank local destination is rejected."""
        with pytest.raises(
            InvalidArgumentException, match="local_destination_directory is required"
        ):
            await directory_client.download_directory(
                f"{self.DIRECTORY_TEST_PREFIX}download-round-trip/", ""
            )

    @pytest.mark.asyncio
    async def test_download_directory_rejects_non_positive_concurrency(
        self, directory_client, tmp_path
    ):
        """Concurrency must be positive, and is checked before transferring."""
        with pytest.raises(
            InvalidArgumentException, match="max_concurrency must be > 0"
        ):
            await directory_client.download_directory(
                f"{self.DIRECTORY_TEST_PREFIX}download-round-trip/",
                str(tmp_path),
                max_concurrency=-1,
            )

    # multipart upload conformance tests
    #
    # The multipart cassettes were recorded against the directory bucket and
    # prefix, so these tests use the ``directory_client`` fixture and
    # ``DIRECTORY_TEST_PREFIX`` -- not the single-object ``async_client`` --
    # to keep the request URIs the cassette matcher compares unchanged.

    async def _safe_abort(self, client: AsyncBucketClient, mpu) -> None:
        """Abort during teardown, ignoring errors; parts are otherwise billed."""
        try:
            await client.abort_multipart_upload(mpu)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Cleanup failed for upload {mpu.upload_id}: {exc}")

    @pytest.mark.asyncio
    async def test_multipart_assembles_parts_in_order(self, directory_client):
        """Parts are assembled by part number, not by upload order.

        Samples the bytes either side of the boundary, so this checks ordering
        rather than the integrity of each part's contents.
        """
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-ordered.bin"
        first, second = b"\x01" * MPU_PART_SIZE, b"\x02" * 16
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            # Uploaded back to front, so ordering cannot come from call order.
            part_two = await directory_client.upload_part(
                mpu, MultipartPart(part_number=2, content=second)
            )
            part_one = await directory_client.upload_part(
                mpu, MultipartPart(part_number=1, content=first)
            )
            result = await directory_client.complete_multipart_upload(
                mpu, [part_two, part_one]
            )
            assert result.e_tag

            # Read the bytes either side of the part boundary rather than the
            # whole object, which would put megabytes into the cassette.
            head = await directory_client.download(key, start=0, end=0)
            assert head.content == b"\x01"
            boundary = await directory_client.download(
                key, start=MPU_PART_SIZE, end=MPU_PART_SIZE
            )
            assert boundary.content == b"\x02"
        finally:
            await self._safe_delete_prefix(directory_client, key)

    @pytest.mark.asyncio
    async def test_multipart_preserves_metadata_and_content_type(
        self, directory_client
    ):
        """Metadata and content type given at initiate survive completion."""
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-metadata.bin"
        mpu = await directory_client.initiate_multipart_upload(
            key, metadata={"owner": "conformance"}, content_type="text/plain"
        )
        try:
            part = await directory_client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"metadata payload")
            )
            await directory_client.complete_multipart_upload(mpu, [part])

            metadata = await directory_client.get_metadata(key)
            assert metadata.metadata.get("owner") == "conformance"
            assert metadata.content_type == "text/plain"
        finally:
            await self._safe_delete_prefix(directory_client, key)

    @pytest.mark.asyncio
    async def test_multipart_abort_discards_the_upload(self, directory_client):
        """After an abort the upload can no longer be completed.

        Asserted through completion, not the object's absence, which would pass
        whether or not the abort ran.
        """
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-aborted.bin"
        mpu = await directory_client.initiate_multipart_upload(key)
        part = await directory_client.upload_part(
            mpu, MultipartPart(part_number=1, content=b"\x03" * MPU_PART_SIZE)
        )
        try:
            await directory_client.abort_multipart_upload(mpu)

            with pytest.raises(ResourceNotFoundException):
                await directory_client.complete_multipart_upload(mpu, [part])
        finally:
            # If abort regressed, the completion above succeeded and left a real
            # object behind, which nothing else would remove.
            await self._safe_abort(directory_client, mpu)
            await self._safe_delete_prefix(directory_client, key)

    @pytest.mark.asyncio
    async def test_multipart_abort_is_idempotent(self, directory_client):
        """A second abort is a no-op, so cleanup paths can call it freely."""
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-abort-twice.bin"
        mpu = await directory_client.initiate_multipart_upload(key)
        part = await directory_client.upload_part(
            mpu, MultipartPart(part_number=1, content=b"\x04" * MPU_PART_SIZE)
        )
        try:
            await directory_client.abort_multipart_upload(mpu)
            await directory_client.abort_multipart_upload(mpu)

            with pytest.raises(ResourceNotFoundException):
                await directory_client.complete_multipart_upload(mpu, [part])
        finally:
            await self._safe_abort(directory_client, mpu)
            await self._safe_delete_prefix(directory_client, key)

    @pytest.mark.asyncio
    async def test_multipart_rejects_part_number_below_one(self, directory_client):
        """A part number below 1 is refused before any request.

        The upper bound is provider-specific, so it is covered per provider in
        the unit tests rather than here.
        """
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-bad-part-number.bin"
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            with pytest.raises(InvalidArgumentException, match="part_number must be"):
                await directory_client.upload_part(
                    mpu, MultipartPart(part_number=0, content=b"x")
                )
        finally:
            await self._safe_abort(directory_client, mpu)

    @pytest.mark.asyncio
    async def test_multipart_unknown_upload_id_behaviour_is_pinned(
        self, directory_client, harness
    ):
        """Pin how an upload id the provider never issued is treated.

        Neither outcome is asserted as correct; this records which one each
        provider gives so a change is visible.
        """
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-unknown-id.bin"
        forged = MultipartUpload(
            bucket=directory_client.bucket, key=key, upload_id="never-issued-by-anyone"
        )
        part = MultipartPart(part_number=1, content=b"orphan")

        if harness.detects_unknown_upload_id():
            with pytest.raises(SubstrateSdkException):
                await directory_client.upload_part(forged, part)
        else:
            # Accepted and stored, so the part has to be cleaned up here.
            await directory_client.upload_part(forged, part)
            await self._safe_abort(directory_client, forged)

    @pytest.mark.asyncio
    async def test_multipart_reuploading_a_part_keeps_the_last_write(
        self, directory_client
    ):
        """Replacing a part before completing is allowed; the last one wins.

        Distinct from listing the same number twice at completion, which is
        refused.
        """
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-part-replaced.bin"
        first, second = b"\x01" * MPU_PART_SIZE, b"\x02" * MPU_PART_SIZE
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            await directory_client.upload_part(
                mpu, MultipartPart(part_number=1, content=first)
            )
            replacement = await directory_client.upload_part(
                mpu, MultipartPart(part_number=1, content=second)
            )

            await directory_client.complete_multipart_upload(mpu, [replacement])

            head = await directory_client.download(key, start=0, end=0)
            assert head.content == b"\x02"
        finally:
            await self._safe_delete_prefix(directory_client, key)

    @pytest.mark.asyncio
    async def test_multipart_listing_a_part_twice_is_refused(self, directory_client):
        """Completing with a repeated part number is rejected before any request.

        One provider refuses it; the other would assemble the part twice.
        """
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-repeated-part.bin"
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            part = await directory_client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"\x05" * MPU_PART_SIZE)
            )
            with pytest.raises(InvalidArgumentException, match="more than once"):
                await directory_client.complete_multipart_upload(mpu, [part, part])
        finally:
            await self._safe_abort(directory_client, mpu)

    @pytest.mark.asyncio
    async def test_multipart_uploads_parts_concurrently(self, directory_client):
        """Parts upload in parallel, which is the reason this API is async."""
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-concurrent.bin"
        payloads = {
            1: b"\x0a" * MPU_PART_SIZE,
            2: b"\x0b" * MPU_PART_SIZE,
            3: b"\x0c" * 32,
        }
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            uploaded = await asyncio.gather(
                *(
                    directory_client.upload_part(
                        mpu, MultipartPart(part_number=n, content=data)
                    )
                    for n, data in payloads.items()
                )
            )
            assert {p.part_number for p in uploaded} == {1, 2, 3}

            await directory_client.complete_multipart_upload(mpu, list(uploaded))

            metadata = await directory_client.get_metadata(key)
            assert metadata.content_length == sum(len(d) for d in payloads.values())
            boundary = await directory_client.download(
                key, start=MPU_PART_SIZE, end=MPU_PART_SIZE
            )
            assert boundary.content == b"\x0b"
        finally:
            await self._safe_abort(directory_client, mpu)
            await self._safe_delete_prefix(directory_client, key)

    @pytest.mark.asyncio
    async def test_multipart_applies_tags_given_at_initiate(self, directory_client):
        """Tags supplied at initiate land on the assembled object."""
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-tagged.bin"
        mpu = await directory_client.initiate_multipart_upload(
            key, tags={"env": "conformance", "team": "multicloud"}
        )
        try:
            part = await directory_client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"tagged payload")
            )
            await directory_client.complete_multipart_upload(mpu, [part])

            assert await directory_client.get_tags(key) == {
                "env": "conformance",
                "team": "multicloud",
            }
        finally:
            await self._safe_abort(directory_client, mpu)
            await self._safe_delete_prefix(directory_client, key)

    @pytest.mark.asyncio
    async def test_multipart_accepts_a_stream_part(self, directory_client):
        """A part may be a binary stream rather than bytes."""
        key = f"{self.DIRECTORY_TEST_PREFIX}multipart-stream.bin"
        payload = b"\x0d" * MPU_PART_SIZE
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            part = await directory_client.upload_part(
                mpu,
                MultipartPart(
                    part_number=1,
                    content=io.BytesIO(payload),
                    content_length=len(payload),
                ),
            )
            assert part.size_in_bytes == len(payload)

            await directory_client.complete_multipart_upload(mpu, [part])

            head = await directory_client.download(key, start=0, end=0)
            assert head.content == b"\x0d"
        finally:
            await self._safe_abort(directory_client, mpu)
            await self._safe_delete_prefix(directory_client, key)

    @pytest.mark.asyncio
    async def test_multipart_list_parts_is_empty_before_any_upload(
        self, directory_client
    ):
        """A session with no parts lists nothing rather than failing."""
        key = f"{self.TEST_PREFIX}multipart-list-empty.bin"
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            assert await directory_client.list_parts(mpu) == []
        finally:
            await self._safe_abort(directory_client, mpu)

    @pytest.mark.asyncio
    async def test_multipart_list_parts_reports_uploaded_parts(self, directory_client):
        """Every part is listed, ordered by number, with its own size and ETag.

        Uploaded out of order with distinct sizes, so ordering cannot come from
        upload order and a size mapped to the wrong part fails. ETags are
        compared against what each upload returned, which is a value check that
        holds without knowing any provider's ETag format. The minimum part size
        applies at completion, so these can stay small.
        """
        key = f"{self.TEST_PREFIX}multipart-list-parts.bin"
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            second = await directory_client.upload_part(
                mpu, MultipartPart(part_number=2, content=b"\x02" * 20)
            )
            first = await directory_client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"\x01" * 10)
            )
            third = await directory_client.upload_part(
                mpu, MultipartPart(part_number=3, content=b"\x03" * 30)
            )

            parts = await directory_client.list_parts(mpu)

            assert [p.part_number for p in parts] == [1, 2, 3]
            assert [p.size_in_bytes for p in parts] == [10, 20, 30]
            assert {p.part_number: p.e_tag for p in parts} == {
                1: first.e_tag,
                2: second.e_tag,
                3: third.e_tag,
            }
        finally:
            await self._safe_abort(directory_client, mpu)

    @pytest.mark.asyncio
    async def test_multipart_list_parts_preserves_gaps(self, directory_client):
        """Part numbers need not be contiguous, and the gap is not filled in."""
        key = f"{self.TEST_PREFIX}multipart-list-gaps.bin"
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            for number in (1, 4):
                await directory_client.upload_part(
                    mpu,
                    MultipartPart(part_number=number, content=b"\x07" * MPU_PART_SIZE),
                )

            assert [p.part_number for p in await directory_client.list_parts(mpu)] == [
                1,
                4,
            ]
        finally:
            await self._safe_abort(directory_client, mpu)

    @pytest.mark.asyncio
    async def test_multipart_list_parts_reflects_a_replaced_part(
        self, directory_client
    ):
        """Re-uploading a part number replaces it rather than adding a second."""
        key = f"{self.TEST_PREFIX}multipart-list-replaced.bin"
        mpu = await directory_client.initiate_multipart_upload(key)
        try:
            await directory_client.upload_part(
                mpu, MultipartPart(part_number=1, content=b"\x01" * MPU_PART_SIZE)
            )
            await directory_client.upload_part(
                mpu,
                MultipartPart(part_number=1, content=b"\x09" * (MPU_PART_SIZE + 16)),
            )

            parts = await directory_client.list_parts(mpu)

            assert [p.part_number for p in parts] == [1]
            assert parts[0].size_in_bytes == MPU_PART_SIZE + 16
        finally:
            await self._safe_abort(directory_client, mpu)

    @pytest.mark.asyncio
    async def test_multipart_list_parts_after_abort(self, directory_client, harness):
        """After an abort the parts are gone, reported per the provider.

        One reports the upload as unknown, the other an empty listing, so the
        outcome is pinned per provider rather than either accepted -- a
        mistranslated exception would otherwise pass.
        """
        key = f"{self.TEST_PREFIX}multipart-list-after-abort.bin"
        mpu = await directory_client.initiate_multipart_upload(key)
        await directory_client.upload_part(
            mpu, MultipartPart(part_number=1, content=b"\x08" * 24)
        )
        await directory_client.abort_multipart_upload(mpu)

        if harness.detects_unknown_upload_id():
            with pytest.raises(ResourceNotFoundException):
                await directory_client.list_parts(mpu)
        else:
            assert await directory_client.list_parts(mpu) == []

    @pytest.mark.asyncio
    async def test_multipart_list_parts_with_invalid_credentials_raises(
        self, invalid_credentials_async_client
    ):
        """An unauthorized listing surfaces as a translated exception.

        The handle names the bucket of the client that will send the request, so
        the driver's bucket check passes and authorization is the only thing left
        to fail. Naming any other client's bucket would fail validation offline
        and never reach the service. The base type is asserted because providers
        report it through different native errors; what matters is that none of
        them leak.
        """
        key = f"{self.TEST_PREFIX}multipart-list-parts-unauthorized.bin"
        handle = MultipartUpload(
            bucket=invalid_credentials_async_client.bucket,
            key=key,
            upload_id="deadbeefdeadbeefdeadbeef",
        )

        with pytest.raises(SubstrateSdkException):
            await invalid_credentials_async_client.list_parts(handle)

    # does_object_exist conformance tests

    @pytest.mark.asyncio
    async def test_does_object_exist_true_for_existing_object(self, async_client):
        """An uploaded object is reported as existing, and only that exact key.

        A key that is a strict prefix of a live object is not itself an object,
        so the negative assertions catch a prefix-listing rather than per-key
        implementation.
        """
        key = f"{self.TEST_PREFIX}exists/present.txt"
        try:
            assert await async_client.does_object_exist(key) is False
            await async_client.upload(key, b"exists conformance payload")
            assert await async_client.does_object_exist(key) is True
            assert await async_client.does_object_exist(key[: -len(".txt")]) is False
            assert (
                await async_client.does_object_exist(f"{self.TEST_PREFIX}exists/")
                is False
            )
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_does_object_exist_false_for_nonexistent_key(self, async_client):
        """A key that was never written is reported as absent, not an error."""
        key = f"{self.TEST_PREFIX}exists/never-written-xyz789.txt"
        assert await async_client.does_object_exist(key) is False

    @pytest.mark.asyncio
    async def test_does_object_exist_tracks_object_lifecycle(self, async_client):
        """Existence flips False -> True -> False across upload and delete.

        The full transition catches an implementation hard-coded either way.
        """
        key = f"{self.TEST_PREFIX}exists/lifecycle.txt"
        try:
            assert await async_client.does_object_exist(key) is False
            await async_client.upload(key, b"now it exists")
            assert await async_client.does_object_exist(key) is True
            await async_client.delete(key)
            assert await async_client.does_object_exist(key) is False
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_does_object_exist_empty_key_rejected(self, async_client):
        """An empty key is rejected client-side rather than queried."""
        with pytest.raises(InvalidArgumentException):
            await async_client.does_object_exist("")

    @pytest.mark.asyncio
    async def test_does_object_exist_by_version_id(self, versioned_client):
        """Each version of an object is individually reported as existing.

        Version ids come from the upload responses since the native format
        differs by cloud.
        """
        client = versioned_client
        key = f"{self.TEST_PREFIX}exists/versioned.txt"
        other_key = f"{self.TEST_PREFIX}exists/versioned-other.txt"
        # (key, version_id) pairs captured from uploads, purged in the finally.
        created: List[Tuple[str, str]] = []
        try:
            first = await client.upload(key, b"version one")
            created.append((key, first.version_id))
            second = await client.upload(key, b"version two")
            created.append((key, second.version_id))
            assert first.version_id, "Provider must return version_id."
            assert second.version_id, "Provider must return version_id."
            assert first.version_id != second.version_id

            assert (
                await client.does_object_exist(key, version_id=first.version_id) is True
            )
            assert (
                await client.does_object_exist(key, version_id=second.version_id)
                is True
            )
            # Omitting the version resolves to the latest, which still exists.
            assert await client.does_object_exist(key) is True

            # A version belonging to a different object must not resolve for
            # this key -- catches an implementation that ignores version_id.
            other = await client.upload(other_key, b"unrelated object")
            created.append((other_key, other.version_id))
            assert other.version_id, "Provider must return version_id."
            assert (
                await client.does_object_exist(key, version_id=other.version_id)
                is False
            )
        finally:
            for created_key, version_id in created:
                await self._safe_delete_version(client, created_key, version_id)

    @pytest.mark.asyncio
    async def test_does_object_exist_false_for_deleted_version(self, versioned_client):
        """A version that existed and was purged is reported as absent.

        Built by upload -> capture -> delete so the id is well-formed and truly
        gone; a newer version keeps the key alive so this tests version
        resolution, not key absence.
        """
        client = versioned_client
        key = f"{self.TEST_PREFIX}exists/deleted-version.txt"
        surviving: Optional[str] = None
        try:
            first = await client.upload(key, b"doomed version")
            second = await client.upload(key, b"surviving version")
            surviving = second.version_id
            assert first.version_id, "Provider must return version_id."

            await client.delete(key, version_id=first.version_id)

            assert (
                await client.does_object_exist(key, version_id=first.version_id)
                is False
            )
            # The key is still present through its remaining version.
            assert await client.does_object_exist(key) is True
        finally:
            # The doomed version is already gone; purge the surviving one too.
            if surviving is not None:
                await self._safe_delete_version(client, key, surviving)

    # does_bucket_exist conformance tests
    #
    # It takes no argument and reports on the client's own bucket, so the two
    # cases need two clients.

    @pytest.mark.asyncio
    async def test_does_bucket_exist_true_for_configured_bucket(self, async_client):
        """The bucket the client is configured against is reported present."""
        assert await async_client.does_bucket_exist() is True

    @pytest.mark.asyncio
    async def test_does_bucket_exist_false_for_absent_bucket(self, harness):
        """A client bound to a nonexistent bucket reports absent, not an error.

        Takes ``harness`` (not ``async_client``) to build a client on a
        different, absent bucket, created and closed inline on this test's loop.
        """
        absent = harness.create_client_for_absent_bucket()
        try:
            assert await absent.does_bucket_exist() is False
        finally:
            await absent.close()

    # generate_presigned_url conformance tests
    #
    # Async API: operation is the HTTP verb ("GET"/"PUT") and the lifetime is
    # expiration_seconds. Header names are provider-prefixed, so assertions match
    # on suffix + value.

    @pytest.mark.asyncio
    async def test_generate_presigned_url_upload_shape(self, async_client):
        """An UPLOAD URL is HTTPS, carries a signature and expiry, and is a PUT."""
        key = f"{self.TEST_PREFIX}presign/upload-shape.txt"
        result = await async_client.generate_presigned_url(
            key, "PUT", expiration_seconds=900
        )
        assert result.url.startswith("https://")
        assert result.http_method == "PUT"
        assert "Signature" in result.url
        assert self._signed_expiry_seconds(result.url) == 900

    @pytest.mark.asyncio
    async def test_generate_presigned_url_download_shape(self, async_client):
        """A DOWNLOAD URL is a GET and requires no headers of the caller."""
        key = f"{self.TEST_PREFIX}presign/download-shape.txt"
        result = await async_client.generate_presigned_url(
            key, "GET", expiration_seconds=900
        )
        assert result.url.startswith("https://")
        assert result.http_method == "GET"
        assert dict(result.required_headers) == {}
        assert self._signed_expiry_seconds(result.url) == 900

    @pytest.mark.asyncio
    async def test_generate_presigned_url_signs_the_requested_duration(
        self, async_client
    ):
        """The requested lifetime reaches the signature rather than a default.

        Two values, because one cannot tell an honored argument from a constant.
        """
        key = f"{self.TEST_PREFIX}presign/duration.txt"
        for expiration in (900, 4500):
            result = await async_client.generate_presigned_url(
                key, "PUT", expiration_seconds=expiration
            )
            assert self._signed_expiry_seconds(result.url) == expiration

    @pytest.mark.asyncio
    async def test_generate_presigned_url_upload_requires_bound_content_type(
        self, async_client
    ):
        """A bound content type comes back as a header the caller must send."""
        key = f"{self.TEST_PREFIX}presign/content-type.txt"
        result = await async_client.generate_presigned_url(
            key, "PUT", expiration_seconds=900, content_type="text/plain"
        )
        assert result.required_headers.get("Content-Type") == "text/plain"

    @pytest.mark.asyncio
    async def test_generate_presigned_url_upload_requires_bound_metadata(
        self, async_client
    ):
        """Bound metadata comes back under a provider-prefixed header."""
        key = f"{self.TEST_PREFIX}presign/metadata.txt"
        result = await async_client.generate_presigned_url(
            key, "PUT", expiration_seconds=900, metadata={"owner": "conformance"}
        )
        headers = dict(result.required_headers)
        matching = [k for k in headers if k.lower().endswith("owner")]
        assert matching, f"metadata not surfaced in required_headers: {headers}"
        assert headers[matching[0]] == "conformance"

    @pytest.mark.asyncio
    async def test_generate_presigned_url_upload_without_bindings_requires_no_headers(
        self, async_client
    ):
        """With nothing bound there is nothing for the caller to replay."""
        key = f"{self.TEST_PREFIX}presign/bare.txt"
        result = await async_client.generate_presigned_url(
            key, "PUT", expiration_seconds=900
        )
        assert dict(result.required_headers) == {}

    @pytest.mark.asyncio
    async def test_generate_presigned_url_binds_the_requested_key(self, async_client):
        """Different keys are signed into the URL's own path.

        Asserts the key lands in each URL's path -- comparing whole URLs would
        pass on timestamp/signature differences alone.
        """
        first_key = f"{self.TEST_PREFIX}presign/first.txt"
        second_key = f"{self.TEST_PREFIX}presign/second.txt"
        first = await async_client.generate_presigned_url(
            first_key, "PUT", expiration_seconds=900
        )
        second = await async_client.generate_presigned_url(
            second_key, "PUT", expiration_seconds=900
        )
        first_path = unquote(urlparse(first.url).path)
        second_path = unquote(urlparse(second.url).path)
        assert first_path.endswith(first_key)
        assert second_path.endswith(second_key)
        assert first_path != second_path

    @pytest.mark.asyncio
    async def test_generate_presigned_url_zero_duration_rejected(self, async_client):
        """A zero lifetime is rejected rather than signed."""
        with pytest.raises(InvalidArgumentException):
            await async_client.generate_presigned_url(
                f"{self.TEST_PREFIX}presign/zero.txt", "PUT", expiration_seconds=0
            )

    @pytest.mark.asyncio
    async def test_generate_presigned_url_negative_duration_rejected(
        self, async_client
    ):
        """A negative lifetime is rejected rather than signed."""
        with pytest.raises(InvalidArgumentException):
            await async_client.generate_presigned_url(
                f"{self.TEST_PREFIX}presign/negative.txt", "PUT", expiration_seconds=-1
            )

    @pytest.mark.asyncio
    async def test_generate_presigned_url_duration_beyond_maximum_rejected(
        self, async_client
    ):
        """A lifetime past the seven-day signing limit is rejected up front."""
        with pytest.raises(InvalidArgumentException):
            await async_client.generate_presigned_url(
                f"{self.TEST_PREFIX}presign/too-long.txt",
                "PUT",
                expiration_seconds=7 * 24 * 60 * 60 + 1,
            )

    @pytest.mark.asyncio
    async def test_generate_presigned_url_empty_key_rejected(self, async_client):
        """An empty key is rejected rather than signed."""
        with pytest.raises(InvalidArgumentException):
            await async_client.generate_presigned_url("", "PUT", expiration_seconds=900)

    @pytest.mark.asyncio
    async def test_generate_presigned_url_invalid_operation_rejected(
        self, async_client
    ):
        """An operation other than GET/PUT is rejected rather than signed."""
        with pytest.raises(InvalidArgumentException):
            await async_client.generate_presigned_url(
                f"{self.TEST_PREFIX}presign/bad-op.txt",
                "DELETE",
                expiration_seconds=900,
            )

    @pytest.mark.asyncio
    async def test_presigned_download_url_returns_the_object(self, async_client):
        """A presigned DOWNLOAD URL returns exactly the bytes that were stored."""
        key = f"{self.TEST_PREFIX}presign/download-roundtrip.txt"
        payload = b"fetched through a presigned url"
        await async_client.upload(key, payload)
        try:
            result = await async_client.generate_presigned_url(
                key, "GET", expiration_seconds=900
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    result.url, headers=dict(result.required_headers)
                ) as response:
                    body = await response.read()
                    assert response.status == 200, body[:200]
                    assert body == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_presigned_upload_url_refuses_omitted_required_headers(
        self, async_client
    ):
        """Omitting a reported header fails the signature check."""
        key = f"{self.TEST_PREFIX}presign/missing-headers.txt"
        result = await async_client.generate_presigned_url(
            key, "PUT", expiration_seconds=900, metadata={"owner": "conformance"}
        )
        assert result.required_headers, "expected bound metadata to be reported"

        async with aiohttp.ClientSession() as session:
            async with session.put(result.url, data=b"no headers sent") as response:
                body = await response.read()
                assert response.status in (400, 403), (
                    f"upload omitting the required headers should be refused, got "
                    f"{response.status}: {body[:200]}"
                )

    # get_tags and set_tags conformance tests

    @pytest.mark.asyncio
    async def test_set_tags_then_get_tags_round_trip(self, async_client):
        key = f"{self.TEST_PREFIX}tags/roundtrip.txt"
        try:
            await async_client.upload(key, b"tag round-trip payload")
            await async_client.set_tags(key, {"environment": "integration"})

            assert await async_client.get_tags(key) == {"environment": "integration"}
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_get_tags_empty_when_no_tags(self, async_client):
        key = f"{self.TEST_PREFIX}tags/none.txt"
        try:
            await async_client.upload(key, b"no tags payload")

            assert await async_client.get_tags(key) == {}
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_set_tags_replaces_existing_tags(self, async_client):
        key = f"{self.TEST_PREFIX}tags/replace.txt"
        try:
            await async_client.upload(key, b"tag replacement payload")
            await async_client.set_tags(key, {"first": "retired"})
            await async_client.set_tags(key, {"second": "current"})

            assert await async_client.get_tags(key) == {"second": "current"}
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_set_tags_multiple_pairs_round_trip(self, async_client):
        key = f"{self.TEST_PREFIX}tags/multi.txt"
        tags = {
            "environment": "integration",
            "owner": "storage-sdk",
            "tier": "archive",
        }
        try:
            await async_client.upload(key, b"multi-tag payload")
            await async_client.set_tags(key, tags)

            assert await async_client.get_tags(key) == tags
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_set_tags_empty_mapping_rejected(self, async_client):
        with pytest.raises(InvalidArgumentException):
            await async_client.set_tags(f"{self.TEST_PREFIX}tags/empty.txt", {})

    # get_metadata conformance tests
    #
    # Every value asserted here is read out of the provider's response, so the
    # mapping from the wire to BlobMetadata is what is under test.

    @pytest.mark.asyncio
    async def test_get_metadata_reports_the_stored_attributes(self, async_client):
        """Size, content type, entity tag and user metadata all come back.

        The payload length, the metadata value and the key are distinct
        strings, so a field wired to the wrong source shows up rather than
        coincidentally matching.
        """
        key = f"{self.TEST_PREFIX}get-metadata/attributes.txt"
        payload = b"metadata attributes payload"
        try:
            uploaded = await async_client.upload(
                key,
                payload,
                content_type="text/plain",
                metadata={"owner": "conformance"},
            )

            fetched = await async_client.get_metadata(key)

            assert fetched.key == key
            assert fetched.content_length == len(payload)
            assert (fetched.content_type or "").startswith("text/plain")
            assert fetched.e_tag == uploaded.e_tag
            assert fetched.metadata.get("owner") == "conformance"
            assert fetched.last_modified is not None
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_get_metadata_reflects_the_most_recent_write(self, async_client):
        """A second write changes what get_metadata reports.

        Two payloads of different lengths, so a stale or cached read is
        visible rather than passing on the first value.
        """
        key = f"{self.TEST_PREFIX}get-metadata/overwritten.txt"
        first = b"first"
        second = b"second write, longer"
        try:
            await async_client.upload(key, first)
            before = await async_client.get_metadata(key)
            await async_client.upload(key, second)
            after = await async_client.get_metadata(key)

            assert before.content_length == len(first)
            assert after.content_length == len(second)
            assert before.e_tag != after.e_tag
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_get_metadata_of_a_missing_key_raises_not_found(self, async_client):
        """A key that does not exist raises the not-found type, not a generic error."""
        key = f"{self.TEST_PREFIX}get-metadata/never-existed-4b7c1e.txt"

        with pytest.raises(ResourceNotFoundException):
            await async_client.get_metadata(key)

    # upload_from_path conformance tests
    #
    # The path may be a str or a Path, so both forms are exercised across these
    # tests rather than in a test of their own.

    @pytest.mark.asyncio
    async def test_upload_from_path_stores_the_file_contents(
        self, async_client, tmp_path
    ):
        """The local file's bytes and length reach the object store."""
        key = f"{self.TEST_PREFIX}upload-from-path/simple.txt"
        payload = b"upload_from_path conformance payload"
        local = tmp_path / "simple.txt"
        local.write_bytes(payload)
        try:
            response = await async_client.upload_from_path(key, local)

            assert response.key == key
            assert response.e_tag
            fetched = await async_client.download(key)
            assert fetched.content == payload
            assert fetched.metadata.content_length == len(payload)
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_from_path_infers_content_type_from_the_extension(
        self, async_client, tmp_path
    ):
        """A recognised extension sets the content type without being told.

        The path is passed as a string here, covering the other accepted form.
        """
        key = f"{self.TEST_PREFIX}upload-from-path/auto-type.txt"
        local = tmp_path / "auto-type.txt"
        local.write_bytes(b"plain text body")
        try:
            await async_client.upload_from_path(key, str(local))

            fetched = await async_client.get_metadata(key)
            # Providers spell the text subtype differently, so match the type.
            assert (fetched.content_type or "").startswith("text/")
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_from_path_applies_metadata(self, async_client, tmp_path):
        """Metadata passed alongside the path lands on the stored object.

        The file name and the metadata value are different strings so a
        provider that echoed the wrong one is visible.
        """
        key = f"{self.TEST_PREFIX}upload-from-path/with-metadata.txt"
        local = tmp_path / "described.txt"
        local.write_bytes(b"described payload")
        try:
            await async_client.upload_from_path(
                key, local, metadata={"owner": "conformance"}
            )

            fetched = await async_client.get_metadata(key)
            assert fetched.metadata.get("owner") == "conformance"
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_from_path_rejects_a_missing_file(
        self, async_client, tmp_path
    ):
        """A path that does not exist is rejected before anything is sent."""
        key = f"{self.TEST_PREFIX}upload-from-path/never-sent.txt"
        missing = tmp_path / "definitely-not-here.txt"

        with pytest.raises(InvalidArgumentException):
            await async_client.upload_from_path(key, missing)

    # list_blob_versions conformance tests
    #
    # The async form hands back the iterator directly rather than a coroutine,
    # and fetches pages as they are consumed, so a provider error surfaces from
    # the loop rather than from the call. Callers are written against both
    # properties, so both are asserted here.

    @pytest.mark.asyncio
    async def test_list_blob_versions_lists_every_version(self, versioned_client):
        """Every uploaded version is listed exactly once, current one flagged.

        The two payloads differ in length so the reported size is checked
        independently of the version identity. Timestamps are compared with
        ``>=`` rather than ``>`` because some providers report the modified
        time only to the second, which two quick uploads can share.
        """
        client = versioned_client
        key = f"{self.TEST_PREFIX}list-versions/all.txt"
        older_payload = b"older version"
        newer_payload = b"newer version payload"
        try:
            older = await client.upload(key, older_payload)
            newer = await client.upload(key, newer_payload)
            assert older.version_id is not None
            assert newer.version_id is not None
            assert older.version_id != newer.version_id

            listed = [info async for info in client.list_blob_versions(key)]

            # Counted as well as keyed: a dict alone would silently collapse a
            # version the provider yielded twice.
            assert len(listed) == 2
            by_id = {info.version_id: info for info in listed}
            assert set(by_id) == {older.version_id, newer.version_id}
            assert by_id[older.version_id].size == len(older_payload)
            assert by_id[newer.version_id].size == len(newer_payload)
            assert all(info.key == key for info in listed)

            latest = [info for info in listed if info.is_latest]
            assert len(latest) == 1
            assert latest[0].version_id == newer.version_id

            assert (
                by_id[newer.version_id].last_modified
                >= by_id[older.version_id].last_modified
            )
        finally:
            await self._safe_delete_all_versions(client, key)

    @pytest.mark.asyncio
    async def test_list_blob_versions_of_a_missing_key_yields_nothing(
        self, versioned_client
    ):
        """A key that never existed yields no versions rather than raising.

        Also pins the call shape: the return value is iterated directly, so
        turning it into a coroutine would break every caller.
        """
        key = f"{self.TEST_PREFIX}list-versions/never-existed-xyz789.txt"

        iterator = versioned_client.list_blob_versions(key)

        assert not asyncio.iscoroutine(iterator)
        assert hasattr(iterator, "__aiter__")
        assert [info async for info in iterator] == []

    @pytest.mark.asyncio
    async def test_list_blob_versions_surfaces_errors_from_the_loop(
        self, invalid_credentials_async_client
    ):
        """A rejected credential surfaces while iterating, not when called.

        Paging is lazy, so nothing reaches the service until the first step of
        the loop. A caller that wrapped only the call would miss the error.
        """
        key = f"{self.TEST_PREFIX}list-versions/unauthorized.txt"

        iterator = invalid_credentials_async_client.list_blob_versions(key)

        with pytest.raises(UnAuthorizedException):
            [info async for info in iterator]

    @pytest.mark.asyncio
    async def test_list_blob_versions_rejects_an_empty_key(self, versioned_client):
        """An empty key is rejected when the iterator is obtained.

        Unlike a provider error, this is validated eagerly rather than on the
        first step of the loop, so the caller learns of it at the call.
        """
        with pytest.raises(InvalidArgumentException):
            versioned_client.list_blob_versions("")

    # update_legal_hold conformance tests
    #
    # These run on an object-lock-enabled bucket (skipped otherwise) and set only
    # a legal hold, never retention: seeding retention needs upload(object_lock=),
    # which the async upload does not yet accept. The hold is toggled through the
    # cloud-agnostic API and read back with get_object_lock, so what is under test
    # is that the flag is applied and reported, not any provider-native format.
    #
    # The read-back alone is not enough on replay: a bodyless read matches
    # positionally, so a write carrying the wrong enable value still replays the
    # recorded read-back and the test passes. ``captured_legal_hold_writes``
    # returns the enable value each write actually put on the wire (captured in
    # conftest), so the tests also assert the value reached the request itself.

    @pytest.mark.asyncio
    async def test_update_legal_hold_places_and_removes(self, object_lock_client, vcr):
        """Toggling a legal hold on then off is observable in both directions."""
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-legal-hold/toggle.txt"
        try:
            await client.upload(key, b"held-then-released")

            await client.update_legal_hold(key, True)
            writes = captured_legal_hold_writes(vcr)
            assert writes, "placing a hold must issue a hold write to the wire"
            assert writes[-1].enabled is True
            lock = await client.get_object_lock(key)
            assert lock is not None
            assert lock.legal_hold is True
            assert lock.mode is None

            await client.update_legal_hold(key, False)
            assert captured_legal_hold_writes(vcr)[-1].enabled is False
            # No retention and no hold remain, so the object has no lock state.
            assert await client.get_object_lock(key) is None
        finally:
            # Release the hold before delete: if an assertion above failed with
            # the hold still on, a plain delete would be refused.
            await self._safe_release_lock_and_delete(client, key)

    @pytest.mark.asyncio
    async def test_update_legal_hold_with_version_id(self, object_lock_client, vcr):
        """A hold placed on a specific version reads back on that version.

        The version id comes from the upload response since the native format
        differs by cloud; targeting it proves version_id is threaded through
        rather than ignored in favour of the latest version. The write's own
        version target is asserted too, so an implementation that drops the
        version_id and holds the latest version instead is caught.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-legal-hold/by-version.txt"
        try:
            uploaded = await client.upload(key, b"held-by-version")
            assert uploaded.version_id is not None

            await client.update_legal_hold(key, True, version_id=uploaded.version_id)
            writes = captured_legal_hold_writes(vcr)
            assert writes, "placing a hold must issue a hold write to the wire"
            assert writes[-1].enabled is True
            assert writes[-1].version_id == uploaded.version_id
            lock = await client.get_object_lock(key, version_id=uploaded.version_id)
            assert lock is not None
            assert lock.legal_hold is True
            assert lock.mode is None
        finally:
            try:
                await client.update_legal_hold(
                    key, False, version_id=uploaded.version_id
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Hold release failed for {key}: {exc}")
            await self._safe_delete_all_versions(client, key)

    @pytest.mark.asyncio
    async def test_update_legal_hold_rejects_blank_key(self, async_client):
        """A blank key is rejected in validation before any request is issued."""
        with pytest.raises(InvalidArgumentException):
            await async_client.update_legal_hold("   ", True)

    @pytest.mark.asyncio
    async def test_update_legal_hold_nonexistent_key_raises(self, object_lock_client):
        """Placing a hold on a missing object raises ResourceNotFoundException."""
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-legal-hold/does-not-exist-xyz123.txt"
        with pytest.raises(ResourceNotFoundException):
            await client.update_legal_hold(key, True)

    @pytest.mark.asyncio
    async def test_update_legal_hold_blocks_delete(self, object_lock_client, vcr):
        """A legal hold refuses deletion of the held version.

        Deletes the specific version, not the key: a keyed delete on a
        versioned bucket only adds a delete marker and never reaches the hold.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}update-legal-hold/blocks-delete.txt"
        try:
            resp = await client.upload(key, b"cannot-delete-while-held")
            assert resp.version_id is not None
            await client.update_legal_hold(key, True)
            writes = captured_legal_hold_writes(vcr)
            assert writes, "placing a hold must issue a hold write to the wire"
            assert writes[-1].enabled is True
            with pytest.raises(UnAuthorizedException):
                await client.delete(key, version_id=resp.version_id)
        finally:
            # Release the hold before delete: the held version is refused
            # otherwise, and the object is hold-only so it deletes freely once
            # the hold is off.
            await self._safe_release_lock_and_delete(client, key)

    # delete_directory conformance tests
    #
    # The prefix is documented as matched verbatim and never normalised, so a
    # prefix without a trailing slash also takes sibling keys that merely share
    # its leading characters. That is a sharp edge for callers, so both forms
    # are pinned. The documented behaviour of an empty or None prefix -- delete
    # every object in the bucket -- is deliberately not exercised against a
    # real bucket.

    @pytest.mark.asyncio
    async def test_delete_directory_removes_every_object_under_the_prefix(
        self, directory_client
    ):
        """Objects at every depth under the prefix are removed, and a repeat is a no-op."""
        prefix = f"{self.DIRECTORY_TEST_PREFIX}delete-dir/tree/"
        keys = [
            f"{prefix}top.txt",
            f"{prefix}nested/deep.txt",
            f"{prefix}nested/more/deeper.txt",
        ]
        try:
            for key in keys:
                await directory_client.upload(key, b"delete me")
            assert await self._keys_under(directory_client, prefix) == set(keys)

            await directory_client.delete_directory(prefix)

            assert await self._keys_under(directory_client, prefix) == set()
            # A prefix matching nothing is documented as a successful no-op.
            await directory_client.delete_directory(prefix)
        finally:
            await self._safe_delete_prefix(directory_client, prefix)

    @pytest.mark.asyncio
    async def test_delete_directory_with_a_trailing_slash_spares_siblings(
        self, directory_client
    ):
        """A slash-terminated prefix takes the directory and nothing beside it.

        ``logs.txt`` and ``logs-archive/`` share the leading characters of
        ``logs`` but are not inside it, so they must survive.
        """
        root = f"{self.DIRECTORY_TEST_PREFIX}delete-dir/scoped/"
        inside = f"{root}logs/inside.txt"
        sibling_file = f"{root}logs.txt"
        sibling_dir = f"{root}logs-archive/outside.txt"
        try:
            for key in (inside, sibling_file, sibling_dir):
                await directory_client.upload(key, b"scoped")

            await directory_client.delete_directory(f"{root}logs/")

            assert await self._keys_under(directory_client, root) == {
                sibling_file,
                sibling_dir,
            }
        finally:
            await self._safe_delete_prefix(directory_client, root)

    @pytest.mark.asyncio
    async def test_delete_directory_without_a_trailing_slash_also_takes_siblings(
        self, directory_client
    ):
        """Omitting the trailing slash takes everything sharing the prefix.

        The counterpart to the test above: this is the documented hazard, so it
        is asserted rather than left for a caller to discover.
        """
        root = f"{self.DIRECTORY_TEST_PREFIX}delete-dir/verbatim/"
        keys = [
            f"{root}logs/inside.txt",
            f"{root}logs.txt",
            f"{root}logs-archive/outside.txt",
        ]
        untouched = f"{root}other.txt"
        try:
            for key in (*keys, untouched):
                await directory_client.upload(key, b"verbatim")

            await directory_client.delete_directory(f"{root}logs")

            assert await self._keys_under(directory_client, root) == {untouched}
        finally:
            await self._safe_delete_prefix(directory_client, root)

    @pytest.mark.asyncio
    async def test_delete_directory_rejects_non_positive_concurrency(
        self, directory_client
    ):
        """A concurrency of zero or less is rejected before anything is listed."""
        prefix = f"{self.DIRECTORY_TEST_PREFIX}delete-dir/never-listed/"
        for concurrency in (0, -1):
            with pytest.raises(InvalidArgumentException):
                await directory_client.delete_directory(
                    prefix, max_concurrency=concurrency
                )

    # list_page conformance tests
    #
    # Each test uses a unique sub-namespace under TEST_PREFIX so a stray
    # object from another test cannot leak into a prefix-scoped listing.
    # prefix, delimiter, max_results and pagination_token are query
    # parameters, so a request that drops or rewrites any of them fails
    # cassette matching on replay. Assertions on the mapped page (keys,
    # sizes, truncation, common_prefixes) catch response-mapping bugs.

    @pytest.mark.asyncio
    async def test_list_page_empty_prefix_returns_nothing(self, async_client):
        """Listing an empty, never-populated prefix yields no blobs."""
        prefix = f"{self.TEST_PREFIX}list-page/empty-ns/"
        resp = await async_client.list_page(prefix=prefix)
        assert resp.blobs == []
        assert resp.common_prefixes == []
        assert resp.is_truncated is False
        assert resp.next_pagination_token is None

    @pytest.mark.asyncio
    async def test_list_page_single_object(self, async_client):
        """A prefix with one object returns exactly that key with its size."""
        prefix = f"{self.TEST_PREFIX}list-page/single-ns/"
        key = f"{prefix}only.txt"
        payload = b"single object listing"
        try:
            await async_client.upload(key, payload, content_type="text/plain")
            resp = await async_client.list_page(prefix=prefix)
            assert len(resp.blobs) == 1
            assert resp.blobs[0].key == key
            assert resp.blobs[0].size == len(payload)
            assert resp.blobs[0].last_modified is not None
            assert resp.is_truncated is False
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_list_page_multiple_objects(self, async_client):
        """All objects under a prefix are returned."""
        prefix = f"{self.TEST_PREFIX}list-page/multi-ns/"
        keys = [f"{prefix}obj-{i}.txt" for i in range(5)]
        try:
            for key in keys:
                await async_client.upload(key, b"multi listing payload")
            resp = await async_client.list_page(prefix=prefix)
            assert {blob.key for blob in resp.blobs} == set(keys)
            assert resp.common_prefixes == []
        finally:
            await self._safe_delete(async_client, *keys)

    @pytest.mark.asyncio
    async def test_list_page_max_results_truncates(self, async_client):
        """max_results caps the page size and flags truncation."""
        prefix = f"{self.TEST_PREFIX}list-page/truncate-ns/"
        keys = [f"{prefix}obj-{i}.txt" for i in range(4)]
        try:
            for key in keys:
                await async_client.upload(key, b"truncation payload")
            resp = await async_client.list_page(prefix=prefix, max_results=2)
            assert len(resp.blobs) == 2
            assert resp.is_truncated is True
            assert resp.next_pagination_token is not None
        finally:
            await self._safe_delete(async_client, *keys)

    @pytest.mark.asyncio
    async def test_list_page_pagination_round_trip(self, async_client):
        """Following the pagination token walks the full result set once."""
        prefix = f"{self.TEST_PREFIX}list-page/paginate-ns/"
        keys = [f"{prefix}obj-{i}.txt" for i in range(4)]
        try:
            for key in keys:
                await async_client.upload(key, b"pagination payload")

            first = await async_client.list_page(prefix=prefix, max_results=2)
            assert len(first.blobs) == 2
            assert first.is_truncated is True

            second = await async_client.list_page(
                prefix=prefix,
                max_results=2,
                pagination_token=first.next_pagination_token,
            )
            assert len(second.blobs) == 2

            first_keys = [blob.key for blob in first.blobs]
            second_keys = [blob.key for blob in second.blobs]
            # No key appears on both pages, and together they cover everything.
            assert set(first_keys).isdisjoint(second_keys)
            assert set(first_keys) | set(second_keys) == set(keys)
            assert second.is_truncated is False
        finally:
            await self._safe_delete(async_client, *keys)

    @pytest.mark.asyncio
    async def test_list_page_delimiter_groups_common_prefixes(self, async_client):
        """A delimiter collapses nested keys into common_prefixes."""
        prefix = f"{self.TEST_PREFIX}list-page/delimiter-ns/"
        top_key = f"{prefix}top.txt"
        nested_keys = [f"{prefix}sub/a.txt", f"{prefix}sub/b.txt"]
        try:
            await async_client.upload(top_key, b"top level")
            for key in nested_keys:
                await async_client.upload(key, b"nested")
            resp = await async_client.list_page(prefix=prefix, delimiter="/")
            # The top-level object is listed directly; the "sub/" group is
            # collapsed into a single common prefix rather than two objects.
            assert [blob.key for blob in resp.blobs] == [top_key]
            assert f"{prefix}sub/" in resp.common_prefixes
        finally:
            await self._safe_delete(async_client, top_key, *nested_keys)

    @pytest.mark.asyncio
    async def test_list_page_prefix_filters_out_others(self, async_client):
        """Objects outside the requested prefix are excluded from the page."""
        prefix = f"{self.TEST_PREFIX}list-page/filter-ns/"
        included = f"{prefix}included.txt"
        excluded = f"{self.TEST_PREFIX}list-page/other-ns/excluded.txt"
        try:
            await async_client.upload(included, b"in")
            await async_client.upload(excluded, b"out")
            resp = await async_client.list_page(prefix=prefix)
            keys = [blob.key for blob in resp.blobs]
            assert included in keys
            assert excluded not in keys
        finally:
            await self._safe_delete(async_client, included, excluded)

    @pytest.mark.asyncio
    async def test_list_page_invalid_max_results_rejected(self, async_client):
        """max_results below 1 is rejected before any I/O."""
        with pytest.raises(InvalidArgumentException):
            await async_client.list_page(max_results=0)

    # copy / copy_from conformance tests
    #
    # Destination key is in the request path, so a copy to the wrong key
    # fails cassette matching. Source identity and source version travel
    # in a request header on some providers, which replay does not match;
    # the download of the destination is what proves the copy landed the
    # right bytes. Tests that need a versioning-enabled bucket skip when
    # the harness has not supplied one.

    @pytest.mark.asyncio
    async def test_copy_simple(self, async_client, harness):
        """Copy an object within the same bucket and verify content matches."""
        src_key = f"{self.TEST_PREFIX}copy/src-simple.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-simple.txt"
        payload = b"copy conformance payload"
        try:
            await async_client.upload(src_key, payload, content_type="text/plain")
            resp = await async_client.copy(
                src_key,
                dest_bucket=harness.get_bucket(),
                dest_key=dest_key,
            )
            assert resp.key == dest_key
            assert resp.e_tag is not None
            downloaded = await async_client.download(dest_key)
            assert downloaded.content == payload
        finally:
            await self._safe_delete(async_client, src_key, dest_key)

    @pytest.mark.asyncio
    async def test_copy_preserves_content_metadata(self, async_client, harness):
        """Copied object has the same content-type and content-length as source."""
        src_key = f"{self.TEST_PREFIX}copy/src-meta.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-meta.txt"
        payload = b"metadata preservation test"
        try:
            await async_client.upload(src_key, payload, content_type="text/plain")
            await async_client.copy(
                src_key,
                dest_bucket=harness.get_bucket(),
                dest_key=dest_key,
            )
            src_download = await async_client.download(src_key)
            dest_download = await async_client.download(dest_key)
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
            await self._safe_delete(async_client, src_key, dest_key)

    @pytest.mark.asyncio
    async def test_copy_overwrites_existing_destination(self, async_client, harness):
        """Copying to an existing key overwrites the destination content."""
        src_key = f"{self.TEST_PREFIX}copy/src-overwrite.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-overwrite.txt"
        src_payload = b"source content wins"
        try:
            await async_client.upload(src_key, src_payload, content_type="text/plain")
            await async_client.upload(dest_key, b"stale destination content")
            resp = await async_client.copy(
                src_key,
                dest_bucket=harness.get_bucket(),
                dest_key=dest_key,
            )
            assert resp.key == dest_key
            downloaded = await async_client.download(dest_key)
            assert downloaded.content == src_payload
        finally:
            await self._safe_delete(async_client, src_key, dest_key)

    @pytest.mark.asyncio
    async def test_copy_nonexistent_source_raises(self, async_client, harness):
        """Copying from a non-existent source key raises ResourceNotFoundException."""
        missing_key = f"{self.TEST_PREFIX}copy/does-not-exist-abc123.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-missing-src.txt"
        with pytest.raises(ResourceNotFoundException):
            await async_client.copy(
                missing_key,
                dest_bucket=harness.get_bucket(),
                dest_key=dest_key,
            )

    @pytest.mark.asyncio
    async def test_copy_nonexistent_destination_bucket_raises(self, async_client):
        """Copying to a non-existent destination bucket raises an SDK exception."""
        src_key = f"{self.TEST_PREFIX}copy/src-bad-dest-bucket.txt"
        dest_key = f"{self.TEST_PREFIX}copy/dest-bad-bucket.txt"
        bogus_bucket = "multicloud-conformance-nonexistent-bucket-xyz-9f8e7d"
        try:
            await async_client.upload(src_key, b"payload", content_type="text/plain")
            with pytest.raises(SubstrateSdkException):
                await async_client.copy(
                    src_key,
                    dest_bucket=bogus_bucket,
                    dest_key=dest_key,
                )
        finally:
            await self._safe_delete(async_client, src_key)

    @pytest.mark.asyncio
    async def test_copy_from_simple(self, async_client, harness):
        """copy_from a source bucket into this bucket; verify content matches."""
        src_key = f"{self.TEST_PREFIX}copyfrom/src-simple.txt"
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-simple.txt"
        payload = b"copy_from conformance payload"
        try:
            await async_client.upload(src_key, payload, content_type="text/plain")
            resp = await async_client.copy_from(
                harness.get_bucket(),
                src_key,
                dest_key=dest_key,
            )
            assert resp.key == dest_key
            assert resp.e_tag is not None
            downloaded = await async_client.download(dest_key)
            assert downloaded.content == payload
        finally:
            await self._safe_delete(async_client, src_key, dest_key)

    @pytest.mark.asyncio
    async def test_copy_from_preserves_content_metadata(self, async_client, harness):
        """copy_from destination has the same content-type and length as source."""
        src_key = f"{self.TEST_PREFIX}copyfrom/src-meta.txt"
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-meta.txt"
        payload = b"copy_from metadata preservation test"
        try:
            await async_client.upload(src_key, payload, content_type="text/plain")
            await async_client.copy_from(
                harness.get_bucket(),
                src_key,
                dest_key=dest_key,
            )
            src_download = await async_client.download(src_key)
            dest_download = await async_client.download(dest_key)
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
            await self._safe_delete(async_client, src_key, dest_key)

    @pytest.mark.asyncio
    async def test_copy_from_overwrites_existing_destination(
        self, async_client, harness
    ):
        """copy_from to an existing key overwrites the destination content."""
        src_key = f"{self.TEST_PREFIX}copyfrom/src-overwrite.txt"
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-overwrite.txt"
        src_payload = b"copy_from source content wins"
        try:
            await async_client.upload(src_key, src_payload, content_type="text/plain")
            await async_client.upload(dest_key, b"stale destination content")
            resp = await async_client.copy_from(
                harness.get_bucket(),
                src_key,
                dest_key=dest_key,
            )
            assert resp.key == dest_key
            downloaded = await async_client.download(dest_key)
            assert downloaded.content == src_payload
        finally:
            await self._safe_delete(async_client, src_key, dest_key)

    @pytest.mark.asyncio
    async def test_copy_from_nonexistent_source_raises(self, async_client, harness):
        """copy_from with a non-existent source key raises ResourceNotFoundException."""
        missing_key = f"{self.TEST_PREFIX}copyfrom/does-not-exist-abc123.txt"
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-missing-src.txt"
        with pytest.raises(ResourceNotFoundException):
            await async_client.copy_from(
                harness.get_bucket(),
                missing_key,
                dest_key=dest_key,
            )

    @pytest.mark.asyncio
    async def test_copy_from_nonexistent_source_bucket_raises(self, async_client):
        """copy_from with a non-existent source bucket raises an SDK exception."""
        dest_key = f"{self.TEST_PREFIX}copyfrom/dest-bad-src-bucket.txt"
        bogus_bucket = "multicloud-conformance-nonexistent-bucket-xyz-9f8e7d"
        with pytest.raises(SubstrateSdkException):
            await async_client.copy_from(
                bogus_bucket,
                "any-key",
                dest_key=dest_key,
            )

    @pytest.mark.asyncio
    async def test_copy_specific_version(self, versioned_client, harness):
        """Copying a specific src_version_id yields that version's content."""
        if not harness.supports_copy_source_version():
            pytest.skip("Provider async copy cannot pin a source version.")
        dest_bucket = harness.get_versioned_bucket()
        src_key = f"{self.TEST_PREFIX}copy-versioned/src-specific.txt"
        dest_key = f"{self.TEST_PREFIX}copy-versioned/dest-specific-v1.txt"
        v1_payload = b"version one content"
        v2_payload = b"version two content"
        try:
            v1 = await versioned_client.upload(
                src_key, v1_payload, content_type="text/plain"
            )
            await versioned_client.upload(
                src_key, v2_payload, content_type="text/plain"
            )
            assert v1.version_id is not None
            await versioned_client.copy(
                src_key,
                dest_bucket=dest_bucket,
                dest_key=dest_key,
                src_version_id=v1.version_id,
            )
            downloaded = await versioned_client.download(dest_key)
            assert downloaded.content == v1_payload
        finally:
            await self._safe_delete_all_versions(versioned_client, src_key)
            await self._safe_delete_all_versions(versioned_client, dest_key)

    @pytest.mark.asyncio
    async def test_copy_latest_version_when_version_id_none(
        self, versioned_client, harness
    ):
        """Copy without src_version_id uses the latest version of the source."""
        dest_bucket = harness.get_versioned_bucket()
        src_key = f"{self.TEST_PREFIX}copy-versioned/src-latest.txt"
        dest_key = f"{self.TEST_PREFIX}copy-versioned/dest-latest.txt"
        v1_payload = b"version one content"
        v2_payload = b"version two content"
        try:
            await versioned_client.upload(
                src_key, v1_payload, content_type="text/plain"
            )
            await versioned_client.upload(
                src_key, v2_payload, content_type="text/plain"
            )
            await versioned_client.copy(
                src_key,
                dest_bucket=dest_bucket,
                dest_key=dest_key,
            )
            downloaded = await versioned_client.download(dest_key)
            assert downloaded.content == v2_payload
        finally:
            await self._safe_delete_all_versions(versioned_client, src_key)
            await self._safe_delete_all_versions(versioned_client, dest_key)

    @pytest.mark.asyncio
    async def test_copy_response_includes_version_id_on_versioned_bucket(
        self, versioned_client, harness
    ):
        """Copy into a versioned bucket returns a non-null version_id."""
        dest_bucket = harness.get_versioned_bucket()
        src_key = f"{self.TEST_PREFIX}copy-versioned/src-response.txt"
        dest_key = f"{self.TEST_PREFIX}copy-versioned/dest-response.txt"
        try:
            await versioned_client.upload(
                src_key, b"payload", content_type="text/plain"
            )
            resp = await versioned_client.copy(
                src_key,
                dest_bucket=dest_bucket,
                dest_key=dest_key,
            )
            assert resp.version_id is not None
        finally:
            await self._safe_delete_all_versions(versioned_client, src_key)
            await self._safe_delete_all_versions(versioned_client, dest_key)

    # delete / delete_objects conformance tests
    #
    # A single-key delete puts the key in the request path, so targeting the
    # wrong object fails cassette matching. Batch keys travel in the request
    # body, which replay does not match; the follow-up download (or version
    # listing) is what proves the identified objects are gone.

    @pytest.mark.asyncio
    async def test_delete_single(self, async_client):
        """Delete an existing blob; it is no longer downloadable."""
        key = f"{self.TEST_PREFIX}delete/single.txt"
        await async_client.upload(key, b"delete me", content_type="text/plain")
        await async_client.delete(key)
        with pytest.raises(ResourceNotFoundException):
            await async_client.download(key)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_no_error(self, async_client):
        """Deleting a non-existent blob is idempotent (no error)."""
        key = f"{self.TEST_PREFIX}delete/never-existed-xyz789.txt"
        assert await async_client.delete(key) is None

    @pytest.mark.asyncio
    async def test_delete_idempotent_subsequent(self, async_client):
        """A second delete of the same key is a no-op (no error)."""
        key = f"{self.TEST_PREFIX}delete/idempotent.txt"
        await async_client.upload(key, b"once")
        await async_client.delete(key)
        assert await async_client.delete(key) is None
        with pytest.raises(ResourceNotFoundException):
            await async_client.download(key)

    @pytest.mark.asyncio
    async def test_delete_objects_batch(self, async_client):
        """Batch delete removes every requested object; none downloadable."""
        keys = [f"{self.TEST_PREFIX}delete/batch-{i}.txt" for i in range(3)]
        try:
            for key in keys:
                await async_client.upload(key, b"batch payload")
            resp = await async_client.delete_objects(
                [BlobIdentifier(key=k) for k in keys]
            )
            assert resp.errors == []
            for key in keys:
                with pytest.raises(ResourceNotFoundException):
                    await async_client.download(key)
        finally:
            await self._safe_delete(async_client, *keys)

    @pytest.mark.asyncio
    async def test_delete_objects_empty_collection_raises(self, async_client):
        """An empty batch is rejected with InvalidArgumentException."""
        with pytest.raises(InvalidArgumentException):
            await async_client.delete_objects([])

    @pytest.mark.asyncio
    async def test_delete_objects_all_nonexistent(self, async_client):
        """Batch delete of only never-existed keys returns no errors."""
        keys = [
            f"{self.TEST_PREFIX}delete/batch-missing-only-a.txt",
            f"{self.TEST_PREFIX}delete/batch-missing-only-b.txt",
            f"{self.TEST_PREFIX}delete/batch-missing-only-c.txt",
        ]
        resp = await async_client.delete_objects([BlobIdentifier(key=k) for k in keys])
        assert resp.errors == []
        for key in keys:
            with pytest.raises(ResourceNotFoundException):
                await async_client.download(key)

    @pytest.mark.asyncio
    async def test_delete_objects_nonexistent_and_duplicates(self, async_client):
        """Batch delete tolerates non-existent keys and duplicate entries."""
        existing = f"{self.TEST_PREFIX}delete/batch-existing.txt"
        missing = f"{self.TEST_PREFIX}delete/batch-missing-abc.txt"
        try:
            await async_client.upload(existing, b"present")
            identifiers = [
                BlobIdentifier(key=existing),
                BlobIdentifier(key=existing),  # duplicate
                BlobIdentifier(key=missing),  # never existed
            ]
            resp = await async_client.delete_objects(identifiers)
            assert resp.errors == []
            with pytest.raises(ResourceNotFoundException):
                await async_client.download(existing)
        finally:
            await self._safe_delete(async_client, existing)

    @pytest.mark.asyncio
    async def test_delete_specific_version_others_remain(self, versioned_client):
        """Deleting one version leaves the other versions intact."""
        key = f"{self.TEST_PREFIX}delete/versioned-specific.txt"
        try:
            first = await versioned_client.upload(key, b"version-one")
            second = await versioned_client.upload(key, b"version-two")
            assert first.version_id is not None
            assert second.version_id is not None
            assert first.version_id != second.version_id

            await versioned_client.delete(key, version_id=first.version_id)

            remaining = {
                info.version_id
                async for info in versioned_client.list_blob_versions(key)
            }
            assert second.version_id in remaining
            assert first.version_id not in remaining

            # Latest remaining version is still downloadable.
            downloaded = await versioned_client.download(key)
            assert downloaded.content == b"version-two"
        finally:
            await self._safe_delete_all_versions(versioned_client, key)

    @pytest.mark.asyncio
    async def test_delete_all_versions(self, versioned_client):
        """Deleting every version leaves the key with no versions."""
        key = f"{self.TEST_PREFIX}delete/versioned-all.txt"
        try:
            first = await versioned_client.upload(key, b"v1")
            second = await versioned_client.upload(key, b"v2")
            for version_id in (first.version_id, second.version_id):
                await versioned_client.delete(key, version_id=version_id)

            leftover = [info async for info in versioned_client.list_blob_versions(key)]
            assert leftover == []
        finally:
            await self._safe_delete_all_versions(versioned_client, key)

    @pytest.mark.asyncio
    async def test_delete_objects_versioned_mixed_with_nonexistent(
        self, versioned_client
    ):
        """Bulk versioned delete handles real versions mixed with a bad key."""
        key = f"{self.TEST_PREFIX}delete/versioned-bulk.txt"
        missing = f"{self.TEST_PREFIX}delete/versioned-bulk-missing.txt"
        try:
            first = await versioned_client.upload(key, b"bulk-v1")
            second = await versioned_client.upload(key, b"bulk-v2")
            identifiers = [
                BlobIdentifier(key=key, version_id=first.version_id),
                BlobIdentifier(key=key, version_id=second.version_id),
                BlobIdentifier(key=missing),  # never existed
            ]
            resp = await versioned_client.delete_objects(identifiers)
            assert resp.errors == []
            leftover = [info async for info in versioned_client.list_blob_versions(key)]
            assert leftover == []
        finally:
            await self._safe_delete_all_versions(versioned_client, key)

    # download / download_to_path / download_to_file / download_stream tests
    #
    # The object body is in the response, so replay has real teeth for the
    # bytes that come back. Range bounds travel in a request header, which
    # replay does not match; those tests still pin the client-layer
    # validation of invalid ranges. Streaming range requests additionally
    # match the Range header so replay catches a missing or malformed header.

    @pytest.mark.asyncio
    async def test_download_simple(self, async_client):
        """Upload then download; verify content matches."""
        key = f"{self.TEST_PREFIX}download/simple.txt"
        payload = b"download conformance test content"
        try:
            await async_client.upload(key, payload, content_type="text/plain")
            resp = await async_client.download(key)
            assert resp.content == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_metadata_populated(self, async_client):
        """Download response includes correct metadata."""
        key = f"{self.TEST_PREFIX}download/with-meta.txt"
        payload = b"metadata download test"
        try:
            await async_client.upload(
                key,
                payload,
                content_type="text/plain",
                metadata={"test-key": "test-value"},
            )
            resp = await async_client.download(key)
            assert resp.metadata is not None
            assert resp.metadata.content_type == "text/plain"
            assert resp.metadata.content_length == len(payload)
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_user_metadata_round_trip(self, async_client):
        """User-defined metadata set on upload is returned on download."""
        key = f"{self.TEST_PREFIX}download/user-meta.txt"
        user_metadata = {"author": "conformance", "purpose": "round-trip"}
        try:
            await async_client.upload(
                key, b"hello", content_type="text/plain", metadata=user_metadata
            )
            resp = await async_client.download(key)
            for meta_key, meta_value in user_metadata.items():
                assert resp.metadata.metadata.get(meta_key) == meta_value
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_nonexistent_key(self, async_client):
        """Downloading a non-existent key raises ResourceNotFoundException."""
        key = f"{self.TEST_PREFIX}download/does-not-exist-xyz123.txt"
        with pytest.raises(ResourceNotFoundException):
            await async_client.download(key)

    @pytest.mark.asyncio
    async def test_download_empty_blob(self, async_client):
        """Download a zero-byte blob."""
        key = f"{self.TEST_PREFIX}download/empty.txt"
        try:
            await async_client.upload(key, b"")
            resp = await async_client.download(key)
            assert resp.content == b""
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_binary_content(self, async_client):
        """Download binary content preserves all bytes."""
        key = f"{self.TEST_PREFIX}download/binary.bin"
        payload = bytes(range(256))
        try:
            await async_client.upload(
                key, payload, content_type="application/octet-stream"
            )
            resp = await async_client.download(key)
            assert resp.content == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_large_blob(self, async_client):
        """Download a 64KiB blob and verify every byte."""
        key = f"{self.TEST_PREFIX}download/large.bin"
        payload = b"Y" * (64 * 1024)
        try:
            await async_client.upload(
                key, payload, content_type="application/octet-stream"
            )
            resp = await async_client.download(key)
            assert len(resp.content) == 64 * 1024
            assert resp.content == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_stream_iterates_chunks(self, async_client):
        """Async iteration reassembles the exact uploaded bytes."""
        key = f"{self.TEST_PREFIX}download-stream/iter.bin"
        payload = b"stream-iteration-payload-" * 512
        try:
            await async_client.upload(
                key, payload, content_type="application/octet-stream"
            )
            async with await async_client.download_stream(key) as stream:
                chunks = [chunk async for chunk in stream]
            assert b"".join(chunks) == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_stream_read_all(self, async_client):
        """read() returns every remaining byte."""
        key = f"{self.TEST_PREFIX}download-stream/read.bin"
        payload = b"async stream read payload " * 128
        try:
            await async_client.upload(
                key, payload, content_type="application/octet-stream"
            )
            async with await async_client.download_stream(key) as stream:
                content = await stream.read()
            assert content == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_stream_large_object_yields_multiple_chunks(
        self, async_client
    ):
        """A payload larger than the native pull size arrives in several chunks."""
        key = f"{self.TEST_PREFIX}download-stream/large.bin"
        payload = b"A" * 8192 + b"B" * 8192 + b"C" * 8192
        try:
            await async_client.upload(
                key, payload, content_type="application/octet-stream"
            )
            async with await async_client.download_stream(key) as stream:
                chunks = [chunk async for chunk in stream]
            assert len(chunks) >= 2
            assert b"".join(chunks) == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_stream_range(self, async_client):
        """Streaming honors inclusive range bounds."""
        key = f"{self.TEST_PREFIX}download-stream/range.txt"
        payload = b"0123456789ABCDEF"
        try:
            await async_client.upload(key, payload)
            async with await async_client.download_stream(
                key, start=2, end=7
            ) as stream:
                content = await stream.read()
                content_length = stream.metadata.content_length
            assert content == b"234567"
            assert content_length == 6
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_stream_context_closes_partial_read(self, async_client):
        """Context exit closes a stream after only part of it was read."""
        key = f"{self.TEST_PREFIX}download-stream/partial.txt"
        payload = b"partial-stream-payload"
        try:
            await async_client.upload(key, payload)
            stream = await async_client.download_stream(key)
            async with stream:
                prefix = await stream.read(7)
            assert prefix == b"partial"
            assert stream.closed is True
            with pytest.raises(ValueError):
                await stream.read(1)
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_stream_nonexistent_key(self, async_client):
        """Opening a missing stream raises the specific translated exception."""
        key = f"{self.TEST_PREFIX}download-stream/does-not-exist-xyz123.txt"
        await async_client.upload(key, b"deleted before download")
        await async_client.delete(key)
        with pytest.raises(ResourceNotFoundException):
            await async_client.download_stream(key)

    @pytest.mark.asyncio
    async def test_download_range_first_bytes(self, async_client):
        """Download the first N bytes of an object."""
        key = f"{self.TEST_PREFIX}download-range/first.txt"
        payload = b"0123456789ABCDEF"
        try:
            await async_client.upload(key, payload)
            resp = await async_client.download(key, start=0, end=3)
            assert resp.content == b"0123"
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_range_middle(self, async_client):
        """Download a middle byte range."""
        key = f"{self.TEST_PREFIX}download-range/middle.txt"
        payload = b"0123456789ABCDEF"
        try:
            await async_client.upload(key, payload)
            resp = await async_client.download(key, start=4, end=9)
            assert resp.content == b"456789"
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_range_single_byte(self, async_client):
        """Download exactly one byte (start == end)."""
        key = f"{self.TEST_PREFIX}download-range/single.txt"
        payload = b"0123456789ABCDEF"
        try:
            await async_client.upload(key, payload)
            resp = await async_client.download(key, start=7, end=7)
            assert resp.content == b"7"
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_range_open_ended_start(self, async_client):
        """Range with only ``start`` returns bytes from ``start`` to EOF."""
        key = f"{self.TEST_PREFIX}download-range/open-start.txt"
        payload = b"0123456789ABCDEF"
        try:
            await async_client.upload(key, payload)
            resp = await async_client.download(key, start=10)
            assert resp.content == b"ABCDEF"
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_range_invalid_boundaries_rejected(self, async_client):
        """A range with ``end`` < ``start`` is rejected before any I/O."""
        key = f"{self.TEST_PREFIX}download-range/invalid.txt"
        with pytest.raises(InvalidArgumentException):
            await async_client.download(key, start=10, end=5)

    @pytest.mark.asyncio
    async def test_download_range_negative_boundary_rejected(self, async_client):
        """A negative range boundary is rejected before any I/O."""
        key = f"{self.TEST_PREFIX}download-range/negative.txt"
        with pytest.raises(InvalidArgumentException):
            await async_client.download(key, start=-1, end=5)

    @pytest.mark.asyncio
    async def test_download_to_path_round_trip(self, async_client, tmp_path):
        """download_to_path writes the object to a local file."""
        key = f"{self.TEST_PREFIX}download-to-path/simple.txt"
        payload = b"download_to_path payload"
        destination = tmp_path / "out.txt"
        try:
            await async_client.upload(key, payload, content_type="text/plain")
            resp = await async_client.download_to_path(key, destination)
            assert destination.exists()
            assert destination.read_bytes() == payload
            assert resp.metadata.content_length == len(payload)
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_to_path_creates_parent_dirs(self, async_client, tmp_path):
        """download_to_path creates missing parent dirs when requested."""
        key = f"{self.TEST_PREFIX}download-to-path/nested.txt"
        payload = b"nested destination payload"
        destination = tmp_path / "a" / "b" / "c" / "nested.txt"
        try:
            await async_client.upload(key, payload)
            await async_client.download_to_path(
                key, destination, create_parent_dirs=True
            )
            assert destination.exists()
            assert destination.read_bytes() == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_to_path_missing_parent_dir_without_flag(
        self, async_client, tmp_path
    ):
        """Without ``create_parent_dirs`` a missing parent dir raises an SDK error."""
        key = f"{self.TEST_PREFIX}download-to-path/missing-parent.txt"
        destination = tmp_path / "does-not-exist" / "out.txt"
        try:
            await async_client.upload(key, b"payload")
            with pytest.raises(SubstrateSdkException):
                await async_client.download_to_path(key, destination)
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_download_to_file_round_trip(self, async_client):
        """download_to_file writes the object into an in-memory buffer."""
        key = f"{self.TEST_PREFIX}download-to-file/simple.txt"
        payload = b"download_to_file payload"
        buffer = io.BytesIO()
        try:
            await async_client.upload(key, payload, content_type="text/plain")
            resp = await async_client.download_to_file(key, buffer)
            assert buffer.getvalue() == payload
            assert resp.metadata.content_length == len(payload)
        finally:
            await self._safe_delete(async_client, key)

    # upload conformance tests
    #
    # Request bodies are not matched on replay, so sending empty bytes still
    # passes these tests in CI. What replay does catch is the follow-up
    # download: the cassette carries the bytes that were actually stored
    # when the cassette was recorded. A mapping bug that returns the wrong
    # download payload fails; a request-side skip of the upload body does
    # not. tags and object_lock are not parameters on the async upload API.

    @pytest.mark.asyncio
    async def test_upload_simple(self, async_client):
        """Upload a simple text blob, then read it back and verify the bytes."""
        key = f"{self.TEST_PREFIX}upload/simple.txt"
        payload = b"Hello, conformance test!"
        try:
            resp = await async_client.upload(key, payload, content_type="text/plain")
            assert resp.key == key
            assert resp.e_tag is not None
            fetched = await async_client.download(key)
            assert fetched.content == payload
            assert fetched.metadata.content_length == len(payload)
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_empty_content(self, async_client):
        """Upload a zero-byte blob and verify it reads back as empty."""
        key = f"{self.TEST_PREFIX}upload/empty.txt"
        try:
            resp = await async_client.upload(key, b"", content_type="text/plain")
            assert resp.key == key
            fetched = await async_client.download(key)
            assert fetched.content == b""
            assert fetched.metadata.content_length == 0
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_binary_content(self, async_client):
        """Upload binary data and verify every byte round-trips intact."""
        key = f"{self.TEST_PREFIX}upload/binary.bin"
        payload = bytes(range(256)) * 4  # 1024 bytes of binary
        try:
            resp = await async_client.upload(
                key, payload, content_type="application/octet-stream"
            )
            assert resp.key == key
            fetched = await async_client.download(key)
            assert fetched.content == payload
            assert fetched.metadata.content_length == len(payload)
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_with_metadata(self, async_client):
        """Upload with user metadata, then read it back and verify it persisted."""
        key = f"{self.TEST_PREFIX}upload/with-metadata.txt"
        metadata = {"author": "conformance-test", "version": "1.0"}
        payload = b"metadata test"
        try:
            resp = await async_client.upload(
                key, payload, content_type="text/plain", metadata=metadata
            )
            assert resp.key == key
            fetched = await async_client.download(key)
            assert fetched.content == payload
            assert fetched.metadata.content_type == "text/plain"
            for meta_key, meta_value in metadata.items():
                assert fetched.metadata.metadata.get(meta_key) == meta_value
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_overwrites_existing(self, async_client):
        """Uploading to the same key overwrites the content."""
        key = f"{self.TEST_PREFIX}upload/overwrite.txt"
        try:
            await async_client.upload(key, b"version1")
            await async_client.upload(key, b"version2")
            resp = await async_client.download(key)
            assert resp.content == b"version2"
        finally:
            await self._safe_delete(async_client, key)

    # Conditional-write conformance tests
    #
    # create_if_absent is supported on every provider. ETag- and
    # version-conditioned overwrites are asymmetric across clouds, so those
    # tests are gated on a harness capability hook each provider must declare.

    @pytest.mark.asyncio
    async def test_create_if_absent(self, async_client):
        """create_if_absent writes a new key and rejects an existing one.

        The first write creates the object; a second create_if_absent against
        the now-existing key raises ResourceExistsException and leaves the
        original content intact.
        """
        key = f"{self.TEST_PREFIX}conditional/create-if-absent.txt"
        try:
            await async_client.upload(key, b"fresh", create_if_absent=True)
            assert (await async_client.download(key)).content == b"fresh"
            with pytest.raises(ResourceExistsException):
                await async_client.upload(key, b"replacement", create_if_absent=True)
            # The failed conditional write must not have replaced the object.
            assert (await async_client.download(key)).content == b"fresh"
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_if_match_overwrite_and_stale(self, async_client, harness):
        """if_match overwrites on the current ETag and fails on a stale one.

        Overwriting with the object's current ETag succeeds; reusing that now
        outdated ETag against the moved-forward object raises
        FailedPreconditionException.
        """
        if not harness.supports_write_if_match():
            pytest.skip("Provider does not enforce ETag preconditions on writes.")
        key = f"{self.TEST_PREFIX}conditional/if-match.txt"
        try:
            await async_client.upload(key, b"v1")
            first_etag = (await async_client.get_metadata(key)).e_tag
            await async_client.upload(key, b"v2", if_match=first_etag)
            assert (await async_client.download(key)).content == b"v2"
            # first_etag is now stale: the object has moved on to v2.
            with pytest.raises(FailedPreconditionException):
                await async_client.upload(key, b"v3", if_match=first_etag)
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_if_match_on_missing_object_raises_failed_precondition(
        self, async_client, harness
    ):
        """if_match against a missing key raises FailedPreconditionException."""
        if not harness.supports_write_if_match():
            pytest.skip("Provider does not enforce ETag preconditions on writes.")
        key = f"{self.TEST_PREFIX}conditional/if-match-missing.txt"
        with pytest.raises(FailedPreconditionException):
            await async_client.upload(key, b"data", if_match='"nonexistent-etag"')

    @pytest.mark.asyncio
    async def test_if_version_match_overwrite_and_stale(self, async_client, harness):
        """if_version_match overwrites on the current version and fails on a stale one.

        Overwriting with the object's current version id succeeds; reusing that
        now outdated version against the moved-forward object raises
        FailedPreconditionException.
        """
        if not harness.supports_write_if_version_match():
            pytest.skip("Provider does not enforce version preconditions on writes.")
        key = f"{self.TEST_PREFIX}conditional/if-version.txt"
        try:
            first_version = (await async_client.upload(key, b"v1")).version_id
            await async_client.upload(key, b"v2", if_version_match=first_version)
            assert (await async_client.download(key)).content == b"v2"
            # first_version is now stale: the object has moved on to v2.
            with pytest.raises(FailedPreconditionException):
                await async_client.upload(key, b"v3", if_version_match=first_version)
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_if_version_match_on_missing_object_raises_failed_precondition(
        self, async_client, harness
    ):
        """if_version_match against a missing key raises FailedPreconditionException."""
        if not harness.supports_write_if_version_match():
            pytest.skip("Provider does not enforce version preconditions on writes.")
        key = f"{self.TEST_PREFIX}conditional/if-version-missing.txt"
        with pytest.raises(FailedPreconditionException):
            await async_client.upload(key, b"data", if_version_match="1")

    # Server-side-encryption conformance tests
    #
    # The cloud-agnostic metadata surface does not expose which key encrypted an
    # object, so the strongest portable assertion is a round-trip. The granular
    # "this exact key/header was applied" assertion lives in the unit tests.

    @pytest.mark.asyncio
    async def test_upload_with_kms_key_roundtrips(self, async_client, kms_key_id):
        """upload with a server-side-encryption key round-trips the content."""
        key = f"{self.TEST_PREFIX}encryption/kms-upload.txt"
        payload = b"kms-encrypted async conformance payload"
        try:
            await async_client.upload(key, payload, kms_key_id=kms_key_id)
            resp = await async_client.download(key)
            assert resp.content == payload
        finally:
            await self._safe_delete(async_client, key)

    # Storage-class conformance tests
    #
    # Storage classes are provider-native, so the harness supplies a non-default
    # class the provider accepts. Listing reports the class an object was written
    # with, which makes the round trip observable without a provider SDK.

    @pytest.mark.asyncio
    async def test_upload_with_storage_class_is_reported_by_listing(
        self, async_client, storage_class
    ):
        """upload with storage_class writes the object in that class."""
        prefix = f"{self.TEST_PREFIX}storage-class/upload-ns/"
        key = f"{prefix}tiered.txt"
        payload = b"storage-class async conformance payload"
        try:
            await async_client.upload(key, payload, storage_class=storage_class)
            assert (await async_client.download(key)).content == payload
            resp = await async_client.list_page(prefix=prefix)
            assert [(b.key, b.storage_class) for b in resp.blobs] == [
                (key, storage_class)
            ]
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_from_path_with_storage_class_is_reported_by_listing(
        self, async_client, storage_class, tmp_path
    ):
        """upload_from_path with storage_class writes the object in that class."""
        prefix = f"{self.TEST_PREFIX}storage-class/upload-from-path-ns/"
        key = f"{prefix}tiered.txt"
        payload = b"storage-class async from-path payload"
        source = tmp_path / "storage-class-source.txt"
        source.write_bytes(payload)
        try:
            await async_client.upload_from_path(
                key, source, storage_class=storage_class
            )
            assert (await async_client.download(key)).content == payload
            resp = await async_client.list_page(prefix=prefix)
            assert [(b.key, b.storage_class) for b in resp.blobs] == [
                (key, storage_class)
            ]
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_with_unknown_storage_class_raises_invalid_argument(
        self, async_client
    ):
        """A storage class the provider does not recognise fails the write."""
        key = f"{self.TEST_PREFIX}storage-class/unknown.txt"
        try:
            with pytest.raises(InvalidArgumentException):
                await async_client.upload(
                    key, b"data", storage_class="NOT_A_STORAGE_CLASS"
                )
            assert await async_client.does_object_exist(key) is False
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_nested_path(self, async_client):
        """Upload to a deeply nested path and verify the content round-trips."""
        key = f"{self.TEST_PREFIX}upload/a/b/c/d/e/deep.txt"
        payload = b"deeply nested"
        try:
            resp = await async_client.upload(key, payload)
            assert resp.key == key
            fetched = await async_client.download(key)
            assert fetched.content == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_binary_io_stream(self, async_client):
        """Upload from an in-memory binary stream (BinaryIO)."""
        key = f"{self.TEST_PREFIX}upload/from-stream.bin"
        payload = b"stream-uploaded payload " * 32
        stream = io.BytesIO(payload)
        try:
            resp = await async_client.upload(
                key, stream, content_type="application/octet-stream"
            )
            assert resp.key == key
            fetched = await async_client.download(key)
            assert fetched.content == payload
        finally:
            await self._safe_delete(async_client, key)

    @pytest.mark.asyncio
    async def test_upload_empty_key_rejected(self, async_client):
        """Uploading with an empty key raises a validation error."""
        with pytest.raises(InvalidArgumentException):
            await async_client.upload("", b"payload")

    # get_object_lock read-back tests.
    #
    # Scope note: these cover the lock states the async *public* API can
    # actually produce. Establishing a retention period is deliberately not
    # among them — the async ``upload`` takes no ``object_lock`` argument, and
    # ``update_object_retention`` refuses to *establish* retention (the shared
    # ``resolve_and_validate`` rule raises when the object has no current
    # retention). With no async path to create a retained object, the
    # governance/compliance read-back cases cannot be exercised here without
    # first extending the async API, so only the legal-hold, no-lock, and
    # missing-key states are covered. The sync suite still covers retention via
    # its ``upload(object_lock=...)`` path.

    @pytest.mark.asyncio
    async def test_get_object_lock_legal_hold_only(self, object_lock_client):
        """A legal hold with no retention reads back via get_object_lock.

        Runs on an object-lock-enabled bucket (skipped otherwise). The object is
        uploaded without retention, then a legal hold is applied and read back —
        so ``mode`` is None while ``legal_hold`` is True.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}object-lock/legal-hold-only.txt"
        try:
            await client.upload(key, b"legally-held")
            await client.update_legal_hold(key, True)
            lock = await client.get_object_lock(key)
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
            await self._safe_release_lock_and_delete(client, key)

    @pytest.mark.asyncio
    async def test_get_object_lock_none_when_unlocked(self, object_lock_client):
        """get_object_lock returns None for an object with no lock configured.

        Runs on an object-lock-enabled bucket (skipped otherwise). An object
        uploaded without retention or a hold has no lock state, so the read
        returns None rather than an empty configuration. This assumes the
        bucket carries no default retention configuration; otherwise new
        objects would be auto-retained and the read would not be None.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}object-lock/unlocked.txt"
        payload = b"no lock here"
        try:
            await client.upload(key, payload)
            # Prove the object is present with its exact bytes before reading the
            # lock, so a None result means "exists but unlocked" rather than
            # "object missing" — the two are otherwise indistinguishable here.
            assert (await client.download(key)).content == payload
            assert await client.get_object_lock(key) is None
        finally:
            # Object-lock buckets are versioning-enabled, so a plain keyed
            # delete only inserts a delete marker; remove every version so the
            # recorded bucket doesn't accumulate objects across record runs.
            await self._safe_delete_all_versions(client, key)

    @pytest.mark.asyncio
    async def test_get_object_lock_nonexistent_key(self, object_lock_client):
        """get_object_lock on a missing key raises ResourceNotFoundException.

        Runs on the object-lock-enabled bucket (skipped otherwise) so the
        lookup reaches the missing-object check rather than a "bucket has no
        object lock" error.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}object-lock/does-not-exist-xyz123.txt"
        with pytest.raises(ResourceNotFoundException):
            await client.get_object_lock(key)
