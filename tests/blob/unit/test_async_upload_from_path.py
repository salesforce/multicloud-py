"""Unit tests for async upload_from_path at the driver and client layers.

Tests validate:
- the file's bytes are what gets uploaded
- content type is detected from the extension, and an explicit one wins
- content length comes from the file, and an explicit one wins
- a missing file, a directory, and an empty key are all rejected before any
  request is made
- str and os.PathLike are accepted, not only Path
- every conditional-write and encryption parameter reaches the provider hook
- the file is read off the event loop
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Mapping, Optional
from unittest.mock import patch

import pytest
from multicloud.blob.blob_client.async_blob.async_client import AsyncBucketClient
from multicloud.blob.blob_client.async_blob.async_driver import (
    ASYNC_BLOB_REGISTRY,
    AsyncUploadResponse,
    BaseAsyncBlobStore,
)
from multicloud.common.exceptions import (
    InvalidArgumentException,
    SubstrateSdkException,
    UnknownException,
)


class RecordingUploadStore(BaseAsyncBlobStore):
    """Records exactly what the driver hands to the provider hook."""

    provider_id = "recording-upload"

    def __init__(self, **kwargs):
        super().__init__(bucket=kwargs.get("bucket", "b"))
        self.uploads: list[dict] = []
        self.raise_native: Optional[Exception] = None

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        if isinstance(exc, SubstrateSdkException):
            return exc
        return UnknownException(str(exc))

    async def do_upload(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: Optional[str],
        content_length: int,
        metadata: Mapping[str, str],
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        storage_class: Optional[str] = None,
    ) -> AsyncUploadResponse:
        if self.raise_native is not None:
            raise self.raise_native
        self.uploads.append(
            {
                "key": key,
                "payload": payload,
                "content_type": content_type,
                "content_length": content_length,
                "metadata": dict(metadata),
                "create_if_absent": create_if_absent,
                "if_match": if_match,
                "if_version_match": if_version_match,
                "kms_key_id": kms_key_id,
                "use_kms_managed_key": use_kms_managed_key,
                "storage_class": storage_class,
            }
        )
        return AsyncUploadResponse(key=key, e_tag="etag-1", version_id="v-1")

    async def do_download(self, request):
        raise NotImplementedError

    async def do_copy(self, request, *, correlation_id):
        raise NotImplementedError

    async def do_delete(self, key, *, version_id=None):
        raise NotImplementedError

    async def do_delete_objects(self, identifiers):
        raise NotImplementedError

    async def do_get_tags(self, key):
        raise NotImplementedError

    async def do_set_tags(self, key, tags):
        raise NotImplementedError

    async def do_upload_directory(self, local_source_directory, **kwargs):
        raise NotImplementedError

    async def do_download_directory(self, prefix, local_destination, **kwargs):
        raise NotImplementedError

    async def do_delete_directory(self, prefix, **kwargs):
        raise NotImplementedError

    async def do_get_metadata(self, key, *, version_id=None):
        raise NotImplementedError

    async def do_list_page(self, *, prefix, delimiter, max_results, pagination_token):
        raise NotImplementedError

    async def do_does_object_exist(self, key, *, version_id):
        raise NotImplementedError

    async def do_does_bucket_exist(self):
        raise NotImplementedError

    async def close(self) -> None:
        return None


@pytest.fixture
def store():
    return RecordingUploadStore(bucket="b")


@pytest.mark.asyncio
class TestDriverUploadFromPath:
    """Tests for BaseAsyncBlobStore.upload_from_path."""

    async def test_uploads_the_file_contents(self, store, tmp_path):
        source = tmp_path / "payload.bin"
        source.write_bytes(b"\x00\x01\x02file bytes")

        response = await store.upload_from_path("k.bin", source)

        assert store.uploads[0]["payload"] == b"\x00\x01\x02file bytes"
        assert response.key == "k.bin"

    async def test_content_type_is_detected_from_the_extension(self, store, tmp_path):
        source = tmp_path / "doc.json"
        source.write_bytes(b"{}")

        await store.upload_from_path("k.json", source)

        assert store.uploads[0]["content_type"] == "application/json"

    async def test_detection_follows_the_extension_not_a_default(self, store, tmp_path):
        """A second extension must map to its own type, not a constant."""
        source = tmp_path / "notes.txt"
        source.write_bytes(b"hello")

        await store.upload_from_path("k.txt", source)

        assert store.uploads[0]["content_type"] == "text/plain"

    async def test_explicit_content_type_wins_over_detection(self, store, tmp_path):
        source = tmp_path / "doc.json"
        source.write_bytes(b"{}")

        await store.upload_from_path("k.json", source, content_type="text/csv")

        assert store.uploads[0]["content_type"] == "text/csv"

    async def test_unknown_extension_falls_back_to_octet_stream(self, store, tmp_path):
        source = tmp_path / "mystery.zzzzz"
        source.write_bytes(b"?")

        await store.upload_from_path("k.zzzzz", source)

        assert store.uploads[0]["content_type"] == "application/octet-stream"

    async def test_content_length_comes_from_the_file(self, store, tmp_path):
        source = tmp_path / "sized.bin"
        source.write_bytes(b"x" * 4321)

        await store.upload_from_path("k.bin", source)

        assert store.uploads[0]["content_length"] == 4321

    async def test_negative_content_length_is_rejected_before_reading(
        self, store, tmp_path
    ):
        """Matches upload(), which refuses the same value.

        Rejected before the filesystem is touched, so a bad argument does not
        cost a read.
        """
        source = tmp_path / "sized.bin"
        source.write_bytes(b"x" * 10)

        with pytest.raises(InvalidArgumentException):
            await store.upload_from_path("k.bin", source, content_length=-5)

        assert store.uploads == []

    async def test_explicit_content_length_wins(self, store, tmp_path):
        source = tmp_path / "sized.bin"
        source.write_bytes(b"x" * 4321)

        await store.upload_from_path("k.bin", source, content_length=99)

        assert store.uploads[0]["content_length"] == 99

    async def test_missing_file_is_rejected_before_any_request(self, store, tmp_path):
        with pytest.raises(InvalidArgumentException):
            await store.upload_from_path("k.bin", tmp_path / "not-here.bin")

        assert store.uploads == []

    async def test_directory_is_rejected_before_any_request(self, store, tmp_path):
        directory = tmp_path / "a-directory"
        directory.mkdir()

        with pytest.raises(InvalidArgumentException):
            await store.upload_from_path("k.bin", directory)

        assert store.uploads == []

    async def test_empty_key_is_rejected(self, store, tmp_path):
        source = tmp_path / "payload.bin"
        source.write_bytes(b"data")

        with pytest.raises(InvalidArgumentException):
            await store.upload_from_path("", source)

        assert store.uploads == []

    async def test_accepts_a_string_path(self, store, tmp_path):
        """A str path is accepted here even though upload() rejects one."""
        source = tmp_path / "as-string.txt"
        source.write_bytes(b"from a string path")

        await store.upload_from_path("k.txt", str(source))

        assert store.uploads[0]["payload"] == b"from a string path"

    async def test_accepts_an_os_pathlike(self, store, tmp_path):
        class Wrapped:
            def __init__(self, target):
                self._target = target

            def __fspath__(self) -> str:
                return str(self._target)

        source = tmp_path / "as-pathlike.txt"
        source.write_bytes(b"from a pathlike")

        await store.upload_from_path("k.txt", Wrapped(source))

        assert store.uploads[0]["payload"] == b"from a pathlike"

    async def test_metadata_and_encryption_reach_the_hook(self, store, tmp_path):
        source = tmp_path / "conditional.txt"
        source.write_bytes(b"conditional")

        await store.upload_from_path(
            "k.txt",
            source,
            metadata={"team": "multicloud"},
            create_if_absent=True,
            kms_key_id="key-arn",
            use_kms_managed_key=True,
        )

        sent = store.uploads[0]
        assert sent["metadata"] == {"team": "multicloud"}
        assert sent["create_if_absent"] is True
        assert sent["kms_key_id"] == "key-arn"
        assert sent["use_kms_managed_key"] is True

    async def test_match_preconditions_reach_the_hook(self, store, tmp_path):
        source = tmp_path / "matched.txt"
        source.write_bytes(b"matched")

        await store.upload_from_path(
            "k.txt", source, if_match="etag-abc", if_version_match="v-9"
        )

        sent = store.uploads[0]
        assert sent["if_match"] == "etag-abc"
        assert sent["if_version_match"] == "v-9"
        assert sent["create_if_absent"] is False

    async def test_mutually_exclusive_preconditions_are_rejected(self, store, tmp_path):
        """The shared write-precondition rules apply here too."""
        source = tmp_path / "clash.txt"
        source.write_bytes(b"clash")

        with pytest.raises(InvalidArgumentException):
            await store.upload_from_path(
                "k.txt", source, create_if_absent=True, if_match="etag-abc"
            )

        assert store.uploads == []

    async def test_create_if_absent_with_a_version_precondition_is_rejected(
        self, store, tmp_path
    ):
        """Asserting absence and a specific version at once is contradictory.

        Covered separately from the if_match clash so that every precondition
        the validator is given is one it actually inspects.
        """
        source = tmp_path / "clash2.txt"
        source.write_bytes(b"clash")

        with pytest.raises(InvalidArgumentException):
            await store.upload_from_path(
                "k.txt", source, create_if_absent=True, if_version_match="v-1"
            )

        assert store.uploads == []

    async def test_blank_version_precondition_is_rejected(self, store, tmp_path):
        """A blank version token matches nothing, so it is a caller mistake."""
        source = tmp_path / "blank.txt"
        source.write_bytes(b"blank")

        with pytest.raises(InvalidArgumentException):
            await store.upload_from_path("k.txt", source, if_version_match="   ")

        assert store.uploads == []

    async def test_defaults_are_not_silently_enabled(self, store, tmp_path):
        source = tmp_path / "plain.txt"
        source.write_bytes(b"plain")

        await store.upload_from_path("k.txt", source)

        sent = store.uploads[0]
        assert sent["create_if_absent"] is False
        assert sent["if_match"] is None
        assert sent["if_version_match"] is None
        assert sent["kms_key_id"] is None
        assert sent["use_kms_managed_key"] is False
        assert sent["metadata"] == {}

    async def test_the_file_is_read_off_the_event_loop(self, store, tmp_path):
        """The loop must keep running for the whole duration of a slow read.

        A ticker runs alongside a read made to take READ_SECONDS. Asserting a
        floor proportional to that duration is what makes this discriminating:
        ``ticks > 0`` would also pass for a blocking read that merely happened
        to follow an await, since one yield is enough to advance the counter
        once.
        """
        read_seconds = 0.2
        tick_seconds = 0.01
        # Generous margin for scheduler jitter; a blocking read yields none of
        # these, while an off-loop read leaves room for roughly 20.
        minimum_ticks = 8

        source = tmp_path / "slow.bin"
        source.write_bytes(b"slow payload")
        real_read_bytes = Path.read_bytes

        def slow_read(self):
            time.sleep(read_seconds)
            return real_read_bytes(self)

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(tick_seconds)
                ticks += 1

        with patch.object(Path, "read_bytes", slow_read):
            spinner = asyncio.create_task(ticker())
            await asyncio.sleep(0)  # let the ticker reach its first await
            try:
                await store.upload_from_path("k.bin", source)
            finally:
                spinner.cancel()

        assert store.uploads[0]["payload"] == b"slow payload"
        assert ticks >= minimum_ticks, (
            f"the loop advanced only {ticks} times during a {read_seconds}s "
            f"read, so it was blocked rather than yielding"
        )


@pytest.mark.asyncio
class TestClientUploadFromPath:
    """Tests for AsyncBucketClient.upload_from_path."""

    async def _client(self, store: RecordingUploadStore) -> AsyncBucketClient:
        with patch.object(ASYNC_BLOB_REGISTRY, "get", return_value=lambda **kw: store):
            return AsyncBucketClient(provider="recording-upload", bucket="b")

    async def test_forwards_the_file_and_returns_the_response(self, tmp_path):
        store = RecordingUploadStore(bucket="b")
        client = await self._client(store)
        source = tmp_path / "client.txt"
        source.write_bytes(b"through the client")

        response = await client.upload_from_path("k.txt", source)

        assert store.uploads[0]["payload"] == b"through the client"
        assert response.e_tag == "etag-1"
        assert response.version_id == "v-1"

    async def test_every_argument_reaches_the_store(self, tmp_path):
        """The client is a forwarder, so each value must arrive unchanged.

        Dropping a write precondition here would quietly turn a conditional
        write into an unconditional overwrite, so this is asserted per
        argument rather than by spot check.
        """
        store = RecordingUploadStore(bucket="b")
        client = await self._client(store)
        source = tmp_path / "forwarded.json"
        source.write_bytes(b'{"forwarded": true}')

        await client.upload_from_path(
            "destination/key.json",
            source,
            content_type="application/vnd.custom+json",
            content_length=19,
            metadata={"team": "multicloud", "purpose": "forwarding"},
            if_match="etag-xyz",
            if_version_match="v-77",
            kms_key_id="arn:aws:kms:key",
            use_kms_managed_key=True,
        )

        sent = store.uploads[0]
        assert sent["key"] == "destination/key.json"
        assert sent["payload"] == b'{"forwarded": true}'
        assert sent["content_type"] == "application/vnd.custom+json"
        assert sent["content_length"] == 19
        assert sent["metadata"] == {"team": "multicloud", "purpose": "forwarding"}
        assert sent["if_match"] == "etag-xyz"
        assert sent["if_version_match"] == "v-77"
        assert sent["kms_key_id"] == "arn:aws:kms:key"
        assert sent["use_kms_managed_key"] is True

    async def test_storage_class_reaches_the_store(self, tmp_path):
        store = RecordingUploadStore(bucket="b")
        client = await self._client(store)
        source = tmp_path / "tiered.txt"
        source.write_bytes(b"tiered")

        await client.upload_from_path("k.txt", source, storage_class="COLDLINE")

        assert store.uploads[0]["storage_class"] == "COLDLINE"

    async def test_blank_storage_class_rejected_before_reading(self, tmp_path):
        store = RecordingUploadStore(bucket="b")
        client = await self._client(store)
        source = tmp_path / "tiered.txt"
        source.write_bytes(b"tiered")

        with pytest.raises(InvalidArgumentException, match="storage_class"):
            await client.upload_from_path("k.txt", source, storage_class="")

        assert store.uploads == []

    async def test_create_if_absent_reaches_the_store(self, tmp_path):
        """Asserted separately: it cannot be combined with if_match."""
        store = RecordingUploadStore(bucket="b")
        client = await self._client(store)
        source = tmp_path / "create-only.txt"
        source.write_bytes(b"create only")

        await client.upload_from_path("k.txt", source, create_if_absent=True)

        assert store.uploads[0]["create_if_absent"] is True

    async def test_client_defaults_are_not_silently_enabled(self, tmp_path):
        """Omitting the optional arguments must not switch any of them on."""
        store = RecordingUploadStore(bucket="b")
        client = await self._client(store)
        source = tmp_path / "bare.txt"
        source.write_bytes(b"bare")

        await client.upload_from_path("k.txt", source)

        sent = store.uploads[0]
        assert sent["metadata"] == {}
        assert sent["create_if_absent"] is False
        assert sent["if_match"] is None
        assert sent["if_version_match"] is None
        assert sent["kms_key_id"] is None
        assert sent["use_kms_managed_key"] is False

    async def test_missing_file_surfaces_as_an_invalid_argument(self, tmp_path):
        """Not a generic transport error, which is what upload() would give."""
        store = RecordingUploadStore(bucket="b")
        client = await self._client(store)

        with pytest.raises(InvalidArgumentException):
            await client.upload_from_path("k.txt", tmp_path / "absent.txt")

    async def test_native_error_is_translated(self, tmp_path):
        store = RecordingUploadStore(bucket="b")
        store.raise_native = RuntimeError("native boom")
        client = await self._client(store)
        source = tmp_path / "boom.txt"
        source.write_bytes(b"boom")

        with pytest.raises(UnknownException):
            await client.upload_from_path("k.txt", source)
