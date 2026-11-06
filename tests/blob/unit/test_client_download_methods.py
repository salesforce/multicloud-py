"""Unit tests for BucketClient download/upload/delete forwarding methods.

Ensures client-layer methods correctly forward to the underlying store
and that the new streaming/file-based download methods are exercised
through the public API.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from multicloud.blob.blob_client import (
    BucketClient,
    DownloadResponse,
    StreamingDownloadResponse,
)
from multicloud.blob.blob_client.driver import BlobMetadata


@pytest.fixture
def client_with_mock_store():
    """Create a BucketClient with its internal store replaced by a mock."""
    client = BucketClient(provider="aws", bucket="b", region="us-west-2")
    mock_store = MagicMock()
    client._store = mock_store
    return client, mock_store


class TestBucketClientUploadForwarding:
    """Verify upload() delegates correctly to the store."""

    def test_upload_forwards_all_params(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        from multicloud.blob.blob_client.driver import UploadResponse

        mock_store.upload.return_value = UploadResponse(
            key="k", e_tag="etag", version_id="v1"
        )

        resp = client.upload(
            "k",
            b"payload",
            content_type="text/plain",
            content_length=7,
            metadata={"env": "dev"},
        )

        mock_store.upload.assert_called_once_with(
            "k",
            b"payload",
            content_type="text/plain",
            content_length=7,
            metadata={"env": "dev"},
            tags=None,
            object_lock=None,
            create_if_absent=False,
            if_match=None,
            if_version_match=None,
            kms_key_id=None,
            use_kms_managed_key=False,
            storage_class=None,
        )
        assert resp.key == "k"
        assert resp.e_tag == "etag"


class TestBucketClientDownloadForwarding:
    """Verify download() delegates correctly to the store."""

    def test_download_forwards_all_params(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        meta = BlobMetadata(key="k", content_length=5)
        mock_store.download.return_value = DownloadResponse(
            key="k", content=b"hello", metadata=meta
        )

        resp = client.download("k", version_id="v2", start=0, end=4)

        mock_store.download.assert_called_once_with(
            "k",
            version_id="v2",
            start=0,
            end=4,
            if_match=None,
        )
        assert resp.content == b"hello"


class TestBucketClientDeleteForwarding:
    """Verify delete() delegates correctly to the store."""

    def test_delete_forwards_key(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        client.delete("to-remove.txt")
        mock_store.delete.assert_called_once_with("to-remove.txt", version_id=None)


class TestBucketClientDownloadToPath:
    """Verify download_to_path() delegates correctly to the store."""

    def test_forwards_all_params(self, client_with_mock_store, tmp_path):
        client, mock_store = client_with_mock_store
        meta = BlobMetadata(key="file.bin", content_length=100)
        mock_store.download_to_path.return_value = DownloadResponse(
            key="file.bin", content=b"", metadata=meta
        )
        dest = tmp_path / "output.bin"

        resp = client.download_to_path(
            "file.bin",
            dest,
            version_id="v1",
            start=10,
            end=109,
            create_parent_dirs=True,
        )

        mock_store.download_to_path.assert_called_once_with(
            "file.bin",
            dest,
            version_id="v1",
            start=10,
            end=109,
            create_parent_dirs=True,
        )
        assert resp.content == b""
        assert resp.metadata.content_length == 100

    def test_defaults_no_create_parent_dirs(self, client_with_mock_store, tmp_path):
        client, mock_store = client_with_mock_store
        meta = BlobMetadata(key="k")
        mock_store.download_to_path.return_value = DownloadResponse(
            key="k", content=b"", metadata=meta
        )

        client.download_to_path("k", tmp_path / "out.bin")

        kwargs = mock_store.download_to_path.call_args
        assert kwargs[1]["create_parent_dirs"] is False


class TestBucketClientDownloadToFile:
    """Verify download_to_file() delegates correctly to the store."""

    def test_forwards_all_params(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        meta = BlobMetadata(key="doc.pdf", content_length=5000)
        mock_store.download_to_file.return_value = DownloadResponse(
            key="doc.pdf", content=b"", metadata=meta
        )
        buf = io.BytesIO()

        resp = client.download_to_file(
            "doc.pdf", buf, version_id="v3", start=0, end=4999
        )

        mock_store.download_to_file.assert_called_once_with(
            "doc.pdf",
            buf,
            version_id="v3",
            start=0,
            end=4999,
        )
        assert resp.content == b""

    def test_defaults_optional_params(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        meta = BlobMetadata(key="k")
        mock_store.download_to_file.return_value = DownloadResponse(
            key="k", content=b"", metadata=meta
        )
        buf = io.BytesIO()

        client.download_to_file("k", buf)

        mock_store.download_to_file.assert_called_once_with(
            "k",
            buf,
            version_id=None,
            start=None,
            end=None,
        )


class TestBucketClientDownloadStream:
    """Verify download_stream() delegates correctly to the store."""

    def test_forwards_all_params(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        meta = BlobMetadata(key="big.bin", content_length=1_000_000)
        streaming_resp = StreamingDownloadResponse(
            key="big.bin", metadata=meta, stream=iter([b"chunk"])
        )
        mock_store.download_stream.return_value = streaming_resp

        resp = client.download_stream("big.bin", version_id="v5", start=0, end=999)

        mock_store.download_stream.assert_called_once_with(
            "big.bin",
            version_id="v5",
            start=0,
            end=999,
        )
        assert isinstance(resp, StreamingDownloadResponse)
        assert resp.key == "big.bin"

    def test_context_manager_through_client(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        close_called = []
        meta = BlobMetadata(key="k")
        streaming_resp = StreamingDownloadResponse(
            key="k",
            metadata=meta,
            stream=iter([b"data"]),
            close_func=lambda: close_called.append(True),
        )
        mock_store.download_stream.return_value = streaming_resp

        with client.download_stream("k") as stream:
            content = stream.read()

        assert content == b"data"
        assert close_called == [True]
        assert stream.closed is True

    def test_defaults_optional_params(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        meta = BlobMetadata(key="k")
        mock_store.download_stream.return_value = StreamingDownloadResponse(
            key="k", metadata=meta, stream=iter([])
        )

        client.download_stream("k")

        mock_store.download_stream.assert_called_once_with(
            "k",
            version_id=None,
            start=None,
            end=None,
        )


class TestBucketClientTagsForwarding:
    """Verify get_tags/set_tags delegate correctly to the store."""

    def test_get_tags_forwards(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        mock_store.get_tags.return_value = {"env": "prod", "team": "sdk"}

        result = client.get_tags("tagged-object.txt")

        mock_store.get_tags.assert_called_once_with("tagged-object.txt")
        assert result == {"env": "prod", "team": "sdk"}

    def test_set_tags_forwards(self, client_with_mock_store):
        client, mock_store = client_with_mock_store
        tags = {"version": "2.0", "owner": "team-a"}

        client.set_tags("my-object.txt", tags)

        mock_store.set_tags.assert_called_once_with("my-object.txt", tags)
