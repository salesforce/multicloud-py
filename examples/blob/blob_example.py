#!/usr/bin/env python3
"""Blob storage examples for the Python Multi-Cloud SDK.

Demonstrates cloud-agnostic blob operations. The same code works with any provider.

Usage:
    # AWS Example
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_SESSION_TOKEN=...  # Optional, if using temporary credentials
    export BUCKET_NAME=my-bucket
    export BUCKET_REGION=us-west-2
    python examples/blob/blob_example.py --provider aws

    # GCP Example
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    export BUCKET_NAME=my-bucket
    export BUCKET_REGION=us-central1
    python examples/blob/blob_example.py --provider gcp

    # Or set provider via environment variable
    export PROVIDER=aws  # or "gcp" (defaults to "aws")
    python examples/blob/blob_example.py

Prerequisites:
    pip install python-multicloud-sdk[aws]  # or [gcp]
"""

import io
import os
import ssl
import sys
import tempfile
import time
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from multicloud.blob.blob_client import (
    BlobIdentifier,
    BucketClient,
    CopyResponse,
    DeleteObjectsResponse,
    MultipartPart,
    ObjectLockConfiguration,
    PresignedOperation,
    RetentionMode,
    StreamingDownloadResponse,
)
from multicloud.common.exceptions import (
    InvalidArgumentException,
    ResourceExhaustedException,
    SubstrateSdkException,
)
from multicloud.common.proxy import ProxyConfig


def upload():
    """Upload content to a specified provider using the given bucket client."""
    provider = get_provider()
    client = get_bucket_client(provider)

    key = "examples/multicloud.txt"
    content = b"Hello from the Python Multi-Cloud SDK!"

    response = client.upload(
        key=key,
        payload=content,
        content_type="text/plain",
        metadata={"key1": "val1"},
    )

    print(f"\n=== Upload ===")
    print(
        f"Received upload response: ETag={response.e_tag}, VersionId={response.version_id}"
    )

    client.close()


def upload_from_path():
    """Upload a file from a local path using automatic content-type detection."""
    provider = get_provider()
    client = get_bucket_client(provider)

    # Create a temporary file to upload
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write('{"message": "Hello from upload_from_path!", "sdk": "multicloud-py"}')
        local_path = Path(f.name)

    try:
        key = "examples/upload-from-path.json"

        # Upload using the file path — content_type auto-detected from .json extension
        response = client.upload_from_path(
            key=key,
            path=local_path,
            metadata={"uploaded-via": "upload_from_path"},
        )

        print(f"\n=== Upload from Path ===")
        print(f"Uploaded file: {local_path.name}")
        print(f"Auto-detected content_type from .json extension")
        print(
            f"Received response: ETag={response.e_tag}, VersionId={response.version_id}"
        )

        # Verify by downloading
        download_resp = client.download(key=key)
        print(f"Verified download: {len(download_resp.content)} bytes")
        print(f"Content-type: {download_resp.metadata.content_type}")
        assert download_resp.metadata.content_type in (
            "application/json",
            "application/json; charset=utf-8",
        ), f"Unexpected content_type: {download_resp.metadata.content_type}"

        # Upload with explicit content_type override
        response2 = client.upload_from_path(
            key="examples/upload-from-path-override.txt",
            path=local_path,
            content_type="text/plain",
        )
        print(f"Upload with override content_type='text/plain': ETag={response2.e_tag}")

        # Clean up remote objects
        client.delete(key)
        client.delete("examples/upload-from-path-override.txt")
        print("Cleaned up remote objects")
    finally:
        local_path.unlink(missing_ok=True)

    client.close()


def upload_with_tags():
    """Upload content with tags attached, then read them back via get_tags."""
    client = get_bucket_client(get_provider())

    key = "examples/multicloud-tagged.txt"
    content = b"Hello with tags!"
    tags = {"environment": "dev", "team": "multicloud", "owner": "rose"}

    print(f"\n=== Upload with Tags ===")
    response = client.upload(
        key=key,
        payload=content,
        content_type="text/plain",
        tags=tags,
    )
    print(f"Uploaded '{key}' with tags={tags}")
    print(f"ETag={response.e_tag}, VersionId={response.version_id}")

    # Round-trip: read the tags back via get_tags to verify they were stored.
    readback = client.get_tags(key)
    print(f"Tags read back via get_tags: {readback}")
    assert readback == tags, f"tags mismatch: expected {tags}, got {readback}"
    print("✓ Tags round-trip verified")

    client.delete(key)
    client.close()


def upload_with_storage_class():
    """Upload into a non-default storage class, then confirm it via list_page."""
    provider = get_provider()
    client = get_bucket_client(provider)

    # Storage class names are provider-native and passed through unchanged.
    storage_class = "STANDARD_IA" if provider == "aws" else "NEARLINE"
    key = "examples/storage-class/multicloud-infrequent.txt"

    print(f"\n=== Upload with Storage Class ===")
    client.upload(
        key=key,
        payload=b"Rarely read content",
        content_type="text/plain",
        storage_class=storage_class,
    )
    print(f"Uploaded '{key}' with storage_class={storage_class}")

    page = client.list_page(prefix=key)
    reported = page.blobs[0].storage_class
    print(f"Storage class reported by list_page: {reported}")
    assert reported == storage_class, f"expected {storage_class}, got {reported}"

    client.delete(key)
    client.close()


def upload_with_object_lock():
    """Upload with first-time retention + a legal hold, then read it back."""
    client = get_bucket_client(get_provider())

    key = "examples/multicloud-locked.txt"
    content = b"Hello with object lock!"
    retain_until = datetime.now(timezone.utc) + timedelta(days=1)
    object_lock = ObjectLockConfiguration(
        mode=RetentionMode.GOVERNANCE,
        retain_until_date=retain_until,
        legal_hold=True,
    )

    print("\n=== Upload with Object Lock ===")
    response = client.upload(
        key=key,
        payload=content,
        content_type="text/plain",
        object_lock=object_lock,
    )
    print(f"Uploaded '{key}' with retention until {retain_until} and a legal hold")
    print(f"ETag={response.e_tag}, VersionId={response.version_id}")

    # Round-trip: read the lock state back to verify it was applied.
    info = client.get_object_lock(key)
    print(f"Lock state read back: {info}")

    client.close()


def multipart_upload():
    """Upload a payload in multiple parts using the multipart upload flow."""
    client = get_bucket_client(get_provider())
    key = "examples/multipart-demo.bin"

    part_size = 5 * 1024 * 1024
    payload = b"X" * (2 * part_size)  # 10 MiB total

    print(f"\n=== Multipart Upload ===")
    mpu = client.initiate_multipart_upload(key, content_type="application/octet-stream")
    print(f"Initiated multipart upload: upload_id={mpu.upload_id}")

    num_parts = (len(payload) + part_size - 1) // part_size
    uploaded = []
    try:
        for part_num in range(1, num_parts + 1):
            start = (part_num - 1) * part_size
            part_data = payload[start : start + part_size]
            resp = client.upload_part(
                mpu, MultipartPart(part_number=part_num, content=part_data)
            )
            print(
                f"Uploaded part {part_num}: etag={resp.e_tag}, size={resp.size_in_bytes}"
            )
            uploaded.append(resp)

        result = client.complete_multipart_upload(mpu, uploaded)
        print(f"Completed multipart upload: etag={result.e_tag}")
    except Exception:
        client.abort_multipart_upload(mpu)
        print("Aborted multipart upload due to error")
        raise

    download_resp = client.download(key=key)
    assert download_resp.content == payload, "multipart round-trip mismatch"
    print(f"Verified download: {len(download_resp.content)} bytes")

    client.delete(key)
    client.close()


def multipart_upload_with_tags():
    """Tag a multipart upload at initiate and verify the result."""
    client = get_bucket_client(get_provider())
    key = f"examples/multipart-tagged-{uuid.uuid4().hex[:8]}.bin"

    payload = b"tagged multipart payload"
    tags = {"env": "example", "note": "a b+c"}
    metadata = {"team": "multicloud"}

    mpu = None
    completed = False
    try:
        print("\n=== Multipart Upload with Tags ===")
        mpu = client.initiate_multipart_upload(key, metadata=metadata, tags=tags)
        print(f"Initiated multipart upload: upload_id={mpu.upload_id}")

        part = client.upload_part(mpu, MultipartPart(part_number=1, content=payload))
        result = client.complete_multipart_upload(mpu, [part])
        completed = True
        print(f"Completed multipart upload: etag={result.e_tag}")

        download_resp = client.download(key=key)
        assert download_resp.content == payload, "multipart round-trip mismatch"
        print(f"Verified download: {len(download_resp.content)} bytes")

        fetched_tags = dict(client.get_tags(key))
        assert fetched_tags == tags, f"tags mismatch: {fetched_tags} != {tags}"
        print(f"Tags verified: {fetched_tags}")

        fetched_metadata = client.get_metadata(key).metadata
        assert (
            fetched_metadata.get("team") == "multicloud"
        ), f"metadata entry lost: {fetched_metadata}"
        print("Metadata survived alongside the tags: team=multicloud")
    finally:
        try:
            if mpu is not None and not completed:
                client.abort_multipart_upload(mpu)
                print("Aborted incomplete multipart upload")
            if completed:
                client.delete(key)
                print(f"Cleaned up '{key}'")
        finally:
            client.close()


def multipart_upload_with_object_lock():
    """End-to-end: set an object lock at initiate and verify it lands.

    Initiates a multipart upload carrying a retention + legal hold, uploads the
    parts, completes, downloads to confirm the bytes round-trip, then reads the
    lock back and asserts every field took effect. Requires a bucket that has
    object lock enabled.
    """
    client = get_bucket_client(get_provider())
    # A retained object cannot be overwritten until its retention expires, so
    # use a unique key per run to keep this example re-runnable.
    key = f"examples/multipart-locked-{uuid.uuid4().hex[:8]}.bin"

    payload = b"locked multipart payload"
    retain_until = datetime.now(timezone.utc) + timedelta(days=1)
    object_lock = ObjectLockConfiguration(
        mode=RetentionMode.GOVERNANCE,
        retain_until_date=retain_until,
        legal_hold=True,
    )

    print("\n=== Multipart Upload with Object Lock ===")
    # The lock is passed at initiate and stamped on the object at completion.
    mpu = client.initiate_multipart_upload(
        key, content_type="application/octet-stream", object_lock=object_lock
    )
    print(f"Initiated multipart upload: upload_id={mpu.upload_id}")

    part = client.upload_part(mpu, MultipartPart(part_number=1, content=payload))
    result = client.complete_multipart_upload(mpu, [part])
    print(f"Completed multipart upload: etag={result.e_tag}")

    # Round-trip the body to prove the parts assembled into the payload.
    download_resp = client.download(key=key)
    assert download_resp.content == payload, "multipart round-trip mismatch"
    print(f"Verified download: {len(download_resp.content)} bytes")

    # Read the lock state back and assert every field took effect — not merely
    # that a result came back.
    info = client.get_object_lock(key)
    assert info is not None, "expected an object lock on the completed object"
    assert info.mode is RetentionMode.GOVERNANCE, f"unexpected mode: {info.mode}"
    assert info.retain_until_date is not None, "expected a retain-until date"
    # Providers round the retain-until instant, so allow a small skew.
    drift = abs((info.retain_until_date - retain_until).total_seconds())
    assert drift < 2, f"retain_until drifted {drift}s from requested"
    assert info.legal_hold is True, "expected the legal hold to be set"
    print(
        f"✓ Lock verified: mode={info.mode.value}, "
        f"retain_until={info.retain_until_date}, legal_hold={info.legal_hold}"
    )

    # Clean up. The retention is in the future, so release the legal hold and
    # best-effort delete; bypassing an unexpired retention is provider-specific
    # and outside the cloud-agnostic API, so note it when the delete is blocked.
    client.update_legal_hold(key, False)
    try:
        client.delete(key)
        print(f"Cleaned up '{key}'")
    except SubstrateSdkException as exc:
        print(
            f"'{key}' stays under retention until {retain_until} "
            f"(cloud-agnostic delete blocked: {type(exc).__name__})"
        )

    client.close()


def download():
    """Download a blob from a specified bucket using the provided key."""
    client = get_bucket_client(get_provider())

    response = client.download(key="examples/multicloud.txt")

    print(f"\n=== Download ===")
    print("Download complete")
    print(f"The downloaded file has a name of {response.key}")
    print(f"The file has an etag of {response.metadata.e_tag}")
    print(f"The file has a versionId of {response.metadata.version_id}")
    print(f"The file has content type of {response.metadata.content_type}")
    print(f"The file has metadata of {response.metadata.metadata}")
    print(f"The file is {response.metadata.content_length} bytes")

    client.close()


def get_metadata():
    """Fetch object metadata without downloading the body (HEAD-style lookup).

    Returns size, content-type, ETag, version, custom metadata and timestamps.
    The same call maps to an S3 HeadObject on AWS and a blob reload on GCP.
    """
    client = get_bucket_client(get_provider())

    key = "examples/multicloud-metadata.txt"
    content = b"Hello from get_metadata!"

    print(f"\n=== Get Metadata ===")
    try:
        # Upload an object with a content-type and custom metadata to inspect.
        client.upload(
            key=key,
            payload=content,
            content_type="text/plain",
            metadata={"author": "sdk-example", "purpose": "metadata-demo"},
        )

        # HEAD-style lookup — retrieves metadata only, never the object body.
        meta = client.get_metadata(key)

        print(f"Metadata for '{meta.key}':")
        print(f"  content_length={meta.content_length} bytes")
        print(f"  content_type={meta.content_type}")
        print(f"  e_tag={meta.e_tag}")
        print(f"  version_id={meta.version_id}")
        print(f"  metadata={meta.metadata}")
        print(f"  last_modified={meta.last_modified}")
        print(f"  created_time={meta.created_time}")

        assert meta.key == key, f"key mismatch: {meta.key}"
        assert meta.content_length == len(
            content
        ), f"content_length mismatch: {meta.content_length} != {len(content)}"
        print("✓ get_metadata verified")

    finally:
        try:
            client.delete(key)
            print(f"Cleaned up '{key}'")
        except Exception:
            pass
        client.close()


def list_page():
    """List blobs in a bucket using pagination."""
    client = get_bucket_client(get_provider())

    response = client.list_page(prefix="examples/", max_results=10)

    print(f"\n=== List Page ===")
    print(f"Found {len(response.blobs)} blobs:")
    for blob in response.blobs:
        print(
            f"  key={blob.key}, size={blob.size}, e_tag={blob.e_tag}, "
            f"last_modified={blob.last_modified}"
        )
    print(f"Common prefixes: {response.common_prefixes}")
    print(f"Is truncated: {response.is_truncated}")
    print(f"Next pagination token: {response.next_pagination_token}")

    client.close()


def does_object_exist():
    """Check if a blob exists in the bucket."""
    client = get_bucket_client(get_provider())

    existing_key = "examples/multicloud.txt"
    missing_key = "examples/nonexistent-file-xyz.txt"

    exists = client.does_object_exist(existing_key)
    print(f"\n=== Does Object Exist ===")
    print(f"  '{existing_key}' exists: {exists}")

    not_exists = client.does_object_exist(missing_key)
    print(f"  '{missing_key}' exists: {not_exists}")

    client.close()


def does_bucket_exist():
    """Check if the configured bucket exists."""
    client = get_bucket_client(get_provider())

    exists = client.does_bucket_exist()
    print(f"\n=== Does Bucket Exist ===")
    print(f"  Bucket '{client.bucket}' exists: {exists}")

    client.close()


def list_blob_versions():
    """List all versions of a blob."""
    client = get_bucket_client(get_provider())

    key = "examples/multicloud.txt"
    print(f"\n=== List Blob Versions ===")
    print(f"Versions of '{key}':")

    count = 0
    for version in client.list_blob_versions(key):
        count += 1
        print(
            f"  version_id={version.version_id}, size={version.size}, "
            f"e_tag={version.e_tag}, is_latest={version.is_latest}, "
            f"last_modified={version.last_modified}"
        )

    if count == 0:
        print("  (no versions found — bucket may not have versioning enabled)")

    client.close()


def set_tags():
    """Set tags on an object, replacing all existing tags."""
    client = get_bucket_client(get_provider())

    key = "examples/multicloud.txt"
    tags = {"environment": "dev", "team": "multicloud", "version": "1.0"}

    print(f"\n=== Set Tags ===")
    print(f"Setting tags on '{key}': {tags}")

    client.set_tags(key, tags)
    print("Tags set successfully")

    client.close()


def get_tags():
    """Retrieve all tags associated with an object."""
    client = get_bucket_client(get_provider())

    key = "examples/multicloud.txt"

    print(f"\n=== Get Tags ===")
    tags = client.get_tags(key)
    print(f"Tags on '{key}':")
    for tag_key, tag_value in tags.items():
        print(f"  {tag_key}={tag_value}")

    if not tags:
        print("  (no tags found)")

    client.close()


def get_object_lock():
    """Retrieve the object lock state (retention + legal hold) for an object."""
    client = get_bucket_client(get_provider())

    key = "examples/multicloud.txt"

    print("\n=== Get Object Lock ===")
    info = client.get_object_lock(key)
    if info is None:
        print(f"'{key}' has no retention or legal hold")
    else:
        print(f"Lock state on '{key}':")
        print(f"  mode={info.mode}")
        print(f"  retain_until_date={info.retain_until_date}")
        print(f"  legal_hold={info.legal_hold}")
        print(f"  event_based_hold={info.event_based_hold}")

    client.close()


def update_legal_hold():
    """Place and then release a legal hold on an object."""
    client = get_bucket_client(get_provider())

    key = "examples/multicloud.txt"

    print("\n=== Update Legal Hold ===")
    # Pass version_id=... to target a specific object version.
    client.update_legal_hold(key, True)
    print(f"Placed legal hold on '{key}'")

    client.update_legal_hold(key, False)
    print(f"Released legal hold on '{key}'")

    client.close()


def download_to_path():
    """Download a blob directly to a local file path without loading into memory."""
    client = get_bucket_client(get_provider())

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / "multicloud.txt"
        response = client.download_to_path("examples/multicloud.txt", dest)

        print(f"\n=== Download to Path ===")
        print(f"Downloaded to: {dest}")
        print(f"File size on disk: {dest.stat().st_size} bytes")
        print(f"Metadata content_length: {response.metadata.content_length}")
        print(f"Content in memory is empty: len={len(response.content)}")
        # Verify content matches
        assert dest.read_bytes() == b"Hello from the Python Multi-Cloud SDK!"

    client.close()


def download_to_path_with_parent_dirs():
    """Download to a nested path, creating parent directories automatically."""
    client = get_bucket_client(get_provider())

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / "deep" / "nested" / "dir" / "multicloud.txt"
        response = client.download_to_path(
            "examples/multicloud.txt", dest, create_parent_dirs=True
        )

        print(f"\n=== Download to Path (create_parent_dirs) ===")
        print(f"Downloaded to nested path: {dest}")
        print(f"Parent dirs auto-created: {dest.parent.exists()}")
        assert dest.read_bytes() == b"Hello from the Python Multi-Cloud SDK!"

    client.close()


def download_to_file():
    """Download a blob to a file-like object without loading into memory."""
    client = get_bucket_client(get_provider())

    # Write to a BytesIO buffer
    buf = io.BytesIO()
    response = client.download_to_file("examples/multicloud.txt", buf)

    print(f"\n=== Download to File ===")
    print(f"Downloaded to BytesIO: {buf.tell()} bytes written")
    print(f"Metadata content_length: {response.metadata.content_length}")
    print(f"Content in memory is empty: len={len(response.content)}")
    assert buf.getvalue() == b"Hello from the Python Multi-Cloud SDK!"

    client.close()


def download_stream():
    """Download a blob as a stream for lazy chunk-by-chunk consumption."""
    client = get_bucket_client(get_provider())

    print(f"\n=== Download Stream ===")

    # Iterate chunk by chunk
    with client.download_stream("examples/multicloud.txt") as stream:
        total_bytes = 0
        chunk_count = 0
        for chunk in stream:
            total_bytes += len(chunk)
            chunk_count += 1

    print(f"Consumed {chunk_count} chunks, {total_bytes} bytes total")
    print(f"Stream closed after context manager: {stream.closed}")

    # Use read() interface
    with client.download_stream("examples/multicloud.txt") as stream:
        content = stream.read()

    print(f"read() returned {len(content)} bytes")
    assert content == b"Hello from the Python Multi-Cloud SDK!"

    client.close()


def download_stream_range():
    """Download a byte range as a stream."""
    client = get_bucket_client(get_provider())

    # Download only first 5 bytes
    with client.download_stream("examples/multicloud.txt", start=0, end=4) as stream:
        content = stream.read()

    print(f"\n=== Download Stream (range) ===")
    print(f"Range bytes 0-4: {content!r}")
    assert content == b"Hello"

    client.close()


def _make_ssl_context():
    """Create an SSL context for presigned URL HTTP calls.

    Defaults to system trust store (secure). Set env var
    MULTICLOUD_EXAMPLE_INSECURE_TLS=1 to disable verification for
    environments where the system trust store is incomplete.
    """
    ctx = ssl.create_default_context()
    if os.environ.get("MULTICLOUD_EXAMPLE_INSECURE_TLS") == "1":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def generate_presigned_upload_url():
    """Generate a presigned upload URL and use it to PUT content."""
    client = get_bucket_client(get_provider())

    key = "examples/presigned-upload.txt"
    content = b"Uploaded via presigned URL!"

    print("\n=== Generate Presigned Upload URL ===")
    try:
        response = client.generate_presigned_url(
            key=key,
            operation=PresignedOperation.UPLOAD,
            duration_seconds=3600,
            content_type="text/plain",
            metadata={"uploaded-by": "presigned-example"},
        )

        print(f"URL: {response.url[:80]}...")
        print(f"HTTP Method: {response.http_method}")
        print(f"Required Headers: {response.required_headers}")

        # Use the presigned URL to upload via raw HTTP.
        # required_headers already includes Content-Type since we passed
        # content_type="text/plain" to generate_presigned_url above.
        headers = dict(response.required_headers)
        req = urllib.request.Request(
            response.url,
            data=content,
            headers=headers,
            method="PUT",
        )
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, context=ctx) as resp:
            print(f"Upload via presigned URL: HTTP {resp.status}")

        # Verify by downloading
        download_resp = client.download(key=key)
        assert download_resp.content == content, "Content mismatch!"
        print("Verified: downloaded content matches uploaded content")

    finally:
        # Clean up
        try:
            client.delete(key)
            print(f"Cleaned up '{key}'")
        except Exception:
            pass
        client.close()


def generate_presigned_download_url():
    """Generate a presigned download URL and use it to GET content."""
    client = get_bucket_client(get_provider())

    key = "examples/presigned-download.txt"
    content = b"Downloaded via presigned URL!"

    print("\n=== Generate Presigned Download URL ===")
    try:
        # First upload something to download
        client.upload(key=key, payload=content, content_type="text/plain")

        response = client.generate_presigned_url(
            key=key,
            operation=PresignedOperation.DOWNLOAD,
            duration_seconds=3600,
        )

        print(f"URL: {response.url[:80]}...")
        print(f"HTTP Method: {response.http_method}")
        print(f"Required Headers: {response.required_headers}")

        # Use the presigned URL to download via raw HTTP
        req = urllib.request.Request(response.url, method="GET")
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, context=ctx) as resp:
            body = resp.read()
            print(
                f"Download via presigned URL: HTTP {resp.status}, "
                f"size={len(body)} bytes"
            )
            assert body == content, "Content mismatch!"
            print("Verified: presigned download content matches original")

    finally:
        # Clean up
        try:
            client.delete(key)
            print(f"Cleaned up '{key}'")
        except Exception:
            pass
        client.close()


def generate_presigned_upload_url_with_tags():
    """Generate a presigned upload URL with tags and metadata."""
    client = get_bucket_client(get_provider())

    key = "examples/presigned-tagged.txt"
    content = b"Tagged presigned upload!"

    print("\n=== Generate Presigned Upload URL with Tags ===")
    try:
        response = client.generate_presigned_url(
            key=key,
            operation=PresignedOperation.UPLOAD,
            duration_seconds=3600,
            content_type="text/plain",
            metadata={"author": "sdk-example"},
            tags={"environment": "dev", "team": "multicloud"},
        )

        print(f"URL: {response.url[:80]}...")
        print(f"HTTP Method: {response.http_method}")
        print(f"Required Headers: {response.required_headers}")

        # Use the presigned URL to upload with required headers.
        # required_headers already contains Content-Type and all
        # metadata/tag headers — just use it directly.
        headers = dict(response.required_headers)
        req = urllib.request.Request(
            response.url,
            data=content,
            headers=headers,
            method="PUT",
        )
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, context=ctx) as resp:
            print(f"Upload via presigned URL: HTTP {resp.status}")

        # Verify tags were applied
        tags = client.get_tags(key)
        print(f"Tags on uploaded object: {dict(tags)}")

    finally:
        # Clean up
        try:
            client.delete(key)
            print(f"Cleaned up '{key}'")
        except Exception:
            pass
        client.close()


def generate_presigned_upload_url_with_kms():
    """Presigned upload to a KMS-encrypted bucket, then verify over HTTP.

    This is the scenario that requires SigV4: a bucket with KMS default
    encryption rejects a legacy SigV2 presigned URL outright with
    SignatureDoesNotMatch / InvalidArgument. SigV4 folds the KMS headers into
    the signature, so the caller must replay every ``required_headers`` entry
    (including ``x-amz-server-side-encryption*``) on the PUT.

    Configure via environment variables (all optional):
      KMS_KEY_ID       — KMS key id/arn/alias to encrypt with. If unset, this
                         example is skipped (it needs a real key).
      KMS_BUCKET_NAME  — bucket to use; falls back to BUCKET_NAME.
      KMS_BUCKET_REGION— region for that bucket; falls back to BUCKET_REGION.
    """
    kms_key_id = os.environ.get("KMS_KEY_ID")
    if not kms_key_id:
        print("\n=== Generate Presigned Upload URL with KMS (skipped) ===")
        print("Set KMS_KEY_ID (and optionally KMS_BUCKET_NAME) to run this.")
        return

    bucket_name = os.environ.get("KMS_BUCKET_NAME") or os.environ.get(
        "BUCKET_NAME", "palsfdc"
    )
    region = os.environ.get("KMS_BUCKET_REGION") or os.environ.get("BUCKET_REGION")
    client = BucketClient(provider=get_provider(), bucket=bucket_name, region=region)

    key = "examples/presigned-kms.txt"
    content = b"Encrypted via presigned URL on a KMS bucket!"

    print("\n=== Generate Presigned Upload URL with KMS ===")
    try:
        response = client.generate_presigned_url(
            key=key,
            operation=PresignedOperation.UPLOAD,
            duration_seconds=3600,
            content_type="text/plain",
            kms_key_id=kms_key_id,
        )

        print(f"Bucket: {bucket_name} (region={region})")
        print(f"URL: {response.url[:80]}...")
        print(f"Required Headers: {response.required_headers}")

        # The KMS headers are signed, so they MUST be replayed on the PUT.
        headers = dict(response.required_headers)
        req = urllib.request.Request(
            response.url, data=content, headers=headers, method="PUT"
        )
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, context=ctx) as resp:
            print(f"Upload via presigned URL to KMS bucket: HTTP {resp.status}")

        # Verify round-trip.
        download_resp = client.download(key=key)
        assert download_resp.content == content, "KMS presigned content mismatch!"
        print("Verified: SigV4 presigned upload accepted by KMS bucket")

    finally:
        try:
            client.delete(key)
            print(f"Cleaned up '{key}'")
        except Exception:
            pass
        client.close()


def delete_objects():
    """Delete multiple objects in a single batch call."""
    client = get_bucket_client(get_provider())

    keys_to_upload = [
        "examples/batch-delete/file1.txt",
        "examples/batch-delete/file2.txt",
        "examples/batch-delete/file3.txt",
    ]

    print("\n=== Delete Objects (Batch) ===")
    try:
        # Upload test objects
        for key in keys_to_upload:
            client.upload(key=key, payload=b"batch delete test content")
        print(f"Uploaded {len(keys_to_upload)} objects for batch delete test")

        # Batch delete using BlobIdentifier list
        identifiers = [BlobIdentifier(key=key) for key in keys_to_upload]
        response: DeleteObjectsResponse = client.delete_objects(objects=identifiers)

        print(f"Batch delete completed")
        print(f"Errors: {len(response.errors)}")
        if response.errors:
            for err in response.errors:
                print(f"  Error: key={err.key}, code={err.code}, msg={err.message}")
        else:
            print("All objects deleted successfully")

        # Verify objects are gone
        for key in keys_to_upload:
            exists = client.does_object_exist(key)
            assert not exists, f"Object {key} should have been deleted"
        print("Verified: all objects no longer exist")

    finally:
        # Clean up any remaining objects
        for key in keys_to_upload:
            try:
                client.delete(key)
            except Exception:
                pass
        client.close()


def copy_blob():
    """Copy a blob to the same bucket (server-side copy)."""
    client = get_bucket_client(get_provider())

    bucket_name = os.environ.get("BUCKET_NAME", "palsfdc")
    source_key = "examples/copy-source.txt"
    dest_key = "examples/copy-destination.txt"
    content = b"Hello from copy test!"

    print("\n=== Copy Blob ===")
    try:
        # Upload source object
        client.upload(key=source_key, payload=content, content_type="text/plain")
        print(f"Uploaded source: {source_key}")

        # Copy to destination within the same bucket
        response: CopyResponse = client.copy(
            source_key,
            dest_bucket=bucket_name,
            dest_key=dest_key,
        )

        print(f"Copied to: {response.key}")
        print(f"Version ID: {response.version_id}")
        print(f"ETag: {response.e_tag}")
        print(f"Last Modified: {response.last_modified}")

        # Verify destination content matches source
        download_resp = client.download(key=dest_key)
        assert download_resp.content == content, "Copy content mismatch!"
        print("Verified: destination content matches source")

    finally:
        # Clean up
        for key in [source_key, dest_key]:
            try:
                client.delete(key)
            except Exception:
                pass
        client.close()


def copy_from_blob():
    """Copy a blob from another bucket into the current bucket.

    This example uses the same bucket as both source and destination
    for convenience. In production, source and destination would
    typically be different buckets.
    """
    client = get_bucket_client(get_provider())

    # Use the same bucket as source for this example (avoids needing two buckets)
    bucket_name = os.environ.get("BUCKET_NAME", "palsfdc")
    source_key = "examples/copy-from-source.txt"
    dest_key = "examples/copy-from-destination.txt"
    content = b"Hello from copy_from test!"

    print("\n=== Copy From Blob ===")
    try:
        # Upload source object
        client.upload(key=source_key, payload=content, content_type="text/plain")
        print(f"Uploaded source: {source_key} in bucket '{bucket_name}'")

        # Copy from same bucket into current bucket (demonstrates the API)
        response: CopyResponse = client.copy_from(
            bucket_name,
            source_key,
            dest_key=dest_key,
        )

        print(f"Copied from bucket '{bucket_name}' key '{source_key}'")
        print(f"Destination key: {response.key}")
        print(f"Version ID: {response.version_id}")
        print(f"ETag: {response.e_tag}")
        print(f"Last Modified: {response.last_modified}")

        # Verify destination content matches source
        download_resp = client.download(key=dest_key)
        assert download_resp.content == content, "Copy from content mismatch!"
        print("Verified: destination content matches source")

    finally:
        # Clean up
        for key in [source_key, dest_key]:
            try:
                client.delete(key)
            except Exception:
                pass
        client.close()


def proxy_inherits_environment_by_default():
    """Without a proxy_config the client uses whatever proxy the environment sets."""
    provider = get_provider()
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")
    region = os.environ.get("BUCKET_REGION")

    print(f"\n=== Proxy: default (inherit environment) ===")
    print(f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY') or '<unset>'}")

    client = BucketClient(provider=provider, bucket=bucket, region=region)
    try:
        print(f"Bucket reachable: {client.does_bucket_exist()}")
    finally:
        client.close()


def proxy_bypassed_for_direct_connection():
    """Ignore the ambient proxy and connect directly.

    What same-region traffic needs in a pod whose environment points at an
    egress proxy for outbound traffic.
    """
    provider = get_provider()
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")
    region = os.environ.get("BUCKET_REGION")

    print(f"\n=== Proxy: bypassed (direct connection) ===")
    print(f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY') or '<unset>'}")

    client = BucketClient(
        provider=provider,
        bucket=bucket,
        region=region,
        proxy_config=ProxyConfig(use_environment_variable_proxy_values=False),
    )
    try:
        print(f"Bucket reachable without the proxy: {client.does_bucket_exist()}")
    finally:
        client.close()


def throttling_is_reported_distinctly():
    """Show which exception a throttled or unavailable response raises.

    Rate limiting cannot be provoked on demand, so this feeds each provider's
    own error shape through the client's translation to show what a caller
    would actually catch. ResourceExhaustedException means back off and retry;
    UnknownException does not.
    """
    provider = get_provider()
    client = get_bucket_client(provider)

    print(f"\n=== Throttling classification ({provider}) ===")
    try:
        if provider == "aws":
            from botocore.exceptions import ClientError

            conditions = [
                ("ThrottlingException (HTTP 400)", "ThrottlingException", 400),
                ("ServiceUnavailable (HTTP 503)", "ServiceUnavailable", 503),
                ("HTTP 429, unfamiliar code", "SomeNewCode", 429),
                ("AccessDenied (HTTP 403)", "AccessDenied", 403),
            ]
            for label, code, status in conditions:
                error = ClientError(
                    {
                        "Error": {"Code": code, "Message": code},
                        "ResponseMetadata": {"HTTPStatusCode": status},
                    },
                    "HeadBucket",
                )
                print(
                    f"  {label:34} -> {type(client._store.translate_error(error)).__name__}"
                )
        else:
            from google.api_core import exceptions as gcp_errors

            conditions = [
                ("TooManyRequests (429)", gcp_errors.TooManyRequests("rate limited")),
                ("ResourceExhausted (429)", gcp_errors.ResourceExhausted("quota")),
                ("ServiceUnavailable (503)", gcp_errors.ServiceUnavailable("backend")),
                ("GatewayTimeout (504)", gcp_errors.GatewayTimeout("slow")),
            ]
            for label, error in conditions:
                print(
                    f"  {label:34} -> {type(client._store.translate_error(error)).__name__}"
                )
    finally:
        client.close()


def proxy_explicit_endpoint():
    """Send requests through a named proxy regardless of the environment."""
    provider = get_provider()
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")
    region = os.environ.get("BUCKET_REGION")
    endpoint = os.environ.get("EXAMPLE_PROXY_ENDPOINT")

    print(f"\n=== Proxy: explicit endpoint ===")
    if not endpoint:
        print(
            "Set EXAMPLE_PROXY_ENDPOINT to run this against a real proxy; "
            "showing configuration only."
        )
        config = ProxyConfig(endpoint="http://proxy.example.net:8443")
        print(f"Configured endpoint: {config.endpoint}")
        return

    client = BucketClient(
        provider=provider,
        bucket=bucket,
        region=region,
        proxy_config=ProxyConfig(endpoint=endpoint),
    )
    try:
        print(f"Bucket reachable via {endpoint}: {client.does_bucket_exist()}")
    finally:
        client.close()


def throttling_retry_pattern():
    """Retry an operation when the service asks the caller to slow down.

    ResourceExhaustedException is the signal to back off; other SDK errors are
    not worth retrying and are re-raised immediately.
    """
    provider = get_provider()
    client = get_bucket_client(provider)

    print(f"\n=== Throttling retry pattern ({provider}) ===")
    attempts = 0
    try:
        for attempt in range(3):
            attempts += 1
            try:
                exists = client.does_bucket_exist()
                print(f"Bucket reachable after {attempts} attempt(s): {exists}")
                break
            except ResourceExhaustedException as exc:
                delay = 2**attempt
                print(f"Throttled ({exc}); retrying in {delay}s")
                time.sleep(delay)
            except SubstrateSdkException:
                raise
    finally:
        client.close()


def proxy_endpoint_validation():
    """A proxy endpoint without a scheme is rejected rather than ignored."""
    print(f"\n=== Proxy: endpoint validation ===")
    try:
        ProxyConfig(endpoint="proxy.example.net:8443")
    except InvalidArgumentException as exc:
        print(f"Rejected endpoint without a scheme: {exc}")


def get_bucket_client(provider):
    """Create and return a BucketClient instance based on the provider.

    Gets configuration from environment variables or system properties.
    """
    bucket_name = os.environ.get("BUCKET_NAME", "palsfdc")
    region = os.environ.get("BUCKET_REGION")
    endpoint = os.environ.get("BUCKET_ENDPOINT")

    if not bucket_name or bucket_name.strip() == "":
        raise ValueError(
            "Bucket name must be provided via 'BUCKET_NAME' environment variable"
        )

    client = BucketClient(
        provider=provider,
        bucket=bucket_name,
        region=region,
    )

    return client


def get_provider():
    """Get the cloud provider to use.

    Can be set via PROVIDER environment variable, otherwise defaults to "aws".
    """
    return os.environ.get("PROVIDER", "aws")  # or "gcp" or "ali"


def main():
    """Run blob storage examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Python Multi-Cloud SDK Blob Examples")
    parser.add_argument(
        "--provider",
        choices=["aws", "gcp", "ali"],
        help="Cloud provider (can also set PROVIDER env var)",
    )
    args = parser.parse_args()

    # Allow command-line arg to override environment variable
    if args.provider:
        os.environ["PROVIDER"] = args.provider

    provider = get_provider()
    print("=== Python Multi-Cloud SDK Blob Examples ===")
    print(f"Provider: {provider}")

    examples = [
        ("upload", upload),
        ("upload_from_path", upload_from_path),
        ("upload_with_tags", upload_with_tags),
        ("upload_with_storage_class", upload_with_storage_class),
        ("upload_with_object_lock", upload_with_object_lock),
        ("multipart_upload", multipart_upload),
        ("multipart_upload_with_tags", multipart_upload_with_tags),
        (
            "proxy_inherits_environment_by_default",
            proxy_inherits_environment_by_default,
        ),
        ("proxy_bypassed_for_direct_connection", proxy_bypassed_for_direct_connection),
        ("proxy_explicit_endpoint", proxy_explicit_endpoint),
        ("proxy_endpoint_validation", proxy_endpoint_validation),
        ("throttling_is_reported_distinctly", throttling_is_reported_distinctly),
        ("throttling_retry_pattern", throttling_retry_pattern),
        ("multipart_upload_with_object_lock", multipart_upload_with_object_lock),
        ("download", download),
        ("get_metadata", get_metadata),
        ("download_to_path", download_to_path),
        ("download_to_path_with_parent_dirs", download_to_path_with_parent_dirs),
        ("download_to_file", download_to_file),
        ("download_stream", download_stream),
        ("download_stream_range", download_stream_range),
        ("list_page", list_page),
        ("does_object_exist", does_object_exist),
        ("does_bucket_exist", does_bucket_exist),
        ("list_blob_versions", list_blob_versions),
        ("set_tags", set_tags),
        ("get_tags", get_tags),
        ("get_object_lock", get_object_lock),
        ("update_legal_hold", update_legal_hold),
        ("delete_objects", delete_objects),
        ("copy_blob", copy_blob),
        ("copy_from_blob", copy_from_blob),
        ("generate_presigned_upload_url", generate_presigned_upload_url),
        ("generate_presigned_download_url", generate_presigned_download_url),
        (
            "generate_presigned_upload_url_with_tags",
            generate_presigned_upload_url_with_tags,
        ),
        (
            "generate_presigned_upload_url_with_kms",
            generate_presigned_upload_url_with_kms,
        ),
    ]

    passed = []
    failed = []

    for name, fn in examples:
        try:
            fn()
            passed.append(name)
        except Exception as exc:
            failed.append((name, exc))
            print(f"\n✗ FAILED: {name}: {exc}", file=sys.stderr)

    # Summary
    print(f"\n{'=' * 50}")
    print(
        f"Results: {len(passed)} passed, {len(failed)} failed "
        f"out of {len(examples)} examples"
    )
    if failed:
        print("\nFailed examples:")
        for name, exc in failed:
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")
        return 1
    else:
        print("\n✓ All examples completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
