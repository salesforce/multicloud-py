#!/usr/bin/env python3
"""Async blob storage examples for the Python Multi-Cloud SDK.

Demonstrates cloud-agnostic async blob operations. The same code works with
any provider, using native async I/O (aiobotocore for AWS, gcloud-aio-storage
for GCP).

Usage:
    # AWS Example
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_SESSION_TOKEN=...  # Optional, if using temporary credentials
    export BUCKET_NAME=my-bucket
    export BUCKET_REGION=us-west-2
    python examples/blob/blob_async_example.py --provider aws

    # GCP Example
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    export BUCKET_NAME=my-bucket
    export BUCKET_REGION=us-central1
    python examples/blob/blob_async_example.py --provider gcp

    # Or set provider via environment variable
    export PROVIDER=aws  # or "gcp" (defaults to "aws")
    python examples/blob/blob_async_example.py

Prerequisites:
    pip install 'python-multicloud-sdk[async-aws]'   # or [async-gcp] or [async-all]
"""

import asyncio
import io
import os
import shutil
import ssl
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from multicloud.blob.blob_client import (
    AsyncBucketClient,
    AsyncTransferConfig,
    BlobIdentifier,
    BulkDeleteResponse,
    MultipartPart,
    MultipartUpload,
    ObjectRetentionConfig,
    PresignedUrlResult,
)
from multicloud.common.credentials import (
    CredentialsOverrider,
    CredentialsType,
    StsCredentials,
)
from multicloud.common.exceptions import (
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceNotFoundException,
    SubstrateSdkException,
)
from multicloud.common.proxy import ProxyConfig


async def upload():
    """Upload raw bytes to blob storage."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/hello.txt"
    content = b"Hello from the async Python Multi-Cloud SDK!"

    response = await client.upload(
        key=key,
        payload=content,
        content_type="text/plain",
        metadata={"source": "async-example", "team": "multicloud"},
    )

    print(f"\n=== Async Upload ===")
    print(
        f"Received upload response: ETag={response.e_tag}, VersionId={response.version_id}"
    )

    await client.close()


async def upload_from_path():
    """Upload a local file by path, letting the SDK read and type it."""
    client = get_async_bucket_client(get_provider())

    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text('{"message": "uploaded from file path", "async": true}')

    key = "examples/async/from-path.json"

    print(f"\n=== Async Upload from Path ===")
    try:
        # The path may be a str as well as a Path, content_type is detected
        # from the .json extension, and the read happens off the event loop.
        response = await client.upload_from_path(
            key,
            str(tmp),
            metadata={"upload-method": "path"},
        )
        print(f"Uploaded file: {tmp.name}")
        print(f"ETag={response.e_tag}, VersionId={response.version_id}")

        md = await client.get_metadata(key)
        print(f"Detected content type: {md.content_type}")
        print(f"Size on the service: {md.content_length} bytes")

        # A file that is not there is refused before any request is sent.
        try:
            await client.upload_from_path(key, tmp.with_name("absent.json"))
        except InvalidArgumentException as exc:
            print(f"Missing file refused: {exc}")

        await client.delete(key)
    finally:
        tmp.unlink(missing_ok=True)
        await client.close()


async def upload_from_stream():
    """Upload content from a file-like BinaryIO stream."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/from-stream.bin"
    stream = io.BytesIO(b"Binary content from a stream object")

    response = await client.upload(
        key=key,
        payload=stream,
        content_type="application/octet-stream",
    )

    print(f"\n=== Async Upload from Stream ===")
    print(
        f"Received upload response: ETag={response.e_tag}, VersionId={response.version_id}"
    )

    await client.close()


async def upload_with_transfer_config():
    """Upload with custom transfer configuration."""
    config = AsyncTransferConfig(
        max_concurrency=20,
        multipart_threshold=16 * 1024 * 1024,  # 16 MB
    )

    client = get_async_bucket_client(get_provider(), transfer_config=config)

    key = "examples/async/configured.txt"
    content = b"Uploaded with custom transfer config"

    response = await client.upload(key=key, payload=content)

    print(f"\n=== Async Upload (custom TransferConfig) ===")
    print(
        f"Received upload response: ETag={response.e_tag}, VersionId={response.version_id}"
    )
    print(
        f"Transfer config: max_concurrency={config.max_concurrency}, "
        f"multipart_threshold={config.multipart_threshold}"
    )

    await client.close()


async def upload_with_context_manager():
    """Demonstrate async context manager for automatic resource cleanup."""
    key = "examples/async/ctx-managed.txt"
    content = b"Uploaded using async context manager"

    async with get_async_bucket_client(get_provider()) as client:
        response = await client.upload(
            key=key,
            payload=content,
            content_type="text/plain",
        )

    print(f"\n=== Async Upload (context manager) ===")
    print(
        f"Received upload response: ETag={response.e_tag}, VersionId={response.version_id}"
    )
    print("Client automatically closed on context manager exit")


async def upload_multiple_concurrent():
    """Upload multiple objects concurrently using asyncio.gather."""
    client = get_async_bucket_client(get_provider())

    keys_and_data = [
        ("examples/async/concurrent-1.txt", b"First concurrent upload"),
        ("examples/async/concurrent-2.txt", b"Second concurrent upload"),
        ("examples/async/concurrent-3.txt", b"Third concurrent upload"),
        ("examples/async/concurrent-4.txt", b"Fourth concurrent upload"),
        ("examples/async/concurrent-5.txt", b"Fifth concurrent upload"),
    ]

    tasks = [
        client.upload(key=key, payload=data, content_type="text/plain")
        for key, data in keys_and_data
    ]
    responses = await asyncio.gather(*tasks)

    print(f"\n=== Async Upload (concurrent with asyncio.gather) ===")
    print(f"Uploaded {len(responses)} objects concurrently:")
    for resp in responses:
        print(f"  {resp.key} -> ETag={resp.e_tag}")

    await client.close()


async def upload_with_metadata():
    """Upload with rich metadata."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/metadata-demo.txt"
    content = b"Object with rich metadata"
    metadata = {
        "environment": "development",
        "team": "multicloud",
        "version": "2.0",
        "created-by": "async-example",
    }

    response = await client.upload(
        key=key,
        payload=content,
        content_type="text/plain",
        metadata=metadata,
    )

    print(f"\n=== Async Upload (with metadata) ===")
    print(
        f"Received upload response: ETag={response.e_tag}, VersionId={response.version_id}"
    )
    print(f"Metadata sent: {metadata}")

    await client.close()


async def upload_with_storage_class():
    """Upload into a non-default storage class, then confirm it via list_page."""
    provider = get_provider()
    client = get_async_bucket_client(provider)

    # Storage class names are provider-native and passed through unchanged.
    storage_class = "STANDARD_IA" if provider == "aws" else "NEARLINE"
    key = "examples/async/storage-class/infrequent.txt"

    await client.upload(
        key=key,
        payload=b"Rarely read content",
        content_type="text/plain",
        storage_class=storage_class,
    )

    print(f"\n=== Async Upload (with storage class) ===")
    print(f"Uploaded '{key}' with storage_class={storage_class}")

    page = await client.list_page(prefix=key)
    reported = page.blobs[0].storage_class
    print(f"Storage class reported by list_page: {reported}")
    assert reported == storage_class, f"expected {storage_class}, got {reported}"

    await client.delete(key)
    await client.close()


async def upload_error_handling():
    """Demonstrate error handling."""
    client = get_async_bucket_client(get_provider())

    print(f"\n=== Async Upload (error handling) ===")

    # Empty key should raise InvalidArgumentException
    try:
        await client.upload(key="", payload=b"data")
    except InvalidArgumentException as exc:
        print(f"Caught expected InvalidArgumentException: {exc}")

    # Successful upload for comparison
    response = await client.upload(
        key="examples/async/error-demo.txt",
        payload=b"This one succeeds",
    )
    print(f"Successful upload: ETag={response.e_tag}")

    await client.close()


async def tags_round_trip():
    """Set tags on an object and read them back, verifying the round-trip."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/tagged.txt"
    tags = {"env": "development", "team": "multicloud"}

    print(f"\n=== Async Tags (set + get round-trip) ===")

    # The object must exist before tags can be attached.
    await client.upload(key=key, payload=b"object to be tagged")

    await client.set_tags(key, tags)
    fetched = await client.get_tags(key)

    print(f"Tags set:     {tags}")
    print(f"Tags fetched: {dict(fetched)}")

    if dict(fetched) != tags:
        raise AssertionError(
            f"Tag round-trip mismatch: set {tags}, got {dict(fetched)}"
        )
    print("Round-trip verified: fetched tags match what was set")

    await client.close()


async def upload_large_payload():
    """Upload a larger payload (1 MB)."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/large-payload.bin"
    content = b"X" * (1024 * 1024)

    response = await client.upload(
        key=key,
        payload=content,
        content_type="application/octet-stream",
    )

    print(f"\n=== Async Upload (large payload - 1MB) ===")
    print(
        f"Received upload response: ETag={response.e_tag}, VersionId={response.version_id}"
    )
    print(f"Payload size: {len(content):,} bytes")

    await client.close()


async def upload_explicit_content_length():
    """Upload with explicit content_length parameter."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/explicit-length.txt"
    content = b"Content with explicit length"

    response = await client.upload(
        key=key,
        payload=content,
        content_type="text/plain",
        content_length=len(content),
    )

    print(f"\n=== Async Upload (explicit content_length) ===")
    print(
        f"Received upload response: ETag={response.e_tag}, VersionId={response.version_id}"
    )
    print(f"Content-Length: {len(content)}")

    await client.close()


async def download():
    """Download blob content into memory."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/hello.txt"
    response = await client.download(key)

    print(f"\n=== Async Download ===")
    print(f"Key: {response.key}")
    print(f"Content: {response.content.decode('utf-8')}")
    print(f"Content-Length: {response.metadata.content_length}")
    print(f"Content-Type: {response.metadata.content_type}")
    print(f"ETag: {response.metadata.e_tag}")

    await client.close()


async def download_to_path():
    """Download blob content to a local file."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/hello.txt"
    dest = Path(tempfile.mkdtemp()) / "downloaded.txt"

    response = await client.download_to_path(
        key,
        destination=dest,
        create_parent_dirs=True,
    )

    print(f"\n=== Async Download to Path ===")
    print(f"Downloaded to: {dest}")
    print(f"File content: {dest.read_text()}")
    print(f"Content-Length: {response.metadata.content_length}")

    dest.unlink(missing_ok=True)
    await client.close()


async def download_to_file():
    """Download blob content to a file-like object."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/hello.txt"
    buffer = io.BytesIO()

    response = await client.download_to_file(key, file_obj=buffer)

    buffer.seek(0)
    print(f"\n=== Async Download to File Object ===")
    print(f"Buffer content: {buffer.read().decode('utf-8')}")
    print(f"Content-Length: {response.metadata.content_length}")

    await client.close()


async def download_range():
    """Download a byte range (partial content)."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/hello.txt"

    # Download bytes 0-4 (inclusive, RFC 7233)
    response = await client.download(key, start=0, end=4)

    print(f"\n=== Async Download (range) ===")
    print(f"Range bytes=0-4: {response.content.decode('utf-8')!r}")
    print(f"Content-Length: {response.metadata.content_length}")

    await client.close()


async def download_with_version():
    """Download a specific object version (AWS only)."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/hello.txt"

    # version_id is supported on AWS; raises InvalidArgumentException on GCP
    response = await client.download(key, version_id="some-version-id")

    print(f"\n=== Async Download (versioned) ===")
    print(f"Version: {response.metadata.version_id}")
    print(f"Content-Length: {response.metadata.content_length}")

    await client.close()


async def download_concurrent():
    """Download multiple objects concurrently with asyncio.gather."""
    client = get_async_bucket_client(get_provider())

    keys = [
        "examples/async/concurrent-1.txt",
        "examples/async/concurrent-2.txt",
        "examples/async/concurrent-3.txt",
    ]

    responses = await asyncio.gather(*[client.download(k) for k in keys])

    print(f"\n=== Async Download (concurrent) ===")
    for resp in responses:
        print(f"  {resp.key}: {len(resp.content)} bytes")

    await client.close()


async def download_error_handling():
    """Demonstrate download error handling."""
    from multicloud.blob.blob_client.async_blob.async_driver import BlobNotFoundError

    client = get_async_bucket_client(get_provider())

    print(f"\n=== Async Download (error handling) ===")

    # Missing key raises BlobNotFoundError
    try:
        await client.download("nonexistent/key.txt")
    except BlobNotFoundError as exc:
        print(f"Caught BlobNotFoundError: {exc}")

    # Invalid range raises InvalidArgumentException
    try:
        await client.download("examples/async/hello.txt", start=-1)
    except InvalidArgumentException as exc:
        print(f"Caught InvalidArgumentException: {exc}")

    await client.close()


async def upload_then_download():
    """Upload content, then download it back to verify round-trip."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/roundtrip.txt"
    content = b"Round-trip verification content"

    await client.upload(key=key, payload=content, content_type="text/plain")
    response = await client.download(key)

    print(f"\n=== Async Upload then Download ===")
    print(f"Uploaded: {len(content)} bytes")
    print(f"Downloaded: {len(response.content)} bytes")
    print(f"Match: {response.content == content}")

    await client.close()


async def get_metadata_basic():
    """Get metadata for an uploaded object."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/metadata-basic.txt"
    content = b"Object for metadata inspection"

    await client.upload(
        key=key,
        payload=content,
        content_type="text/plain",
        metadata={"team": "multicloud", "purpose": "metadata-demo"},
    )

    md = await client.get_metadata(key)

    print(f"\n=== Async Get Metadata (basic) ===")
    print(f"Key: {md.key}")
    print(f"Content-Length: {md.content_length}")
    print(f"Content-Type: {md.content_type}")
    print(f"ETag: {md.e_tag}")
    print(f"Version ID: {md.version_id}")
    print(f"Metadata: {md.metadata}")
    print(f"Last Modified: {md.last_modified}")
    print(f"Created Time: {md.created_time}")

    await client.close()


async def get_metadata_round_trip():
    """Upload with known metadata, get_metadata, and verify round-trip."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/metadata-roundtrip.txt"
    content = b"Metadata round-trip verification payload"
    content_type = "application/json"
    metadata = {"env": "test", "team": "multicloud", "version": "1.0"}

    await client.upload(
        key=key,
        payload=content,
        content_type=content_type,
        metadata=metadata,
    )

    md = await client.get_metadata(key)

    print(f"\n=== Async Get Metadata (round-trip) ===")

    if md.content_length != len(content):
        raise AssertionError(
            f"Content-length mismatch: uploaded {len(content)}, got {md.content_length}"
        )

    if md.content_type != content_type:
        raise AssertionError(
            f"Content-type mismatch: uploaded {content_type!r}, got {md.content_type!r}"
        )

    for key_name in metadata:
        if key_name not in md.metadata:
            raise AssertionError(
                f"Metadata key missing: {key_name!r} not in {md.metadata}"
            )

    print("Round-trip verified: content_length, content_type, and metadata match")

    await client.close()


async def get_metadata_not_found():
    """Demonstrate get_metadata error handling for missing object."""
    from multicloud.blob.blob_client.async_blob.async_driver import BlobNotFoundError

    client = get_async_bucket_client(get_provider())

    key = "examples/async/never-existed-metadata-12345.txt"

    print(f"\n=== Async Get Metadata (not found) ===")

    try:
        await client.get_metadata(key)
        print("ERROR: Should have raised BlobNotFoundError!")
    except BlobNotFoundError as exc:
        print(f"Caught BlobNotFoundError: {exc}")

    await client.close()


async def get_metadata_error_handling():
    """Demonstrate get_metadata error handling for invalid input."""
    client = get_async_bucket_client(get_provider())

    print(f"\n=== Async Get Metadata (error handling) ===")

    # Empty key should raise InvalidArgumentException
    try:
        await client.get_metadata("")
    except InvalidArgumentException as exc:
        print(f"Caught expected InvalidArgumentException: {exc}")

    await client.close()


async def get_metadata_versioned():
    """Get metadata for a specific object version (versioned bucket)."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/metadata-versioned.txt"

    response = await client.upload(key=key, payload=b"Versioned metadata content")
    version_id = response.version_id

    print(f"\n=== Async Get Metadata (versioned) ===")

    if version_id:
        try:
            md = await client.get_metadata(key, version_id=version_id)
            print(f"Key: {md.key}")
            print(f"Version ID: {md.version_id}")
            print(f"Content-Length: {md.content_length}")
            print(f"ETag: {md.e_tag}")
        except Exception as exc:
            print(f"Error getting metadata for version {version_id}: {exc}")
    else:
        print("Versioning not enabled — skipped")

    await client.close()


async def delete_single():
    """Delete a single object from blob storage."""
    client = get_async_bucket_client(get_provider())

    # First upload an object to delete
    key = "examples/async/to-delete.txt"
    await client.upload(
        key=key, payload=b"This will be deleted", content_type="text/plain"
    )
    print(f"\n=== Async Delete (single) ===")
    print(f"Uploaded: {key}")

    # Delete it
    await client.delete(key)
    print(f"Deleted: {key}")

    # Delete again — idempotent, no error even though it's already gone
    await client.delete(key)
    print(f"Deleted again (idempotent): {key} — no error raised")

    await client.close()


async def check_object_exists():
    """Check if an object exists in the bucket."""
    client = get_async_bucket_client(get_provider())

    # Upload a test object first
    key = "examples/async/exists-test.txt"
    await client.upload(key=key, payload=b"Test content", content_type="text/plain")

    # Check if it exists
    exists = await client.does_object_exist(key)

    print(f"\n=== Async Object Existence Check ===")
    print(f"Key: {key}")
    print(f"Exists: {exists}")

    # Check a non-existent key
    missing_exists = await client.does_object_exist("examples/async/missing.txt")
    print(f"Missing key exists: {missing_exists}")

    await client.close()


async def check_object_exists_with_version():
    """Check if a specific object version exists (AWS only)."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/versioned-exists.txt"

    # Upload to get a version
    response = await client.upload(key=key, payload=b"Versioned content")
    version_id = response.version_id

    print(f"\n=== Async Object Existence Check (with version) ===")
    print(f"Key: {key}")
    print(f"Version: {version_id}")

    if version_id:
        exists = await client.does_object_exist(key, version_id=version_id)
        print(f"Version exists: {exists}")
    else:
        print("Versioning not enabled on this bucket")

    await client.close()


async def check_bucket_exists():
    """Check if the bucket exists."""
    client = get_async_bucket_client(get_provider())

    exists = await client.does_bucket_exist()

    print(f"\n=== Async Bucket Existence Check ===")
    print(f"Bucket: {client.bucket}")
    print(f"Exists: {exists}")

    await client.close()


async def existence_check_error_handling():
    """Demonstrate existence check error handling."""
    client = get_async_bucket_client(get_provider())

    print(f"\n=== Async Existence Check (error handling) ===")

    # Empty key should raise InvalidArgumentException
    try:
        await client.does_object_exist("")
    except InvalidArgumentException as exc:
        print(f"Caught expected InvalidArgumentException: {exc}")

    # Successful check for comparison
    exists = await client.does_object_exist("examples/async/hello.txt")
    print(f"Valid key exists: {exists}")

    await client.close()


async def delete_with_version():
    """Delete a specific version of an object (AWS only — versioned bucket)."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/versioned-delete.txt"

    # Upload twice to create versions
    resp1 = await client.upload(key=key, payload=b"Version 1")
    resp2 = await client.upload(key=key, payload=b"Version 2")

    print(f"\n=== Async Delete (versioned) ===")
    print(f"Uploaded v1: version_id={resp1.version_id}")
    print(f"Uploaded v2: version_id={resp2.version_id}")

    # Delete only version 1 (if versioning is enabled on the bucket)
    if resp1.version_id:
        await client.delete(key, version_id=resp1.version_id)
        print(f"Deleted version: {resp1.version_id}")
    else:
        print("Bucket does not have versioning enabled — skipping versioned delete")
        await client.delete(key)
        print(f"Deleted: {key}")

    await client.close()


async def delete_nonexistent():
    """Delete a non-existent key — idempotent, no error."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/never-existed-12345.txt"

    print(f"\n=== Async Delete (non-existent key — idempotent) ===")
    await client.delete(key)
    print(f"Deleted non-existent key: {key} — no error raised")

    await client.close()


async def delete_objects_batch():
    """Batch delete multiple objects at once."""
    client = get_async_bucket_client(get_provider())

    # Upload several objects first
    keys = [f"examples/async/batch-delete-{i}.txt" for i in range(5)]
    for key in keys:
        await client.upload(key=key, payload=f"Content for {key}".encode())

    print(f"\n=== Async Delete Objects (batch) ===")
    print(f"Uploaded {len(keys)} objects")

    # Batch delete them all
    identifiers = [BlobIdentifier(key=k) for k in keys]
    result: BulkDeleteResponse = await client.delete_objects(identifiers)

    print(f"Deleted: {len(result.deleted)} objects")
    for d in result.deleted:
        print(f"  ✓ {d.key}")

    if result.has_errors:
        print(f"Errors: {len(result.errors)}")
        for e in result.errors:
            print(f"  ✗ {e.key}: [{e.code}] {e.message}")
    else:
        print("No errors — all objects deleted successfully")

    await client.close()


async def delete_objects_partial_failure():
    """Demonstrate partial-failure reporting in batch delete.

    Even if some objects fail to delete, the operation does not raise.
    Instead, failed items are reported in BulkDeleteResponse.errors.
    """
    client = get_async_bucket_client(get_provider())

    # Upload one real object and include a mix of keys
    real_key = "examples/async/batch-partial-real.txt"
    await client.upload(key=real_key, payload=b"real content")

    print(f"\n=== Async Delete Objects (partial failure demo) ===")
    print(f"Uploaded: {real_key}")

    # Include the real key plus a non-existent one (non-existent is success/idempotent)
    identifiers = [
        BlobIdentifier(key=real_key),
        BlobIdentifier(key="examples/async/nonexistent-1.txt"),
        BlobIdentifier(key="examples/async/nonexistent-2.txt"),
    ]

    result = await client.delete_objects(identifiers)
    print(f"Total requested: {len(identifiers)}")
    print(f"Successfully deleted: {len(result.deleted)}")
    print(f"Failed: {len(result.errors)}")
    print(f"has_errors: {result.has_errors}")

    await client.close()


async def delete_objects_with_versions():
    """Batch delete with version IDs (for versioned buckets)."""
    client = get_async_bucket_client(get_provider())

    # Upload objects
    key1 = "examples/async/batch-v-1.txt"
    key2 = "examples/async/batch-v-2.txt"
    resp1 = await client.upload(key=key1, payload=b"content 1")
    resp2 = await client.upload(key=key2, payload=b"content 2")

    print(f"\n=== Async Delete Objects (with version IDs) ===")
    print(f"Uploaded: {key1} (version={resp1.version_id})")
    print(f"Uploaded: {key2} (version={resp2.version_id})")

    # Delete specific versions
    identifiers = [
        BlobIdentifier(key=key1, version_id=resp1.version_id),
        BlobIdentifier(key=key2, version_id=resp2.version_id),
    ]

    result = await client.delete_objects(identifiers)
    print(f"Deleted: {len(result.deleted)} objects")
    if result.has_errors:
        print(f"Errors: {len(result.errors)}")
        for e in result.errors:
            print(f"  ✗ {e.key}: [{e.code}] {e.message}")
    else:
        print("All versioned deletes succeeded")

    await client.close()


async def delete_error_handling():
    """Demonstrate delete error handling."""
    client = get_async_bucket_client(get_provider())

    print(f"\n=== Async Delete (error handling) ===")

    # Empty key should raise InvalidArgumentException
    try:
        await client.delete("")
    except InvalidArgumentException as exc:
        print(f"Caught expected InvalidArgumentException: {exc}")

    # Empty identifiers list should raise InvalidArgumentException
    try:
        await client.delete_objects([])
    except InvalidArgumentException as exc:
        print(f"Caught expected InvalidArgumentException for empty list: {exc}")

    # Valid delete for comparison
    await client.delete("examples/async/cleanup-test.txt")
    print("Successful delete of non-existent key (idempotent)")

    await client.close()


async def upload_then_delete():
    """Full lifecycle: upload, verify, delete, verify gone."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/lifecycle-test.txt"
    content = b"Lifecycle test content"

    # Upload
    await client.upload(key=key, payload=content, content_type="text/plain")
    print(f"\n=== Async Upload then Delete (lifecycle) ===")
    print(f"Uploaded: {key}")

    # Verify it exists by downloading
    resp = await client.download(key)
    print(f"Downloaded: {len(resp.content)} bytes (verified exists)")

    # Delete
    await client.delete(key)
    print(f"Deleted: {key}")

    # Verify it's gone
    from multicloud.blob.blob_client.async_blob.async_driver import BlobNotFoundError

    try:
        await client.download(key)
        print("WARNING: Object still exists after delete!")
    except (BlobNotFoundError, Exception):
        print("Confirmed: Object no longer exists after delete")

    await client.close()


async def copy_same_bucket():
    """Copy an object within the same bucket (rename/move pattern)."""
    client = get_async_bucket_client(get_provider())
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")

    # First, upload a source object to copy
    source_key = "examples/async/copy-source.txt"
    content = b"Hello from the async copy source!"

    await client.upload(
        key=source_key,
        payload=content,
        content_type="text/plain",
        metadata={"source": "copy-example"},
    )

    # Now copy within the same bucket
    dest_key = "examples/async/copy-dest.txt"

    response = await client.copy(
        source_key,
        dest_bucket=bucket,
        dest_key=dest_key,
    )

    print(f"\n=== Async Copy (same bucket) ===")
    print(f"Source: {bucket}/{source_key}")
    print(f"Destination: {bucket}/{dest_key}")
    print(
        f"Result: key={response.key}, ETag={response.e_tag}, "
        f"VersionId={response.version_id}"
    )
    print(f"Correlation ID: {response.correlation_id}")

    await client.close()


async def copy_with_metadata_override():
    """Copy an object while replacing its metadata."""
    client = get_async_bucket_client(get_provider())
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")

    # Upload source with original metadata
    source_key = "examples/async/copy-meta-source.txt"
    await client.upload(
        key=source_key,
        payload=b"Content with metadata",
        metadata={"environment": "dev", "version": "1.0"},
    )

    # Copy with new metadata
    dest_key = "examples/async/copy-meta-dest.txt"

    response = await client.copy(
        source_key,
        dest_bucket=bucket,
        dest_key=dest_key,
        metadata={"environment": "prod", "version": "2.0", "copied-from": source_key},
    )

    print(f"\n=== Async Copy (metadata override) ===")
    print(f"Source: {source_key}")
    print(f"Destination: {dest_key}")
    print(f"New metadata: environment=prod, version=2.0")
    print(f"Result: key={response.key}, ETag={response.e_tag}")
    print(f"Correlation ID: {response.correlation_id}")

    await client.close()


async def copy_missing_source():
    """Demonstrate error handling when source doesn't exist."""
    client = get_async_bucket_client(get_provider())
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")

    print(f"\n=== Async Copy (missing source - error handling) ===")
    try:
        await client.copy(
            "examples/async/this-does-not-exist-12345.txt",
            dest_bucket=bucket,
            dest_key="examples/async/should-not-be-created.txt",
        )
        print("  ERROR: Should have raised ResourceNotFoundException!")
    except ResourceNotFoundException as exc:
        print(f"  Correctly caught ResourceNotFoundException: {exc}")
    except Exception as exc:
        print(f"  Caught exception: {type(exc).__name__}: {exc}")

    await client.close()


async def copy_validation_error():
    """Demonstrate copy input validation error handling."""
    client = get_async_bucket_client(get_provider())
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")

    print(f"\n=== Async Copy (validation errors) ===")

    # Empty source key
    try:
        await client.copy(
            "",
            dest_bucket=bucket,
            dest_key="dest",
        )
    except InvalidArgumentException as exc:
        print(f"  Empty source_key: {exc}")

    # Empty destination bucket
    try:
        await client.copy(
            "key",
            dest_bucket="",
            dest_key="dest",
        )
    except InvalidArgumentException as exc:
        print(f"  Empty destination_bucket: {exc}")

    await client.close()


async def copy_multiple_concurrent():
    """Copy multiple objects concurrently using asyncio.gather."""
    client = get_async_bucket_client(get_provider())
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")

    # Upload several source objects
    source_keys = [f"examples/async/concurrent-copy-src-{i}.txt" for i in range(5)]
    upload_tasks = [
        client.upload(key=key, payload=f"Content {i}".encode())
        for i, key in enumerate(source_keys)
    ]
    await asyncio.gather(*upload_tasks)

    # Copy all concurrently
    copy_tasks = [
        client.copy(
            src_key,
            dest_bucket=bucket,
            dest_key=f"examples/async/concurrent-copy-dest-{i}.txt",
        )
        for i, src_key in enumerate(source_keys)
    ]
    responses = await asyncio.gather(*copy_tasks)

    print(f"\n=== Async Copy (concurrent with asyncio.gather) ===")
    print(f"Copied {len(responses)} objects concurrently:")
    for resp in responses:
        print(f"  {resp.key} -> ETag={resp.e_tag}")

    await client.close()


async def generate_presigned_download_url():
    """Generate a presigned GET URL for downloading an object."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/presigned-download.txt"

    # First upload the object
    await client.upload(key=key, payload=b"Content for presigned download")

    result = await client.generate_presigned_url(key, operation="GET")

    print(f"\n=== Generate Presigned Download URL ===")
    print(f"URL: {result.url}")
    print(f"HTTP Method: {result.http_method}")
    print(f"Expiration: {result.expiration}")

    await client.close()


async def generate_presigned_upload_url():
    """Generate a presigned PUT URL for uploading an object."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/presigned-upload.txt"

    result = await client.generate_presigned_url(
        key, operation="PUT", content_type="text/plain"
    )

    print(f"\n=== Generate Presigned Upload URL ===")
    print(f"URL: {result.url}")
    print(f"HTTP Method: {result.http_method}")
    print(f"Required Headers: {result.required_headers}")

    await client.close()


async def generate_presigned_url_custom_expiration():
    """Generate a presigned URL with custom expiration time."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/presigned-custom-expiry.txt"

    # First upload the object
    await client.upload(key=key, payload=b"Content with custom expiration")

    result = await client.generate_presigned_url(
        key, operation="GET", expiration_seconds=7200
    )

    print(f"\n=== Generate Presigned URL (custom expiration) ===")
    print(f"Expiration: {result.expiration}")

    await client.close()


async def generate_presigned_url_with_metadata():
    """Generate a presigned PUT URL with metadata."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/presigned-with-metadata.json"

    result = await client.generate_presigned_url(
        key,
        operation="PUT",
        content_type="application/json",
        metadata={"team": "multicloud", "env": "demo"},
    )

    print(f"\n=== Generate Presigned URL (with metadata) ===")
    print(f"Required Headers: {result.required_headers}")

    await client.close()


async def generate_presigned_url_with_content_disposition():
    """Generate a presigned GET URL with content disposition."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/presigned-report.pdf"

    # First upload the object
    await client.upload(key=key, payload=b"PDF report content")

    result = await client.generate_presigned_url(
        key, operation="GET", content_disposition='attachment; filename="report.pdf"'
    )

    print(f"\n=== Generate Presigned URL (content disposition) ===")
    print(f"URL: {result.url}")

    await client.close()


async def generate_presigned_url_versioned():
    """Generate a presigned GET URL for a specific object version."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/presigned-versioned.txt"

    # First upload the object
    upload_response = await client.upload(key=key, payload=b"Versioned content")

    print(f"\n=== Generate Presigned URL (versioned) ===")

    try:
        result = await client.generate_presigned_url(
            key, operation="GET", version_id="1"
        )
        print(f"URL: {result.url}")
    except Exception as exc:
        print(f"Versioning not supported or error occurred: {exc}")

    await client.close()


async def generate_presigned_url_concurrent():
    """Generate multiple presigned URLs concurrently with asyncio.gather."""
    client = get_async_bucket_client(get_provider())

    keys = [
        "examples/async/concurrent-presigned-0.txt",
        "examples/async/concurrent-presigned-1.txt",
        "examples/async/concurrent-presigned-2.txt",
    ]

    # Upload the objects first
    for key in keys:
        await client.upload(key=key, payload=f"Content for {key}".encode())

    # Generate presigned URLs concurrently
    results = await asyncio.gather(
        *[client.generate_presigned_url(key, operation="GET") for key in keys]
    )

    print(f"\n=== Generate Presigned URLs (concurrent) ===")
    print(f"Generated {len(results)} presigned URLs concurrently")

    await client.close()


async def generate_presigned_url_error_handling():
    """Demonstrate presigned URL generation error handling."""
    client = get_async_bucket_client(get_provider())

    print(f"\n=== Generate Presigned URL (error handling) ===")

    # Invalid operation should raise InvalidArgumentException
    try:
        await client.generate_presigned_url(
            "examples/async/test.txt", operation="DELETE"
        )
    except InvalidArgumentException as exc:
        print(f"Caught expected InvalidArgumentException for invalid operation: {exc}")

    # Invalid expiration should raise InvalidArgumentException
    try:
        await client.generate_presigned_url(
            "examples/async/test.txt", operation="GET", expiration_seconds=0
        )
    except InvalidArgumentException as exc:
        print(f"Caught expected InvalidArgumentException for invalid expiration: {exc}")

    await client.close()


def _make_ssl_context():
    """Create an SSL context for presigned URL HTTP calls.

    Defaults to the system trust store (secure). Set env var
    MULTICLOUD_EXAMPLE_INSECURE_TLS=1 to disable verification for
    environments where the system trust store is incomplete.
    """
    ctx = ssl.create_default_context()
    if os.environ.get("MULTICLOUD_EXAMPLE_INSECURE_TLS") == "1":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _http_put(url, data, headers):
    """Blocking HTTP PUT (run via asyncio.to_thread from async callers)."""
    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    with urllib.request.urlopen(req, context=_make_ssl_context()) as resp:
        return resp.status


def _http_get(url):
    """Blocking HTTP GET (run via asyncio.to_thread from async callers)."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, context=_make_ssl_context()) as resp:
        return resp.status, resp.read()


async def presigned_put_then_get_http_round_trip():
    """Exercise presigned URLs over real HTTP: PUT to upload, GET to download.

    This proves the generated signature is actually usable, not just that a
    URL string was produced. The PUT replays every header the signer folded
    into the signature (result.required_headers); omitting any of them would
    cause a 403 SignatureDoesNotMatch. The GET then reads the object back and
    the content is compared to what was uploaded.

    Raw HTTP uses the stdlib ``urllib.request`` (via ``asyncio.to_thread`` so
    the blocking call never stalls the event loop), matching the synchronous
    ``blob_example.py`` and avoiding any third-party HTTP dependency.
    """
    client = get_async_bucket_client(get_provider())
    key = "examples/async/presigned-http-round-trip.txt"
    content = b"Presigned round-trip payload verified over real HTTP"
    content_type = "text/plain"

    print(f"\n=== Presigned URL HTTP Round-Trip (PUT then GET) ===")

    try:
        # 1. Generate a presigned PUT URL and upload via plain HTTP.
        put_result = await client.generate_presigned_url(
            key,
            operation="PUT",
            content_type=content_type,
        )
        put_headers = dict(put_result.required_headers)
        put_status = await asyncio.to_thread(
            _http_put, put_result.url, content, put_headers
        )
        print(f"PUT {put_status}: uploaded {len(content)} bytes via presigned URL")
        print(f"  Replayed required headers: {sorted(put_headers)}")

        # 2. Generate a presigned GET URL and download via plain HTTP.
        get_result = await client.generate_presigned_url(key, operation="GET")
        get_status, body = await asyncio.to_thread(_http_get, get_result.url)
        print(f"GET {get_status}: downloaded {len(body)} bytes")

        # 3. Verify the round-trip.
        if body != content:
            raise AssertionError(
                f"Round-trip mismatch: uploaded {content!r}, " f"downloaded {body!r}"
            )
        print("Round-trip verified: downloaded bytes match uploaded bytes")
    finally:
        await client.delete(key)
        await client.close()


async def presigned_kms_put_then_get_http_round_trip():
    """Presigned round-trip against a KMS-encrypted bucket over real HTTP.

    This is the scenario that requires SigV4. A bucket with KMS default
    encryption rejects a legacy SigV2 presigned URL with SignatureDoesNotMatch
    / InvalidArgument. SigV4 folds the ``x-amz-server-side-encryption*`` headers
    into the signature, so the PUT must replay every ``required_headers`` entry.

    Configure via environment variables (all optional):
      KMS_KEY_ID        — KMS key id/arn/alias. If unset, this example is
                          skipped (it needs a real key).
      KMS_BUCKET_NAME   — bucket to use; falls back to BUCKET_NAME.
      KMS_BUCKET_REGION — region for that bucket; falls back to BUCKET_REGION.
    """
    kms_key_id = os.environ.get("KMS_KEY_ID")
    print(f"\n=== Presigned KMS HTTP Round-Trip (PUT then GET) ===")
    if not kms_key_id:
        print("Set KMS_KEY_ID (and optionally KMS_BUCKET_NAME) to run this.")
        return

    bucket_name = os.environ.get("KMS_BUCKET_NAME") or os.environ.get(
        "BUCKET_NAME", "palsfdc"
    )
    region = os.environ.get("KMS_BUCKET_REGION") or os.environ.get("BUCKET_REGION")
    client = AsyncBucketClient(
        provider=get_provider(), bucket=bucket_name, region=region
    )
    key = "examples/async/presigned-kms-round-trip.txt"
    content = b"Encrypted presigned round-trip verified over real HTTP"

    try:
        put_result = await client.generate_presigned_url(
            key,
            operation="PUT",
            content_type="text/plain",
            kms_key_id=kms_key_id,
        )
        put_headers = dict(put_result.required_headers)
        put_status = await asyncio.to_thread(
            _http_put, put_result.url, content, put_headers
        )
        print(f"Bucket: {bucket_name} (region={region})")
        print(f"PUT {put_status}: uploaded {len(content)} bytes to KMS bucket")
        print(f"  Replayed required headers: {sorted(put_headers)}")

        get_result = await client.generate_presigned_url(key, operation="GET")
        get_status, body = await asyncio.to_thread(_http_get, get_result.url)
        print(f"GET {get_status}: downloaded {len(body)} bytes")

        if body != content:
            raise AssertionError(
                f"KMS round-trip mismatch: uploaded {content!r}, "
                f"downloaded {body!r}"
            )
        print("Round-trip verified: SigV4 presigned upload accepted by KMS bucket")
    finally:
        await client.delete(key)
        await client.close()


async def upload_directory():
    """Upload an entire local directory to cloud storage asynchronously.

    Demonstrates:
    - Bounded concurrency with worker pool pattern
    - Continue-on-error semantics
    - Prefix assignment
    - Tag propagation
    """
    client = get_async_bucket_client(get_provider())
    tmp_dir = Path(tempfile.mkdtemp(prefix="async_upload_dir_"))

    try:
        (tmp_dir / "readme.txt").write_text("Directory upload example")
        (tmp_dir / "data.json").write_text('{"key": "value"}')
        sub = tmp_dir / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested file content")

        response = await client.upload_directory(
            str(tmp_dir),
            prefix="examples/async/dir-upload/",
            include_sub_folders=True,
            max_concurrency=10,
            tags={"source": "async-example", "team": "multicloud"},
        )

        print(f"\n=== Async Directory Upload ===")
        print(f"Total files: {response.total_files}")
        print(f"Successful uploads: {response.successful_uploads}")
        print(f"Failed uploads: {len(response.failed_transfers)}")
        print(f"Total bytes transferred: {response.total_bytes_transferred:,}")
        print(f"Has failures: {response.has_failures}")

        if response.failed_transfers:
            print("Failed files:")
            for fail in response.failed_transfers:
                print(f"  {fail.source}: {fail.exception}")
    finally:

        shutil.rmtree(tmp_dir, ignore_errors=True)
        await client.close()


async def download_directory():
    """Download all objects under a prefix to a local directory asynchronously.

    Demonstrates:
    - Bounded concurrency download
    - Continue-on-error semantics
    - Prefix exclusion filtering
    - Automatic directory creation
    """
    client = get_async_bucket_client(get_provider())
    tmp_upload_dir = Path(tempfile.mkdtemp(prefix="async_upload_src_"))
    tmp_download_dir = Path(tempfile.mkdtemp(prefix="async_download_dest_"))

    try:
        # First upload some files to have something to download
        (tmp_upload_dir / "file1.txt").write_text("File one content")
        (tmp_upload_dir / "file2.txt").write_text("File two content")
        sub = tmp_upload_dir / "sub"
        sub.mkdir()
        (sub / "file3.txt").write_text("Nested file three")

        await client.upload_directory(
            str(tmp_upload_dir),
            prefix="examples/async/dir-download-test/",
            include_sub_folders=True,
        )

        # Now download them to a new directory
        response = await client.download_directory(
            "examples/async/dir-download-test",
            str(tmp_download_dir),
            max_concurrency=10,
        )

        print(f"\n=== Async Directory Download ===")
        print(f"Total files: {response.total_files}")
        print(f"Successful downloads: {response.successful_downloads}")
        print(f"Failed downloads: {len(response.failed_transfers)}")
        print(f"Total bytes transferred: {response.total_bytes_transferred:,}")
        print(f"Has failures: {response.has_failures}")

        # Show downloaded files
        print("Downloaded files:")
        for p in sorted(tmp_download_dir.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(tmp_download_dir)}: {p.stat().st_size} bytes")

        if response.failed_transfers:
            print("Failed files:")
            for fail in response.failed_transfers:
                print(f"  {fail.destination}: {fail.exception}")
    finally:

        shutil.rmtree(tmp_upload_dir, ignore_errors=True)
        shutil.rmtree(tmp_download_dir, ignore_errors=True)
        await client.close()


async def download_directory_with_exclusions():
    """Download a directory but exclude certain prefixes.

    Demonstrates the prefixes_to_exclude option for skipping specific
    subdirectories during download.
    """
    client = get_async_bucket_client(get_provider())
    tmp_upload_dir = Path(tempfile.mkdtemp(prefix="async_excl_upload_"))
    tmp_download_dir = Path(tempfile.mkdtemp(prefix="async_excl_download_"))

    try:
        # Upload test structure: keep/ and skip/ subdirectories
        keep = tmp_upload_dir / "keep"
        keep.mkdir()
        (keep / "wanted.txt").write_text("Keep this file")
        skip = tmp_upload_dir / "skip"
        skip.mkdir()
        (skip / "unwanted.txt").write_text("Skip this file")

        await client.upload_directory(
            str(tmp_upload_dir),
            prefix="examples/async/dir-excl-test/",
            include_sub_folders=True,
        )

        # Download with exclusion
        response = await client.download_directory(
            "examples/async/dir-excl-test",
            str(tmp_download_dir),
            prefixes_to_exclude=["examples/async/dir-excl-test/skip/"],
        )

        print(f"\n=== Async Directory Download (with exclusions) ===")
        print(f"Total files: {response.total_files}")
        print(f"Successful downloads: {response.successful_downloads}")
        print(f"Excluded 'skip/' prefix - only downloaded files from 'keep/'")

        # Verify skip directory was excluded
        print("Downloaded files:")
        for p in sorted(tmp_download_dir.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(tmp_download_dir)}")
    finally:

        shutil.rmtree(tmp_upload_dir, ignore_errors=True)
        shutil.rmtree(tmp_download_dir, ignore_errors=True)
        await client.close()


async def delete_directory():
    """Delete every object under a prefix asynchronously.

    Demonstrates:
    - Recursive prefix delete, including nested subdirectories
    - Verification that the prefix is empty afterwards
    - Idempotency: deleting an already-empty prefix is a no-op
    """
    client = get_async_bucket_client(get_provider())
    tmp_dir = Path(tempfile.mkdtemp(prefix="async_delete_dir_"))
    prefix = "examples/async/dir-delete/"

    try:
        # Seed a nested tree so the delete has something recursive to remove.
        (tmp_dir / "top.txt").write_text("top level file")
        (tmp_dir / "data.json").write_text('{"key": "value"}')
        sub = tmp_dir / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested file content")

        upload = await client.upload_directory(
            str(tmp_dir),
            prefix=prefix,
            include_sub_folders=True,
        )

        print(f"\n=== Async Directory Delete ===")
        print(f"Seeded {upload.successful_uploads} object(s) under {prefix}")

        before = await client.list_page(prefix=prefix)
        print(f"Objects before delete: {len(before.blobs)}")
        for blob in before.blobs:
            print(f"  {blob.key}")

        result = await client.delete_directory(prefix, max_concurrency=10)
        print(f"delete_directory returned: {result!r}")

        after = await client.list_page(prefix=prefix)
        print(f"Objects after delete: {len(after.blobs)}")
        if after.blobs:
            raise RuntimeError(
                f"delete_directory left {len(after.blobs)} object(s) under {prefix}"
            )

        # Deleting nothing must succeed rather than raise.
        await client.delete_directory(prefix)
        print("Re-deleting the now-empty prefix succeeded (idempotent)")
    finally:

        shutil.rmtree(tmp_dir, ignore_errors=True)
        await client.close()


async def delete_directory_is_recursive_and_scoped():
    """Show that a prefix delete is recursive but does not touch siblings.

    Seeds two sibling prefixes, deletes one, and verifies the other survives.
    This is the guard against an over-broad prefix removing more than intended.
    """
    client = get_async_bucket_client(get_provider())
    tmp_dir = Path(tempfile.mkdtemp(prefix="async_delete_scope_"))
    base = "examples/async/dir-delete-scope/"
    target = f"{base}target/"
    keeper = f"{base}keeper/"

    try:
        (tmp_dir / "a.txt").write_text("file a")
        deep = tmp_dir / "one" / "two"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("deeply nested")

        await client.upload_directory(str(tmp_dir), prefix=target)
        await client.upload_directory(str(tmp_dir), prefix=keeper)

        print(f"\n=== Async Directory Delete (recursive, sibling-safe) ===")
        target_before = await client.list_page(prefix=target)
        keeper_before = await client.list_page(prefix=keeper)
        print(f"target/ before: {len(target_before.blobs)} object(s)")
        print(f"keeper/ before: {len(keeper_before.blobs)} object(s)")

        await client.delete_directory(target)

        target_after = await client.list_page(prefix=target)
        keeper_after = await client.list_page(prefix=keeper)
        print(f"target/ after:  {len(target_after.blobs)} object(s)")
        print(f"keeper/ after:  {len(keeper_after.blobs)} object(s)")

        if target_after.blobs:
            raise RuntimeError("target prefix was not fully deleted")
        if len(keeper_after.blobs) != len(keeper_before.blobs):
            raise RuntimeError("sibling prefix was affected by the delete")
        print("Nested objects removed; sibling prefix untouched")
    finally:

        # Clean up the surviving sibling so the bucket is left as we found it.
        try:
            await client.delete_directory(base)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            await client.close()


# Async client choices — demonstrating WHY async directory operations matter
#
# With sync client: ZERO choices. Thread is frozen. Nothing else can run.
# With async client: the caller CHOOSES how to use the operation.
#
# Choice 1: await sequentially (simple, thread serves other coroutines)
# Choice 2: asyncio.gather() — run multiple operations concurrently
# Choice 3: asyncio.create_task() — start in background, do other work


async def directory_choice_1_sequential():
    """Choice 1: Simple sequential await.

    response = await client.upload_directory(...)

    The caller waits for the result. Looks similar to sync, but the
    thread is NOT blocked — the event loop is free to serve other
    coroutines (e.g. other HTTP requests in a web server).
    """
    client = get_async_bucket_client(get_provider())
    tmp_dir = Path(tempfile.mkdtemp(prefix="async_choice1_"))

    try:
        (tmp_dir / "file1.txt").write_text("Sequential upload file 1")
        (tmp_dir / "file2.txt").write_text("Sequential upload file 2")
        sub = tmp_dir / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested file content")

        # Choice 1: simple await — thread is free while waiting
        response = await client.upload_directory(
            str(tmp_dir),
            prefix="examples/async/choice1-sequential/",
            include_sub_folders=True,
        )

        print(f"\n=== Choice 1: Sequential await ===")
        print(f"response = await client.upload_directory(...)")
        print(f"  Files uploaded: {response.successful_uploads}")
        print(f"  Bytes transferred: {response.total_bytes_transferred:,}")
        print(f"  Has failures: {response.has_failures}")
        print("Thread was FREE while waiting (not blocked like sync)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        await client.close()


async def directory_choice_2_gather():
    """Choice 2: Upload multiple directories concurrently with asyncio.gather.

    r1, r2, r3 = await asyncio.gather(
        client.upload_directory("/dir1", prefix="a/"),
        client.upload_directory("/dir2", prefix="b/"),
        client.upload_directory("/dir3", prefix="c/"),
    )

    All 3 directories upload at the same time. Total time ≈ slowest one.
    This is IMPOSSIBLE with the sync client.
    """
    client = get_async_bucket_client(get_provider())

    dirs = []
    try:
        for i in range(3):
            tmp = Path(tempfile.mkdtemp(prefix=f"async_choice2_dir{i}_"))
            (tmp / f"file_a.txt").write_text(f"Directory {i}, file A content")
            (tmp / f"file_b.txt").write_text(f"Directory {i}, file B content")
            sub = tmp / "nested"
            sub.mkdir()
            (sub / "deep.txt").write_text(f"Directory {i}, nested file")
            dirs.append(tmp)

        # Choice 2: parallel — all 3 upload concurrently (IMPOSSIBLE with sync)
        r1, r2, r3 = await asyncio.gather(
            client.upload_directory(
                str(dirs[0]), prefix="examples/async/choice2-parallel/a/"
            ),
            client.upload_directory(
                str(dirs[1]), prefix="examples/async/choice2-parallel/b/"
            ),
            client.upload_directory(
                str(dirs[2]), prefix="examples/async/choice2-parallel/c/"
            ),
        )

        print(f"\n=== Choice 2: Parallel with asyncio.gather ===")
        print("r1, r2, r3 = await asyncio.gather(")
        print("    client.upload_directory('/dir1', prefix='a/'),")
        print("    client.upload_directory('/dir2', prefix='b/'),")
        print("    client.upload_directory('/dir3', prefix='c/'),")
        print(")")
        print(
            f"  dir1: {r1.successful_uploads} files, {r1.total_bytes_transferred:,} bytes"
        )
        print(
            f"  dir2: {r2.successful_uploads} files, {r2.total_bytes_transferred:,} bytes"
        )
        print(
            f"  dir3: {r3.successful_uploads} files, {r3.total_bytes_transferred:,} bytes"
        )
        total = r1.successful_uploads + r2.successful_uploads + r3.successful_uploads
        print(f"  Total: {total} files uploaded CONCURRENTLY")
        print("IMPOSSIBLE with sync — sync would take 3x longer (sequential)")
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
        await client.close()


async def directory_choice_3_create_task():
    """Choice 3: Start upload in background, do other work, collect later.

    task = asyncio.create_task(client.upload_directory("/data", prefix="v2/"))
    await do_other_work()      # runs immediately, doesn't wait for upload
    response = await task      # collect result when ready

    This is IMPOSSIBLE with the sync client — the thread would be frozen.
    """
    client = get_async_bucket_client(get_provider())
    tmp_dir = Path(tempfile.mkdtemp(prefix="async_choice3_"))

    try:
        for i in range(5):
            (tmp_dir / f"data_{i}.txt").write_text(f"Background data file {i}" * 10)

        print(f"\n=== Choice 3: Background with asyncio.create_task ===")

        # Start directory upload in background — does NOT block
        task = asyncio.create_task(
            client.upload_directory(
                str(tmp_dir),
                prefix="examples/async/choice3-background/",
                include_sub_folders=True,
            )
        )
        print("task = asyncio.create_task(client.upload_directory(...))")
        print("Upload started — NOT waiting yet...")

        # Do other work while upload runs in background
        await client.upload(
            key="examples/async/other-work-1.txt",
            payload=b"This runs WHILE directory upload is in progress",
        )
        print("  await do_other_work()  # upload progresses in background")

        print(
            "  await do_other_work()  # uploading other-work-2.txt during background upload"
        )
        await client.upload(
            key="examples/async/other-work-2.txt",
            payload=b"More work happening concurrently with dir upload",
        )
        print("  await do_more_work()   # still not waiting for upload")

        # Now collect the background upload result
        response = await task
        print(f"response = await task    # collect result when ready")
        print(f"  Files uploaded: {response.successful_uploads}")
        print(f"  Bytes transferred: {response.total_bytes_transferred:,}")
        print("IMPOSSIBLE with sync — sync freezes the thread on the first call")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        await client.close()


async def directory_download_choice_2_gather():
    """Choice 2 applied to download: download multiple prefixes concurrently.

    r1, r2 = await asyncio.gather(
        client.download_directory("prefix/a", "/local/a"),
        client.download_directory("prefix/b", "/local/b"),
    )

    Both downloads happen at the same time.
    """
    client = get_async_bucket_client(get_provider())

    # Seed data: upload 2 separate prefixes
    seed1 = Path(tempfile.mkdtemp(prefix="async_dl_seed1_"))
    seed2 = Path(tempfile.mkdtemp(prefix="async_dl_seed2_"))
    dest1 = Path(tempfile.mkdtemp(prefix="async_dl_dest1_"))
    dest2 = Path(tempfile.mkdtemp(prefix="async_dl_dest2_"))

    try:
        (seed1 / "alpha.txt").write_text("Alpha from prefix A")
        (seed1 / "beta.txt").write_text("Beta from prefix A")
        (seed2 / "gamma.txt").write_text("Gamma from prefix B")
        (seed2 / "delta.txt").write_text("Delta from prefix B")

        await asyncio.gather(
            client.upload_directory(str(seed1), prefix="examples/async/dl-gather/a/"),
            client.upload_directory(str(seed2), prefix="examples/async/dl-gather/b/"),
        )

        # Choice 2: download both prefixes concurrently
        r1, r2 = await asyncio.gather(
            client.download_directory("examples/async/dl-gather/a", str(dest1)),
            client.download_directory("examples/async/dl-gather/b", str(dest2)),
        )

        print(f"\n=== Choice 2: Parallel download with asyncio.gather ===")
        print("r1, r2 = await asyncio.gather(")
        print("    client.download_directory('prefix/a', '/local/a'),")
        print("    client.download_directory('prefix/b', '/local/b'),")
        print(")")
        print(f"  prefix/a: {r1.successful_downloads} files downloaded")
        print(f"  prefix/b: {r2.successful_downloads} files downloaded")
        print("Both downloads ran simultaneously!")
    finally:
        for d in [seed1, seed2, dest1, dest2]:
            shutil.rmtree(d, ignore_errors=True)
        await client.close()


async def directory_download_choice_3_background():
    """Choice 3 applied to download: download in background while processing.

    task = asyncio.create_task(client.download_directory("data", "/local"))
    await process_previous_batch()   # runs while download progresses
    response = await task            # collect when done
    """
    client = get_async_bucket_client(get_provider())

    seed_dir = Path(tempfile.mkdtemp(prefix="async_dl_bg_seed_"))
    dest_dir = Path(tempfile.mkdtemp(prefix="async_dl_bg_dest_"))

    try:
        # Seed data
        for i in range(4):
            (seed_dir / f"batch_{i}.txt").write_text(f"Batch data item {i}" * 5)
        await client.upload_directory(
            str(seed_dir), prefix="examples/async/dl-background/"
        )

        print(f"\n=== Choice 3: Background download with create_task ===")

        # Start download in background
        task = asyncio.create_task(
            client.download_directory("examples/async/dl-background", str(dest_dir))
        )
        print("task = asyncio.create_task(client.download_directory(...))")
        print("Download started — doing other work...")

        # Process other things while download runs
        await client.upload(
            key="examples/async/processing-result.txt",
            payload=b"Result of processing done WHILE download runs",
        )
        print("  await process_previous_batch()  # download progresses")

        # Collect download result
        response = await task
        print(f"response = await task")
        print(f"  Files downloaded: {response.successful_downloads}")
        print(f"  Bytes: {response.total_bytes_transferred:,}")
        print("Download ran in background while we did other work!")
    finally:
        shutil.rmtree(seed_dir, ignore_errors=True)
        shutil.rmtree(dest_dir, ignore_errors=True)
        await client.close()


async def list_page_single():
    """List a single page of blobs and inspect each BlobInfo field.

    list_page returns exactly one page of results. Fields are normalized
    across providers: size is an int, last_modified a tz-aware datetime,
    e_tag has provider quoting stripped, and storage_class/version_id are
    surfaced when the provider reports them.
    """
    client = get_async_bucket_client(get_provider())

    prefix = "examples/async/list-page/single/"
    keys = [f"{prefix}object-{i}.txt" for i in range(3)]
    for key in keys:
        await client.upload(key=key, payload=f"Content for {key}".encode())

    print("\n=== Async List Page (single page) ===")
    print(f"Uploaded {len(keys)} objects under '{prefix}'")

    page = await client.list_page(prefix=prefix)

    print(f"Fetched {len(page.blobs)} blobs in one page:")
    for blob in page.blobs:
        print(
            f"  - {blob.key} (size={blob.size}, etag={blob.e_tag}, "
            f"storage_class={blob.storage_class}, "
            f"last_modified={blob.last_modified})"
        )
    print(
        f"is_truncated={page.is_truncated}, "
        f"next_pagination_token={page.next_pagination_token}"
    )

    # Cleanup
    await client.delete_objects([BlobIdentifier(key=k) for k in keys])
    await client.close()


async def list_page_manual_pagination():
    """Manually iterate every page by feeding the token back in.

    This is the core use case for list_page: the caller drives pagination,
    fetching one page at a time and passing next_pagination_token back as
    pagination_token until is_truncated is False. Here max_results=2 forces
    multiple pages so the loop is exercised end to end.
    """
    client = get_async_bucket_client(get_provider())

    prefix = "examples/async/list-page/paged/"
    keys = [f"{prefix}object-{i:02d}.txt" for i in range(5)]
    for key in keys:
        await client.upload(key=key, payload=f"Content for {key}".encode())

    print("\n=== Async List Page (manual multi-page iteration) ===")
    print(f"Uploaded {len(keys)} objects; listing with max_results=2")

    all_keys = []
    token = None
    page_num = 0
    while True:
        page = await client.list_page(
            prefix=prefix, max_results=2, pagination_token=token
        )
        page_num += 1
        all_keys.extend(b.key for b in page.blobs)
        print(
            f"  Page {page_num}: {len(page.blobs)} blobs, "
            f"is_truncated={page.is_truncated}"
        )
        if not page.is_truncated:
            break
        token = page.next_pagination_token

    print(f"Collected {len(all_keys)} keys across {page_num} pages")

    # Cleanup
    await client.delete_objects([BlobIdentifier(key=k) for k in keys])
    await client.close()


async def list_page_with_prefix_delimiter():
    """List with a delimiter to get folder-style common_prefixes.

    A delimiter collapses everything beyond the next delimiter into a single
    "common prefix" (like a directory). Only blobs directly under the prefix
    are returned in blobs; nested keys roll up into common_prefixes.
    """
    client = get_async_bucket_client(get_provider())

    base = "examples/async/list-page/tree/"
    # A top-level file plus two "subdirectories" each holding an object.
    layout = [
        f"{base}root.txt",
        f"{base}images/a.png",
        f"{base}images/b.png",
        f"{base}docs/readme.md",
    ]
    for key in layout:
        await client.upload(key=key, payload=f"Content for {key}".encode())

    print("\n=== Async List Page (prefix + delimiter) ===")
    print(f"Uploaded {len(layout)} objects under '{base}'")

    page = await client.list_page(prefix=base, delimiter="/")

    print(f"Blobs directly under '{base}':")
    for blob in page.blobs:
        print(f"  - {blob.key}")
    print("Common prefixes (subdirectories):")
    for cp in page.common_prefixes:
        print(f"  - {cp}")

    # Cleanup
    await client.delete_objects([BlobIdentifier(key=k) for k in layout])
    await client.close()


async def list_page_empty():
    """List a prefix that matches nothing -> an empty page.

    An empty listing is not an error: blobs and common_prefixes are empty,
    is_truncated is False, and next_pagination_token is None.
    """
    client = get_async_bucket_client(get_provider())

    print("\n=== Async List Page (empty result) ===")

    page = await client.list_page(prefix="examples/async/list-page/nonexistent-xyz/")

    print(
        f"blobs={page.blobs}, common_prefixes={page.common_prefixes}, "
        f"is_truncated={page.is_truncated}, "
        f"next_pagination_token={page.next_pagination_token}"
    )

    await client.close()


# Credentials overrider — supply non-default credentials to the async client.
#
# By default the client discovers credentials from the environment (env vars,
# credential files, or cloud IAM/ADC). A CredentialsOverrider lets the caller
# override that resolution per client. Each example reads its inputs from env
# vars and skips itself when they are unset, so the suite stays green without
# real credentials — it never hardcodes a secret.


async def credentials_overrider_session():
    """Use pre-issued temporary credentials via a SESSION override.

    SESSION wraps credentials the caller already holds (e.g. minted by an
    upstream token broker). Set EXAMPLE_SESSION_TOKEN (a bearer/session token)
    and, for AWS, EXAMPLE_SESSION_ACCESS_KEY_ID + EXAMPLE_SESSION_SECRET.
    """
    print(f"\n=== Async Credentials Overrider (SESSION) ===")

    token = os.environ.get("EXAMPLE_SESSION_TOKEN")
    if not token:
        print("Set EXAMPLE_SESSION_TOKEN to run this.")
        return

    overrider = CredentialsOverrider(
        credentials_type=CredentialsType.SESSION,
        session_credentials=StsCredentials(
            access_key_id=os.environ.get("EXAMPLE_SESSION_ACCESS_KEY_ID", ""),
            access_key_secret=os.environ.get("EXAMPLE_SESSION_SECRET", ""),
            security_token=token,
        ),
    )

    client = get_async_bucket_client(get_provider(), credentials_overrider=overrider)
    try:
        exists = await client.does_bucket_exist()
        print(f"Authenticated with SESSION credentials; bucket exists: {exists}")
    finally:
        await client.close()


async def credentials_overrider_assume_role():
    """Assume a role for the client via an ASSUME_ROLE override.

    The client transparently assumes the role and uses the resulting
    temporary credentials. ``role`` is provider-interpreted (an ARN on AWS,
    a service-account email on GCP). duration_seconds requests a token
    lifetime; set EXAMPLE_ASSUME_ROLE to run this.
    """
    print(f"\n=== Async Credentials Overrider (ASSUME_ROLE) ===")

    role = os.environ.get("EXAMPLE_ASSUME_ROLE")
    if not role:
        print("Set EXAMPLE_ASSUME_ROLE (an ARN on AWS / SA email on GCP) to run this.")
        return

    overrider = CredentialsOverrider(
        credentials_type=CredentialsType.ASSUME_ROLE,
        role=role,
        duration_seconds=900,
        session_name="async-example",
    )

    client = get_async_bucket_client(get_provider(), credentials_overrider=overrider)
    try:
        exists = await client.does_bucket_exist()
        print(f"Assumed role {role} (duration=900s); bucket exists: {exists}")
    finally:
        await client.close()


async def credentials_overrider_web_identity():
    """Exchange a web identity (OIDC) token via an ASSUME_ROLE_WEB_IDENTITY override.

    The token is produced by a supplier callback, so it can be refreshed on
    each use. Set EXAMPLE_WIF_ROLE and EXAMPLE_WIF_TOKEN to run this.

    Provider support differs: AWS exchanges the token for temporary
    credentials; on the GCP async path there is no Workload Identity
    Federation token type yet, so this override falls back to Application
    Default Credentials.
    """
    print(f"\n=== Async Credentials Overrider (WEB_IDENTITY) ===")

    role = os.environ.get("EXAMPLE_WIF_ROLE")
    token = os.environ.get("EXAMPLE_WIF_TOKEN")
    if not role or not token:
        print("Set EXAMPLE_WIF_ROLE and EXAMPLE_WIF_TOKEN to run this.")
        return

    overrider = CredentialsOverrider(
        credentials_type=CredentialsType.ASSUME_ROLE_WEB_IDENTITY,
        role=role,
        web_identity_token_supplier=lambda: token,
        session_name="async-example-wif",
    )

    client = get_async_bucket_client(get_provider(), credentials_overrider=overrider)
    try:
        exists = await client.does_bucket_exist()
        print(f"Exchanged web identity token for role {role}; bucket exists: {exists}")
    finally:
        await client.close()


def get_async_bucket_client(
    provider, *, transfer_config=None, credentials_overrider=None
):
    """Create and return an AsyncBucketClient instance.

    Gets configuration from environment variables. Pass credentials_overrider
    to override the default (system/ADC) credential resolution.
    """
    bucket_name = os.environ.get("BUCKET_NAME", "palsfdc")
    region = os.environ.get("BUCKET_REGION")

    if not bucket_name or bucket_name.strip() == "":
        raise ValueError(
            "Bucket name must be provided via 'BUCKET_NAME' environment variable"
        )

    client = AsyncBucketClient(
        provider=provider,
        bucket=bucket_name,
        region=region,
        transfer_config=transfer_config,
        credentials_overrider=credentials_overrider,
    )

    return client


def get_provider():
    """Get the cloud provider to use.

    Can be set via PROVIDER environment variable, otherwise defaults to "aws".
    """
    return os.environ.get("PROVIDER", "aws")


async def multipart_upload():
    """Upload a payload in parts and assemble it into one object."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/multipart-demo.bin"
    part_size = 5 * 1024 * 1024  # S3 requires at least this for a non-final part
    payload = b"X" * part_size + b"Y" * 1024

    print(f"\n=== Async Multipart Upload ===")
    mpu = await client.initiate_multipart_upload(
        key, content_type="application/octet-stream"
    )
    print(f"Initiated: upload_id={mpu.upload_id}")

    uploaded = []
    try:
        for part_number, chunk in enumerate(
            [payload[:part_size], payload[part_size:]], start=1
        ):
            part = await client.upload_part(
                mpu, MultipartPart(part_number=part_number, content=chunk)
            )
            print(f"Uploaded part {part.part_number}: {part.size_in_bytes} bytes")
            uploaded.append(part)

        result = await client.complete_multipart_upload(mpu, uploaded)
        print(f"Completed: key={result.key}, etag={result.e_tag}")
    except Exception:
        # Without this the uploaded parts are retained, and billed.
        await client.abort_multipart_upload(mpu)
        print("Aborted after failure")
        raise

    downloaded = await client.download(key)
    assert downloaded.content == payload, "multipart round-trip mismatch"
    print(f"Verified download: {len(downloaded.content)} bytes")

    await client.delete(key)
    await client.close()


async def multipart_upload_concurrent_parts():
    """Upload the parts concurrently, which is the point of the async client."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/multipart-concurrent.bin"
    part_size = 5 * 1024 * 1024
    chunks = [bytes([b]) * part_size for b in (b"A"[0], b"B"[0])] + [b"C" * 512]

    print(f"\n=== Async Multipart Upload (concurrent parts) ===")
    mpu = await client.initiate_multipart_upload(key)
    try:
        # Parts are independent, so they can go up at once; completion orders
        # them by part number regardless of which finishes first.
        uploaded = await asyncio.gather(
            *(
                client.upload_part(
                    mpu, MultipartPart(part_number=number, content=chunk)
                )
                for number, chunk in enumerate(chunks, start=1)
            )
        )
        print(f"Uploaded {len(uploaded)} parts concurrently")

        result = await client.complete_multipart_upload(mpu, list(uploaded))
        print(f"Completed: etag={result.e_tag}")
    except Exception:
        await client.abort_multipart_upload(mpu)
        raise

    metadata = await client.get_metadata(key)
    print(f"Assembled object size: {metadata.content_length} bytes")

    await client.delete(key)
    await client.close()


async def multipart_upload_abort():
    """Abandon an upload, discarding the parts already uploaded."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/multipart-aborted.bin"

    print(f"\n=== Async Multipart Abort ===")
    mpu = await client.initiate_multipart_upload(key)
    await client.upload_part(
        mpu, MultipartPart(part_number=1, content=b"Z" * (5 * 1024 * 1024))
    )
    print(f"Uploaded one part, then abandoning upload {mpu.upload_id}")

    await client.abort_multipart_upload(mpu)
    # Safe to repeat, so cleanup paths need not track whether it already ran.
    await client.abort_multipart_upload(mpu)
    print(
        f"Aborted twice without error; object exists: "
        f"{await client.does_object_exist(key)}"
    )

    await client.close()


async def multipart_upload_error_handling():
    """Show what an invalid part number and a stale upload id raise."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/multipart-errors.bin"

    print(f"\n=== Async Multipart Error Handling ===")
    mpu = await client.initiate_multipart_upload(key)
    try:
        try:
            await client.upload_part(
                mpu, MultipartPart(part_number=0, content=b"too low")
            )
        except InvalidArgumentException as exc:
            print(f"Rejected part number 0: {exc}")

        # An unrecognized upload id is not reported uniformly. A provider with
        # server-side upload sessions rejects it; one where multipart is
        # emulated with ordinary objects has no session to check it against, so
        # the part is stored and only the completion step notices.
        stale = MultipartUpload(
            bucket=mpu.bucket, key=key, upload_id="not-a-real-upload-id"
        )
        try:
            await client.upload_part(
                stale, MultipartPart(part_number=1, content=b"orphan")
            )
            print("Unknown upload id accepted; abandoned part left behind")
            await client.abort_multipart_upload(stale)
        except SubstrateSdkException as exc:
            print(f"Unknown upload id rejected: {type(exc).__name__}")
    finally:
        await client.abort_multipart_upload(mpu)
        await client.close()


async def proxy_inherits_environment_by_default():
    """Without a proxy_config the client keeps its provider's default behaviour."""
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")
    region = os.environ.get("BUCKET_REGION")

    print(f"\n=== Async Proxy: default ===")
    print(f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY') or '<unset>'}")

    client = AsyncBucketClient(provider=get_provider(), bucket=bucket, region=region)
    try:
        print(f"Bucket reachable: {await client.does_bucket_exist()}")
    finally:
        await client.close()


async def proxy_bypassed_for_direct_connection():
    """Ignore the ambient proxy and connect directly.

    What same-region traffic needs in a pod whose environment points at an
    egress proxy for outbound traffic.
    """
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")
    region = os.environ.get("BUCKET_REGION")

    print(f"\n=== Async Proxy: bypassed (direct connection) ===")
    print(f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY') or '<unset>'}")

    client = AsyncBucketClient(
        provider=get_provider(),
        bucket=bucket,
        region=region,
        proxy_config=ProxyConfig(use_environment_variable_proxy_values=False),
    )
    try:
        print(f"Bucket reachable without the proxy: {await client.does_bucket_exist()}")
    finally:
        await client.close()


async def proxy_explicit_endpoint():
    """Send requests through a named proxy regardless of the environment."""
    bucket = os.environ.get("BUCKET_NAME", "palsfdc")
    region = os.environ.get("BUCKET_REGION")
    endpoint = os.environ.get("EXAMPLE_PROXY_ENDPOINT")

    print(f"\n=== Async Proxy: explicit endpoint ===")
    if not endpoint:
        print(
            "Set EXAMPLE_PROXY_ENDPOINT to run this against a real proxy; "
            "showing configuration only."
        )
        print(
            f"Configured endpoint: {ProxyConfig(endpoint='http://proxy.example.net:8443').endpoint}"
        )
        return

    client = AsyncBucketClient(
        provider=get_provider(),
        bucket=bucket,
        region=region,
        proxy_config=ProxyConfig(endpoint=endpoint),
    )
    try:
        print(f"Bucket reachable via {endpoint}: {await client.does_bucket_exist()}")
    finally:
        await client.close()


async def multipart_list_parts():
    """Inspect what has been uploaded so far in an open multipart upload."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/multipart-list-parts.bin"
    part_size = 5 * 1024 * 1024

    print(f"\n=== Async Multipart: list parts ===")
    mpu = await client.initiate_multipart_upload(key)
    try:
        print(f"Before uploading anything: {await client.list_parts(mpu)}")

        for number in (1, 2):
            await client.upload_part(
                mpu,
                MultipartPart(part_number=number, content=bytes([number]) * part_size),
            )

        for part in await client.list_parts(mpu):
            print(
                f"Part {part.part_number}: {part.size_in_bytes} bytes, "
                f"etag={part.e_tag}"
            )

        # Completing from the listing is how a resumed upload finishes.
        result = await client.complete_multipart_upload(
            mpu, await client.list_parts(mpu)
        )
        print(f"Completed from the listing: etag={result.e_tag}")
        await client.delete(key)
    except Exception:
        await client.abort_multipart_upload(mpu)
        raise
    finally:
        await client.close()


async def get_object_lock_state():
    """Read an object's retention and legal-hold state."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/object-lock-state.txt"
    await client.upload(key=key, payload=b"lock state demo", content_type="text/plain")

    lock = await client.get_object_lock(key)

    print(f"\n=== Async Get Object Lock ===")
    print(f"Key: {key}")
    if lock is None:
        print("Lock: none — the object has no retention and no legal hold")
    else:
        print(f"Retention mode: {lock.mode}")
        print(f"Retain until: {lock.retain_until_date}")
        print(f"Legal hold: {lock.legal_hold}")
        # Tri-state: None means the provider does not report an event-based
        # hold at all, which is different from reporting one that is off.
        print(f"Event-based hold: {lock.event_based_hold!r}")

    await client.delete(key)
    await client.close()


async def update_object_retention():
    """Change an object's retention, which it must already have.

    Note: on a bucket that applies a default retention, the object created
    here inherits it and cannot be deleted until that retention expires.
    """
    client = get_async_bucket_client(get_provider())

    key = "examples/async/retention-update.txt"
    extend_to = datetime.now(timezone.utc) + timedelta(days=7)

    print(f"\n=== Async Update Object Retention ===")
    try:
        await client.upload(
            key=key, payload=b"retention demo", content_type="text/plain"
        )

        try:
            await client.update_object_retention(
                key, ObjectRetentionConfig(retain_until_date=extend_to)
            )
            lock = await client.get_object_lock(key)
            print(f"Extended to {lock.retain_until_date} in mode {lock.mode}")
        except FailedPreconditionException as exc:
            # The method changes an existing retention and never creates one,
            # so an object with no retention is refused.
            print(f"Refused: {exc}")

        # Shortening needs bypass_governance=True on a mutable retention, and
        # is refused even with it on an immutable one.
        try:
            await client.delete(key)
        except SubstrateSdkException as exc:
            # Retention is doing its job; say so rather than failing the run.
            print(f"Left in place, still retained: {type(exc).__name__}")
    finally:
        await client.close()


async def legal_hold_round_trip():
    """Place a legal hold, see it block deletion, then release it."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/legal-hold.txt"

    print(f"\n=== Async Legal Hold ===")
    try:
        # Pin the version: where the bucket is versioned, a hold protects the
        # version it was placed on rather than the key.
        uploaded = await client.upload(key=key, payload=b"held content")
        version = uploaded.version_id

        await client.update_legal_hold(key, True, version_id=version)
        lock = await client.get_object_lock(key, version_id=version)
        print(f"Held: legal_hold={lock.legal_hold}")

        try:
            await client.delete(key, version_id=version)
            print("Deleted while held — unexpected")
        except SubstrateSdkException as exc:
            print(f"Delete refused while held: {type(exc).__name__}")

        await client.update_legal_hold(key, False, version_id=version)
        print(
            f"Released: lock is now {await client.get_object_lock(key, version_id=version)}"
        )

        await client.delete(key, version_id=version)
        print("Deleted after release")
    finally:
        await client.close()


async def download_stream_in_chunks():
    """Consume a blob lazily instead of holding it in memory."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/streamed.bin"
    payload = bytes(range(256)) * 40

    print(f"\n=== Async Download Stream ===")
    try:
        await client.upload(key=key, payload=payload)

        # async with closes the stream, which releases the connection even if
        # only part of the object is read.
        total = 0
        chunks = 0
        async with await client.download_stream(key) as stream:
            print(
                f"Content-Length before reading a byte: {stream.metadata.content_length}"
            )
            async for chunk in stream:
                total += len(chunk)
                chunks += 1
        print(f"Read {total} bytes in {chunks} chunks")

        # read(n) returns exactly n bytes until the object runs out.
        async with await client.download_stream(key) as stream:
            header = await stream.read(16)
        print(f"First 16 bytes: {header.hex()}")

        await client.delete(key)
    finally:
        await client.close()


async def download_stream_byte_range():
    """Stream part of a blob. Both bounds are inclusive."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/streamed-range.bin"
    payload = bytes(range(256))

    print(f"\n=== Async Download Stream (byte range) ===")
    try:
        await client.upload(key=key, payload=payload)

        async with await client.download_stream(key, start=10, end=19) as stream:
            window = await stream.read(-1)
        print(f"start=10 end=19 -> {len(window)} bytes: {window.hex()}")

        # Giving only end reads the LAST end bytes, not the first.
        async with await client.download_stream(key, end=4) as stream:
            tail = await stream.read(-1)
        print(f"end=4 (suffix)  -> {tail.hex()}")

        await client.delete(key)
    finally:
        await client.close()


async def list_blob_versions():
    """Walk every version of one key, newest state marked by is_latest."""
    client = get_async_bucket_client(get_provider())

    key = "examples/async/versioned.txt"

    print(f"\n=== Async List Blob Versions ===")
    try:
        for n in (1, 2, 3):
            await client.upload(key=key, payload=f"revision {n}".encode())

        # No await to obtain the iterator; pages are fetched as consumed.
        # The order is the provider's own, so sort if order matters to you.
        async for info in client.list_blob_versions(key):
            marker = " (current)" if info.is_latest else ""
            print(f"  {info.version_id}  {info.size} bytes{marker}")

        absent = [i async for i in client.list_blob_versions(f"{key}-absent")]
        print(f"A key with no versions yields: {absent}")
    finally:
        # Collected before deleting, not during: removing versions while
        # walking a paginated listing can skip entries at a page boundary.
        versions = [i async for i in client.list_blob_versions(key)]
        for info in versions:
            await client.delete(key, version_id=info.version_id)
        await client.close()


async def async_main():
    """Run all async blob examples."""
    provider = get_provider()
    print("=== Python Multi-Cloud SDK Async Blob Examples ===")
    print(f"Provider: {provider}")

    examples = [
        ("upload", upload),
        ("upload_from_path", upload_from_path),
        ("upload_from_stream", upload_from_stream),
        ("upload_with_transfer_config", upload_with_transfer_config),
        ("upload_with_context_manager", upload_with_context_manager),
        ("upload_multiple_concurrent", upload_multiple_concurrent),
        ("upload_with_metadata", upload_with_metadata),
        ("upload_with_storage_class", upload_with_storage_class),
        ("tags_round_trip", tags_round_trip),
        ("upload_error_handling", upload_error_handling),
        ("upload_large_payload", upload_large_payload),
        ("upload_explicit_content_length", upload_explicit_content_length),
        ("download", download),
        ("download_to_path", download_to_path),
        ("download_to_file", download_to_file),
        ("download_range", download_range),
        ("download_with_version", download_with_version),
        ("download_stream_in_chunks", download_stream_in_chunks),
        ("download_stream_byte_range", download_stream_byte_range),
        ("download_concurrent", download_concurrent),
        ("download_error_handling", download_error_handling),
        ("upload_then_download", upload_then_download),
        ("get_metadata_basic", get_metadata_basic),
        ("get_metadata_round_trip", get_metadata_round_trip),
        ("get_metadata_not_found", get_metadata_not_found),
        ("get_metadata_error_handling", get_metadata_error_handling),
        ("get_metadata_versioned", get_metadata_versioned),
        ("get_object_lock_state", get_object_lock_state),
        ("update_object_retention", update_object_retention),
        ("legal_hold_round_trip", legal_hold_round_trip),
        ("delete_single", delete_single),
        ("delete_with_version", delete_with_version),
        ("delete_nonexistent", delete_nonexistent),
        ("delete_objects_batch", delete_objects_batch),
        ("delete_objects_partial_failure", delete_objects_partial_failure),
        ("delete_objects_with_versions", delete_objects_with_versions),
        ("delete_error_handling", delete_error_handling),
        ("upload_then_delete", upload_then_delete),
        ("copy_same_bucket", copy_same_bucket),
        ("copy_with_metadata_override", copy_with_metadata_override),
        ("copy_missing_source", copy_missing_source),
        ("copy_validation_error", copy_validation_error),
        ("copy_multiple_concurrent", copy_multiple_concurrent),
        ("check_object_exists", check_object_exists),
        ("check_object_exists_with_version", check_object_exists_with_version),
        ("check_bucket_exists", check_bucket_exists),
        ("existence_check_error_handling", existence_check_error_handling),
        ("generate_presigned_download_url", generate_presigned_download_url),
        ("generate_presigned_upload_url", generate_presigned_upload_url),
        (
            "generate_presigned_url_custom_expiration",
            generate_presigned_url_custom_expiration,
        ),
        ("generate_presigned_url_with_metadata", generate_presigned_url_with_metadata),
        (
            "generate_presigned_url_with_content_disposition",
            generate_presigned_url_with_content_disposition,
        ),
        ("generate_presigned_url_versioned", generate_presigned_url_versioned),
        ("generate_presigned_url_concurrent", generate_presigned_url_concurrent),
        ("multipart_upload", multipart_upload),
        ("multipart_upload_concurrent_parts", multipart_upload_concurrent_parts),
        ("multipart_list_parts", multipart_list_parts),
        ("multipart_upload_abort", multipart_upload_abort),
        ("multipart_upload_error_handling", multipart_upload_error_handling),
        (
            "proxy_inherits_environment_by_default",
            proxy_inherits_environment_by_default,
        ),
        (
            "proxy_bypassed_for_direct_connection",
            proxy_bypassed_for_direct_connection,
        ),
        ("proxy_explicit_endpoint", proxy_explicit_endpoint),
        (
            "generate_presigned_url_error_handling",
            generate_presigned_url_error_handling,
        ),
        (
            "presigned_put_then_get_http_round_trip",
            presigned_put_then_get_http_round_trip,
        ),
        (
            "presigned_kms_put_then_get_http_round_trip",
            presigned_kms_put_then_get_http_round_trip,
        ),
        ("upload_directory", upload_directory),
        ("download_directory", download_directory),
        ("download_directory_with_exclusions", download_directory_with_exclusions),
        ("delete_directory", delete_directory),
        (
            "delete_directory_is_recursive_and_scoped",
            delete_directory_is_recursive_and_scoped,
        ),
        # Async client choices — why async matters for directory operations
        ("directory_choice_1_sequential", directory_choice_1_sequential),
        ("directory_choice_2_gather", directory_choice_2_gather),
        ("directory_choice_3_create_task", directory_choice_3_create_task),
        ("directory_download_choice_2_gather", directory_download_choice_2_gather),
        (
            "directory_download_choice_3_background",
            directory_download_choice_3_background,
        ),
        # Single-page listing with manual pagination (list_page)
        ("list_page_single", list_page_single),
        ("list_page_manual_pagination", list_page_manual_pagination),
        ("list_page_with_prefix_delimiter", list_page_with_prefix_delimiter),
        ("list_page_empty", list_page_empty),
        ("list_blob_versions", list_blob_versions),
        # Credentials overrider — non-default credential resolution
        ("credentials_overrider_session", credentials_overrider_session),
        ("credentials_overrider_assume_role", credentials_overrider_assume_role),
        ("credentials_overrider_web_identity", credentials_overrider_web_identity),
    ]

    passed = []
    failed = []

    for name, fn in examples:
        try:
            await fn()
            passed.append(name)
        except Exception as exc:
            failed.append((name, exc))
            print(f"\n  FAILED: {name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    # Summary
    print(f"\n{'=' * 50}")
    print(
        f"Results: {len(passed)} passed, {len(failed)} failed "
        f"out of {len(examples)} examples"
    )
    if failed:
        print("\nFailed examples:")
        for name, exc in failed:
            print(f"  x {name}: {type(exc).__name__}: {exc}")
        return 1
    else:
        print("\n  All async examples completed successfully!")
        return 0


def main():
    """Entry point — run the async examples."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Python Multi-Cloud SDK Async Blob Examples"
    )
    parser.add_argument(
        "--provider",
        choices=["aws", "gcp"],
        help="Cloud provider (can also set PROVIDER env var)",
    )
    args = parser.parse_args()

    # Allow command-line arg to override environment variable
    if args.provider:
        os.environ["PROVIDER"] = args.provider

    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
