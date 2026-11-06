# Blob Storage Guide

The Blob Storage service provides cloud-agnostic object storage operations — upload, download, and delete objects across AWS S3, Google Cloud Storage, and Ali OSS.

## Creating a Client

`BucketClient` is scoped to a single bucket. All operations target that bucket.

```python
from multicloud.blob import BucketClient

# AWS S3
with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    ...

# Google Cloud Storage
with BucketClient(provider="gcp", bucket="my-bucket", region="us-central1") as client:
    ...
```

### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `provider` | `str` | ✅ | Provider id: `"aws"`, `"gcp"`, or `"ali"` |
| `bucket` | `str` | ✅ | Bucket name |
| `region` | `str` | | Region (AWS region / GCS location) |
| `endpoint` | `str` | | Endpoint URL override |

All parameters are **keyword-only** (enforced by `*` separator).

!!! note
    Credentials are discovered automatically from environment variables, config files, or IAM/ADC. The SDK never accepts explicit credential parameters.

## Uploading Objects

```python
from multicloud.blob import BucketClient

with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    # Upload raw bytes
    resp = client.upload("hello.txt", b"Hello, World!")
    print(f"Uploaded: {resp.key}, ETag: {resp.e_tag}")

    # Upload with metadata and content type
    resp = client.upload(
        "data.json",
        b'{"key": "value"}',
        content_type="application/json",
        metadata={"author": "sdk-user", "version": "1"},
    )
```

### Upload from a File

```python
with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    with open("large-file.bin", "rb") as f:
        resp = client.upload("backups/large-file.bin", f)
```

### Upload Parameters

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `key` | `str` | ✅ | Object key (path) in the bucket |
| `payload` | `bytes \| BinaryIO` | ✅ | Content to upload |
| `content_type` | `str` | | MIME type (e.g., `"text/plain"`) |
| `content_length` | `int` | | Size in bytes (auto-detected if omitted) |
| `metadata` | `Mapping[str, str]` | | User-defined metadata |
| `storage_class` | `str` | | Provider storage class for the new object; omit to use the bucket default |

### Storage Class

Pass `storage_class` to `upload` or `upload_from_path` to choose the object's
storage tier. The value is the provider's native storage class name and is sent
unchanged; a blank string raises `InvalidArgumentException`, and a class the
provider does not recognize raises `InvalidArgumentException` from the
provider call.

| Provider | Example values |
|----------|----------------|
| AWS S3 | `STANDARD`, `STANDARD_IA`, `ONEZONE_IA`, `INTELLIGENT_TIERING`, `GLACIER_IR`, `GLACIER`, `DEEP_ARCHIVE` |
| Google Cloud Storage | `STANDARD`, `NEARLINE`, `COLDLINE`, `ARCHIVE` |

```python
with BucketClient(provider="gcp", bucket="my-bucket") as client:
    client.upload("archive/2025.tar", payload, storage_class="NEARLINE")

    page = client.list_page(prefix="archive/")
    print(page.blobs[0].storage_class)  # "NEARLINE"
```

The async `AsyncBucketClient.upload` and `upload_from_path` accept the same
keyword. Multipart uploads and presigned upload URLs do not take a storage
class.

### UploadResponse

`UploadResponse` is a frozen dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Object key |
| `e_tag` | `str \| None` | Entity tag (content hash) |
| `version_id` | `str \| None` | Version ID (if versioning enabled) |

## Downloading Objects

```python
with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    resp = client.download("hello.txt")
    print(f"Content: {resp.content.decode()}")
    print(f"Content-Type: {resp.metadata.content_type}")
    print(f"Size: {resp.metadata.content_length} bytes")
```

### Range Downloads

Download a specific byte range:

```python
with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    # Download bytes 0–99 (first 100 bytes)
    resp = client.download("large-file.bin", start=0, end=99)
```

### Download with Version

```python
with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    resp = client.download("hello.txt", version_id="abc123")
```

### Download Parameters

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `key` | `str` | ✅ | Object key |
| `version_id` | `str` | | Specific version to download |
| `start` | `int` | | Start byte for range download (inclusive) |
| `end` | `int` | | End byte for range download (inclusive) |

### DownloadResponse

`DownloadResponse` is a frozen dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Object key |
| `content` | `bytes` | Downloaded content |
| `metadata` | `BlobMetadata` | Object metadata (see below) |

### BlobMetadata

Nested within `DownloadResponse.metadata`:

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Object key |
| `content_length` | `int \| None` | Size in bytes |
| `content_type` | `str \| None` | MIME type |
| `e_tag` | `str \| None` | Entity tag |
| `version_id` | `str \| None` | Version ID |
| `metadata` | `Mapping[str, str]` | User-defined metadata |

## Deleting Objects

```python
with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    client.delete("hello.txt")
```

Delete is **idempotent** — deleting a non-existent key does not raise an error.

## Multipart Uploads

Large objects can be uploaded in parts. Initiate a session, upload each part, then complete to assemble the final object:

```python
from multicloud.blob import BucketClient, MultipartPart

with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    mpu = client.initiate_multipart_upload("big-file.bin", content_type="application/octet-stream")

    parts = []
    parts.append(client.upload_part(mpu, MultipartPart(part_number=1, content=chunk_1)))
    parts.append(client.upload_part(mpu, MultipartPart(part_number=2, content=chunk_2)))

    client.complete_multipart_upload(mpu, parts)
```

### Object Lock at Initiate

Pass `object_lock` to `initiate_multipart_upload` to apply retention and/or a legal hold to the final assembled object. The lock is set on the object produced when the upload completes:

```python
from datetime import datetime, timedelta, timezone

from multicloud.blob import (
    BucketClient,
    MultipartPart,
    ObjectLockConfiguration,
    RetentionMode,
)

with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    lock = ObjectLockConfiguration(
        mode=RetentionMode.GOVERNANCE,
        retain_until_date=datetime.now(timezone.utc) + timedelta(days=7),
        legal_hold=True,
    )
    mpu = client.initiate_multipart_upload("locked-file.bin", object_lock=lock)

    part = client.upload_part(mpu, MultipartPart(part_number=1, content=payload))
    client.complete_multipart_upload(mpu, [part])

    # The lock is present on the completed object:
    info = client.get_object_lock("locked-file.bin")
    print(info.mode, info.retain_until_date, info.legal_hold)
```

`ObjectLockConfiguration` fields:

| Field | Type | Description |
|-------|------|-------------|
| `mode` | `RetentionMode` | `GOVERNANCE` or `COMPLIANCE`; requires `retain_until_date` |
| `retain_until_date` | `datetime` | Timezone-aware instant until which the object is retained; requires `mode` |
| `legal_hold` | `bool` | Place a legal hold on the object |
| `use_event_based_hold` | `bool` | Apply a requested legal hold as an event-based hold where the provider distinguishes the two |

!!! note
    `GOVERNANCE` retention can be shortened or removed by sufficiently privileged callers; `COMPLIANCE` retention cannot be shortened or removed by anyone until it expires. A retained object cannot be deleted until its retention expires (or is shortened, for `GOVERNANCE`), and an object under legal hold cannot be deleted until the hold is released.

## Complete Example

A full round-trip: upload, download, verify, delete.

```python
from multicloud.blob import BucketClient

with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    # Upload
    upload_resp = client.upload(
        "test/round-trip.txt",
        b"round-trip content",
        content_type="text/plain",
        metadata={"purpose": "testing"},
    )
    print(f"Uploaded: {upload_resp.key}")

    # Download and verify
    download_resp = client.download("test/round-trip.txt")
    assert download_resp.content == b"round-trip content"
    assert download_resp.metadata.metadata["purpose"] == "testing"

    # Cleanup
    client.delete("test/round-trip.txt")
    print("Deleted successfully")
```

## Switching Providers

The same code works across clouds — change only the `provider` and `region`:

```python
from multicloud.blob import BucketClient

providers = [
    {"provider": "aws", "bucket": "my-aws-bucket", "region": "us-west-2"},
    {"provider": "gcp", "bucket": "my-gcp-bucket", "region": "us-central1"},
]

for config in providers:
    with BucketClient(**config) as client:
        client.upload("hello.txt", b"same code, different cloud")
        resp = client.download("hello.txt")
        print(f"[{config['provider']}] Content: {resp.content.decode()}")
        client.delete("hello.txt")
```

## Error Handling

All blob operations raise exceptions from the common `multicloud.common.exceptions` hierarchy:

```python
from multicloud.blob import BucketClient
from multicloud.common.exceptions import (
    ResourceNotFoundException,
    InvalidArgumentException,
    UnAuthorizedException,
    SubstrateSdkException,
)

with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    try:
        resp = client.download("does-not-exist.txt")
    except ResourceNotFoundException as e:
        print(f"Object not found: {e}")
    except UnAuthorizedException as e:
        print(f"Access denied: {e}")
    except SubstrateSdkException as e:
        print(f"SDK error: {e}")
```

| Exception | When Raised |
|-----------|-------------|
| `ResourceNotFoundException` | Object or bucket does not exist |
| `InvalidArgumentException` | Empty key, invalid content_type, bad range |
| `UnAuthorizedException` | Credentials missing, expired, or insufficient |
| `ResourceExistsException` | Duplicate resource on create |
| `UnknownException` | Unmapped provider error |

## Client Lifecycle

`BucketClient` supports the context manager protocol:

```python
# Preferred — automatic cleanup
with BucketClient(provider="aws", bucket="b", region="us-west-2") as client:
    client.upload("key", b"data")

# Manual cleanup
client = BucketClient(provider="aws", bucket="b", region="us-west-2")
try:
    client.upload("key", b"data")
finally:
    client.close()
```
