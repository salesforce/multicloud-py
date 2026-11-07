"""Cloud-agnostic blob client package.

Public entry point — :class:`BucketClient`::

    from multicloud.blob.blob_client import BucketClient

    with BucketClient(
        provider="aws", bucket="my-bucket", region="us-west-2"
    ) as client:
        client.upload("hello.txt", b"hi")
        resp = client.download("hello.txt")
        client.delete("hello.txt")

Switch clouds by changing the ``provider`` kwarg only::

    BucketClient(provider="gcp", bucket="my-bucket", region="us-central1")

Provider classes (``AwsBlobStore``, ``GcpBlobStore``, ...) live in
sibling distributions (``multicloud-blob-aws``, ``-gcp``, ``-ali``)
and are discovered at runtime via setuptools entry points. They are
hidden behind the registry; users never reference them directly.
"""

from multicloud.blob.blob_client.client import BucketClient
from multicloud.blob.blob_client.driver import (
    BLOB_REGISTRY,
    BaseBlobStore,
    BlobIdentifier,
    BlobInfo,
    BlobMetadata,
    BlobStoreValidator,
    ChecksumAlgorithm,
    CopyResponse,
    DeleteError,
    DeleteObjectsResponse,
    DownloadResponse,
    ListBlobsPageResponse,
    MultipartPart,
    MultipartUpload,
    MultipartUploadResponse,
    ObjectLockConfiguration,
    ObjectLockInfo,
    ObjectRetentionConfig,
    PresignedOperation,
    PresignedUrlResponse,
    ReplicationStatus,
    RetentionMode,
    StreamingDownloadResponse,
    UploadPartResponse,
    UploadResponse,
)

__all__ = [
    "BucketClient",
    "BlobIdentifier",
    "CopyResponse",
    "DeleteError",
    "DeleteObjectsResponse",
    "UploadResponse",
    "ChecksumAlgorithm",
    "DownloadResponse",
    "StreamingDownloadResponse",
    "BlobMetadata",
    "PresignedOperation",
    "PresignedUrlResponse",
    "BlobInfo",
    "ListBlobsPageResponse",
    # Multipart upload types
    "MultipartPart",
    "MultipartUpload",
    "MultipartUploadResponse",
    "UploadPartResponse",
    # Object lock / retention types
    "ObjectLockConfiguration",
    "ObjectLockInfo",
    "ObjectRetentionConfig",
    "RetentionMode",
    # Replication
    "ReplicationStatus",
]

from multicloud.blob.blob_client.async_blob import AsyncBucketClient
from multicloud.blob.blob_client.async_blob.async_driver import (
    ASYNC_BLOB_REGISTRY,
    AsyncDirectoryDownloadResponse,
    AsyncDirectoryUploadResponse,
    AsyncDownloadRequest,
    AsyncDownloadResponse,
    AsyncTransferConfig,
    AsyncUploadResponse,
    BaseAsyncBlobStore,
    BlobAsyncError,
    BlobIdentifier,
    BlobNotFoundError,
    BulkDeleteResponse,
)
from multicloud.blob.blob_client.async_blob.async_driver import (
    CopyRequest as AsyncCopyRequest,
)
from multicloud.blob.blob_client.async_blob.async_driver import (
    CopyResponse as AsyncCopyResponse,
)
from multicloud.blob.blob_client.async_blob.async_driver import (
    DeletedBlob,
    DeleteError,
    FailedAsyncBlobDownload,
    FailedAsyncBlobUpload,
    PresignedUrlResult,
)

# Re-export CopyRequest from async module (named to avoid conflict with sync CopyResponse)
CopyRequest = AsyncCopyRequest

__all__ += [
    "AsyncBucketClient",
    "AsyncCopyRequest",
    "AsyncCopyResponse",
    "AsyncDirectoryDownloadResponse",
    "AsyncDirectoryUploadResponse",
    "AsyncUploadResponse",
    "AsyncDownloadRequest",
    "AsyncDownloadResponse",
    "AsyncTransferConfig",
    "BlobAsyncError",
    "BlobIdentifier",
    "BlobNotFoundError",
    "BulkDeleteResponse",
    "CopyRequest",
    "AsyncCopyResponse",
    "DeletedBlob",
    "DeleteError",
    "FailedAsyncBlobDownload",
    "FailedAsyncBlobUpload",
    "PresignedUrlResult",
]
