"""AWS S3 blob conformance tests with VCR.py record/replay.

Extends :class:`BaseBlobStoreConformance`, supplying an AWS-specific
``Harness`` that wires up the :class:`BucketClient`.

Run modes::

    # Record mode (requires AWS credentials)
    pytest tests/blob/conformance/test_aws_blob_conformance.py \\
        --record-mode=rewrite -v

    # Replay mode (no credentials needed, uses cassettes)
    pytest tests/blob/conformance/test_aws_blob_conformance.py \\
        --record-mode=none -v
"""

from __future__ import annotations

import os
from typing import Optional

import pytest
from multicloud.blob.blob_client import BucketClient
from multicloud.common.credentials import (
    CredentialsOverrider,
    CredentialsType,
    StsCredentials,
)

from tests.blob.conformance.base_blob_store_conformance import (
    BaseBlobStoreConformance,
    Harness,
)

AWS_BUCKET = os.environ.get("CONFORMANCE_AWS_BUCKET", "example-test-bucket-barry")
AWS_REGION = os.environ.get("CONFORMANCE_AWS_REGION", "us-west-2")
# Deliberately not env-overridable, like AWS_ABSENT_BUCKET below. The name
# appears in the request URI, which the cassette matcher compares, so
# overriding it would send a request no committed cassette can answer. Unlike
# AWS_ABSENT_BUCKET this bucket is only ever addressed, never listed among the
# recorder's real buckets, so a single fixed name serves every account. If the
# name ever does get taken, changing it here and re-recording is the fix.
AWS_NONEXISTENT_BUCKET = "multicloud-conformance-nonexistent-bucket-aws-9f8e7d"
AWS_VERSIONED_BUCKET = os.environ.get(
    "CONFORMANCE_AWS_VERSIONED_BUCKET", "chameleon-python-sdk-versioned"
)
AWS_OBJECT_LOCK_BUCKET = os.environ.get(
    "CONFORMANCE_AWS_OBJECT_LOCK_BUCKET", "chameleon-test-bucket-with-object-lock"
)
# Deliberately not env-overridable, unlike the constants above. Those name
# buckets that must exist in whichever account is recording, so each recorder
# needs to point them somewhere. This one names a bucket that must exist in no
# account, so there is nothing to redirect -- and the name appears in the request
# URI, which the cassette matcher compares, so overriding it would send a request
# no committed cassette can answer. If the name ever does get taken, changing it
# here and re-recording is the fix.
AWS_ABSENT_BUCKET = "multicloudj-conformance-absent-bucket-do-not-create"


class AwsHarness(Harness):
    """AWS S3 test harness."""

    def __init__(self) -> None:
        self._client = BucketClient(
            provider="aws", bucket=AWS_BUCKET, region=AWS_REGION
        )
        self._versioned_client: Optional[BucketClient] = BucketClient(
            provider="aws", bucket=AWS_VERSIONED_BUCKET, region=AWS_REGION
        )
        self._object_lock_client: Optional[BucketClient] = BucketClient(
            provider="aws", bucket=AWS_OBJECT_LOCK_BUCKET, region=AWS_REGION
        )

    def create_client(self) -> BucketClient:
        return self._client

    def create_client_for_absent_bucket(self) -> BucketClient:
        return BucketClient(provider="aws", bucket=AWS_ABSENT_BUCKET, region=AWS_REGION)

    def create_versioned_client(self) -> Optional[BucketClient]:
        return self._versioned_client

    def create_object_lock_client(self) -> Optional[BucketClient]:
        return self._object_lock_client

    def create_nonexistent_bucket_client(self) -> BucketClient:
        # A syntactically valid but non-existent bucket in the real region.
        # boto3 constructs the client offline; the missing bucket is only
        # discovered when an operation actually calls S3.
        return BucketClient(
            provider="aws",
            bucket=AWS_NONEXISTENT_BUCKET,
            region=AWS_REGION,
        )

    def create_invalid_credentials_client(self) -> BucketClient:
        # Point at the real bucket but hand boto3 deliberately bogus session
        # credentials. Building a session from static keys is offline, so the
        # 403 only surfaces when an operation signs and sends a request.
        overrider = CredentialsOverrider(
            credentials_type=CredentialsType.SESSION,
            session_credentials=StsCredentials(
                access_key_id="INVALIDACCESSKEY",
                access_key_secret="invalid-secret-key-invalid-secret-key",
                security_token="invalid-session-token",
            ),
        )
        return BucketClient(
            provider="aws",
            bucket=AWS_BUCKET,
            region=AWS_REGION,
            credentials_overrider=overrider,
        )

    def get_provider_id(self) -> str:
        return "aws"

    def get_bucket(self) -> str:
        return AWS_BUCKET

    def get_versioned_bucket(self) -> Optional[str]:
        return AWS_VERSIONED_BUCKET if self._versioned_client is not None else None

    def supports_write_if_match(self) -> bool:
        # S3 compare-and-swap is ETag-based (IfMatch).
        return True

    def supports_write_if_version_match(self) -> bool:
        # S3 has no version-id precondition on writes; if_match is the CAS token.
        return False

    def get_kms_key_id(self) -> str:
        # Fixed customer-managed key in the recording account (us-west-2). The
        # id only has to be live at record time; on replay it is matched from
        # the cassette. So the encryption tests always run rather than skip.
        return "arn:aws:kms:us-west-2:123456789012:key/00000000-0000-4000-8000-000000000000"

    def delete_retained_object(self, client: BucketClient, key: str) -> None:
        # Object-lock buckets are versioning-enabled, so a keyed delete only
        # inserts a delete marker — the locked version survives (masked) until
        # its retain-until elapses, and BypassGovernanceRetention is a no-op
        # unless a specific version is targeted. Resolve every version of the
        # key and delete each one by VersionId, bypassing governance, so the
        # locked version itself is reclaimed rather than left behind.
        s3 = client._store._s3  # type: ignore[attr-defined]
        bucket = client._store.bucket  # type: ignore[attr-defined]
        versions = s3.list_object_versions(Bucket=bucket, Prefix=key)
        for entry in (
            *versions.get("Versions", ()),
            *versions.get("DeleteMarkers", ()),
        ):
            if entry["Key"] != key:
                continue
            s3.delete_object(
                Bucket=bucket,
                Key=key,
                VersionId=entry["VersionId"],
                BypassGovernanceRetention=True,
            )

    def close(self) -> None:
        self._client.close()
        if self._versioned_client is not None:
            self._versioned_client.close()
        if self._object_lock_client is not None:
            self._object_lock_client.close()


@pytest.fixture(scope="module")
def harness():
    h = AwsHarness()
    yield h
    h.close()


@pytest.fixture(scope="module")
def client(harness):
    return harness.create_client()


@pytest.mark.vcr
class TestAwsBlobConformance(BaseBlobStoreConformance):
    """AWS S3 blob conformance tests.

    All test methods are inherited from
    :class:`BaseBlobStoreConformance`. Cassettes live at
    ``tests/blob/conformance/cassettes/aws/`` and are keyed by
    ``TestAwsBlobConformance.<test_method>.yaml``.
    """

    pass
