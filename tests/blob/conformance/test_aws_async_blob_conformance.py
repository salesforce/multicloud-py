"""AWS S3 async blob conformance tests with VCR.py record/replay.

Runs :class:`BaseAsyncBlobStoreConformance` against S3 via ``AsyncBucketClient``,
supplying the AWS ``AsyncHarness``. Record with ``--record-mode=rewrite`` (needs
AWS credentials); replay with ``--record-mode=none``.
"""

from __future__ import annotations

import os
from typing import Optional

import pytest
from multicloud.blob.blob_client import AsyncBucketClient
from multicloud.common.credentials import (
    CredentialsOverrider,
    CredentialsType,
    StsCredentials,
)

from .base_async_blob_store_conformance import (
    AsyncHarness,
    BaseAsyncBlobStoreConformance,
)

AWS_BUCKET = os.environ.get("CONFORMANCE_AWS_BUCKET", "example-test-bucket-barry")
AWS_REGION = os.environ.get("CONFORMANCE_AWS_REGION", "us-west-2")
# Not env-overridable: the name is part of the matched request URI. If it is
# ever taken, change it here and re-record.
AWS_ABSENT_BUCKET = "multicloudj-conformance-absent-bucket-do-not-create"
AWS_VERSIONED_BUCKET = os.environ.get(
    "CONFORMANCE_AWS_VERSIONED_BUCKET", "chameleon-python-sdk-versioned"
)
AWS_OBJECT_LOCK_BUCKET = os.environ.get(
    "CONFORMANCE_AWS_OBJECT_LOCK_BUCKET", "chameleon-test-bucket-with-object-lock"
)


class AwsAsyncHarness(AsyncHarness):
    """AWS S3 async test harness."""

    def create_client(self) -> AsyncBucketClient:
        return AsyncBucketClient(provider="aws", bucket=AWS_BUCKET, region=AWS_REGION)

    def create_client_for_absent_bucket(self) -> AsyncBucketClient:
        return AsyncBucketClient(
            provider="aws", bucket=AWS_ABSENT_BUCKET, region=AWS_REGION
        )

    def create_versioned_client(self) -> Optional[AsyncBucketClient]:
        return AsyncBucketClient(
            provider="aws", bucket=AWS_VERSIONED_BUCKET, region=AWS_REGION
        )

    def create_object_lock_client(self) -> Optional[AsyncBucketClient]:
        return AsyncBucketClient(
            provider="aws", bucket=AWS_OBJECT_LOCK_BUCKET, region=AWS_REGION
        )

    def get_provider_id(self) -> str:
        return "aws"

    def get_bucket(self) -> str:
        return AWS_BUCKET

    def get_versioned_bucket(self) -> Optional[str]:
        return AWS_VERSIONED_BUCKET

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

    def create_invalid_credentials_client(self) -> AsyncBucketClient:
        # Real bucket, deliberately bogus static session credentials. Building
        # the session is offline, so the 403 only appears once a request signs.
        return AsyncBucketClient(
            provider="aws",
            bucket=AWS_BUCKET,
            region=AWS_REGION,
            credentials_overrider=CredentialsOverrider(
                credentials_type=CredentialsType.SESSION,
                session_credentials=StsCredentials(
                    access_key_id="INVALIDACCESSKEY",
                    access_key_secret="invalid-secret-key-invalid-secret-key",
                    security_token="invalid-session-token",
                ),
            ),
        )


@pytest.fixture(scope="module")
def harness():
    return AwsAsyncHarness()


@pytest.mark.vcr
class TestAwsAsyncBlobConformance(BaseAsyncBlobStoreConformance):
    """AWS S3 async blob conformance tests.

    The class name must contain ``Aws`` so ``conftest._provider_for`` resolves
    the cassette directory and header filters.
    """

    pass
