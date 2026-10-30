"""GCP Cloud Storage async blob conformance tests with VCR.py record/replay.

Runs :class:`BaseAsyncBlobStoreConformance` against Cloud Storage via
``AsyncBucketClient``, supplying the GCP ``AsyncHarness``. Record with
``--record-mode=rewrite`` (needs GCP credentials); replay with
``--record-mode=none``.

The directory operations run the synchronous ``transfer_manager`` in a thread
(over urllib3), while single-object operations go over gcloud-aio/aiohttp; both
are recorded by VCR. Directory and single-object cassettes were recorded against
different buckets, so the harness binds each family to its own bucket.
"""

from __future__ import annotations

import os
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest
from multicloud.blob.blob_client import AsyncBucketClient
from multicloud.common.credentials import (
    CredentialsOverrider,
    CredentialsType,
    StsCredentials,
)

from tests.gcp_common.mock_gcp_credentials import make_mock_credentials
from tests.gcp_common.recording import certifi_trust_store

from .base_async_blob_store_conformance import (
    AsyncHarness,
    BaseAsyncBlobStoreConformance,
)

GCP_BUCKET = os.environ.get("CONFORMANCE_GCP_BUCKET", "python-async-test-bucket")
# Directory cassettes were recorded against a separate bucket.
GCP_DIRECTORY_BUCKET = os.environ.get(
    "CONFORMANCE_GCP_DIRECTORY_BUCKET", "example-test-bucket-barry"
)
# Not env-overridable: the name is part of the matched request URI. If it is
# ever taken, change it here and re-record.
GCP_ABSENT_BUCKET = "multicloudj-conformance-absent-bucket-do-not-create"
# Dedicated versioning-enabled bucket, isolated so its un-reclaimable noncurrent
# versions don't accumulate in GCP_BUCKET.
GCP_VERSIONED_BUCKET = os.environ.get(
    "CONFORMANCE_GCP_VERSIONED_BUCKET", "chameleon-python-sdk-versioned"
)
GCP_OBJECT_LOCK_BUCKET = os.environ.get(
    "CONFORMANCE_GCP_OBJECT_LOCK_BUCKET", "sdk-blob-with-object-lock"
)


class GcpAsyncHarness(AsyncHarness):
    """GCP Cloud Storage async test harness."""

    def create_client(self) -> AsyncBucketClient:
        return AsyncBucketClient(provider="gcp", bucket=GCP_BUCKET, region=None)

    def create_directory_client(self) -> AsyncBucketClient:
        return AsyncBucketClient(
            provider="gcp", bucket=GCP_DIRECTORY_BUCKET, region=None
        )

    def create_client_for_absent_bucket(self) -> AsyncBucketClient:
        return AsyncBucketClient(provider="gcp", bucket=GCP_ABSENT_BUCKET, region=None)

    def create_versioned_client(self) -> Optional[AsyncBucketClient]:
        return AsyncBucketClient(
            provider="gcp", bucket=GCP_VERSIONED_BUCKET, region=None
        )

    def create_object_lock_client(self) -> Optional[AsyncBucketClient]:
        return AsyncBucketClient(
            provider="gcp", bucket=GCP_OBJECT_LOCK_BUCKET, region=None
        )

    def get_provider_id(self) -> str:
        return "gcp"

    def get_bucket(self) -> str:
        return GCP_BUCKET

    def get_versioned_bucket(self) -> Optional[str]:
        return GCP_VERSIONED_BUCKET

    def supports_copy_source_version(self) -> bool:
        # gcloud-aio-storage copy() has no sourceGeneration parameter and
        # refuses source_version_id rather than silently copying latest.
        return False

    def supports_write_if_match(self) -> bool:
        # GCS conditional overwrite is generation-based, not ETag-based, so the
        # write path rejects if_match with UnsupportedOperationException.
        return False

    def supports_write_if_version_match(self) -> bool:
        # GCS compare-and-swap uses the object generation (ifGenerationMatch).
        return True

    def get_kms_key_id(self) -> str:
        # Fixed customer-managed key in the recording project. The id only has
        # to be live at record time; on replay it is matched from the cassette.
        # So the encryption tests always run rather than skip.
        return (
            "projects/example-project/locations/us"
            "/keyRings/multicloud-test/cryptoKeys/multicloud-test"
        )

    def detects_unknown_upload_id(self) -> bool:
        """Multipart is emulated with scratch objects, so there is no session
        to check a forged upload id against."""
        return False

    def create_invalid_credentials_client(self) -> AsyncBucketClient:
        # Real bucket, bogus session token. A non-empty token is required, since
        # an empty one is rejected while the credentials are constructed.
        return AsyncBucketClient(
            provider="gcp",
            bucket=GCP_BUCKET,
            region=None,
            credentials_overrider=CredentialsOverrider(
                credentials_type=CredentialsType.SESSION,
                session_credentials=StsCredentials(
                    access_key_id="",
                    access_key_secret="",
                    security_token="invalid-session-token",
                ),
            ),
        )


@pytest.fixture(scope="module", autouse=True)
def mock_gcp_auth(request):
    """Bypass credential acquisition in replay mode (real creds in record mode).

    Two mechanisms need stubbing offline:

    - ``google.auth.default`` (used for URL signing and the thread-run
      synchronous ``storage.Client``). The mock credentials mimic keyless creds
      so signing takes the recorded IAM signBlob path; the returned project is
      required or ``storage.Client()`` raises ``OSError``.
    - ``gcloud-aio``'s ``Token.get`` mints an OAuth token from an un-recorded
      host, so it is stubbed to a static token.
    """
    record_mode = request.config.getoption("--record-mode", default="none")
    if record_mode != "none":
        # Recording mints a real OAuth token over aiohttp; certifi_trust_store
        # makes that TLS independent of the host OS trust store. Replay does not
        # reach this branch (Token.get is patched below).
        with certifi_trust_store():
            yield
        return

    # Local signing disallowed so the keyless path is the only one under test.
    fake_creds = make_mock_credentials(allow_local_signing=False)
    with (
        patch("google.auth.default", return_value=(fake_creds, "fake-project")),
        patch(
            "gcloud.aio.auth.token.Token.get",
            new=AsyncMock(return_value="mock-gcloud-aio-token"),
        ),
    ):
        yield


@pytest.fixture(scope="module")
def harness(mock_gcp_auth):
    return GcpAsyncHarness()


@pytest.mark.vcr
class TestGcpAsyncBlobConformance(BaseAsyncBlobStoreConformance):
    """GCP Cloud Storage async blob conformance tests.

    All object-lock test methods are inherited from
    :class:`BaseAsyncBlobStoreConformance`. The class name must contain ``Gcp``
    so ``conftest._provider_for`` resolves the cassette directory and header
    filters.

    Note: there is no GCP-only event-based-hold test here (unlike the sync
    suite). Requesting an event-based hold needs ``use_event_based_hold`` at
    upload time, which the async ``upload`` does not expose, so the flag cannot
    be set through the async public API — see the scope note on the shared
    object-lock tests.
    """

    pass
