"""GCP Cloud Storage blob conformance tests with VCR.py record/replay.

Extends :class:`BaseBlobStoreConformance`, supplying a GCP-specific
``Harness`` that wires up the :class:`BucketClient`.
"""

from __future__ import annotations

import os
from typing import Optional
from unittest.mock import patch

import pytest
from multicloud.blob.blob_client import BucketClient, ObjectLockConfiguration
from multicloud.common.credentials import (
    CredentialsOverrider,
    CredentialsType,
    StsCredentials,
)

from tests.blob.conformance.base_blob_store_conformance import (
    BaseBlobStoreConformance,
    Harness,
)
from tests.gcp_common.mock_gcp_credentials import make_mock_credentials

GCP_BUCKET = os.environ.get("CONFORMANCE_GCP_BUCKET", "example-test-bucket-barry")
# Deliberately not env-overridable, like GCP_ABSENT_BUCKET below. The name
# appears in the request URI, which the cassette matcher compares, so
# overriding it would send a request no committed cassette can answer. Unlike
# GCP_ABSENT_BUCKET this bucket is only ever addressed, never listed among the
# recorder's real buckets, so a single fixed name serves every project. If the
# name ever does get taken, changing it here and re-recording is the fix.
GCP_NONEXISTENT_BUCKET = "multicloud-conformance-nonexistent-bucket-gcp-9f8e7d"
GCP_VERSIONED_BUCKET = os.environ.get(
    "CONFORMANCE_GCP_VERSIONED_BUCKET", "chameleon-python-sdk-versioned"
)
GCP_OBJECT_LOCK_BUCKET = os.environ.get(
    "CONFORMANCE_GCP_OBJECT_LOCK_BUCKET", "sdk-blob-with-object-lock"
)
# Deliberately not env-overridable, unlike the constants above. Those name
# buckets that must exist in whichever project is recording, so each recorder
# needs to point them somewhere. This one names a bucket that must exist in no
# project, so there is nothing to redirect -- and the name appears in the request
# URI, which the cassette matcher compares, so overriding it would send a request
# no committed cassette can answer. If the name ever does get taken, changing it
# here and re-recording is the fix.
GCP_ABSENT_BUCKET = "multicloudj-conformance-absent-bucket-do-not-create"


class GcpHarness(Harness):
    """GCP Cloud Storage test harness."""

    def __init__(self) -> None:
        self._client = BucketClient(
            provider="gcp",
            bucket=GCP_BUCKET,
            region=None,
        )
        self._versioned_client: Optional[BucketClient] = BucketClient(
            provider="gcp",
            bucket=GCP_VERSIONED_BUCKET,
            region=None,
        )
        self._object_lock_client: Optional[BucketClient] = BucketClient(
            provider="gcp",
            bucket=GCP_OBJECT_LOCK_BUCKET,
            region=None,
        )

    def create_client(self) -> BucketClient:
        return self._client

    def create_client_for_absent_bucket(self) -> BucketClient:
        return BucketClient(
            provider="gcp",
            bucket=GCP_ABSENT_BUCKET,
            region=None,
        )

    def create_versioned_client(self) -> Optional[BucketClient]:
        return self._versioned_client

    def create_object_lock_client(self) -> Optional[BucketClient]:
        return self._object_lock_client

    def create_nonexistent_bucket_client(self) -> BucketClient:
        # A syntactically valid but non-existent bucket. The storage client is
        # built offline (auth is patched in replay); the missing bucket is only
        # discovered when an operation actually calls the API.
        return BucketClient(
            provider="gcp",
            bucket=GCP_NONEXISTENT_BUCKET,
            region=None,
        )

    def create_invalid_credentials_client(self) -> BucketClient:
        # Point at the real bucket but supply a bogus session token. Building
        # OAuth2 credentials from a static token is offline, so the 401/403
        # only surfaces when an operation actually sends a request. A non-empty
        # token is required — an empty one is rejected during construction.
        overrider = CredentialsOverrider(
            credentials_type=CredentialsType.SESSION,
            session_credentials=StsCredentials(
                access_key_id="",
                access_key_secret="",
                security_token="invalid-session-token",
            ),
        )
        return BucketClient(
            provider="gcp",
            bucket=GCP_BUCKET,
            region=None,
            credentials_overrider=overrider,
        )

    def get_provider_id(self) -> str:
        return "gcp"

    def get_bucket(self) -> str:
        return GCP_BUCKET

    def get_versioned_bucket(self) -> Optional[str]:
        return GCP_VERSIONED_BUCKET if self._versioned_client is not None else None

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

    def delete_retained_object(self, client: BucketClient, key: str) -> None:
        # GCS Unlocked retention cannot be shortened to a past instant (the API
        # rejects it) and blocks a plain delete — it must be removed entirely
        # first. The cloud-agnostic retention API requires a retain-until
        # instant and so cannot express removal, so this drops to the
        # provider's blob handle, clears the retention, then deletes.
        blob = client._store._bucket_handle.blob(key)  # type: ignore[attr-defined]
        blob.reload()
        blob.retention.mode = None
        blob.retention.retain_until_time = None
        blob.patch(override_unlocked_retention=True)
        blob.delete()

    def validates_part_etags(self) -> bool:
        # Compose-based multipart assembles parts by re-reading the stored
        # temp objects in part-number order and never checks the ETag the
        # caller passes to complete, so a wrong ETag cannot be detected.
        return False

    def close(self) -> None:
        self._client.close()
        if self._versioned_client is not None:
            self._versioned_client.close()
        if self._object_lock_client is not None:
            self._object_lock_client.close()


@pytest.fixture(scope="module", autouse=True)
def mock_gcp_auth(request):
    """Patch google.auth.default() in replay mode to bypass ADC lookup.

    In record mode the patch is NOT applied — real credentials are used.

    Nothing about URL signing is stubbed. The mock credentials look like the
    keyless credentials production resolves, so signing takes the library's IAM
    signBlob branch, and that call is recorded like any other -- see the
    presigned branch in ``conftest.vcr_config``. Replaying it rather than
    stubbing it is what keeps the keyless path itself under test.
    """
    record_mode = request.config.getoption("--record-mode") or "none"
    if record_mode == "none":
        # Local signing is disallowed so the keyless path is the only one that
        # can succeed; otherwise a regression that dropped it would still pass
        # by falling back to signing locally.
        fake_creds = make_mock_credentials(allow_local_signing=False)
        with patch("google.auth.default", return_value=(fake_creds, "fake-project")):
            yield
    else:
        yield


@pytest.fixture(scope="module")
def harness(mock_gcp_auth):
    h = GcpHarness()
    yield h
    h.close()


@pytest.fixture(scope="module")
def client(harness):
    return harness.create_client()


@pytest.mark.vcr
class TestGcpBlobConformance(BaseBlobStoreConformance):
    """GCP Cloud Storage blob conformance tests.

    Most test methods are inherited from
    :class:`BaseBlobStoreConformance`. Cassettes live at
    ``tests/blob/conformance/cassettes/gcp/`` and are keyed by
    ``TestGcpBlobConformance.<test_method>.yaml``.

    The event-based-hold test below is GCP-only: ``use_event_based_hold``
    maps to GCS's ``eventBasedHold`` (distinct from ``temporaryHold``),
    whereas S3 has no such concept and ignores the flag — so the read-back
    value differs by provider and the assertion cannot live in the shared
    suite.
    """

    def test_get_object_lock_event_based_hold(self, object_lock_client, harness):
        """A legal hold requested as an event-based hold reads back on GCS.

        Uploads with ``legal_hold=True`` and ``use_event_based_hold=True`` so
        the hold is applied as GCS's ``eventBasedHold`` rather than the default
        ``temporaryHold``. ``get_object_lock`` must report the hold
        (``legal_hold is True``) and surface the event-based flag distinctly
        (``event_based_hold is True``) — proving the None/False/True
        distinction on ``ObjectLockInfo.event_based_hold`` is populated, not
        left at its "not reported" default.
        """
        client = object_lock_client
        key = f"{self.TEST_PREFIX}object-lock/event-based-hold.txt"
        lock_cfg = ObjectLockConfiguration(
            legal_hold=True,
            use_event_based_hold=True,
        )
        try:
            client.upload(key, b"event-held", object_lock=lock_cfg)
            lock = client.get_object_lock(key)
            assert lock is not None
            assert lock.mode is None
            assert lock.retain_until_date is None
            assert lock.legal_hold is True
            assert lock.event_based_hold is True
        finally:
            self._safe_release_lock_and_delete(client, harness, key)
