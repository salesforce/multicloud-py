"""Shared VCR.py configuration for blob conformance tests.

Each provider's cassettes live in a sibling subdirectory (`cassettes/aws/`,
`cassettes/gcp/`, `cassettes/ali/`). The provider is detected from the
running test's class name so a single `vcr_config` fixture serves all
provider tests.
"""

import io
import itertools
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest
import vcr.patch
import vcr.stubs.aiohttp_stubs

from tests.blob.conformance.vcr_filters import (
    DEFAULT_IGNORE_HOSTS,
    DIRECT_TAG_MATCHERS,
    MULTIPART_TAG_MATCHERS,
    RANGE_HEADER_MATCHER,
    decode_legal_hold_write,
    get_default_vcr_config,
    make_before_record_request,
    make_mpu_before_record_request,
    register_blob_tag_matchers,
    register_range_header_matcher,
)

CASSETTES_ROOT = Path(__file__).parent / "cassettes"

_TAGGED_MULTIPART_TESTS = {
    "test_multipart_applies_tags_given_at_initiate",
    "test_multipart_tags_and_metadata_both_survive_initiate",
    "test_multipart_tags_with_reserved_characters_round_trip",
}

_RANGE_MATCHED_TESTS = {"test_download_stream_range"}


def pytest_recording_configure(config, vcr):  # type: ignore[no-untyped-def]
    """Register blob conformance request matchers."""
    register_blob_tag_matchers(vcr)
    register_range_header_matcher(vcr)


def _install_connection_remover_snapshot_patch() -> None:
    """Make VCR's ``ConnectionRemover.__exit__`` iterate over snapshots.

    VCR patches urllib3's connection pools during a cassette and, at teardown,
    ``ConnectionRemover.__exit__`` walks its internal ``{pool: {connections}}``
    registry to close the connections it opened, iterating the live dict and
    the live per-pool set directly. The google-cloud-storage transport keeps a
    pooled connection alive and mutates that pool while VCR is tearing down,
    which makes the un-snapshotted iteration raise
    ``RuntimeError: Set changed size during iteration`` (seen at record time on
    the first recorded cassette).

    That teardown runs during un-patching, so the raise leaves urllib3
    partially patched. The fix preserves stock behavior exactly and only
    iterates over copies so a concurrent mutation can't abort the cleanup.
    Idempotent: applied once per process.
    """
    remover_cls = vcr.patch.ConnectionRemover
    if getattr(remover_cls, "_snapshot_exit_patched", False):
        return

    def __exit__(self, *args):  # type: ignore[no-untyped-def]
        for pool, connections in list(self._connection_pool_to_connections.items()):
            readd_connections = []
            while pool.pool and not pool.pool.empty() and connections:
                connection = pool.pool.get()
                if isinstance(connection, self._connection_class):
                    connections.discard(connection)
                    connection.close()
                else:
                    readd_connections.append(connection)
            for connection in readd_connections:
                pool._put_conn(connection)
            for connection in list(connections):
                connection.close()

    remover_cls.__exit__ = __exit__
    remover_cls._snapshot_exit_patched = True


_install_connection_remover_snapshot_patch()


# Body types VCR can round-trip through YAML. Anything else is a stream or a
# wrapper object whose repr would be meaningless in a cassette at best, and
# unserializable at worst.
_SERIALIZABLE_BODY_TYPES = (
    type(None),
    bytes,
    bytearray,
    str,
    dict,
    list,
    tuple,
    io.BytesIO,
)


def _install_aiohttp_streaming_body_patch() -> None:
    """Let VCR record aiohttp requests whose body is a streaming wrapper.

    VCR deepcopies the request before any user hook, so a body holding a
    coroutine raises there. S3's default checksum wraps uploads in one, and it
    cannot be avoided: Object Lock requires the checksum. The body is not part
    of ``match_on``, so a placeholder is safe, and the live request is untouched.

    Scoped to the aiohttp stub, leaving the urllib3 stub the sync tests use.
    Idempotent.
    """
    if getattr(vcr.stubs.aiohttp_stubs, "_streaming_body_patched", False):
        return

    stock_request_cls = vcr.stubs.aiohttp_stubs.Request

    class _StreamingSafeRequest(stock_request_cls):  # type: ignore[misc,valid-type]
        def __init__(self, method, uri, body, headers):
            if not isinstance(body, _SERIALIZABLE_BODY_TYPES):
                body = f"<streaming body omitted: {type(body).__name__}>"
            super().__init__(method, uri, body, headers)

    vcr.stubs.aiohttp_stubs.Request = _StreamingSafeRequest
    vcr.stubs.aiohttp_stubs._streaming_body_patched = True


_install_aiohttp_streaming_body_patch()


def _install_aiohttp_raw_headers_patch() -> None:
    """Populate ``raw_headers`` on VCR's replayed aiohttp responses.

    aiobotocore reads it directly and it is left at ``None``, so every replayed
    request raises ``TypeError``. Writes the backing ``_raw_headers``, since the
    property itself is a read-only cached property. Idempotent.

    vcrpy 9.1.0 and later set it themselves. Only Python 3.10 still uses an
    older vcrpy (vcrpy 9 needs Python 3.11), so remove this with Python 3.10.
    """
    if tuple(int(part) for part in vcr.__version__.split(".")[:2]) >= (9, 1):
        return
    stubs = vcr.stubs.aiohttp_stubs
    if getattr(stubs, "_raw_headers_patched", False):
        return

    stock_build_response = stubs.build_response

    def build_response(vcr_request, vcr_response, history):  # type: ignore[no-untyped-def]
        response = stock_build_response(vcr_request, vcr_response, history)
        pairs = tuple(
            (name.encode("utf-8"), value.encode("utf-8"))
            for name, value in response.headers.items()
        )
        # Deliberately not read back through `raw_headers` first: that property
        # memoizes, so probing it would cache the `None` this is fixing.
        if hasattr(response, "_raw_headers"):
            response._raw_headers = pairs
        else:
            response.raw_headers = pairs
        return response

    stubs.build_response = build_response
    stubs._raw_headers_patched = True


_install_aiohttp_raw_headers_patch()


def _install_legal_hold_capture_patch() -> None:
    """Record the intent of every legal-hold write on the wire, in both modes.

    A legal-hold round-trip test verifies the hold through a separate bodyless
    read-back, and a bodyless read matches positionally on play count -- unbound
    to the write that preceded it. So a write that carries the wrong enable value
    still replays the recorded read-back and the test passes: the value never
    reaching the wire goes uncaught. ``match_on`` cannot close this either, since
    the request body is deliberately excluded so streaming uploads still match.

    The outgoing request, body included, converges on two ``Cassette`` methods
    both the urllib3 (sync) and aiohttp (async) stubs route through: on replay
    every served request goes through ``play_response``; on record every real
    request is stored through ``append``. A request hits exactly one of the two
    per interaction, so wrapping both decodes each hold write into a
    cloud-agnostic ``LegalHoldWrite`` and appends it to the live cassette exactly
    once -- in replay *and* record -- which pytest-recording exposes to the test
    through its ``vcr`` fixture, so the test can assert the value actually put on
    the wire (including while re-recording). A fresh cassette per test means the
    list starts empty each time. Idempotent.
    """
    from vcr.cassette import Cassette

    if getattr(Cassette, "_legal_hold_capture_patched", False):
        return

    def _capture(cassette: Any, request: Any) -> None:
        write = decode_legal_hold_write(request)
        if write is None:
            return
        captured = getattr(cassette, "_captured_legal_hold_writes", None)
        if captured is None:
            captured = []
            cassette._captured_legal_hold_writes = captured
        captured.append(write)

    stock_play_response = Cassette.play_response
    stock_append = Cassette.append

    def play_response(self, request):  # type: ignore[no-untyped-def]
        _capture(self, request)
        return stock_play_response(self, request)

    def append(self, request, response):  # type: ignore[no-untyped-def]
        _capture(self, request)
        return stock_append(self, request, response)

    Cassette.play_response = play_response
    Cassette.append = append
    Cassette._legal_hold_capture_patched = True


_install_legal_hold_capture_patch()

# Provider-specific extra headers — request IDs and other noise that's not a
# security risk but creates cassette diff churn if left in.
AWS_EXTRA_HEADERS = [
    "X-Amz-Request-Id",
    "x-amz-id-2",
]

GCP_EXTRA_HEADERS = [
    "x-goog-generation",
    "x-goog-metageneration",
    "x-guploader-uploadid",
]


def _provider_for(class_name: str) -> str:
    """Map a test class name to its provider id.

    ``TestAwsBlobConformance`` → ``"aws"``,
    ``TestGcpBlobConformance`` → ``"gcp"``,
    ``TestAliBlobConformance`` → ``"ali"``.
    """
    name = class_name.lower()
    if "aws" in name:
        return "aws"
    if "gcp" in name:
        return "gcp"
    if "ali" in name:
        return "ali"
    raise RuntimeError(
        f"Cannot determine provider for test class '{class_name}'. "
        "Class name must contain 'aws', 'gcp', or 'ali'."
    )


def _is_gcs_object_upload(request) -> bool:
    """Whether this is an object upload, whose body is content, not credentials."""
    return "/upload/storage/v1/b/" in getattr(request, "uri", "")


@pytest.fixture(scope="module")
def vcr_cassette_dir(request) -> str:
    """Override pytest-recording's default cassette directory.

    pytest-recording's built-in ``vcr_cassette_dir`` fixture resolves to
    ``<test_dir>/cassettes/<module_name>/`` (e.g.
    ``cassettes/test_aws_blob_conformance/``). Our layout organizes cassettes
    by *provider* instead (``cassettes/aws/``, ``cassettes/gcp/``).

    This override is **required** for ``--record-mode=rewrite`` to work.
    The rewrite logic in pytest-recording deletes the old cassette using the
    path from this fixture before re-recording. Without this override, the
    delete targets the wrong (non-existent) path, silently fails, and VCR
    replays the stale cassette instead of making a real HTTP call.

    The ``vcr_config`` fixture below also references ``cassette_library_dir``
    which must agree with this value — both are derived from the same
    ``CASSETTES_ROOT / provider`` computation to stay in sync.
    """
    test_class = getattr(request, "cls", None)
    if test_class is not None:
        class_name = test_class.__name__
    else:
        class_name = request.module.__name__

    provider = _provider_for(class_name)
    return str(CASSETTES_ROOT / provider)


def _test_name(request) -> str:
    """Return the running test's function name (without parametrize id)."""
    node = getattr(request, "node", None)
    return getattr(node, "originalname", None) or getattr(node, "name", "") or ""


def _is_tagged_multipart_test(request) -> bool:
    return _test_name(request) in _TAGGED_MULTIPART_TESTS


def _is_tagged_multipart_sync_test(request) -> bool:
    test_class = getattr(request, "cls", None)
    class_name = getattr(test_class, "__name__", "")
    return "Async" not in class_name and _is_tagged_multipart_test(request)


def _is_direct_tag_test(request) -> bool:
    test_class = getattr(request, "cls", None)
    class_name = getattr(test_class, "__name__", "")
    name = _test_name(request)
    return "Async" in class_name and (
        name.startswith("test_get_tags") or name.startswith("test_set_tags")
    )


def _requires_complete_playback(request) -> bool:
    return _is_direct_tag_test(request) or _is_tagged_multipart_sync_test(request)


def _played_requests_in_recorded_order(vcr) -> bool:
    played = [request for request, _ in vcr._played_interactions]
    return played == vcr.requests


@pytest.fixture(autouse=True)
def require_complete_playback(request, vcr, record_mode):
    """Require every significant write interaction during replay."""
    yield
    if record_mode == "none" and _requires_complete_playback(request):
        assert vcr is not None
        unused = [
            str(recorded)
            for index, recorded in enumerate(vcr.requests)
            if vcr.play_counts[index] == 0
        ]
        assert not unused, f"cassette has unused interactions: {unused}"
        if _is_direct_tag_test(request):
            assert _played_requests_in_recorded_order(
                vcr
            ), "cassette interactions played out of recorded order"


@pytest.fixture(autouse=True)
def mpu_deterministic_uuid(request, monkeypatch):
    """Make client-generated multipart upload ids deterministic for replay.

    Some providers mint the multipart ``upload_id`` client-side with
    ``uuid.uuid4()`` (no server round-trip) and then embed it in the URL
    query and path of later list/abort/complete requests. Those components
    are matched during replay, so a freshly-random id would never match the
    recorded cassette. Seeding ``uuid.uuid4`` with a per-test counter makes
    the recorded and replayed ids identical.

    Applied only to multipart tests so no other cassette is affected.
    """
    if "multipart" not in _test_name(request):
        yield
        return

    counter = itertools.count(1)

    def _fake_uuid4() -> uuid.UUID:
        # Encode a monotonic counter into a fixed uuid layout so every part /
        # list / abort within one test reuses the same deterministic id space.
        return uuid.UUID(int=next(counter))

    monkeypatch.setattr(uuid, "uuid4", _fake_uuid4)
    yield


@pytest.fixture
def vcr_config(request, vcr_cassette_dir) -> Dict[str, Any]:
    """Per-provider VCR.py configuration.

    Applies provider-specific header filters and body scrubbing rules.
    The cassette directory is sourced from the ``vcr_cassette_dir`` fixture
    (single source of truth) to guarantee that ``--record-mode=rewrite``
    deletes from the same path VCR reads/writes.
    """
    test_class = getattr(request, "cls", None)
    if test_class is not None:
        class_name = test_class.__name__
    else:
        class_name = request.module.__name__

    provider = _provider_for(class_name)

    config = get_default_vcr_config(cassette_dir=vcr_cassette_dir)
    if _is_tagged_multipart_test(request):
        config["match_on"].extend(MULTIPART_TAG_MATCHERS)
    if _is_direct_tag_test(request):
        config["match_on"].extend(DIRECT_TAG_MATCHERS)
    if _test_name(request) in _RANGE_MATCHED_TESTS:
        config["match_on"].append(RANGE_HEADER_MATCHER)

    # Multipart-upload tests send multi-MiB part bodies on the wire; truncate
    # them in the cassette so committed fixtures stay small. Detected by test
    # name so the whole suite doesn't pay for a rarely-needed filter.
    is_multipart = "multipart" in _test_name(request)

    if provider == "aws":
        config["filter_headers"].extend(AWS_EXTRA_HEADERS)
        # boto3 stores BytesIO request bodies that VCR's request-body scrubber
        # can't handle, and AWS auth lives entirely in headers (already
        # filtered), so request-body scrubbing is unnecessary. The default
        # before_record_response is kept, though: S3 list responses echo the
        # caller's principal/role/canonical-id in <Initiator>/<Owner>, which
        # the response scrubber redacts.
        if is_multipart:
            config["before_record_request"] = make_mpu_before_record_request(
                scrub_bodies=False
            )
        else:
            # A presigned URL carries the signature, the signing time and the
            # session token in its query string, so a test that dereferences one
            # needs those replaced -- to keep the token out of the fixture, and
            # so the request can be matched on a later run.
            config["before_record_request"] = make_before_record_request(
                scrub_bodies=False,
                normalize_signed_url_query="presign" in _test_name(request),
            )
        if "presign" in _test_name(request):
            config["allow_playback_repeats"] = True
    elif provider == "gcp":
        config["filter_headers"].extend(GCP_EXTRA_HEADERS)
        if is_multipart:
            # A part upload reaches VCR as a file-like body the scrubber cannot
            # read, so those requests skip it. Exempting them individually keeps
            # every other body scrubbed, including the metadata patch that
            # carries the caller's own metadata and tags.
            config["before_record_request"] = make_mpu_before_record_request(
                body_scrub_exempt=_is_gcs_object_upload
            )
        elif "presign" in _test_name(request):
            # URL signing with keyless credentials signs through the IAM signBlob
            # API. That host is on the ignore list with the other credential
            # endpoints, which would leave these tests with no recorded
            # interaction at all, so it is recorded here instead. The signing
            # identity is normalized out of the path so a recording made under
            # real credentials matches the offline mock.
            config["before_record_request"] = make_before_record_request(
                ignore_hosts=DEFAULT_IGNORE_HOSTS - {"iamcredentials.googleapis.com"},
                normalize_service_account_identity=True,
                normalize_signed_url_query=True,
            )
            # Every signing request is byte-identical once the identity is
            # normalized out, and the signing library decides how many to issue.
            # Consuming one recorded interaction per call would make replay
            # depend on that count; allowing repeats makes it depend only on the
            # request itself.
            config["allow_playback_repeats"] = True
    # ali: no extra headers yet; impl is a placeholder.

    return config
