"""Synchronous blob benchmarks.

Serial (one op at a time) and concurrent (``ThreadPoolExecutor`` at 4/8
workers) bodies. They use ``blob_bench_client`` and ``timed()``, so the
same tests run against ``benchfake`` or a real backend.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from multicloud.blob.blob_client.driver import (
    BlobIdentifier,
    CopyResponse,
    DeleteObjectsResponse,
    MultipartPart,
    MultipartUploadResponse,
    UploadResponse,
)

from benchmarks._bench import resolve_rounds, timed

_KB = 1024
_MB = 1024 * 1024

# Payload-size tiers exercised by the serial upload/download benchmarks.
_SIZES = [
    pytest.param(_KB, id="1KB"),
    pytest.param(_MB, id="1MB"),
    pytest.param(10 * _MB, id="10MB"),
]

# Number of objects seeded for the listing benchmarks.
_LIST_COUNT = 100

# Page size for the listing benchmarks — small relative to the seeded count so
# a single page is truncated and the paginate-to-completion loop takes several
# pages.
_LIST_PAGE_MAX = 20

# Multipart upload shape: a 10 MB object uploaded as two 5 MB parts.
_MULTIPART_PART_SIZE = 5 * _MB
_MULTIPART_PARTS = 2

# Number of objects deleted in one batch by the bulk-delete benchmark.
_BULK_DELETE_BATCH = 25

# Concurrent worker counts. Keep t1: it is the single-worker baseline the
# t4/t8 numbers are read against. Healthy scaling is roughly linear off t1
# (t4 ~= t1/4, t8 ~= t1/8, so t8 ~= t4/2); a live plateau (t8 ~= t4 ~= t1)
# means calls aren't overlapping. Both live and benchfake run all three.
_THREAD_COUNTS = [
    pytest.param(1, id="t1"),
    pytest.param(4, id="t4"),
    pytest.param(8, id="t8"),
]

# Operations issued per timed batch in the concurrent benchmarks. Fixed so
# throughput is comparable across thread counts; payload is small on purpose so
# scheduling and GIL-handoff overhead dominate the measurement.
_BATCH = 64

# Concurrent sample count, separate from the serial BLOB_BENCH_ROUNDS: the t8/t4
# ratio is round-count-insensitive, so it defaults lower.
_CONCURRENT_ROUNDS = int(os.environ.get("BLOB_BENCH_CONCURRENT_ROUNDS", "30"))

# Per-op ceiling on future.result() so a stalled live call fails its round
# instead of hanging the run. <=0 waits unbounded; benchfake never trips it.
_raw_op_timeout = float(os.environ.get("BLOB_BENCH_OP_TIMEOUT", "120"))
_OP_TIMEOUT = _raw_op_timeout if _raw_op_timeout > 0 else None


def _payload(size: int) -> bytes:
    return b"x" * size


@pytest.mark.benchmark(group="blob-sync-upload")
@pytest.mark.parametrize("size", _SIZES)
def test_upload(benchmark, blob_bench_client, bench_prefix, size):
    """Single upload, timed across payload-size tiers.

    Unique key per round: reusing one name would trip single-object
    write-rate limits and measure throttling, not upload latency.
    """
    payload = _payload(size)
    # Pre-generate one unique key per invocation outside the timed body: at the
    # benchfake tier a sample is tens of ns, so an in-body f-string would be a
    # real fraction of the measurement. next() on a list iterator is a pointer
    # bump, not a format.
    rounds, warmup = resolve_rounds()
    keys = iter(
        [f"{bench_prefix}upload-{size}-{i}.dat" for i in range(rounds + warmup)]
    )

    def _upload() -> UploadResponse:
        return blob_bench_client.upload(next(keys), payload)

    result = timed(benchmark, _upload)

    assert result.key.startswith(f"{bench_prefix}upload-{size}-")
    assert result.e_tag is not None
    # Round-trip (untimed): prove the bytes landed, not just a response shape.
    assert blob_bench_client.download(result.key).content == payload


@pytest.mark.benchmark(group="blob-sync-download")
@pytest.mark.parametrize("size", _SIZES)
def test_download(benchmark, blob_bench_client, bench_prefix, size):
    """Single download, timed across payload-size tiers."""
    payload = _payload(size)
    key = f"{bench_prefix}download-{size}.dat"
    blob_bench_client.upload(key, payload)  # seed — not timed

    result = timed(benchmark, blob_bench_client.download, key)

    assert result.key == key
    assert result.content == payload  # round-trip


@pytest.mark.benchmark(group="blob-sync-lifecycle")
def test_write_read_delete(benchmark, blob_bench_client, bench_prefix):
    """Full upload -> download -> delete lifecycle, timed as one op."""
    payload = _payload(_KB)
    # Unique key per round (reusing one name would measure single-object write
    # throttling, not lifecycle latency), pre-generated outside the timed body
    # so key formatting stays out of the sample.
    rounds, warmup = resolve_rounds()
    keys = iter([f"{bench_prefix}lifecycle-{i}.dat" for i in range(rounds + warmup)])
    last_key: list[str] = []

    def _cycle() -> bytes:
        key = next(keys)
        last_key[:] = [key]
        blob_bench_client.upload(key, payload)
        content = blob_bench_client.download(key).content
        blob_bench_client.delete(key)
        return content

    content = timed(benchmark, _cycle)

    assert content == payload
    assert last_key  # the timed body ran at least once
    assert blob_bench_client.does_object_exist(last_key[0]) is False


@pytest.mark.benchmark(group="blob-sync-metadata")
def test_get_metadata(benchmark, blob_bench_client, bench_prefix):
    """HEAD-style metadata lookup, timed."""
    payload = _payload(_MB)
    key = f"{bench_prefix}metadata.dat"
    blob_bench_client.upload(key, payload, content_type="application/octet-stream")

    meta = timed(benchmark, blob_bench_client.get_metadata, key)

    assert meta.key == key
    assert meta.content_length == len(payload)
    # The content_type set on upload round-tripped through the metadata path.
    assert meta.content_type == "application/octet-stream"


@pytest.mark.benchmark(group="blob-sync-list")
def test_list_page(benchmark, blob_bench_client, bench_prefix):
    """Single listing page over a seeded prefix, timed."""
    prefix = f"{bench_prefix}list/"
    for i in range(_LIST_COUNT):
        blob_bench_client.upload(f"{prefix}{i}.dat", _payload(_KB))

    page = timed(
        benchmark,
        blob_bench_client.list_page,
        prefix=prefix,
        max_results=_LIST_PAGE_MAX,
    )

    keys = {b.key for b in page.blobs}
    assert f"{prefix}0.dat" in keys
    assert len(page.blobs) == _LIST_PAGE_MAX
    assert page.is_truncated is True  # seeded count exceeds one page


@pytest.mark.benchmark(group="blob-sync-list-all")
def test_list_all(benchmark, blob_bench_client, bench_prefix):
    """Paginate a prefix to completion, timed as one op."""
    prefix = f"{bench_prefix}listall/"
    for i in range(_LIST_COUNT):
        blob_bench_client.upload(f"{prefix}{i}.dat", _payload(_KB))

    def _list_all() -> set[str]:
        keys: set[str] = set()
        token = None
        while True:
            page = blob_bench_client.list_page(
                prefix=prefix, max_results=_LIST_PAGE_MAX, pagination_token=token
            )
            keys.update(b.key for b in page.blobs)
            token = page.next_pagination_token
            if not page.is_truncated or not token:
                break
        return keys

    keys = timed(benchmark, _list_all)

    assert len(keys) == _LIST_COUNT  # every page was walked
    assert f"{prefix}0.dat" in keys


@pytest.mark.benchmark(group="blob-sync-copy")
def test_copy(benchmark, blob_bench_client, bench_prefix):
    """Server-side copy to a fresh destination each round, timed."""
    src_key = f"{bench_prefix}copy-src.dat"
    payload = _payload(_KB)
    blob_bench_client.upload(src_key, payload)

    # Unique dest per round (reusing one name would measure single-object write
    # throttling, not copy latency), pre-generated outside the timed body.
    rounds, warmup = resolve_rounds()
    dest_keys = iter(
        [f"{bench_prefix}copy-dst-{i}.dat" for i in range(rounds + warmup)]
    )

    def _copy() -> CopyResponse:
        return blob_bench_client.copy(
            src_key, dest_bucket=blob_bench_client.bucket, dest_key=next(dest_keys)
        )

    result = timed(benchmark, _copy)

    assert result.key.startswith(f"{bench_prefix}copy-dst-")
    assert blob_bench_client.download(result.key).content == payload  # round-trip


@pytest.mark.benchmark(group="blob-sync-multipart")
def test_multipart_upload(benchmark, blob_bench_client, bench_prefix):
    """Full multipart upload (initiate -> upload parts -> complete), timed."""
    part_payload = _payload(_MULTIPART_PART_SIZE)
    # Unique key per round (reusing one name would measure single-object write
    # throttling), pre-generated outside the timed body.
    rounds, warmup = resolve_rounds()
    keys = iter([f"{bench_prefix}multipart-{i}.dat" for i in range(rounds + warmup)])
    last_key: list[str] = []

    def _multipart() -> MultipartUploadResponse:
        key = next(keys)
        last_key[:] = [key]
        mpu = blob_bench_client.initiate_multipart_upload(key)
        # Abort on any failure between initiate and complete: a mid-round error
        # would otherwise orphan billable multipart parts that teardown's
        # list-based scan cannot see (in-progress uploads aren't listable).
        try:
            parts = [
                blob_bench_client.upload_part(
                    mpu, MultipartPart(part_number=n, content=part_payload)
                )
                for n in range(1, _MULTIPART_PARTS + 1)
            ]
            return blob_bench_client.complete_multipart_upload(mpu, parts)
        except Exception:
            try:
                blob_bench_client.abort_multipart_upload(mpu)
            except Exception:
                pass
            raise

    result = timed(benchmark, _multipart)

    # Only e_tag is guaranteed on the completion response; the assembled object
    # is verified by round-trip download (proving it landed under the last key).
    assert result.e_tag is not None
    assert last_key  # the timed body ran at least once
    assert (
        blob_bench_client.download(last_key[0]).content
        == part_payload * _MULTIPART_PARTS
    )


@pytest.mark.benchmark(group="blob-sync-bulk-delete")
def test_bulk_delete_lifecycle(benchmark, blob_bench_client, bench_prefix):
    """Upload a fresh batch then delete it in one call, timed as one op.

    Upload is inside the timed region on purpose: seeding once and timing only
    the delete would, after round 1, delete already-gone keys. So the cost is
    labelled upload+delete, not a bare delete.
    """
    payload = _payload(_KB)
    # Fresh key set per round (so post-round-1 deletes hit live keys, not
    # already-gone ones), pre-generated outside the timed body.
    rounds, warmup = resolve_rounds()
    batches = iter(
        [
            [f"{bench_prefix}bulk-{b}-{i}.dat" for i in range(_BULK_DELETE_BATCH)]
            for b in range(rounds + warmup)
        ]
    )
    last_keys: list[str] = []

    def _seed_and_delete() -> DeleteObjectsResponse:
        keys = next(batches)
        for k in keys:
            blob_bench_client.upload(k, payload)
        last_keys[:] = keys
        return blob_bench_client.delete_objects([BlobIdentifier(key=k) for k in keys])

    result = timed(benchmark, _seed_and_delete)

    assert result.errors == []  # every object deleted without error
    assert last_keys  # the timed body ran at least once
    # Final batch fully gone — every key, not just the first.
    assert not any(blob_bench_client.does_object_exist(k) for k in last_keys)


@pytest.mark.benchmark(group="blob-sync-concurrent-upload")
@pytest.mark.parametrize("threads", _THREAD_COUNTS)
def test_concurrent_upload(benchmark, blob_bench_client, bench_prefix, threads):
    """A batch of uploads drained by a fixed thread pool."""
    payload = _payload(_KB)
    # Fresh key set per round (reusing names would overwrite the same objects
    # and measure per-key write throttling), pre-generated outside the timed
    # body — mirrors test_concurrent_download.
    rounds, warmup = resolve_rounds(rounds=_CONCURRENT_ROUNDS)
    batches = iter(
        [
            [f"{bench_prefix}cc-upload-{threads}-{r}-{i}.dat" for i in range(_BATCH)]
            for r in range(rounds + warmup)
        ]
    )
    last_keys: list[str] = []

    with ThreadPoolExecutor(max_workers=threads) as pool:

        def _batch() -> int:
            keys = next(batches)
            last_keys[:] = keys
            futures = [pool.submit(blob_bench_client.upload, k, payload) for k in keys]
            return sum(1 for f in futures if f.result(timeout=_OP_TIMEOUT).key)

        completed = timed(benchmark, _batch, rounds=_CONCURRENT_ROUNDS)

    assert completed == _BATCH
    assert last_keys  # the timed body ran at least once
    # Verify every object in the final batch landed with the right bytes.
    assert all(blob_bench_client.download(k).content == payload for k in last_keys)


@pytest.mark.benchmark(group="blob-sync-concurrent-download")
@pytest.mark.parametrize("threads", _THREAD_COUNTS)
def test_concurrent_download(benchmark, blob_bench_client, bench_prefix, threads):
    """A batch of downloads drained by a fixed thread pool."""
    payload = _payload(_KB)
    keys = [f"{bench_prefix}cc-download-{threads}-{i}.dat" for i in range(_BATCH)]
    for k in keys:
        blob_bench_client.upload(k, payload)  # seed — not timed

    with ThreadPoolExecutor(max_workers=threads) as pool:

        def _batch() -> list[bytes]:
            futures = [pool.submit(blob_bench_client.download, k) for k in keys]
            return [f.result(timeout=_OP_TIMEOUT).content for f in futures]

        contents = timed(benchmark, _batch, rounds=_CONCURRENT_ROUNDS)

    assert len(contents) == _BATCH
    assert all(c == payload for c in contents)  # every download round-tripped
