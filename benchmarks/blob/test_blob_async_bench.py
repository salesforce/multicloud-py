"""Asynchronous blob benchmarks.

Serial (one awaited op), concurrent (``asyncio.gather`` under a
``Semaphore`` at 4/8), and directory (whole-tree) bodies. They use
``async_blob_bench_client`` and ``timed()``, so the same tests run against
``benchfake`` or a real backend. Each body is a sync function that drives
its coroutine to completion via ``run_async`` on the shared session loop.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Awaitable, Iterable, List, TypeVar

import pytest
from multicloud.blob.blob_client.async_blob.async_driver import (
    AsyncUploadResponse,
    BlobIdentifier,
    BulkDeleteResponse,
    CopyResponse,
    MultipartPart,
    MultipartUploadResponse,
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

# Page size for the listing benchmarks -- small relative to the seeded count so
# a single page is truncated and the paginate-to-completion loop takes several
# pages.
_LIST_PAGE_MAX = 20

# Multipart upload shape: a 10 MB object uploaded as two 5 MB parts.
_MULTIPART_PART_SIZE = 5 * _MB
_MULTIPART_PARTS = 2

# Number of objects deleted in one batch by the bulk-delete benchmark.
_BULK_DELETE_BATCH = 25

# Concurrency levels for the gather-based group. No c1 baseline: the serial
# group already measures single-op latency, and c4-vs-c8 (healthy: c8 ~= c4/2)
# is the scaling signal.
_CONCURRENCY = [
    pytest.param(4, id="c4"),
    pytest.param(8, id="c8"),
]

# Operations issued per timed batch in the concurrent benchmarks. Fixed so
# throughput is comparable across concurrency levels; payload is small on
# purpose so scheduling overhead dominates the measurement.
_BATCH = 64

# Concurrent sample count, separate from the serial BLOB_BENCH_ROUNDS: the
# c8/c4 ratio is round-count-insensitive, so it defaults lower.
_CONCURRENT_ROUNDS = int(os.environ.get("BLOB_BENCH_CONCURRENT_ROUNDS", "30"))

# Directory-op tree shape: many small files so the measured cost is the
# client's fan-out orchestration, not per-file byte movement.
_DIR_FILE_COUNT = 50
_DIR_FILE_SIZE = _KB


def _payload(size: int) -> bytes:
    return b"x" * size


def _make_local_tree(root: Path, count: int, size: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    data = _payload(size)
    for i in range(count):
        (root / f"file-{i}.dat").write_bytes(data)
    return root


# Serial benchmarks


@pytest.mark.benchmark(group="blob-async-upload")
@pytest.mark.parametrize("size", _SIZES)
def test_upload(benchmark, async_blob_bench_client, bench_prefix, run_async, size):
    """Single upload, timed across payload-size tiers.

    Unique key per round: reusing one name would trip single-object
    write-rate limits and measure throttling, not upload latency.
    """
    payload = _payload(size)
    # Pre-generate one unique key per invocation outside the timed body so key
    # formatting stays out of the sample, mirroring the sync suite. next() on a
    # list iterator is a pointer bump, not a format.
    rounds, warmup = resolve_rounds()
    keys = iter(
        [f"{bench_prefix}upload-{size}-{i}.dat" for i in range(rounds + warmup)]
    )

    def _upload() -> AsyncUploadResponse:
        return run_async(async_blob_bench_client.upload(next(keys), payload))

    result = timed(benchmark, _upload)

    assert result.key.startswith(f"{bench_prefix}upload-{size}-")
    assert result.e_tag is not None
    # Round-trip (untimed): prove the bytes landed, not just a response shape.
    assert run_async(async_blob_bench_client.download(result.key)).content == payload


@pytest.mark.benchmark(group="blob-async-download")
@pytest.mark.parametrize("size", _SIZES)
def test_download(benchmark, async_blob_bench_client, bench_prefix, run_async, size):
    """Single download, timed across payload-size tiers."""
    payload = _payload(size)
    key = f"{bench_prefix}download-{size}.dat"
    run_async(async_blob_bench_client.upload(key, payload))  # seed -- not timed

    result = timed(benchmark, lambda: run_async(async_blob_bench_client.download(key)))

    assert result.key == key
    assert result.content == payload  # round-trip


@pytest.mark.benchmark(group="blob-async-lifecycle")
def test_write_read_delete(benchmark, async_blob_bench_client, bench_prefix, run_async):
    """Full upload -> download -> delete lifecycle, timed as one op."""
    payload = _payload(_KB)
    # Unique key per round (reusing one name would measure single-object write
    # throttling, not lifecycle latency), pre-generated outside the timed body.
    rounds, warmup = resolve_rounds()
    keys = iter([f"{bench_prefix}lifecycle-{i}.dat" for i in range(rounds + warmup)])
    last_key: List[str] = []

    async def _cycle_coro(key: str) -> bytes:
        await async_blob_bench_client.upload(key, payload)
        content = (await async_blob_bench_client.download(key)).content
        await async_blob_bench_client.delete(key)
        return content

    def _cycle() -> bytes:
        key = next(keys)
        last_key[:] = [key]
        return run_async(_cycle_coro(key))

    content = timed(benchmark, _cycle)

    assert content == payload
    assert last_key  # the timed body ran at least once
    assert run_async(async_blob_bench_client.does_object_exist(last_key[0])) is False


@pytest.mark.benchmark(group="blob-async-metadata")
def test_get_metadata(benchmark, async_blob_bench_client, bench_prefix, run_async):
    """HEAD-style metadata lookup, timed."""
    payload = _payload(_MB)
    key = f"{bench_prefix}metadata.dat"
    run_async(
        async_blob_bench_client.upload(
            key, payload, content_type="application/octet-stream"
        )
    )

    meta = timed(
        benchmark, lambda: run_async(async_blob_bench_client.get_metadata(key))
    )

    assert meta.key == key
    assert meta.content_length == len(payload)
    # The content_type set on upload round-tripped through the metadata path.
    assert meta.content_type == "application/octet-stream"


@pytest.mark.benchmark(group="blob-async-list")
def test_list_page(benchmark, async_blob_bench_client, bench_prefix, run_async):
    """Single listing page over a seeded prefix, timed."""
    prefix = f"{bench_prefix}list/"
    for i in range(_LIST_COUNT):
        run_async(async_blob_bench_client.upload(f"{prefix}{i}.dat", _payload(_KB)))

    page = timed(
        benchmark,
        lambda: run_async(
            async_blob_bench_client.list_page(prefix=prefix, max_results=_LIST_PAGE_MAX)
        ),
    )

    keys = {b.key for b in page.blobs}
    assert f"{prefix}0.dat" in keys
    assert len(page.blobs) == _LIST_PAGE_MAX
    assert page.is_truncated is True  # seeded count exceeds one page


@pytest.mark.benchmark(group="blob-async-list-all")
def test_list_all(benchmark, async_blob_bench_client, bench_prefix, run_async):
    """Paginate a prefix to completion, timed as one op."""
    prefix = f"{bench_prefix}listall/"
    for i in range(_LIST_COUNT):
        run_async(async_blob_bench_client.upload(f"{prefix}{i}.dat", _payload(_KB)))

    async def _list_all_coro() -> set:
        keys: set = set()
        token = None
        while True:
            page = await async_blob_bench_client.list_page(
                prefix=prefix, max_results=_LIST_PAGE_MAX, pagination_token=token
            )
            keys.update(b.key for b in page.blobs)
            token = page.next_pagination_token
            if not page.is_truncated or not token:
                break
        return keys

    keys = timed(benchmark, lambda: run_async(_list_all_coro()))

    assert len(keys) == _LIST_COUNT  # every page was walked
    assert f"{prefix}0.dat" in keys


@pytest.mark.benchmark(group="blob-async-copy")
def test_copy(benchmark, async_blob_bench_client, bench_prefix, run_async):
    """Server-side copy to a fresh destination each round, timed."""
    src_key = f"{bench_prefix}copy-src.dat"
    payload = _payload(_KB)
    run_async(async_blob_bench_client.upload(src_key, payload))

    # Unique dest per round (reusing one name would measure single-object write
    # throttling, not copy latency), pre-generated outside the timed body.
    rounds, warmup = resolve_rounds()
    dest_keys = iter(
        [f"{bench_prefix}copy-dst-{i}.dat" for i in range(rounds + warmup)]
    )

    def _copy() -> CopyResponse:
        return run_async(
            async_blob_bench_client.copy(
                src_key,
                dest_bucket=async_blob_bench_client.bucket,
                dest_key=next(dest_keys),
            )
        )

    result = timed(benchmark, _copy)

    assert result.key.startswith(f"{bench_prefix}copy-dst-")
    # Round-trip: the copied bytes are readable at the destination.
    assert run_async(async_blob_bench_client.download(result.key)).content == payload


@pytest.mark.benchmark(group="blob-async-multipart")
def test_multipart_upload(benchmark, async_blob_bench_client, bench_prefix, run_async):
    """Full multipart upload (initiate -> upload parts -> complete), timed."""
    part_payload = _payload(_MULTIPART_PART_SIZE)
    # Unique key per round (reusing one name would measure single-object write
    # throttling), pre-generated outside the timed body.
    rounds, warmup = resolve_rounds()
    keys = iter([f"{bench_prefix}multipart-{i}.dat" for i in range(rounds + warmup)])
    last_key: List[str] = []

    async def _multipart_coro(key: str) -> MultipartUploadResponse:
        mpu = await async_blob_bench_client.initiate_multipart_upload(key)
        # Abort on any failure between initiate and complete: a mid-round error
        # would otherwise orphan billable multipart parts that teardown's
        # list-based scan cannot see (in-progress uploads aren't listable).
        # CancelledError is included because run_async's timeout cancels this
        # coroutine, and cancellation mid-upload leaves the same orphan.
        try:
            parts = [
                await async_blob_bench_client.upload_part(
                    mpu, MultipartPart(part_number=n, content=part_payload)
                )
                for n in range(1, _MULTIPART_PARTS + 1)
            ]
            return await async_blob_bench_client.complete_multipart_upload(mpu, parts)
        except (Exception, asyncio.CancelledError):
            try:
                await async_blob_bench_client.abort_multipart_upload(mpu)
            except Exception:
                pass
            raise

    def _multipart() -> MultipartUploadResponse:
        key = next(keys)
        last_key[:] = [key]
        return run_async(_multipart_coro(key))

    result = timed(benchmark, _multipart)

    # Only e_tag is guaranteed on the completion response; the assembled object
    # is verified by round-trip download (proving it landed under the last key).
    assert result.e_tag is not None
    assert last_key  # the timed body ran at least once
    assert (
        run_async(async_blob_bench_client.download(last_key[0])).content
        == part_payload * _MULTIPART_PARTS
    )


@pytest.mark.benchmark(group="blob-async-bulk-delete")
def test_bulk_delete_lifecycle(
    benchmark, async_blob_bench_client, bench_prefix, run_async
):
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
    last_keys: List[str] = []

    async def _seed_and_delete_coro(keys: List[str]) -> BulkDeleteResponse:
        for k in keys:
            await async_blob_bench_client.upload(k, payload)
        return await async_blob_bench_client.delete_objects(
            [BlobIdentifier(key=k) for k in keys]
        )

    def _seed_and_delete() -> BulkDeleteResponse:
        keys = next(batches)
        last_keys[:] = keys
        return run_async(_seed_and_delete_coro(keys))

    result = timed(benchmark, _seed_and_delete)

    assert result.errors == []  # every object deleted without error
    assert last_keys  # the timed body ran at least once
    # Final batch fully gone -- every key, not just the first.
    assert not any(
        run_async(async_blob_bench_client.does_object_exist(k)) for k in last_keys
    )


# Concurrent benchmarks

_T = TypeVar("_T")


async def _gather(coros: Iterable[Awaitable[_T]]) -> List[_T]:
    """Run ``coros`` concurrently; cancel siblings on the first failure.

    Bare ``asyncio.gather`` raises on the first exception and leaves the
    rest running. This suite reuses one session loop across ``pedantic``
    rounds, so those orphans would overlap the next sample. Cancel them
    and drain with ``return_exceptions=True`` before re-raising.
    """
    tasks = [asyncio.create_task(coro) for coro in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _gather_upload(client, keys: List[str], payload: bytes, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def _one(key: str) -> AsyncUploadResponse:
        async with sem:
            return await client.upload(key, payload)

    return await _gather(_one(k) for k in keys)


async def _gather_download(client, keys: List[str], concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def _one(key: str) -> bytes:
        async with sem:
            return (await client.download(key)).content

    return await _gather(_one(k) for k in keys)


@pytest.mark.benchmark(group="blob-async-concurrent-upload")
@pytest.mark.parametrize("concurrency", _CONCURRENCY)
def test_concurrent_upload(
    benchmark, async_blob_bench_client, bench_prefix, run_async, concurrency
):
    """A batch of uploads fanned out with gather under a bounded semaphore."""
    payload = _payload(_KB)
    # Fresh key set per round (reusing names would overwrite the same objects
    # and measure per-key write throttling), pre-generated outside the timed
    # body -- mirrors test_concurrent_download.
    rounds, warmup = resolve_rounds(rounds=_CONCURRENT_ROUNDS)
    batches = iter(
        [
            [
                f"{bench_prefix}cc-upload-{concurrency}-{r}-{i}.dat"
                for i in range(_BATCH)
            ]
            for r in range(rounds + warmup)
        ]
    )
    last_keys: List[str] = []

    def _batch() -> int:
        keys = next(batches)
        last_keys[:] = keys
        results = run_async(
            _gather_upload(async_blob_bench_client, keys, payload, concurrency)
        )
        return sum(1 for r in results if r.key)

    completed = timed(benchmark, _batch, rounds=_CONCURRENT_ROUNDS)

    assert completed == _BATCH
    assert last_keys  # the timed body ran at least once
    # Verify every object in the final batch landed with the right bytes.
    contents = run_async(
        _gather_download(async_blob_bench_client, last_keys, concurrency)
    )
    assert all(c == payload for c in contents)


@pytest.mark.benchmark(group="blob-async-concurrent-download")
@pytest.mark.parametrize("concurrency", _CONCURRENCY)
def test_concurrent_download(
    benchmark, async_blob_bench_client, bench_prefix, run_async, concurrency
):
    """A batch of downloads fanned out with gather under a bounded semaphore."""
    payload = _payload(_KB)
    keys = [f"{bench_prefix}cc-download-{concurrency}-{i}.dat" for i in range(_BATCH)]
    run_async(_gather_upload(async_blob_bench_client, keys, payload, concurrency))

    def _batch() -> list:
        return run_async(_gather_download(async_blob_bench_client, keys, concurrency))

    contents = timed(benchmark, _batch, rounds=_CONCURRENT_ROUNDS)

    assert len(contents) == _BATCH
    assert all(c == payload for c in contents)  # every download round-tripped


# Directory benchmarks


@pytest.mark.benchmark(group="blob-async-upload-dir")
def test_upload_directory(
    benchmark, async_blob_bench_client, bench_prefix, run_async, tmp_path
):
    """Whole-tree upload, timed as one op."""
    source_dir = _make_local_tree(tmp_path / "src", _DIR_FILE_COUNT, _DIR_FILE_SIZE)
    rounds, warmup = resolve_rounds()
    prefixes = iter([f"{bench_prefix}updir-{i}/" for i in range(rounds + warmup)])
    last_prefix: List[str] = []

    def _upload_dir():
        prefix = next(prefixes)
        last_prefix[:] = [prefix]
        return run_async(
            async_blob_bench_client.upload_directory(str(source_dir), prefix=prefix)
        )

    result = timed(benchmark, _upload_dir)

    assert result.total_files == _DIR_FILE_COUNT
    assert result.successful_uploads == _DIR_FILE_COUNT
    assert result.has_failures is False
    assert last_prefix  # the timed body ran at least once
    # Round-trip: every file is listable under the final prefix.
    page = run_async(
        async_blob_bench_client.list_page(
            prefix=last_prefix[0], max_results=_DIR_FILE_COUNT + 1
        )
    )
    assert len(page.blobs) == _DIR_FILE_COUNT


@pytest.mark.benchmark(group="blob-async-download-dir")
def test_download_directory(
    benchmark, async_blob_bench_client, bench_prefix, run_async, tmp_path
):
    """Whole-tree download, timed as one op."""
    source_dir = _make_local_tree(tmp_path / "src", _DIR_FILE_COUNT, _DIR_FILE_SIZE)
    prefix = f"{bench_prefix}dldir/"
    run_async(
        async_blob_bench_client.upload_directory(str(source_dir), prefix=prefix)
    )  # seed -- not timed

    # Fresh local destination per round so a re-download is real work, not a
    # no-op over already-present files.
    rounds, warmup = resolve_rounds()
    dests = iter([tmp_path / f"dl-{i}" for i in range(rounds + warmup)])
    last_dest: List[Path] = []

    def _download_dir():
        dest = next(dests)
        last_dest[:] = [dest]
        return run_async(async_blob_bench_client.download_directory(prefix, str(dest)))

    result = timed(benchmark, _download_dir)

    assert result.total_files == _DIR_FILE_COUNT
    assert result.successful_downloads == _DIR_FILE_COUNT
    assert result.has_failures is False
    assert last_dest  # the timed body ran at least once
    # Round-trip: a downloaded file exists with the seeded byte count.
    written = list(last_dest[0].rglob("*.dat"))
    assert len(written) == _DIR_FILE_COUNT
    assert written[0].stat().st_size == _DIR_FILE_SIZE


@pytest.mark.benchmark(group="blob-async-delete-dir")
def test_delete_directory(
    benchmark, async_blob_bench_client, bench_prefix, run_async, tmp_path
):
    """Seed a fresh tree then delete it by prefix, timed as one op."""
    source_dir = _make_local_tree(tmp_path / "src", _DIR_FILE_COUNT, _DIR_FILE_SIZE)
    rounds, warmup = resolve_rounds()
    prefixes = iter([f"{bench_prefix}deldir-{i}/" for i in range(rounds + warmup)])
    last_prefix: List[str] = []

    async def _seed_and_delete_coro(prefix: str) -> None:
        await async_blob_bench_client.upload_directory(str(source_dir), prefix=prefix)
        await async_blob_bench_client.delete_directory(prefix)

    def _delete_dir() -> str:
        prefix = next(prefixes)
        last_prefix[:] = [prefix]
        run_async(_seed_and_delete_coro(prefix))
        return prefix

    timed(benchmark, _delete_dir)

    assert last_prefix  # the timed body ran at least once
    # Final prefix fully gone: nothing lists under it.
    page = run_async(
        async_blob_bench_client.list_page(prefix=last_prefix[0], max_results=1)
    )
    assert page.blobs == []
