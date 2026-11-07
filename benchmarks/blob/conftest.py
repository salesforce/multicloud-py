"""Fixtures for blob benchmarks (sync + async).

Importing this module registers ``"benchfake"`` in both the sync and async
registries. Sync bodies depend on ``blob_bench_client``; async bodies on
``async_blob_bench_client`` driven through ``run_async``. Neither depends on a
provider name — by default both fixtures are the in-memory store (no
credentials). Point the same bodies at a real backend with:

- ``BLOB_BENCH_PROVIDER`` — e.g. ``aws`` (default ``benchfake``)
- ``BLOB_BENCH_BUCKET`` — required unless benchfake
- ``BLOB_BENCH_REGION`` — optional

Credentials come from ambient discovery, never from these fixtures. On a
real backend, each test gets a unique ``bench-<uuid>/`` key prefix and
teardown deletes only objects under that run-scoped prefix, so a shared
bucket's pre-existing objects are never touched.

Async bodies are sync functions that drive their coroutine to completion
via ``run_async`` on one session-scoped event loop (``bench_loop``), reused
so loop setup/teardown stays out of the measured region.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
import warnings
from typing import Any, Awaitable, Callable

import pytest
from multicloud.blob.blob_client.async_blob.async_client import AsyncBucketClient
from multicloud.blob.blob_client.async_blob.async_driver import (
    BlobIdentifier as AsyncBlobIdentifier,
)
from multicloud.blob.blob_client.client import BucketClient
from multicloud.blob.blob_client.driver import BlobIdentifier

from benchmarks._bench import resolve_rounds

# Import for the registration side effect (register "benchfake" in both the
# sync and async registries).
from benchmarks.blob import _async_fakes, _sync_fakes  # noqa: F401

_LOG = logging.getLogger(__name__)

# Every benchmark key is namespaced under a unique ``bench-<uuid>/`` prefix
# (the ``bench_prefix`` fixture) so real-backend teardown only ever deletes
# objects the current test created — never a shared bucket's existing objects.
_BENCH_KEY_ROOT = "bench-"

_DEFAULT_PROVIDER = "benchfake"
_DEFAULT_BUCKET = "bench-bucket"

# Per-round ceiling on run_async's awaited work so a stalled live call fails its
# round instead of hanging the run. <=0 waits unbounded; benchfake never trips it.
_raw_op_timeout = float(os.environ.get("BLOB_BENCH_OP_TIMEOUT", "120"))
_OP_TIMEOUT = _raw_op_timeout if _raw_op_timeout > 0 else None


def _bench_provider() -> str:
    return os.environ.get("BLOB_BENCH_PROVIDER", _DEFAULT_PROVIDER)


def _make_sync_client() -> BucketClient:
    provider = _bench_provider()
    if provider == _DEFAULT_PROVIDER:
        return BucketClient(provider=_DEFAULT_PROVIDER, bucket=_DEFAULT_BUCKET)
    bucket = os.environ["BLOB_BENCH_BUCKET"]
    region = os.environ.get("BLOB_BENCH_REGION")
    return BucketClient(provider=provider, bucket=bucket, region=region)


def _make_async_client() -> AsyncBucketClient:
    provider = _bench_provider()
    if provider == _DEFAULT_PROVIDER:
        return AsyncBucketClient(provider=_DEFAULT_PROVIDER, bucket=_DEFAULT_BUCKET)
    bucket = os.environ["BLOB_BENCH_BUCKET"]
    region = os.environ.get("BLOB_BENCH_REGION")
    return AsyncBucketClient(provider=provider, bucket=bucket, region=region)


def _cleanup_bench_objects(client: BucketClient, prefix: str) -> None:
    """Bulk-delete every object under this run's prefix, one listing page per call.

    Failures are logged, never raised or swallowed — a silently skipped delete is
    billable residue on a disposable bucket.
    """
    token = None
    while True:
        page = client.list_page(prefix=prefix, max_results=1000, pagination_token=token)
        identifiers = [BlobIdentifier(key=blob.key) for blob in page.blobs]
        if identifiers:
            try:
                response = client.delete_objects(identifiers)
            except Exception:
                # Log, don't raise: teardown must not mask the test result, but
                # it also must not silently leave objects behind.
                _LOG.exception(
                    "Benchmark teardown: batch delete failed under prefix %s; "
                    "%d object(s) may remain",
                    prefix,
                    len(identifiers),
                )
            else:
                for err in response.errors:
                    _LOG.warning(
                        "Benchmark teardown: could not delete %s: %s %s",
                        err.key,
                        err.code,
                        err.message,
                    )
        token = page.next_pagination_token
        if not page.is_truncated or not token:
            break


async def _cleanup_bench_objects_async(client: AsyncBucketClient, prefix: str) -> None:
    """Bulk-delete every object under this run's prefix, one listing page per call.

    Failures (listing or delete) are logged, never raised or swallowed — a
    silently skipped cleanup is billable residue on a disposable bucket.
    """
    token = None
    while True:
        try:
            page = await client.list_page(
                prefix=prefix, max_results=1000, pagination_token=token
            )
        except Exception:
            _LOG.exception(
                "Benchmark teardown: listing failed under prefix %s; "
                "remaining objects may not be deleted",
                prefix,
            )
            break
        identifiers = [AsyncBlobIdentifier(key=blob.key) for blob in page.blobs]
        if identifiers:
            try:
                response = await client.delete_objects(identifiers)
            except Exception:
                # Log, don't raise: teardown must not mask the test result, but
                # it also must not silently leave objects behind.
                _LOG.exception(
                    "Benchmark teardown: batch delete failed under prefix %s; "
                    "%d object(s) may remain",
                    prefix,
                    len(identifiers),
                )
            else:
                for err in response.errors:
                    _LOG.warning(
                        "Benchmark teardown: could not delete %s: %s %s",
                        err.key,
                        err.code,
                        err.message,
                    )
        token = page.next_pagination_token
        if not page.is_truncated or not token:
            break


@pytest.fixture
def bench_prefix() -> str:
    """Unique per-test key prefix (``bench-<uuid>/``).

    Every key a benchmark body creates is namespaced under this, so
    real-backend teardown deletes only what the test created.
    """
    return f"{_BENCH_KEY_ROOT}{uuid.uuid4().hex}/"


@pytest.fixture
def blob_bench_client(bench_prefix: str) -> BucketClient:
    """BucketClient for benchmark bodies.

    ``benchfake`` unless ``BLOB_BENCH_PROVIDER`` is set. Real backends are
    cleaned up in teardown, scoped to this test's ``bench_prefix``.
    """
    is_real = _bench_provider() != _DEFAULT_PROVIDER
    if is_real and resolve_rounds()[1] == 0:
        # On a live backend the first sample pays one-time TLS/DNS/pool-fill/
        # credential-resolution cost that lands in the recorded tail (p99/max).
        # Enforce the doc'd guidance rather than only stating it.
        warnings.warn(
            f"Benchmarking live provider {_bench_provider()!r} with "
            "warmup_rounds=0: cold-start cost will pollute the recorded tail. "
            "Set BLOB_BENCH_WARMUP=1 (or 2) for a live run.",
            stacklevel=2,
        )
    client = _make_sync_client()
    try:
        yield client
    finally:
        # Guard cleanup so a teardown listing/delete error can't skip close()
        # and leak the underlying provider client / connection pool.
        try:
            if is_real:
                _cleanup_bench_objects(client, bench_prefix)
        finally:
            client.close()


@pytest.fixture(scope="session")
def bench_loop() -> asyncio.AbstractEventLoop:
    """One event loop for the whole session, reused across all samples."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture
def run_async(
    bench_loop: asyncio.AbstractEventLoop,
) -> Callable[[Awaitable[Any]], Any]:
    """Drive a coroutine to completion on the shared session loop."""

    def _run(coro: Awaitable[Any]) -> Any:
        if _OP_TIMEOUT is not None:
            coro = asyncio.wait_for(coro, _OP_TIMEOUT)
        return bench_loop.run_until_complete(coro)

    return _run


@pytest.fixture
def async_blob_bench_client(
    bench_prefix: str, bench_loop: asyncio.AbstractEventLoop
) -> AsyncBucketClient:
    """AsyncBucketClient for benchmark bodies.

    ``benchfake`` unless ``BLOB_BENCH_PROVIDER`` is set. Real backends are
    cleaned up in teardown, scoped to this test's ``bench_prefix``.
    """
    is_real = _bench_provider() != _DEFAULT_PROVIDER
    if is_real and resolve_rounds()[1] == 0:
        # On a live backend the first sample pays one-time TLS/DNS/pool-fill/
        # credential-resolution cost that lands in the recorded tail (p99/max).
        # Enforce the doc'd guidance rather than only stating it.
        warnings.warn(
            f"Benchmarking live provider {_bench_provider()!r} with "
            "warmup_rounds=0: cold-start cost will pollute the recorded tail. "
            "Set BLOB_BENCH_WARMUP=1 (or 2) for a live run.",
            stacklevel=2,
        )
    client = _make_async_client()
    try:
        yield client
    finally:
        # Guard cleanup so a teardown listing/delete error can't skip close()
        # and leak the underlying provider client / connection pool. Both run
        # on the shared loop since they are coroutines.
        try:
            if is_real:
                bench_loop.run_until_complete(
                    _cleanup_bench_objects_async(client, bench_prefix)
                )
        finally:
            bench_loop.run_until_complete(client.close())
