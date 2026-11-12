"""Pubsub benchmarks: serial, consume-side batch, concurrency, pipeline, nack.

Every body runs through ``timed`` (one call per sample) so ``p99.py`` reads a
real tail. ``send_acks``/``send_nack`` only enqueue into the ack Batcher — the
broker RPC is forced by ``close()`` -> ``shutdown_and_drain()``, so the ack/nack
bodies time a fresh client's ``send_* + close()``, and every receive body
``send_ack``s each message (cheap enqueue) so a live backend won't redeliver into
a later round.

Payload tiers are 1KB/64KB/256KB (SNS/SQS cap at 256 KiB). Defaults target the
benchfake tier; per-group round counts come from ``PUBSUB_BENCH_{ROUNDS,
BATCH_ROUNDS,CONCURRENT_ROUNDS,PIPELINE_ROUNDS,WARMUP}``. See ``conftest.py`` for
the live tier's shared-queue seeding.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable, List, Tuple

import pytest
from multicloud.pubsub.pubsub_client.client import SubscriptionClient, TopicClient
from multicloud.pubsub.pubsub_client.driver import Message

from benchmarks._bench import resolve_rounds, timed

_ENV_PREFIX = "PUBSUB_BENCH"
_KB = 1024

# Payload tiers for publish (256 KiB is the largest all providers accept).
_SIZES = [("1kb", _KB), ("64kb", 64 * _KB), ("256kb", 256 * _KB)]

# Consume-side batch dimension for test_ack_batch. Stops at 100 on purpose: that
# body serially receives every round's ids un-acked on one client up front, and a
# larger tier (e.g. 1000) can't be drained on a backend whose non-blocking receive
# returns empty under backlog.
_ACK_BATCH_SIZES = [1, 10, 100]
# No n1: single-message receive is already covered by test_receive. These sizes
# show how much prefetch amortizes the per-message cost across a drain.
_DRAIN_SIZES = [10, 100]

# Concurrency / pipeline fan-out: messages per timed sample, split across workers.
_TOTAL = 64
_WORKERS = [1, 4, 8]

# Per-group round counts. Batch/concurrent/pipeline samples each do many ops, so
# they sample fewer times than the serial group to keep wall-clock (and live
# billing) reasonable.
_BATCH_ROUNDS = int(os.environ.get("PUBSUB_BENCH_BATCH_ROUNDS", "50"))
_CONCURRENT_ROUNDS = int(os.environ.get("PUBSUB_BENCH_CONCURRENT_ROUNDS", "30"))
_PIPELINE_ROUNDS = int(os.environ.get("PUBSUB_BENCH_PIPELINE_ROUNDS", "20"))

# Bounds each concurrent/pipeline future's wait so a stalled round fails in
# bounded time. Callers shut the pool down with wait=False so this TimeoutError
# propagates instead of re-blocking on the pool's exit join; a worker wedged in a
# blocking call still lingers until interpreter exit (pure threading can't kill
# it -- a CI job timeout is the backstop). 0/negative waits unbounded. Serial
# receive bodies are not covered -- see the README.
_raw_op_timeout = float(os.environ.get("PUBSUB_BENCH_OP_TIMEOUT", "120"))
_OP_TIMEOUT = _raw_op_timeout if _raw_op_timeout > 0 else None


def _drain_and_ack(client: SubscriptionClient, n: int) -> List[Message]:
    """Receive ``n`` messages serially, then ack them all (ack keeps live clean)."""
    messages = [client.receive() for _ in range(n)]
    client.send_acks([m.ack_id for m in messages])
    return messages


def _fan_out(pool: ThreadPoolExecutor, fn: Callable[[], object], total: int) -> list:
    """Submit ``fn`` ``total`` times on ``pool`` and collect every result."""
    futures = [pool.submit(fn) for _ in range(total)]
    return [f.result(timeout=_OP_TIMEOUT) for f in futures]


def _run_pipeline(
    topic_client: TopicClient,
    sub_client: SubscriptionClient,
    workers: int,
    total: int,
    body: bytes,
) -> int:
    """Publish and consume ``total`` messages concurrently; return consumed count.

    A shared counter reserves each receive slot, so sends and receives stay
    balanced (exactly ``total`` each) and no pre-seed is needed — a consumer that
    reserves the last slot blocks until its matching publish lands. The count is
    only meaningful because ``pipeline_clients`` gives this body a queue nothing
    else writes to.
    """
    consumed = 0
    lock = Lock()

    def produce(count: int) -> None:
        message = Message.builder().with_body(body).build()
        for _ in range(count):
            topic_client.send(message)

    def consume() -> None:
        nonlocal consumed
        while True:
            with lock:
                if consumed >= total:
                    return
                consumed += 1
            message = sub_client.receive()
            sub_client.send_ack(message.ack_id)

    # Split total across producers (front-load the remainder).
    per_producer = [total // workers] * workers
    for i in range(total % workers):
        per_producer[i] += 1

    # shutdown(wait=False) on the way out so a timed-out round's TimeoutError
    # propagates instead of re-blocking on the pool's exit join.
    pool = ThreadPoolExecutor(max_workers=workers * 2)
    try:
        futures = [pool.submit(produce, count) for count in per_producer]
        futures += [pool.submit(consume) for _ in range(workers)]
        for future in futures:
            future.result(timeout=_OP_TIMEOUT)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return consumed


@pytest.mark.benchmark(group="send")
@pytest.mark.parametrize("label,size", _SIZES, ids=[s[0] for s in _SIZES])
def test_send(
    benchmark, topic_bench_client, is_benchfake_topic, label: str, size: int
) -> None:
    """Serial publish latency across payload-size tiers."""
    message = Message.builder().with_body(b"x" * size).build()

    timed(benchmark, topic_bench_client.send, message, env_prefix=_ENV_PREFIX)

    if is_benchfake_topic:
        # send() blocks on the Batcher future, so do_send_batch has run at least
        # once by now. Use >= 1 (not >= rounds) so this holds under
        # --benchmark-disable too, where pedantic invokes the body a single time.
        assert topic_bench_client._topic.sent_count >= 1


@pytest.mark.benchmark(group="receive")
def test_receive(benchmark, sub_bench_client, seed) -> None:
    """Serial receive latency (through prefetch + adaptive batch sizing)."""
    rounds, warmup = resolve_rounds(env_prefix=_ENV_PREFIX)
    seed(rounds + warmup)

    def _receive_and_ack() -> Message:
        message = sub_bench_client.receive()
        sub_bench_client.send_ack(message.ack_id)
        return message

    message = timed(benchmark, _receive_and_ack, env_prefix=_ENV_PREFIX)

    # ack_id may be 0, so test identity against None, not truthiness.
    assert message.body is not None
    assert message.ack_id is not None


@pytest.mark.benchmark(group="ack_batch")
@pytest.mark.parametrize("n", _ACK_BATCH_SIZES, ids=[f"n{n}" for n in _ACK_BATCH_SIZES])
def test_ack_batch(
    benchmark, sub_bench_client, sub_client_factory, seed, is_benchfake, n: int
) -> None:
    """Batched-ack broker round-trip vs. batch size (times ``send_acks`` + ``close()``).

    Each round acks a distinct id batch on a fresh client (re-acking one batch
    would delete nothing after round 1); ``close()`` is the flush barrier.
    """
    rounds, warmup = resolve_rounds(rounds=_BATCH_ROUNDS, env_prefix=_ENV_PREFIX)
    seed(n * (rounds + warmup))

    # Build one client and receive one distinct id batch per invocation up front
    # (untimed); pedantic's per-round setup hook cannot coexist with warmup_rounds.
    # Constructing clients here (registry lookup, driver + batcher + pool objects)
    # keeps that overhead out of the timed body -- otherwise, on benchfake, it
    # dwarfs the ack itself. The timed body is then just send_acks + close().
    clients = iter([sub_client_factory() for _ in range(rounds + warmup)])
    ack_id_batches = iter(
        [
            [sub_bench_client.receive().ack_id for _ in range(n)]
            for _ in range(rounds + warmup)
        ]
    )

    def _ack_and_flush() -> SubscriptionClient:
        client = next(clients)
        client.send_acks(next(ack_id_batches))
        client.close()
        return client

    last_client = timed(
        benchmark, _ack_and_flush, rounds=_BATCH_ROUNDS, env_prefix=_ENV_PREFIX
    )

    if is_benchfake:
        # close() drained the ack Batcher, so do_send_acks ran on the client used
        # last (>= 1 holds under --benchmark-disable's single invocation too).
        assert last_client._subscription.ack_count >= 1


@pytest.mark.benchmark(group="drain")
@pytest.mark.parametrize("n", _DRAIN_SIZES, ids=[f"n{n}" for n in _DRAIN_SIZES])
def test_receive_drain(benchmark, sub_bench_client, seed, n: int) -> None:
    """Cost of draining ``n`` messages per call (adaptive batch sizing at work)."""
    warmup = resolve_rounds(rounds=_BATCH_ROUNDS, env_prefix=_ENV_PREFIX)[1]
    seed(n * (_BATCH_ROUNDS + warmup))

    messages = timed(
        benchmark,
        _drain_and_ack,
        sub_bench_client,
        n,
        rounds=_BATCH_ROUNDS,
        env_prefix=_ENV_PREFIX,
    )

    assert len(messages) == n
    assert all(m.body is not None and m.ack_id is not None for m in messages)


@pytest.mark.benchmark(group="send_concurrent")
@pytest.mark.parametrize("workers", _WORKERS, ids=[f"w{w}" for w in _WORKERS])
def test_send_concurrent(
    benchmark, topic_bench_client, is_benchfake_topic, workers: int
) -> None:
    """Publish throughput under concurrent senders sharing one client/Batcher."""
    message = Message.builder().with_body(b"x" * _KB).build()

    def send_one() -> None:
        topic_bench_client.send(message)

    # Pool is created OUTSIDE the timed body so thread-spawn cost is not counted.
    # shutdown(wait=False) on the way out so a timed-out round's TimeoutError
    # propagates instead of re-blocking on the pool's exit join.
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        results = timed(
            benchmark,
            _fan_out,
            pool,
            send_one,
            _TOTAL,
            rounds=_CONCURRENT_ROUNDS,
            env_prefix=_ENV_PREFIX,
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    assert len(results) == _TOTAL
    if is_benchfake_topic:
        # Prove the publishes landed, not just that the futures returned.
        assert topic_bench_client._topic.sent_count >= _TOTAL


@pytest.mark.benchmark(group="receive_concurrent")
@pytest.mark.parametrize("workers", _WORKERS, ids=[f"w{w}" for w in _WORKERS])
def test_receive_concurrent(benchmark, sub_bench_client, seed, workers: int) -> None:
    """Receive throughput under concurrent consumers sharing one client/queue."""
    warmup = resolve_rounds(rounds=_CONCURRENT_ROUNDS, env_prefix=_ENV_PREFIX)[1]
    seed(_TOTAL * (_CONCURRENT_ROUNDS + warmup))

    def receive_one() -> Message:
        # Ack on receipt so unacked messages don't redeliver on live; send_ack
        # takes the Batcher lock, so concurrent callers on one client are safe.
        message = sub_bench_client.receive()
        sub_bench_client.send_ack(message.ack_id)
        return message

    # shutdown(wait=False) on the way out so a timed-out round's TimeoutError
    # propagates instead of re-blocking on the pool's exit join.
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        messages = timed(
            benchmark,
            _fan_out,
            pool,
            receive_one,
            _TOTAL,
            rounds=_CONCURRENT_ROUNDS,
            env_prefix=_ENV_PREFIX,
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    assert len(messages) == _TOTAL
    assert all(m.ack_id is not None for m in messages)


@pytest.mark.benchmark(group="pipeline")
@pytest.mark.parametrize("workers", _WORKERS, ids=[f"w{w}" for w in _WORKERS])
def test_pipeline(benchmark, pipeline_clients, workers: int) -> None:
    """End-to-end throughput (``workers`` producers + consumers): msgs/sec =
    ``_TOTAL`` / time. ``pipeline_clients`` gives this body its own queue (a fresh
    SQS queue on live), so every receive is one of its publishes; on live the
    fixture also fails the test if any message is left behind."""
    topic_client, sub_client = pipeline_clients
    body = b"x" * _KB

    consumed = timed(
        benchmark,
        _run_pipeline,
        topic_client,
        sub_client,
        workers,
        _TOTAL,
        body,
        rounds=_PIPELINE_ROUNDS,
        env_prefix=_ENV_PREFIX,
    )

    assert consumed == _TOTAL


@pytest.mark.benchmark(group="nack")
def test_nack(benchmark, sub_client_factory, seed, is_benchfake) -> None:
    """Full receive -> nack -> ``close()`` cycle on a fresh client (nack reaches
    the broker only at close; the redelivery round-trip is realized only on live)."""
    rounds, warmup = resolve_rounds(env_prefix=_ENV_PREFIX)
    seed(rounds + warmup)

    # Build one client per invocation up front (untimed); the timed body is then
    # just receive + send_nack + close(), not client construction.
    clients = iter([sub_client_factory() for _ in range(rounds + warmup)])

    def _receive_nack_flush() -> Tuple[SubscriptionClient, Message]:
        client = next(clients)
        message = client.receive()
        client.send_nack(message.ack_id)
        client.close()
        return client, message

    last_client, message = timed(benchmark, _receive_nack_flush, env_prefix=_ENV_PREFIX)

    assert last_client.can_nack()
    assert message.body is not None
    assert message.ack_id is not None
    if is_benchfake:
        # close() drained the ack Batcher, so do_send_nacks ran on the client used
        # last (>= 1 holds under --benchmark-disable's single invocation too).
        assert last_client._subscription.nack_count >= 1
