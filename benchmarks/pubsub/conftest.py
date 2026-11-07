"""Fixtures for the pubsub benchmarks. Two tiers, both defaulting to ``benchfake``.

* **benchfake** (default, network-free, $0) — the in-memory drivers in ``_fakes``.
  The fake subscription fabricates messages, so receive-side bodies never block
  and need no seeding; the pipeline body instead uses the ``benchqueue``
  shared-queue driver so it is genuinely end-to-end.
* **live** — real providers, selected per side. An SQS queue is both a send and
  receive endpoint, so pointing the topic and subscription at one queue
  (``PUBSUB_BENCH_QUEUE``) gives the receive-side bodies a self-contained backlog.

The ``seed`` fixture fills that backlog on live (a no-op on benchfake). Each live
round is billable, so keep the ``PUBSUB_BENCH_*_ROUNDS`` counts low. Credentials
come from ambient discovery, never these fixtures.
"""

from __future__ import annotations

import os
import uuid
import warnings
from contextlib import ExitStack
from typing import Callable, Dict, List, Tuple

import pytest
from multicloud.pubsub.pubsub_client.client import SubscriptionClient, TopicClient
from multicloud.pubsub.pubsub_client.driver import Message

from benchmarks._bench import resolve_rounds

# Import for the registration side effect: it runs the register decorators for
# both fake providers, so get("benchfake")/get("benchqueue") resolve without
# entry points.
from benchmarks.pubsub import _fakes  # noqa: F401

_BENCHFAKE = "benchfake"
# Shared in-memory queue provider used by the end-to-end pipeline body.
_BENCHQUEUE = "benchqueue"
_ENV_PREFIX = "PUBSUB_BENCH"


def _topic_provider() -> str:
    return os.environ.get("PUBSUB_BENCH_TOPIC_PROVIDER", _BENCHFAKE)


def _sub_provider() -> str:
    return os.environ.get("PUBSUB_BENCH_SUB_PROVIDER", _BENCHFAKE)


def _is_live() -> bool:
    # The subscription tier decides whether receive-side bodies need seeding: a
    # benchfake subscription fabricates messages, a live one must be pre-filled.
    return _sub_provider() != _BENCHFAKE


def _topic_name() -> str:
    # SNS publishes to a distinct topic ARN (PUBSUB_BENCH_TOPIC); the SQS queue
    # the subscription drains is a separate resource. Only the SQS-both-endpoints
    # model shares one name, so let a shared queue take precedence just there.
    if _topic_provider() == "awssns":
        return os.environ.get("PUBSUB_BENCH_TOPIC", "bench-topic")
    return os.environ.get("PUBSUB_BENCH_QUEUE") or os.environ.get(
        "PUBSUB_BENCH_TOPIC", "bench-topic"
    )


def _subscription_name() -> str:
    return os.environ.get("PUBSUB_BENCH_QUEUE") or os.environ.get(
        "PUBSUB_BENCH_SUBSCRIPTION", "bench-sub"
    )


def _region_kwargs() -> Dict[str, str]:
    region = os.environ.get("PUBSUB_BENCH_REGION")
    return {"region": region} if region is not None else {}


def _seed_body_size() -> int:
    return int(os.environ.get("PUBSUB_BENCH_BODY_SIZE", "1024"))


def _warn_if_live_without_warmup() -> None:
    # A live backend pays one-time costs (TLS, DNS, pool fill, credentials) on
    # the first call; without warmup those land in the recorded tail (p99/max).
    if _is_live() and resolve_rounds(env_prefix=_ENV_PREFIX)[1] == 0:
        warnings.warn(
            f"Benchmarking live pubsub (subscription provider {_sub_provider()!r}) "
            "with PUBSUB_BENCH_WARMUP=0: cold-start cost will pollute the "
            "recorded tail. Set PUBSUB_BENCH_WARMUP=1 (or 2) for a live run.",
            stacklevel=2,
        )


def _make_topic_client() -> TopicClient:
    return TopicClient(
        provider=_topic_provider(), topic_name=_topic_name(), **_region_kwargs()
    )


@pytest.fixture
def is_benchfake() -> bool:
    """True when the subscription side runs on the in-memory fake.

    Single source of truth for the "am I on the fake?" check the receive-side
    bodies use to gate their fake-only assertions — derived from the same
    provider resolution the fixtures key off, so it can't drift.
    """
    return not _is_live()


@pytest.fixture
def is_benchfake_topic() -> bool:
    """True when the topic side runs on the in-memory fake (see ``is_benchfake``)."""
    return _topic_provider() == _BENCHFAKE


@pytest.fixture
def topic_bench_client() -> TopicClient:
    """A ``TopicClient`` for the configured topic provider (benchfake default)."""
    _warn_if_live_without_warmup()
    client = _make_topic_client()
    try:
        yield client
    finally:
        client.close()


def _make_sub_client() -> SubscriptionClient:
    return SubscriptionClient(
        provider=_sub_provider(),
        subscription_name=_subscription_name(),
        **_region_kwargs(),
    )


@pytest.fixture
def sub_bench_client() -> SubscriptionClient:
    """A ``SubscriptionClient`` for the configured subscription provider."""
    _warn_if_live_without_warmup()
    client = _make_sub_client()
    try:
        yield client
    finally:
        client.close()


class _SubClientFactory:
    """Builds and tracks fresh ``SubscriptionClient``s for ``close()``-timing bodies.

    ``close()`` is the only public ack-Batcher flush barrier but also shuts the
    client down, so those bodies need a new client each round. Teardown closes any
    left open (``close()`` is idempotent).
    """

    def __init__(self) -> None:
        self.clients: List[SubscriptionClient] = []

    def __call__(self) -> SubscriptionClient:
        client = _make_sub_client()
        self.clients.append(client)
        return client


@pytest.fixture
def sub_client_factory() -> _SubClientFactory:
    """Factory for per-round ``SubscriptionClient`` instances (see class doc)."""
    _warn_if_live_without_warmup()
    factory = _SubClientFactory()
    try:
        yield factory
    finally:
        for client in factory.clients:
            client.close()


@pytest.fixture
def pipeline_clients() -> Tuple[TopicClient, SubscriptionClient]:
    """A ``(topic, subscription)`` pair sharing one backend, for the pipeline body.

    Live: the normal per-side providers on the shared queue. Benchfake: both on
    the ``benchqueue`` shared-queue provider keyed to one unique name, so a publish
    is genuinely drained by a receive (the fabricating fake could not be).
    """
    _warn_if_live_without_warmup()
    # ExitStack closes the topic even if the subscription's construction raises.
    with ExitStack() as stack:
        if _is_live():
            topic = stack.enter_context(_make_topic_client())
            subscription = stack.enter_context(_make_sub_client())
        else:
            name = f"bench-pipeline-{uuid.uuid4().hex}"
            topic = stack.enter_context(
                TopicClient(provider=_BENCHQUEUE, topic_name=name)
            )
            subscription = stack.enter_context(
                SubscriptionClient(provider=_BENCHQUEUE, subscription_name=name)
            )
        yield topic, subscription


@pytest.fixture
def seed() -> Callable[[int], None]:
    """Return ``seed(n)``: ensure ``n`` messages are waiting to be received.

    A no-op on benchfake (the fake fabricates messages). On live it publishes ``n``
    into the shared queue before the timed region; closing the topic client flushes
    the Batcher, so all ``n`` land before the body runs.
    """

    def _seed(n: int) -> None:
        if not _is_live() or n <= 0:
            return
        body = b"x" * _seed_body_size()
        with _make_topic_client() as client:
            for _ in range(n):
                client.send(Message.builder().with_body(body).build())

    return _seed
