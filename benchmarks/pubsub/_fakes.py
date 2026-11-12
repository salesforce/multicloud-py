"""In-memory pubsub drivers for the benchmarks.

They isolate the client -> driver -> Batcher path (no credentials or network) so
a slower result here means that path got slower, not the broker. Two providers:

* ``"benchfake"`` -- counts sends/acks/nacks and *fabricates* messages on receive,
  so ``receive`` never blocks on a publisher. The default for one-sided bodies.
* ``"benchqueue"`` -- a topic and subscription sharing one in-memory queue keyed
  by name: send enqueues, receive dequeues (and block-polls when empty). A publish
  is really drained by a receive, so the pipeline body is genuinely end-to-end.
"""

from __future__ import annotations

import itertools
from collections import deque
from threading import Condition, Lock
from typing import Any, Deque, Dict, List

from multicloud.common.exceptions import SubstrateSdkException, UnknownException
from multicloud.pubsub.pubsub_client.driver import (
    SUBSCRIPTION_REGISTRY,
    TOPIC_REGISTRY,
    AckID,
    AckInfo,
    BaseSubscription,
    BaseTopic,
    GetAttributeResult,
    Message,
)


@TOPIC_REGISTRY.register("benchfake")
class BenchTopic(BaseTopic):
    """In-memory topic for ``benchfake``: ``do_send_batch`` counts, no network."""

    provider_id = "benchfake"

    # Generous ceiling so every payload tier validates; real providers cap lower.
    _MAX_MESSAGE_SIZE = 10 * 1024 * 1024

    def __init__(self, *, topic_name: str = "bench-topic", **kwargs: Any) -> None:
        super().__init__(topic_name=topic_name, **kwargs)
        # Observable side effects for benchfake-tier assertions.
        self.sent_count = 0
        self.send_batch_calls = 0
        self.last_batch_size = 0

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        if isinstance(exc, SubstrateSdkException):
            return exc
        return UnknownException(str(exc))

    def get_max_message_size_bytes(self) -> int:
        return self._MAX_MESSAGE_SIZE

    def do_send_batch(self, messages: List[Message]) -> None:
        # No network: count the batch. The Batcher's single handler serializes
        # these increments even under concurrent client sends.
        self.send_batch_calls += 1
        self.last_batch_size = len(messages)
        self.sent_count += len(messages)


@SUBSCRIPTION_REGISTRY.register("benchfake")
class BenchSubscription(BaseSubscription):
    """In-memory subscription for ``benchfake``.

    ``do_receive_batch`` fabricates ``batch_size`` messages from a monotonic
    counter (no backlog to seed, never blocks); ``do_send_acks`` /
    ``do_send_nacks`` count.
    """

    provider_id = "benchfake"

    def __init__(
        self,
        *,
        subscription_name: str = "bench-sub",
        body_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(subscription_name=subscription_name, **kwargs)
        self._body = b"x" * body_size
        self._counter = itertools.count()
        # Observable side effects for benchfake-tier assertions / batcher introspection.
        self.ack_count = 0
        self.nack_count = 0
        self.receive_batch_calls = 0
        self.last_batch_size = 0

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        if isinstance(exc, SubstrateSdkException):
            return exc
        return UnknownException(str(exc))

    def do_receive_batch(self, batch_size: int) -> List[Message]:
        # Fabricate the requested count, each with a fresh unique ack_id.
        # batch_size comes from BaseSubscription's adaptive sizing; record it.
        self.receive_batch_calls += 1
        self.last_batch_size = batch_size
        body = self._body
        counter = self._counter
        return [Message(body=body, ack_id=next(counter)) for _ in range(batch_size)]

    def do_send_acks(self, ack_ids: List[AckID]) -> None:
        self.ack_count += len(ack_ids)

    def do_send_nacks(self, nacks: List[AckInfo]) -> None:
        self.nack_count += len(nacks)

    def can_nack(self) -> bool:
        return True

    def is_retryable(self, error: BaseException) -> bool:
        # Benchfake never raises, so nothing is ever retried.
        return False

    def get_attributes(self) -> GetAttributeResult:
        return GetAttributeResult(
            name=self.subscription_name,
            topic="bench-topic",
            approximate_number_of_messages=0,
        )


# How long a consumer parks on an empty queue before returning [] so the driver
# can re-poll. On a publish it wakes immediately (notify); the timeout only bounds
# teardown so a blocked prefetch can't wedge close()'s pool shutdown.
_RECEIVE_WAIT_SECONDS = 0.5


class _SharedQueue:
    """Bounded-wait in-memory queue: producers ``put`` + notify, consumers
    ``get_batch`` block until an item lands (or the wait elapses).

    An empty ``get_batch`` parks the consumer on the condition instead of
    returning ``[]`` at once, which would spin the driver's prefetch loop
    (empty prefetch -> notify -> re-prefetch) hot until the next publish. A real
    long-poll / streaming-pull provider blocks the same way.
    """

    def __init__(self) -> None:
        self._items: Deque[bytes] = deque()
        self._cond = Condition()

    def put(self, body: bytes) -> None:
        with self._cond:
            self._items.append(body)
            self._cond.notify_all()

    def get_batch(self, max_items: int, wait_seconds: float) -> List[bytes]:
        with self._cond:
            if not self._items:
                self._cond.wait(wait_seconds)
            batch: List[bytes] = []
            while self._items and len(batch) < max_items:
                batch.append(self._items.popleft())
            return batch

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)


# Shared queues keyed by name, so a "benchqueue" topic and subscription built with
# the same name pass messages to each other (the pipeline fixture uses a unique
# name per test). Bodies, not Message objects, are stored: the subscription assigns
# a fresh ack_id on receive, like a broker's receipt handle.
_QUEUES: Dict[str, _SharedQueue] = {}
_QUEUES_LOCK = Lock()


def _shared_queue(name: str) -> _SharedQueue:
    with _QUEUES_LOCK:
        queue = _QUEUES.get(name)
        if queue is None:
            queue = _QUEUES[name] = _SharedQueue()
        return queue


@TOPIC_REGISTRY.register("benchqueue")
class SharedQueueTopic(BaseTopic):
    """In-memory topic that enqueues into the queue named by ``topic_name``."""

    provider_id = "benchqueue"

    _MAX_MESSAGE_SIZE = 10 * 1024 * 1024

    def __init__(self, *, topic_name: str = "bench-queue", **kwargs: Any) -> None:
        super().__init__(topic_name=topic_name, **kwargs)
        # Not "_queue": BaseSubscription uses that name for its prefetch buffer.
        self._backlog = _shared_queue(topic_name)

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        if isinstance(exc, SubstrateSdkException):
            return exc
        return UnknownException(str(exc))

    def get_max_message_size_bytes(self) -> int:
        return self._MAX_MESSAGE_SIZE

    def do_send_batch(self, messages: List[Message]) -> None:
        # put() notifies waiting consumers, so a parked receive wakes at once.
        for message in messages:
            self._backlog.put(message.body)


@SUBSCRIPTION_REGISTRY.register("benchqueue")
class SharedQueueSubscription(BaseSubscription):
    """In-memory subscription that drains the queue named by ``subscription_name``.

    ``do_receive_batch`` blocks on the shared queue when empty and wakes on the
    next publish, so a consumer that outruns producers parks rather than spinning
    -- the coordination a live backend imposes, without network.
    """

    provider_id = "benchqueue"

    def __init__(
        self, *, subscription_name: str = "bench-queue", **kwargs: Any
    ) -> None:
        super().__init__(subscription_name=subscription_name, **kwargs)
        # Not "_queue": BaseSubscription uses that name for its prefetch buffer.
        self._backlog = _shared_queue(subscription_name)
        self._counter = itertools.count()
        self.ack_count = 0
        self.nack_count = 0

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        if isinstance(exc, SubstrateSdkException):
            return exc
        return UnknownException(str(exc))

    def do_receive_batch(self, batch_size: int) -> List[Message]:
        bodies = self._backlog.get_batch(batch_size, _RECEIVE_WAIT_SECONDS)
        return [Message(body=body, ack_id=next(self._counter)) for body in bodies]

    def do_send_acks(self, ack_ids: List[AckID]) -> None:
        self.ack_count += len(ack_ids)

    def do_send_nacks(self, nacks: List[AckInfo]) -> None:
        self.nack_count += len(nacks)

    def can_nack(self) -> bool:
        return True

    def is_retryable(self, error: BaseException) -> bool:
        return False

    def get_attributes(self) -> GetAttributeResult:
        return GetAttributeResult(
            name=self.subscription_name,
            topic="bench-queue",
            approximate_number_of_messages=len(self._backlog),
        )
