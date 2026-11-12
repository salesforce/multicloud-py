"""Shared ``timed()`` helper for the benchmark suite.

pytest-benchmark's default auto-calibration records the *mean* of many calls,
hiding the tail. ``timed`` uses ``benchmark.pedantic(iterations=1)`` so each
sample is one wall-clock call — what ``p99.py`` needs. ``rounds`` is the sample
count; ``warmup_rounds`` runs the body untimed first (keeps a live backend's
one-time TLS/DNS/pool/cred costs out of the tail). Omitted values come from
``<env_prefix>_ROUNDS`` (default 100) / ``<env_prefix>_WARMUP`` (default 0);
``env_prefix`` picks the family (``BLOB_BENCH`` default, ``PUBSUB_BENCH``).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple

_ITERATIONS = 1
_DEFAULT_ENV_PREFIX = "BLOB_BENCH"
_DEFAULT_ROUNDS = 100
_DEFAULT_WARMUP = 0


def _env_int(env_prefix: str, suffix: str, default: int) -> int:
    return int(os.environ.get(f"{env_prefix}_{suffix}", str(default)))


def resolve_rounds(
    rounds: Optional[int] = None,
    warmup_rounds: Optional[int] = None,
    *,
    env_prefix: str = _DEFAULT_ENV_PREFIX,
) -> Tuple[int, int]:
    """Resolve the ``(rounds, warmup_rounds)`` a matching ``timed()`` will use.

    The body runs exactly ``warmup_rounds + rounds`` times, so a bench can
    pre-generate one input per invocation outside the timed body.

    Args:
        rounds: Sample count; ``<env_prefix>_ROUNDS`` or 100 if omitted.
        warmup_rounds: Untimed calls before sampling; ``<env_prefix>_WARMUP`` or 0.
        env_prefix: Environment-variable family (e.g. ``BLOB_BENCH``).

    Returns:
        The resolved ``(rounds, warmup_rounds)`` pair.
    """
    return (
        (
            rounds
            if rounds is not None
            else _env_int(env_prefix, "ROUNDS", _DEFAULT_ROUNDS)
        ),
        (
            warmup_rounds
            if warmup_rounds is not None
            else _env_int(env_prefix, "WARMUP", _DEFAULT_WARMUP)
        ),
    )


def timed(
    benchmark: Any,
    fn: Callable[..., Any],
    *args: Any,
    rounds: Optional[int] = None,
    warmup_rounds: Optional[int] = None,
    env_prefix: str = _DEFAULT_ENV_PREFIX,
    **kwargs: Any,
) -> Any:
    """Time ``fn(*args, **kwargs)`` with one call per sample.

    Args:
        benchmark: pytest-benchmark fixture.
        fn: Callable under test.
        rounds: Sample count; ``<env_prefix>_ROUNDS`` or 100 if omitted.
        warmup_rounds: Untimed calls before sampling; ``<env_prefix>_WARMUP`` or 0.
        env_prefix: Environment-variable family (e.g. ``BLOB_BENCH``).

    Returns:
        Whatever ``fn`` returns on its last invocation.
    """
    resolved_rounds, resolved_warmup = resolve_rounds(
        rounds, warmup_rounds, env_prefix=env_prefix
    )
    return benchmark.pedantic(
        fn,
        args=args,
        kwargs=kwargs,
        iterations=_ITERATIONS,
        rounds=resolved_rounds,
        warmup_rounds=resolved_warmup,
    )
