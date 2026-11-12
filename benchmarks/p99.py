"""Print p50/p95/p99/max from a pytest-benchmark ``--benchmark-json`` file.

pytest-benchmark has no percentile column; this reads every sample from
``stats.data``. Numbers are only real percentiles if the run used ``timed()``
(one call per sample). Read p50 as reliable and **p95 as the tail metric**
(rows sort by p95, worst first). Don't quote p99/max off a ~100-round run: at
100 samples nearest-rank p99 is a single order statistic (high variance; below
100 it just equals ``max``, and the script warns). Trust p99 only at round
counts well past 100; report p50/p95 on the live tier.

Usage::

    uv run python benchmarks/p99.py $HOME/benchmarks/results/bench.json
"""

from __future__ import annotations

import json
import math
import sys
from typing import List, Sequence, Tuple

# One row per benchmark: name, p50, p95, p99, max (seconds), sample count.
_Row = Tuple[str, float, float, float, float, int]

# Below this sample count the nearest-rank p99 index lands on the last sample,
# so the reported p99 is identical to max and carries no tail information.
_P99_MIN_SAMPLES = 100


def _percentile(samples: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of ``samples`` for ``0 < pct <= 100``."""
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    rank = math.ceil(pct / 100.0 * len(ordered))
    idx = min(max(rank, 1), len(ordered)) - 1
    return ordered[idx]


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("usage: python benchmarks/p99.py <benchmark-json-file>", file=sys.stderr)
        return 2

    try:
        with open(argv[1]) as fh:
            report = json.load(fh)
    except FileNotFoundError:
        print(f"file not found: {argv[1]}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"not valid JSON ({argv[1]}): {exc}", file=sys.stderr)
        return 2

    rows: List[_Row] = []
    for bench in report.get("benchmarks", []):
        try:
            data = bench["stats"]["data"]
        except (KeyError, TypeError):
            print(
                f"not a pytest-benchmark file ({argv[1]}): missing stats.data",
                file=sys.stderr,
            )
            return 2
        rows.append(
            (
                bench["name"],
                _percentile(data, 50),
                _percentile(data, 95),
                _percentile(data, 99),
                max(data) if data else float("nan"),
                len(data),
            )
        )

    if not rows:
        print(f"no benchmarks found in {argv[1]}", file=sys.stderr)
        return 1

    name_w = max(len(r[0]) for r in rows)
    header = (
        f"{'benchmark':<{name_w}}  {'p50(us)':>10}  {'p95(us)':>10}  "
        f"{'p99(us)':>10}  {'max(us)':>10}  {'samples':>7}"
    )
    print(header)
    print("-" * len(header))
    # Sort by p95, worst first: p95 is the tail metric to trust at ~100
    # samples (p99/max are directional), so the slowest p95 should be on top.
    for name, p50, p95, p99, mx, n in sorted(rows, key=lambda r: r[2], reverse=True):
        print(
            f"{name:<{name_w}}  {p50 * 1e6:>10.2f}  {p95 * 1e6:>10.2f}  "
            f"{p99 * 1e6:>10.2f}  {mx * 1e6:>10.2f}  {n:>7}"
        )

    print("\nread: p50 reliable · p95 tail metric (sort key) · p99/max directional")

    min_n = min(r[5] for r in rows)
    if min_n < _P99_MIN_SAMPLES:
        print(
            f"\nnote: at {min_n} samples the nearest-rank p99 equals max — they "
            f"diverge only at >={_P99_MIN_SAMPLES} samples. Raise the round count "
            "(more billable requests) for a meaningful p99 tail.",
            file=sys.stderr,
        )
    elif min_n == _P99_MIN_SAMPLES:
        print(
            f"\nnote: at {min_n} samples the nearest-rank p99 is a single order "
            "statistic (the 99th of 100), so it has high run-to-run variance — "
            "report p50/p95 here and raise the round count before trusting p99.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
