# Benchmarks

Performance benchmarks for the multicloud Python SDK (`multicloud-py`), built on
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/). Benchmarks are
`pytest` tests using the `benchmark` fixture, so they reuse the same runner,
fixtures, and per-provider harness as the test suite.

## Layout

Benchmarks live here, **outside** `tests/`, and are **not** in `testpaths` — so
`uv run pytest` (the normal unit run) never collects them. Run them by pointing
pytest at this directory.

```
benchmarks/
  _bench.py                   # shared timed() helper (one call per sample, p99-friendly)
  p99.py                      # computes p50/p95/p99/max from --benchmark-json output
  blob/
    _sync_fakes.py            # in-memory "benchfake" sync store (no network)
    _async_fakes.py           # in-memory "benchfake" async store (no network)
    conftest.py               # registers benchfake + the sync/async client & event-loop fixtures
    test_blob_sync_bench.py   # sync: serial ops (payload tiers) + thread-pool concurrency (4/8)
    test_blob_async_bench.py  # async: serial ops + gather concurrency (4/8) + directory ops
  pubsub/
    _fakes.py                 # in-memory fakes: "benchfake" (fabricates) + "benchqueue" (shared queue)
    conftest.py               # registers benchfake + topic/subscription/seed fixtures
    test_pubsub_bench.py      # serial + consume-side batch + concurrency + pipeline + nack
```

`_bench.py` reads its round/warmup defaults from a per-suite environment family
selected by `env_prefix`: `BLOB_BENCH_*` for blob (the default), `PUBSUB_BENCH_*`
for pubsub. `p99.py` is suite-agnostic — point it at any `--benchmark-json`.

## Provider-agnostic benchmark bodies

Blob benchmark bodies depend on a client fixture — `blob_bench_client` (sync) or
`async_blob_bench_client` (async) — not a provider, and never name a cloud, so the
same body runs against `benchfake` or a real client by changing only environment
variables. Defaults resolve to `benchfake` (network-free, no credentials); set
these to point at a real backend:

| Variable | Meaning | Default |
|---|---|---|
| `BLOB_BENCH_PROVIDER` | provider id, e.g. `aws` | `benchfake` |
| `BLOB_BENCH_BUCKET` | target bucket (required unless benchfake) | — |
| `BLOB_BENCH_REGION` | provider region | unset |

Credentials come from the provider's ambient discovery, never the fixtures. On a
real backend each test namespaces keys under a unique `bench-<uuid>/` prefix and
teardown deletes only that prefix, so a shared bucket's other objects are safe.

> **Use a non-versioned (or lifecycle-managed) bucket for the live tier.**
> Teardown deletes by key, not version — on a versioned bucket every prior
> version stays retained and billable. A single run makes tens of thousands of
> writes, so point `BLOB_BENCH_*` at a disposable bucket.

## Sync benchmarks

In `test_blob_sync_bench.py`:

- **Serial** — one op timed at a time: upload/download across `1KB / 1MB / 10MB`
  tiers, write-read-delete lifecycle, metadata, one listing page,
  paginate-to-completion, copy, multipart upload, batch delete. Read-path seeding
  happens outside the timed region.
- **Concurrent** — a `ThreadPoolExecutor` (`1 / 4 / 8` workers) drains a batch of
  uploads and downloads; `t1` is the single-worker baseline and healthy scaling
  is roughly linear off it (`t4 ≈ t1 / 4`, `t8 ≈ t1 / 8`, so `t8 ≈ t4 / 2`).
  Against `benchfake` there's no I/O to overlap, so throughput stays flat — you
  measure GIL-handoff, not concurrency. Live `t8 ≈ t4` means calls aren't
  overlapping (internal serialization or a pool/rate-limit plateau) — cross-check
  serial latency.

**Scope: buffered path, sync only.** These cover buffered `upload(bytes)` / full
`download` plus multipart. They do **not** exercise `download_stream`, the
file-path variants, range reads, `copy_from`, or versioned ops — read the numbers
as a baseline for the buffered path, not the full surface.

## How async bodies are timed

`pytest-benchmark` times synchronous callables, so each async body is an ordinary
sync function that builds a coroutine and drives it to completion with the
`run_async` helper. `run_async` runs on one **session-scoped** event loop (the
`bench_loop` fixture), created once and reused for every sample, so loop
setup/teardown never lands in the measured region. `benchmark.pedantic(iterations=1)`
then times one full await chain per sample.

## Async benchmarks

In `test_blob_async_bench.py`:

- **Serial** — one awaited op timed at a time: upload/download across
  `1KB / 1MB / 10MB` tiers, write-read-delete lifecycle, metadata, one listing
  page, paginate-to-completion, copy, multipart upload, batch delete. Read-path
  seeding happens outside the timed region.
- **Concurrent** — a batch fanned out with `asyncio.gather` under an
  `asyncio.Semaphore` (`4 / 8` in-flight) drains uploads and downloads; healthy
  scaling is `c8 ≈ c4 / 2` (no `c1` point — the serial group already covers
  single-op latency). This measures the client's own coroutine scheduling, not
  thread hand-off. Against `benchfake` there's no I/O to overlap, so throughput
  stays flat — you measure event-loop scheduling overhead, not concurrency. Live
  `c8 ≈ c4` means calls aren't overlapping (internal serialization or a
  pool/rate-limit plateau) — cross-check serial latency.
- **Directory** — whole-tree `upload_directory` / `download_directory` /
  `delete_directory` over a small-file tree, timed as one op. These exercise the
  client's internal bounded-concurrency fan-out. `download`/`delete` seed outside
  the timed region where the op is read-only; `delete_directory` re-seeds inside
  the timed region (a seed-once delete would, after round 1, delete an
  already-gone prefix), so its cost is labelled upload+delete.

**Scope: buffered path, async only.** These cover buffered `upload(bytes)` / full
`download`, multipart, and directory ops. They do **not** exercise
`download_to_path`/`download_to_file`, range reads, `copy_from`, presigned URLs,
or versioned ops — read the numbers as a baseline for the buffered async path, not
the full surface.

## PubSub benchmarks

In `test_pubsub_bench.py`. Pubsub has two sides — a topic (`send`) and a
subscription (`receive` / `send_ack` / `send_nack`) — so cases sit on one side or
run both at once. Bodies depend on the `topic_bench_client` / `sub_bench_client`
fixtures, never a provider, so the same body runs against `benchfake` or a real
backend by changing only environment variables.

- **Serial** — one op timed at a time: `send` across `1KB / 64KB / 256KB` tiers
  (256 KiB is the largest all providers accept — SNS/SQS cap there), single
  `receive`.
- **Batch (consume-side)** — the public API sends one message at a time (the
  driver's Batcher coalesces internally), so the batch dimension lives on the
  consume side: `send_acks` across batch sizes `1 / 10 / 100`, and drain
  of `10 / 100` messages per call (no `1` — `receive` already covers a single
  message). `send_acks` only enqueues into the ack Batcher and returns an
  already-completed future, so the broker delete is forced by `close()` →
  `shutdown_and_drain()`; the ack body therefore times `send_acks` **then**
  `close()` on a fresh per-round client and re-receives a distinct id batch each
  round, so batch size drives a real batched delete rather than 50 no-op
  re-acks. Every receive-side body `send_ack`s each message it consumes (cheap
  fire-and-forget) so a live backend does not redeliver unacked messages into a
  later round.
- **Concurrency** — a `ThreadPoolExecutor` (`1 / 4 / 8` workers) fans out `send`
  and `receive` through one client. `send_concurrent` stays roughly flat on
  `benchfake` — there is no network latency to overlap, so you measure
  GIL-handoff, not concurrency. On a **live** backend it typically *scales*: the
  topic Batcher coalesces concurrent publishes into fewer, larger broker RPCs,
  so more senders means bigger batches and higher throughput. Cross-check serial
  `send` latency to read the plateau.
- **Pipeline** — `w` producers and `w` consumers (`1 / 4 / 8`) drive a fixed
  message count through publish and consume at once. The headline throughput
  case: msgs/sec = count / measured time. Producers publish exactly what
  consumers drain, so no pre-seed is needed. On `benchfake` this body uses the
  shared-queue driver (`benchqueue`: topic enqueues, subscription drains the
  same queue) so a publish is genuinely received — the fabricating fake would
  let a consumer "receive" a message no producer sent.
- **Nack** — a full receive → nack → `close()` cycle on a fresh per-round client,
  so the nack actually reaches the broker (like ack, `send_nack` alone only
  enqueues). The redelivery round-trip itself is only realized on live.

Payload tiers are smaller than blob's because broker limits sit far below
object-store limits. Consume-side bodies vary message *count*, not body size —
the Batcher groups by count, so throughput is largely size-insensitive there.

### PubSub live tier

An empty real subscription makes `receive()` block, and pubsub has no idempotent
round-trip like blob's `upload`→`download`→`delete`, so the receive-side bodies
need a backlog. An SQS queue is **both** a send and a receive endpoint, so
pointing the topic and the subscription at one queue gives those bodies a real,
self-contained backlog to drain. The `seed` fixture publishes exactly the number
of messages a body will consume into that queue before the timed region (a no-op
on `benchfake`, which fabricates messages).

> **Keep `PUBSUB_BENCH_BATCH_ROUNDS` low on live for `test_ack_batch`.** That
> body re-receives a distinct id batch each round and pre-receives them all up
> front, which starts every message's ack deadline / visibility timeout at once.
> A long ack run can outlast the deadline, so a late round acks an already
> redelivered id. Small round counts (the default 50, less on live) keep the run
> inside the deadline.

| Variable | Meaning | Default |
|---|---|---|
| `PUBSUB_BENCH_TOPIC_PROVIDER` | topic provider id, e.g. `awssqs` | `benchfake` |
| `PUBSUB_BENCH_SUB_PROVIDER` | subscription provider id, e.g. `aws` | `benchfake` |
| `PUBSUB_BENCH_QUEUE` | shared queue name/URL used as **both** endpoints | — |
| `PUBSUB_BENCH_TOPIC` | topic name; the SNS topic **ARN** when the topic provider is `awssns`. For SQS, ignored if `PUBSUB_BENCH_QUEUE` is set | `bench-topic` |
| `PUBSUB_BENCH_SUBSCRIPTION` | subscription name (ignored if `PUBSUB_BENCH_QUEUE` is set) | `bench-sub` |
| `PUBSUB_BENCH_REGION` | provider region | unset |
| `PUBSUB_BENCH_BODY_SIZE` | seeded/fabricated message body size (bytes) | `1024` |
| `PUBSUB_BENCH_OP_TIMEOUT` | per concurrent/pipeline future wait (s); `<=0` unbounded | `120` |

```bash
# Live pubsub against one SQS queue (send + receive share it). Credentials come
# from ambient discovery. Keep round counts low — each round is billable.
PUBSUB_BENCH_TOPIC_PROVIDER=awssqs \
PUBSUB_BENCH_SUB_PROVIDER=aws \
PUBSUB_BENCH_QUEUE=my-bench-queue \
PUBSUB_BENCH_REGION=us-west-2 \
PUBSUB_BENCH_ROUNDS=20 \
PUBSUB_BENCH_BATCH_ROUNDS=5 \
PUBSUB_BENCH_CONCURRENT_ROUNDS=5 \
PUBSUB_BENCH_PIPELINE_ROUNDS=5 \
PUBSUB_BENCH_WARMUP=1 \
  uv run pytest benchmarks/pubsub/
```

The topic side also accepts the `awssns` provider. Unlike SQS, SNS is not a
single send+receive endpoint: the topic is a distinct SNS ARN and the
subscription drains a **separate** SQS queue subscribed to it (SNS→SQS fan-out).
Set `PUBSUB_BENCH_TOPIC` to the ARN and `PUBSUB_BENCH_QUEUE` to that queue, and
provision the topic, the queue, and the subscription yourself first.

> **Subscribe with raw message delivery on.** Otherwise SNS wraps each body in a
> JSON envelope: the `256kb` tier then exceeds the 256 KB SNS/SQS limit and the
> publish fails, and the smaller tiers carry envelope overhead that skews the
> per-size numbers.

```bash
# Live pubsub with an SNS topic fanning out to a subscribed SQS queue.
PUBSUB_BENCH_TOPIC_PROVIDER=awssns \
PUBSUB_BENCH_SUB_PROVIDER=aws \
PUBSUB_BENCH_TOPIC=arn:aws:sns:us-west-2:123456789012:my-bench-topic \
PUBSUB_BENCH_QUEUE=my-bench-queue \
PUBSUB_BENCH_REGION=us-west-2 \
PUBSUB_BENCH_ROUNDS=20 \
PUBSUB_BENCH_BATCH_ROUNDS=5 \
PUBSUB_BENCH_CONCURRENT_ROUNDS=5 \
PUBSUB_BENCH_PIPELINE_ROUNDS=5 \
PUBSUB_BENCH_WARMUP=1 \
  uv run pytest benchmarks/pubsub/
```

## Two kinds of benchmark

- **SDK-overhead (network-free).** The `benchfake` runs (default): drive the real
  client → driver → provider path with the transport removed. Deterministic, `$0`,
  safe on every change.
- **Live end-to-end (opt-in).** The same bodies against a real backend via
  `BLOB_BENCH_*`, to track real latency and exercise I/O concurrency. Needs
  credentials, so run on a schedule, not every PR.

## Reading the numbers: within-cloud trends, not a cross-cloud race

Live-tier results track **one provider's latency over time**, not A-vs-B. The
client runs from one location, and different regions sit at different network
distances, so absolute latency to provider A vs. B is dominated by round-trip
time, not SDK or service behavior. Two providers are only comparable if the
client is co-located with each region. Treat a cross-provider delta from a single
client as noise — and note the vantage point (especially over a VPN), which skews
absolute latency too.

## Running

Write JSON results to a directory outside the repo (e.g. `$HOME/benchmarks/results/`).

```bash
# All benchmarks (network-free benchfake store by default)
uv run pytest benchmarks/

# One service
uv run pytest benchmarks/blob/

# Against a real backend (opt-in). Credentials come from ambient discovery;
# teardown deletes only the run's bench-<uuid>/ prefix. Always set
# BLOB_BENCH_WARMUP>=1 on a live backend so the cold first call (TLS, DNS,
# pool fill, cred resolution) stays out of the recorded tail.
BLOB_BENCH_PROVIDER=aws \
BLOB_BENCH_BUCKET=my-bench-bucket \
BLOB_BENCH_REGION=us-west-2 \
BLOB_BENCH_WARMUP=1 \
  uv run pytest benchmarks/blob/

# Save results / compare against a baseline (outside the repo)
uv run pytest benchmarks/ --benchmark-json=$HOME/benchmarks/results/benchmark_results.json
uv run pytest benchmarks/ --benchmark-compare
```

## Worst-case latency (p99)

`pytest-benchmark` has no p99 column and averages the slow tail away. Two pieces
close that gap:

- **One call per sample.** Every body goes through the shared `timed` helper
  (`benchmarks/_bench.py`), which uses `benchmark.pedantic(iterations=1)` so each
  sample is a single call — without it, auto-calibration records the *mean* and
  the tail disappears.
- **Compute percentiles from raw data.** `--benchmark-json` retains every
  sample; `benchmarks/p99.py` reads it and prints p50/p95/p99/max per benchmark,
  sorted by p95 (the tail metric to trust at ~100 samples):

```bash
uv run pytest benchmarks/blob/ \
  --benchmark-columns=min,median,max,rounds \
  --benchmark-json=$HOME/benchmarks/results/bench.json
uv run python benchmarks/p99.py $HOME/benchmarks/results/bench.json
```

> **Reading the tail.** On the live tier report **`p50`/`p95` only, never
> `p99`/`max`**, and always run with `BLOB_BENCH_WARMUP=1` (or 2) so cold start
> (TLS, DNS, pool fill, creds) stays out of the tail. At 100 rounds nearest-rank
> p99 is a single order statistic (the 99th of 100), so it carries high
> run-to-run variance and can move between identical runs — "distinct from
> `max`" is not "stable". Treat `p99` as a **`benchfake`-tier** metric only:
> rounds are free there, so push them well past 100 (raise `BLOB_BENCH_ROUNDS`)
> before trusting a p99. `p50` is reliable at 100 rounds and `p95` (~5 tail
> samples) is a sound tail metric on either tier.

## Controlling sample count (live tier)

Each group's round count comes from a **separate** env var, so one command can
sample them differently. The blob suite uses the `BLOB_BENCH_*` family, pubsub
the `PUBSUB_BENCH_*` family:

| Group | Blob | PubSub | Default | Why |
|---|---|---|---|---|
| Serial | `BLOB_BENCH_ROUNDS` | `PUBSUB_BENCH_ROUNDS` | `100` | `p50`/`p95` tail needs ~100 samples |
| Batch | — | `PUBSUB_BENCH_BATCH_ROUNDS` | `50` | each sample handles many messages |
| Concurrent | `BLOB_BENCH_CONCURRENT_ROUNDS` | `PUBSUB_BENCH_CONCURRENT_ROUNDS` | `30` | ratio is round-count-insensitive |
| Pipeline | — | `PUBSUB_BENCH_PIPELINE_ROUNDS` | `20` | each sample drains a full message batch |

Warmup is `BLOB_BENCH_WARMUP` / `PUBSUB_BENCH_WARMUP` (default `0`; set `1`-`2`
on any live run).

`BLOB_BENCH_OP_TIMEOUT` / `PUBSUB_BENCH_OP_TIMEOUT` (default `120`, seconds) bound
one round's work so a stalled live call fails its round in bounded time. The blob
sync suite caps each concurrent op's `future.result()` wait; the blob async suite
caps the awaited work `run_async` drives (serial and concurrent alike). The pubsub
suite caps each concurrent/pipeline `future.result()` wait **and shuts its pool
down without waiting**, so the timeout propagates instead of re-blocking on pool
exit; a worker wedged in a blocking call still lingers until interpreter exit, and
teardown (`close()`) can hang the same way — rely on a CI job timeout as the
backstop. The **serial** receive bodies block unbounded — so keep per-group round
counts low on live. Set `<=0` to wait unbounded.

Each live round is a billable request, so pin them low. A single live run:

```bash
BLOB_BENCH_PROVIDER=aws \
BLOB_BENCH_BUCKET=my-bench-bucket \
BLOB_BENCH_REGION=us-west-2 \
BLOB_BENCH_ROUNDS=100 \
BLOB_BENCH_CONCURRENT_ROUNDS=30 \
BLOB_BENCH_WARMUP=1 \
  uv run pytest benchmarks/blob/ \
    --benchmark-json=$HOME/benchmarks/results/bench-live.json
uv run python benchmarks/p99.py $HOME/benchmarks/results/bench-live.json
```
