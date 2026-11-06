---
name: multicloudpy-feature-dev
description: Use when adding or extending MultiCloud-Py SDK operations across providers or adding conformance tests for those operations.
---

# Cross-provider feature development

Read `AGENTS.md`, the target service's implementation and tests, and
`.github/workflows/ci.yml`. Use the public repository's current contracts,
package names, and checks. Bug fixes without an API change can use a focused
debugging workflow; documentation-only changes use `multi-cloudpy-docs-guides`.

## Establish the contract

Identify the service, operation, intended observable behavior, and supported
providers from the request and existing code. Ask only for missing decisions
that materially change the API. Research the official provider SDK operations,
limits, pagination, consistency, and error behavior before implementation.
Present material provider differences and resolve the cloud-agnostic contract
with the user when the request has not already settled it.

Use direct arguments and keyword-only optional parameters, with existing typed
result/dataclass conventions. Do not add request wrappers. If a capability is
provider-specific, discuss the scope in an issue before adding it to a shared
API. Do not claim support by implementing a silent fallback that changes its
meaning. Ali distributions are placeholders; preserve that status unless
implementing that provider is explicitly part of the task.

## Follow all three layers

Source paths are under
`packages/SERVICE/multicloud-SERVICE/multicloud/SERVICE/SERVICE_client/`.
Provider distributions sit alongside that directory; published names use
`multicloud-py-`, and the provider identifier is `ali` for Ali.

1. Extend the driver contract in `driver.py`, using the existing validator
   and a `do_*` hook. Put common normalization in the driver.
2. Add the public client method in `client.py`. It forwards to the driver
   and owns exception translation through the provider's `translate_error`.
3. Implement each supported provider's hook independently, using its native
   SDK. Do not catch merely to translate inside provider operations.
4. Extend error classification as needed. Mapping returns the common exception
   type and preserves exceptions already translated; request-specific behavior
   stays at the operation boundary.

Keep common and client/driver code independent of provider dependencies.
Providers must not reference sibling providers, including in comments.
Discover providers through distribution entry points and the existing registry;
do not introduce eager imports into implicit namespace packages.

Check service-specific differences:

- Pub/Sub has separate topic and subscription registries. AWS topic provider
  IDs distinguish `awssns` and `awssqs`; inspect the correct contract.
- Blob has an async stack in `blob_client/async_blob/` and provider-specific
  `async_blob/` directories. Include matching behavior there, or state an
  explicitly agreed async scope. Use the existing async dependency-injection
  fixtures rather than sync-client mocks.
- DocStore uses action/query contracts. Inspect its driver before assuming
  every service shares blob's operation shape or validation helpers.

## Test observable behavior

Add unit tests under `tests/SERVICE/unit/`, injecting mocks for native SDK
collaborators rather than mocking the provider under test. Check request
transformation, returned values, validation, and translated errors. Assert
specific failures, not just non-null results or broadly caught exceptions.

Add shared happy-path and error-path conformance cases under
`tests/SERVICE/conformance/`. Run concrete provider classes, not abstract
base suites. Use round trips, unique resource names, and cleanup; keep shared
assertions cloud-agnostic. Consult existing HTTP and gRPC fixture machinery
for the service instead of assuming all recordings use one transport.

Replay uses `--record-mode=none`. Recording with `--record-mode=rewrite`
requires live cloud access and overwrites fixtures. Do not fabricate
recordings or weaken assertions to make a test pass.

## Record conformance fixtures

Each provider's harness module, `tests/SERVICE/conformance/test_PROVIDER_*.py`,
defines its cloud resources as module constants read from `CONFORMANCE_*`
environment variables with hardcoded defaults. Replay matches recorded hosts
and paths, so new fixtures must be recorded against those defaults; an
override is for exploratory runs, not for fixtures you commit. The blob
defaults are the team's shared test buckets:

| Purpose | AWS (`us-west-2`) | GCP |
| --- | --- | --- |
| Main | `example-test-bucket-barry` | `example-test-bucket-barry` (sync), `python-async-test-bucket` (async) |
| Versioned | `chameleon-python-sdk-versioned` | `chameleon-python-sdk-versioned` |
| Object lock | `chameleon-test-bucket-with-object-lock` | `sdk-blob-with-object-lock` |

Read the harness module rather than trusting this table if they disagree.
Before recording, confirm the account behind the credentials can reach every
resource the impacted tests use, for example with `aws s3api head-bucket` or
`gcloud storage buckets describe`. If a default names a placeholder or a
resource the account cannot reach, stop and ask the user which account owns
it. Do not change defaults, rename hosts inside cassettes, or re-record
unrelated fixtures to work around it.

Ask the user for credentials when recording is needed, and keep them in the
provider's standard credential chain:

- AWS: an `AWS_PROFILE` the user has configured, or `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`.
- GCP: application default credentials or `GOOGLE_APPLICATION_CREDENTIALS`
  pointing at a credentials file, plus any service-account impersonation the
  user specifies.
- Ali: not recorded locally; leave its conformance status unchanged.

Prefer a profile or credentials file the user sets up outside the
conversation, so only its name reaches you. Never echo, write to disk, or
commit credential values.

Record only the tests impacted by the change, one provider at a time, by
selecting their node IDs:

```bash
uv run --frozen python -m pytest -m conformance --record-mode=rewrite \
  "tests/SERVICE/conformance/test_PROVIDER_SERVICE_conformance.py::CLASS::TEST"
```

After recording, inspect every new or changed cassette for credentials,
tokens, and account identifiers the existing fixtures scrub. Then run the
whole service's conformance suite in replay mode. If recording is blocked by
unavailable credentials or resources, report the exact tests left to record
rather than claiming conformance passed.

## Validate and submit

Use the locked uv workspace. Run the target service's checks during development
and the CI checks appropriate to the completed change before opening the PR:

```bash
uv sync --frozen --all-groups --all-packages
uv run --frozen python -m black --check packages tests scripts
uv run --frozen python -m isort --check-only packages tests scripts
uv run --frozen python scripts/sync_version.py --check
uv run --frozen python -m pytest -m 'not conformance'
uv run --frozen python -m pytest -m conformance --record-mode=none
```

Consult `Makefile` for coverage targets and `pyproject.toml` for typing settings;
do not invent a coverage threshold or claim an advisory check is a CI gate.
Update Google-style docstrings, relevant examples, and guides. Verify example
imports against the real `SERVICE_client` exports.

## Open the PR

The only PR target is `main` of `https://github.com/salesforce/multicloud-py`.
This holds regardless of what `origin` points to, which remotes exist, or where
earlier PRs in the checkout went. Never open the PR against an internal mirror,
another host, or a fork's own default branch. Do not add internal work-item
prefixes to the branch, commit, or title.

1. Run `git remote -v` and `gh auth status --hostname github.com`. If no
   remote's push URL is a github.com fork of `salesforce/multicloud-py` (or
   upstream itself, with write access), stop and ask the user which GitHub
   fork to push to.
2. Branch from an up-to-date `salesforce/multicloud-py:main`, commit with a
   conventional `feat:` title, and push the branch to that GitHub fork.
3. Create the PR with an explicit upstream target:

   ```bash
   gh pr create --repo salesforce/multicloud-py --base main \
     --head FORK_OWNER:BRANCH --title "feat(SERVICE): ..." --body-file BODY.md
   ```

   The body describes the contract, supported providers, tests run with their
   results, and any unresolved limits.
4. Confirm the target with
   `gh pr view URL --json url,baseRefName,headRepositoryOwner` and check that
   the URL starts with `https://github.com/salesforce/multicloud-py/pull/`.

Finish by giving the user the PR URL printed by `gh pr create`, as a full
link, in the final message. If the PR could not be created, say so and report
the pushed branch and the exact failing step instead. No internal ticket or CI
access is required for public contribution.
