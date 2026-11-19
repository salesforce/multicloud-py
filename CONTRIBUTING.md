# Contributing to MultiCloud Python SDK

Thank you for your interest in contributing. This guide describes how to propose changes and what maintainers expect from contributions.

## Governance

MultiCloud Python SDK is a Salesforce-sponsored open source project. The goal is to grow its contributor and user community while maintaining a consistent, cloud-agnostic API. Salesforce employees hold repository administration rights and make final decisions about accepted contributions.

## Before You Start

Search the [existing issues](https://github.com/salesforce/multicloud-py/issues) before opening a new one. For significant features or API changes, open an issue first so the approach and cross-provider behavior can be discussed before implementation begins.

## Development Setup

The repository is a [uv](https://docs.astral.sh/uv/) workspace. Clone your fork and install the workspace dependencies:

```bash
git clone https://github.com/<your-github-user>/multicloud-py.git
cd multicloud-py
uv sync --frozen
```

Run the unit tests and replay-mode conformance tests before submitting a pull request:

```bash
uv run pytest tests/ -m "not conformance"
uv run pytest tests/ -m conformance --record-mode=none
uv run black --check packages tests
uv run isort --check-only packages tests
```

CI tests Python 3.10 through 3.13. Python 3.14 is temporarily excluded while the
two AWS Blob version-copy recordings' malformed base64 headers are repaired;
restore its CI job after validating those recordings on all tested versions.
This temporary CI exception does not change package Python-version metadata.
The required `test` check succeeds only when
every runtime, formatting/version checks, benchmark smoke tests and package
validation pass. On Python 3.12, CI also
enforces **90% total SDK line coverage** across unit and replay tests:

```bash
uv run --frozen python -m pytest --record-mode=none \
  --cov=packages --cov-report=term-missing --cov-report=xml --cov-fail-under=90
```

Coverage includes the SDK source under `packages/`, including unimported
namespace modules. Do not omit provider code or add exclusions to meet the
threshold. Codecov project and changed-line targets are also 90%; its upload is
best-effort and does not replace the local CI coverage gate. Maintainers must
confirm Codecov repository access and reporting before requiring its statuses.

Conformance tests validate consistent behavior across supported providers. Recording new cassettes requires credentials for the relevant cloud provider; never commit credentials or unsanitized recordings.

CI also runs the in-memory Blob/Pub/Sub benchmarks with `--benchmark-disable`.
This is a functional smoke test, not a live-cloud test or a performance budget.
The package job builds all 21 wheels and source distributions and checks their
metadata, versions and Apache-2.0 licenses using the release artifact validator.
It does not publish anything. The separate required `zizmor` check scans GitHub
Actions and Dependabot configuration for workflow-security issues.

## Issues and Feature Requests

- Report reproducible bugs through [GitHub Issues](https://github.com/salesforce/multicloud-py/issues/new).
- Describe the expected cloud-agnostic behavior and affected providers.
- Include a minimal reproduction when possible.
- Discuss provider-specific features before implementation. Public APIs should remain implementable consistently across supported providers.

## Pull Requests

1. Create a focused branch in your fork.
2. Keep commits atomic and use descriptive commit messages.
3. Add or update unit and conformance tests for behavior changes.
4. Update user-facing documentation when APIs or behavior change.
5. Run the validation commands above.
6. Open a pull request against `main` and link the relevant issue.
7. Sign the Salesforce Contributor License Agreement when prompted.

Keep pull requests focused and avoid unrelated formatting or refactoring changes. All changes require peer review before merge.

Use a conventional PR title such as `fix(blob): preserve metadata` or
`docs: clarify onboarding`. Allowed types are `feat`, `fix`, `perf`, `refactor`,
`revert`, `docs`, `style`, `test`, `build`, `ci`, and `chore`. A scope is optional;
add `!` for a breaking change. The title becomes the squash commit subject used
by Release Please. Existing dependency-update PRs may need a maintainer to
retitle them before merging.

The `main` ruleset requires one eligible review, resolved review threads, an
up-to-date branch and successful CI, documentation, title, security and CLA
checks, without an admin bypass. New changes dismiss old approvals. Code-owner
review is separate and is not enabled merely by adding a CODEOWNERS file.

The existing `conventional-title` PR check validates proposed validator changes.
The additional `trusted-conventional-title` check uses `pull_request_target`
with a read-only token and checks out only the base revision's validator. It
never checks out or runs PR code and has no publishing credentials. This keeps
a PR from weakening its own authoritative title validator.

Bootstrap: a new `pull_request_target` workflow cannot validate its own first
PR from the trusted base. After this workflow merges, a maintainer must confirm
`trusted-conventional-title` runs on a new or updated PR and then add that exact
GitHub Actions check to the required statuses. Retain `conventional-title`
during this transition; never require a context that cannot run yet.

Release PRs follow the same review and check requirements. GitHub may require a
maintainer to approve workflow runs created by automation. That approval covers
selected `pull_request` events, not `pull_request_target`. After the final
automated version-sync push, a maintainer must edit the PR title or body using
their own account to trigger the trusted title workflow. Verify both title
checks and the remaining required checks on the final release head before
approving and squash-merging. Confirm this flow on a real release PR before
making `trusted-conventional-title` required; never bypass the ruleset to work
around missing checks. See [GitHub's token-trigger behavior](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs).
Publishing remains in the existing
release workflow; this change does not add extra PyPI approval prompts or alter
the package Trusted Publishers.

## Contributor License Agreement

Contributors must sign the [Salesforce Contributor License Agreement](https://cla.salesforce.com/sign-cla). You only need to sign it once for Salesforce open source projects.

## Code of Conduct

All contributors must follow the project [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contribution will be licensed under the [Apache License 2.0](LICENSE.txt).
