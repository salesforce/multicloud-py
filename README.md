# Python Multi-Cloud SDK

A cloud-agnostic Python library providing unified interfaces for cloud services across AWS, GCP, and Ali. Write code once, run it on any supported cloud — only the `provider` keyword argument changes.

[Documentation](https://opensource.salesforce.com/multicloud-py/) ·
[Getting started](https://opensource.salesforce.com/multicloud-py/getting-started/) ·
[API reference](https://opensource.salesforce.com/multicloud-py/api/)

## Overview

The Python Multi-Cloud SDK follows a **three-layer architecture** (Client → Driver → Provider) with an in-process provider registry. Provider-specific implementations register themselves at import time using decorators; user code stays cloud-agnostic.

### Design Principles

- **Consistent API**: Single interface across AWS, GCP, and Ali
- **Direct SDK Usage**: Uses official provider SDKs (boto3, google-cloud-storage, etc.) — no custom HTTP calls
- **Provider Registry**: Decorator-based registration with lazy import and helpful error messages
- **Unified Error Hierarchy**: Provider-specific exceptions translated to a common `SubstrateSdkException` tree
- **Conformance Testing**: VCR.py-based record/replay tests ensure cross-provider behavioral parity

## Architecture

```
Three-layer pattern (per service):

┌─────────────────────────────────────────────────────────────────┐
│  Client Layer    │  BucketClient / StsClient                    │  ← user-facing API
├─────────────────────────────────────────────────────────────────┤
│  Driver Layer    │  BaseBlobStore / BaseSts                     │  ← abstract contract + validation
├─────────────────────────────────────────────────────────────────┤
│  Provider Layer  │  AwsBlobStore / GcpBlobStore / AwsSts / ...  │  ← cloud-specific implementations
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
packages/                                  # uv workspace — one directory per publishable distribution
├── common/
│   ├── multicloud-common/                 #   multicloud-py-common: exceptions + ProviderRegistry + Provider ABC
│   ├── multicloud-aws-common/             #   multicloud-py-aws-common: shared AWS/boto3 helpers
│   └── multicloud-gcp-common/             #   multicloud-py-gcp-common: shared GCP helpers
├── blob/                                  # Blob storage service
│   ├── multicloud-blob/                   #   Cloud-agnostic API (BucketClient + BaseBlobStore, sync + async)
│   ├── multicloud-blob-aws/               #   AWS S3 implementation
│   ├── multicloud-blob-gcp/               #   GCP Cloud Storage implementation
│   └── multicloud-blob-ali/           #   Ali OSS implementation (placeholder)
├── sts/                                   # Security Token Service
│   ├── multicloud-sts/                    #   Cloud-agnostic API (StsClient + BaseSts)
│   ├── multicloud-sts-aws/                #   AWS STS implementation
│   ├── multicloud-sts-gcp/                #   GCP implementation
│   └── multicloud-sts-ali/            #   Ali STS implementation (placeholder)
├── iam/                                   # Identity and Access Management
│   ├── multicloud-iam/                    #   Cloud-agnostic API (IamClient + BaseIam)
│   ├── multicloud-iam-aws/                #   AWS IAM implementation
│   └── multicloud-iam-gcp/                #   GCP IAM implementation
├── docstore/                              # Document store service
│   ├── multicloud-docstore/               #   Cloud-agnostic API
│   ├── multicloud-docstore-aws/           #   AWS DynamoDB implementation
│   └── multicloud-docstore-gcp-firestore/ #   GCP Firestore implementation
└── pubsub/                                # Publish/subscribe messaging
    ├── multicloud-pubsub/                 #   Cloud-agnostic API
    ├── multicloud-pubsub-aws/             #   AWS SNS/SQS implementation
    ├── multicloud-pubsub-gcp/             #   GCP Pub/Sub implementation
    └── multicloud-pubsub-ali/         #   Ali MNS implementation (placeholder)

# Each distribution directory holds a PEP 420 namespace tree, e.g.
#   packages/blob/multicloud-blob-aws/multicloud/blob/blob_aws/aws_blob_store.py
# so every package contributes to the shared `multicloud.*` import namespace.
# (Directory names are unprefixed `multicloud-*`; the published distributions
#  carry the `sfdc-` prefix — see Installation below.)

tests/                                     # Unit + conformance tests, mirrored per service
├── blob/                                  #   unit/ (mock-based) + conformance/ (VCR.py record/replay)
├── sts/
├── iam/
├── docstore/
├── pubsub/
├── common/                                #   exceptions + provider-registry tests
├── aws_common/                            #   shared AWS helper tests
├── gcp_common/                            #   shared GCP helper tests
├── e2e/                                   #   Live E2E harness (developer laptop only, not CI)
├── conftest.py                            #   auto-tags tests under any conformance/ directory
└── test_multicloud_init.py

benchmarks/                                # pytest-benchmark suites (blob, pubsub); not in testpaths

examples/
├── blob/
├── sts/
├── iam/
├── pubsub/
└── README.md
```

## Services

| Service | AWS | GCP | Ali |
|---------|-----|-----|---------------|
| Blob Storage | ✓ S3 | ✓ Cloud Storage | ⌛ OSS |
| Security Token Service | ✓ STS | ✓ IAM | ⌛ STS |
| Identity (IAM) | ✓ IAM | ✓ IAM | ⌛ RAM |
| Document Store | ✓ DynamoDB | ✓ Firestore | ⌛ TableStore |
| PubSub | ✓ SNS/SQS | ✓ Pub/Sub | ⌛ MNS |

✓ = Implemented, ⌛ = Planned

## Installation

The SDK ships as multiple distributions sharing the `multicloud.*` namespace,
modeled on the Azure SDK for Python. Distributions publish under the
`multicloud-py-` prefix; the **import namespace stays `multicloud.*`**
regardless of which distribution you install. Install only the service +
provider you need; transitive dependencies (the cloud-agnostic client +
`multicloud-py-common`, plus any shared provider-common package) are pulled
in automatically.

### From PyPI (Not Available Yet)

```bash
# Blob storage on AWS — installs multicloud-py-blob-aws + multicloud-py-blob + multicloud-py-common + boto3
pip install multicloud-py-blob-aws

# Blob storage on GCP
pip install multicloud-py-blob-gcp

# Mix and match: AWS for blob, GCP for the Firestore document store
pip install multicloud-py-blob-aws multicloud-py-docstore-gcp-firestore
```

Available distributions:

| Distribution | Pulls in |
|---|---|
| `multicloud-py-blob-aws` / `-blob-gcp` | Blob storage backed by S3 / GCS |
| `multicloud-py-sts-aws` / `-sts-gcp` | Security Token Service (AWS STS / GCP) |
| `multicloud-py-iam-aws` / `-iam-gcp` | Identity (IAM) management |
| `multicloud-py-docstore-aws` / `-docstore-gcp-firestore` | Document store backed by DynamoDB / Firestore |
| `multicloud-py-pubsub-aws` / `-pubsub-gcp` | Publish/subscribe backed by SNS-SQS / Pub-Sub |

The Ali distributions — `multicloud-py-blob-ali`,
`multicloud-py-sts-ali`, and `multicloud-py-pubsub-ali` — are
**placeholder packages** (distribution name reserved, implementation pending).
Installing one registers no provider yet, so the client raises
`No provider registered`. See the Services table above (`⌛` = Planned) for
per-provider status.

The shared `multicloud-py-aws-common` / `-gcp-common` packages are pulled in
transitively by the provider distributions; you don't install them directly.

A blob-only AWS deployment lands `boto3` only — no GCP or Ali SDKs on
disk. Strict isolation, no opt-in flags.

### For Local Development

The repo is a [uv](https://docs.astral.sh/uv/) workspace. One command sets
up an isolated `.venv/` with every distribution installed editable plus the
dev tooling:

```bash
git clone https://github.com/salesforce/multicloud-py.git
cd multicloud-py
uv sync       # creates .venv, installs all 21 workspace members + dev deps
```

If you don't have `uv` yet, install it with `pip install uv` or follow
the [official installer](https://docs.astral.sh/uv/getting-started/installation/).

## Quick Start

### Blob Storage

```python
from multicloud.blob import BucketClient

# Create a client — only 'provider' changes between clouds
with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    # Upload
    response = client.upload(
        key="hello.txt",
        payload=b"Hello from the Python Multi-Cloud SDK!",
        content_type="text/plain",
        metadata={"author": "multicloud"},
    )
    print(f"Uploaded: ETag={response.e_tag}")

    # Download
    response = client.download(key="hello.txt")
    print(f"Content: {response.content.decode()}")
    print(f"Size: {response.metadata.content_length} bytes")

    # Delete
    client.delete(key="hello.txt")
```

Switch to GCP by changing one argument:

```python
with BucketClient(provider="gcp", bucket="my-bucket", region="us-central1") as client:
    # Same API — identical code works
    client.upload(key="hello.txt", payload=b"Hello from GCP!")
```

### Security Token Service (STS)

```python
from multicloud.sts import StsClient

with StsClient(provider="aws", region="us-east-1") as sts:
    # Get current identity
    identity = sts.get_caller_identity()
    print(f"Account: {identity.account_id}, Resource: {identity.cloud_resource_name}")

    # Assume a role -> returns StsCredentials
    creds = sts.assume_role(
        "arn:aws:iam::123456789012:role/MyRole",
        session_name="my-session",
        expiration=3600,
    )
    print(f"Temporary credentials expire at: {creds.expiration}")
```

STS also supports federated sign-in (`assume_role_with_web_identity`), access
tokens (`get_access_token` with `audience`/`scopes`), and service-account token
generation (`generate_service_account_token`). See the
[STS guide](docs/guides/sts-guide.md) for details and provider support.

### Error Handling

All provider-specific errors are translated to a unified hierarchy:

```python
from multicloud.blob import BucketClient
from multicloud.common.exceptions import (
    ResourceNotFoundException,
    UnAuthorizedException,
    SubstrateSdkException,
)

with BucketClient(provider="aws", bucket="my-bucket", region="us-west-2") as client:
    try:
        client.download(key="nonexistent.txt")
    except ResourceNotFoundException:
        print("Object not found — works the same on AWS, GCP, or Ali")
    except UnAuthorizedException:
        print("Credentials missing or expired")
    except SubstrateSdkException as e:
        print(f"SDK error: {e}")
```

## Development

### Setting Up

```bash
git clone https://github.com/salesforce/multicloud-py.git
cd multicloud-py
uv sync
```

`uv sync` reads the `[tool.uv.workspace]` block in the root `pyproject.toml`,
installs every package under `packages/` editable into `.venv/`, and pulls
in the `dev` dependency group (pytest, vcrpy, black, isort, mypy).

### Running Tests

Prefix every command with ``uv run`` so it executes inside the workspace
venv. Conformance tests are auto-tagged with the ``conformance`` marker by
``tests/conftest.py`` (any test under a ``conformance/`` directory).

```bash
# Run every unit test (no cloud credentials needed)
uv run pytest tests/ -m "not conformance"

# Run a single service's unit tests
uv run pytest tests/blob/unit/
uv run pytest tests/sts/unit/
uv run pytest tests/iam/unit/

# Replay-mode conformance tests (no credentials needed, uses cassettes)
uv run pytest tests/ -m conformance --record-mode=none

# Record-mode conformance (requires real cloud credentials)
uv run pytest tests/blob/conformance/test_aws_blob_conformance.py --record-mode=rewrite
uv run pytest tests/blob/conformance/test_gcp_blob_conformance.py --record-mode=rewrite

# Coverage report
uv run pytest --cov=multicloud --cov-report=term-missing
```

### Conformance Testing

Conformance tests ensure behavioral parity across providers using [VCR.py](https://vcrpy.readthedocs.io/) for HTTP record/replay:

- **Replay mode** (default): Uses pre-recorded cassettes — fast, deterministic, no credentials needed. This is what CI runs.
- **Record mode** (`--record-mode=rewrite`): Hits live cloud services, records HTTP interactions as YAML cassettes. Requires credentials.

The conformance suite is written once in `base_blob_store_conformance.py` and executed by each provider's harness class. Credential scrubbing ensures no secrets are persisted in cassettes.

**Credentials for recording:**
- AWS: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- GCP: `GOOGLE_APPLICATION_CREDENTIALS` pointing to service account JSON

### Code Quality

```bash
# Format code
black packages tests

# Sort imports
isort packages tests

# Type checking
mypy packages
```

## Adding a New Provider

1. Create a new package under the service (e.g., `multicloud/blob/blob_ali/`)
2. Implement the driver's abstract `do_*` methods
3. Register with the service registry using the decorator:
   ```python
   from multicloud.blob.blob_client.driver import BLOB_REGISTRY, BaseBlobStore

   @BLOB_REGISTRY.register("ali")
   class AliBlobStore(BaseBlobStore):
       def do_upload(self, key, payload, *, content_type, content_length, metadata):
           ...
   ```
4. Import in the driver module's provider-loading section (with `ImportError` guard)
5. Add conformance test harness in `tests/blob/conformance/`
6. Record cassettes with real credentials

## Adding a New Service

1. Create a new top-level package (e.g., `multicloud/docstore/`)
2. Define the client in `docstore_client/client.py`
3. Define the abstract driver in `docstore_client/driver.py` with a `ProviderRegistry`
4. Implement providers in `docstore_aws/`, `docstore_gcp/`, etc.
5. Add unit and conformance tests following the existing pattern

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the [Apache License 2.0](LICENSE.txt).
