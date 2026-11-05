"""Google Cloud Storage implementation of :class:`BaseBlobStore`."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, List, Mapping, Optional

import google.auth  # type: ignore[import-not-found]
import google.auth.exceptions  # type: ignore[import-not-found]
import google.auth.transport.requests  # type: ignore[import-not-found]
import requests  # type: ignore[import-not-found]
from google.api_core import (
    exceptions as gcs_exceptions,  # type: ignore[import-not-found]
)
from google.auth.transport.requests import (  # type: ignore[import-not-found]
    AuthorizedSession,
)
from google.cloud import storage  # type: ignore[import-not-found]
from multicloud.blob.blob_client.driver import (
    BLOB_REGISTRY,
    BaseBlobStore,
    BlobIdentifier,
    BlobInfo,
    BlobMetadata,
    ChecksumAlgorithm,
    CopyResponse,
    DeleteError,
    DeleteObjectsResponse,
    DownloadResponse,
    ListBlobsPageResponse,
    MultipartPart,
    MultipartUpload,
    MultipartUploadResponse,
    ObjectLockConfiguration,
    ObjectLockInfo,
    ObjectRetentionConfig,
    PresignedOperation,
    PresignedUrlResponse,
    RetentionMode,
    StreamingDownloadResponse,
    UploadPartResponse,
    UploadResponse,
)
from multicloud.blob.blob_client.retention_rules import resolve_and_validate
from multicloud.blob.blob_client.validators import CHECKSUM_MISMATCH_MSG
from multicloud.blob.blob_gcp._gcp_common import resolve_generation_match
from multicloud.common.credentials import CredentialsOverrider
from multicloud.common.exceptions import (
    DeadlineExceededException,
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceExhaustedException,
    ResourceExistsException,
    ResourceNotFoundException,
    SubstrateSdkException,
    UnAuthorizedException,
    UnknownException,
)
from multicloud.common.proxy import ProxyConfig
from multicloud.gcp_common.gcp_credentials import DEFAULT_SCOPES, get_credentials

#: Matches google.cloud.client, whose session setup is bypassed by _http.
_CREDENTIALS_REFRESH_TIMEOUT = 300

_logger = logging.getLogger(__name__)

_GCP_TAG_PREFIX = "gcp-tag-"

# GCS object compose API caps the number of source components at 32 per call.
# Through iterative composition (composing intermediates of intermediates) the
# practical ceiling for a single assembled object is 1024 components. Going
# past this surfaces as an opaque HTTP 400 from the service; we pre-flight
# both upload_part and complete to fail fast with InvalidArgumentException.
_GCS_COMPOSE_MAX_PARTS = 1024


# Blob property holding each algorithm's base64 digest. Set before upload it is
# sent in the object resource for GCS to validate; after upload it holds the
# digest GCS computed over the stored bytes.
_CHECKSUM_PROPERTIES = {
    ChecksumAlgorithm.CRC32C: "crc32c",
    ChecksumAlgorithm.MD5: "md5_hash",
}
# GCS rejects a mismatched digest with HTTP 400 and a message of the form
# 'Provided CRC32C "..." doesn't match calculated CRC32C "..."'.
_CHECKSUM_MISMATCH_MARKER = "doesn't match calculated"


@BLOB_REGISTRY.register("gcp")
class GcpBlobStore(BaseBlobStore):
    """Google Cloud Storage-backed blob store."""

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        *,
        storage_client: Any = None,
        credentials_overrider: Optional[CredentialsOverrider] = None,
        proxy_config: Optional[ProxyConfig] = None,
    ) -> None:
        super().__init__(
            bucket=bucket,
            region=region,
            credentials_overrider=credentials_overrider,
            proxy_config=proxy_config,
        )

        # Set only when a proxy policy makes this store own its sessions.
        self._auth_request: Any = None
        self._auth_session: Any = None

        if storage_client is not None:
            # Caller-supplied client takes precedence over credentials_overrider.
            self._client = storage_client
            self._credentials = None
        elif credentials_overrider is not None:
            resolved = get_credentials(credentials_overrider, scopes=DEFAULT_SCOPES)
            if resolved is not None:
                self._credentials = resolved
                self._client = storage.Client(
                    credentials=resolved, **self._proxy_client_kwargs(resolved)
                )
            else:
                # Unsupported override type (e.g. WEB_IDENTITY) → fall through to ADC.
                self._credentials, project = google.auth.default(scopes=DEFAULT_SCOPES)
                self._client = storage.Client(
                    credentials=self._credentials,
                    project=project,
                    **self._proxy_client_kwargs(self._credentials),
                )
        else:
            # Use google-cloud-storage's default Application Default Credentials (ADC):
            # 1. GOOGLE_APPLICATION_CREDENTIALS environment variable (service account key)
            # 2. gcloud auth application-default login credentials
            # 3. GCE/GKE metadata service (if running on GCP)
            # Project is auto-detected from ADC credentials.
            #
            # We request cloud-platform scope so the access token can be used for
            # IAM signBlob calls (needed by generate_signed_url when credentials
            # don't have a local private key, e.g. impersonated service accounts).
            self._credentials, project = google.auth.default(scopes=DEFAULT_SCOPES)
            self._client = storage.Client(
                credentials=self._credentials,
                project=project,
                **self._proxy_client_kwargs(self._credentials),
            )
        self._bucket_handle = self._client.bucket(bucket)

    def _apply_proxy_policy(self, session: Any) -> Any:
        """Point a session at the configured proxy, or at none at all.

        Clearing ``trust_env`` is the only thing that works: requests merges the
        environment's proxies into the per-request settings, which override the
        session's. It also stops the certificate bundle variables being read,
        hence carrying the bundle over.
        """
        assert self._proxy_config is not None
        if self._proxy_config.use_environment_variable_proxy_values and (
            self._proxy_config.endpoint is None
        ):
            return session

        # An explicit proxy and no proxy both need the environment ignored.
        ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get(
            "CURL_CA_BUNDLE"
        )
        session.trust_env = False
        if ca_bundle:
            session.verify = ca_bundle
        if self._proxy_config.endpoint is not None:
            session.proxies = {
                "http": self._proxy_config.endpoint,
                "https": self._proxy_config.endpoint,
            }
        return session

    def _proxy_client_kwargs(self, credentials: Any) -> dict:
        """Client kwargs applying the proxy policy, empty if none is needed.

        AuthorizedSession refreshes credentials over a session of its own, so
        the token endpoint needs its own ``auth_request`` or it would keep
        following the ambient proxy. Supplying ``_http`` bypasses the session
        setup google.cloud.client does, so that is reproduced here.
        """
        if self._proxy_config is None or not self._proxy_needs_session():
            return {}
        self._auth_session = self._apply_proxy_policy(requests.Session())
        self._auth_request = google.auth.transport.requests.Request(self._auth_session)
        session = AuthorizedSession(
            credentials,
            auth_request=self._auth_request,
            refresh_timeout=_CREDENTIALS_REFRESH_TIMEOUT,
        )
        session.configure_mtls_channel(None)
        return {"_http": self._apply_proxy_policy(session)}

    def _proxy_needs_session(self) -> bool:
        """Whether the policy differs from what the environment already gives."""
        config = self._proxy_config
        return config is not None and (
            config.endpoint is not None
            or not config.use_environment_variable_proxy_values
        )

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        """Translate GCP SDK exceptions to common hierarchy.

        Maps google-api-core exceptions to cloud-agnostic exception types.
        """
        # Already translated
        if isinstance(exc, SubstrateSdkException):
            return exc

        # GCP-specific exceptions
        if isinstance(exc, gcs_exceptions.NotFound):
            return ResourceNotFoundException(str(exc))
        if isinstance(exc, (gcs_exceptions.Unauthorized, gcs_exceptions.Forbidden)):
            return UnAuthorizedException(str(exc))
        # A failed precondition (if_match mismatch on read, or a conditional
        # write) returns 412 PreconditionFailed — the caller's precondition was
        # not satisfied.
        if isinstance(exc, gcs_exceptions.PreconditionFailed):
            return FailedPreconditionException(str(exc))
        if isinstance(exc, gcs_exceptions.BadRequest):
            return InvalidArgumentException(str(exc))
        # 429 means the request was rate limited; 504 is reported as a deadline.
        # 503 is deliberately left to the fallback below: an unavailable backend
        # is not the caller exceeding a quota.
        # ResourceExhausted and DeadlineExceeded are the gRPC-flavoured
        # subclasses of these, so each check covers both.
        if isinstance(exc, gcs_exceptions.TooManyRequests):
            return ResourceExhaustedException(str(exc))
        if isinstance(exc, gcs_exceptions.GatewayTimeout):
            return DeadlineExceededException(str(exc))

        # A rejected or expired token surfaces from the auth layer as
        # RefreshError, not the storage API, so google-api-core never sees it
        # and it would otherwise fall through to UnknownException. It reaches
        # here on any operation whose credential refresh is rejected -- the
        # 401-triggered refresh-and-retry, or the explicit refresh before an
        # IAM signBlob call in do_generate_presigned_url. From the caller's
        # perspective this is an authorization failure.
        if isinstance(exc, google.auth.exceptions.RefreshError):
            return UnAuthorizedException(str(exc))

        # Catch signBlob permission errors (e.g. when generating signed URLs
        # without iam.serviceAccounts.signBlob permission)
        msg = str(exc).lower()
        if "permission_denied" in msg or "sign_bytes" in msg:
            return UnAuthorizedException(str(exc))

        # Fallback for unknown exceptions
        return UnknownException(str(exc))

    @staticmethod
    def _apply_object_lock(
        blob: Any, object_lock: Optional[ObjectLockConfiguration]
    ) -> None:
        """Apply object-lock retention to a GCS blob before upload.

        Retention maps to the blob's retention config: COMPLIANCE becomes a
        "Locked" retention (immutable until the retain-until time), any other
        mode becomes "Unlocked". Retention is a writable object field, so it
        travels with the upload (objects.insert) request.

        Legal holds are NOT set here — GCS does not accept ``temporaryHold`` /
        ``eventBasedHold`` on the insert request, so they are applied after the
        upload via :meth:`_apply_hold_after_upload`.
        """
        if object_lock is None:
            return
        if object_lock.mode is not None and object_lock.retain_until_date is not None:
            # Mutates the blob in memory only; the retention config is serialized
            # onto the objects.insert request by the subsequent upload call. Do
            # NOT turn this into a pre-upload blob.patch() — the object does not
            # exist yet.
            retention = blob.retention
            retention.mode = (
                "Locked" if object_lock.mode is RetentionMode.COMPLIANCE else "Unlocked"
            )
            retention.retain_until_time = object_lock.retain_until_date

    @staticmethod
    def _apply_hold_after_upload(
        blob: Any, object_lock: Optional[ObjectLockConfiguration]
    ) -> None:
        """Set the legal hold on an already-uploaded GCS blob.

        GCS rejects ``temporaryHold`` / ``eventBasedHold`` on the upload
        (objects.insert) request — they are only writable via a metadata
        update (objects.patch). This must therefore run after the upload
        call, once the object exists. Legal hold maps to a temporary hold by
        default, or an event-based hold when ``use_event_based_hold`` is set.
        """
        if object_lock is None or not object_lock.legal_hold:
            return
        if object_lock.use_event_based_hold:
            blob.event_based_hold = True
        else:
            blob.temporary_hold = True
        blob.patch()

    @staticmethod
    def _resolve_upload_conditions(
        blob: Any,
        *,
        create_if_absent: bool,
        if_match: Optional[str],
        if_version_match: Optional[str],
        kms_key_id: Optional[str],
    ) -> dict[str, Any]:
        """Adapt shared write-precondition policy onto the GCS blob handle.

        Policy (ETag-only reject, generation mapping) lives in
        :func:`resolve_generation_match`. This adapter applies an explicit
        ``kms_key_id`` as ``kms_key_name`` on *blob* and returns
        ``if_generation_match`` kwargs for ``upload_from_string`` /
        ``upload_from_filename``.
        """
        if kms_key_id is not None:
            blob.kms_key_name = kms_key_id
        generation = resolve_generation_match(
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
        )
        if generation is None:
            return {}
        return {"if_generation_match": generation}

    @staticmethod
    def _put_object_conditional(
        upload_call: Callable[[], None], create_if_absent: bool
    ) -> None:
        """Run *upload_call*, re-labelling a create-if-absent conflict.

        With ``if_generation_match=0`` GCS returns HTTP 412 when a live object
        already exists at the key; surface that as ResourceExistsException so
        create-if-absent callers get a semantically precise error. Every other
        412 (none currently reachable on upload) propagates for the client layer
        to translate. This local catch exists solely for that semantic
        disambiguation and re-raises everything else untouched.
        """
        if not create_if_absent:
            upload_call()
            return
        try:
            upload_call()
        except gcs_exceptions.PreconditionFailed as exc:
            raise ResourceExistsException(str(exc)) from exc

    def do_upload(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: Optional[str],
        content_length: Optional[int],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration],
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
        checksum_value: Optional[str] = None,
        checksum_algorithm: Optional[ChecksumAlgorithm] = None,
    ) -> UploadResponse:
        blob = self._bucket_handle.blob(key)
        # GCS has no native object tagging API; encode tags as
        # ``gcp-tag-<key>`` entries in the per-object metadata so they
        # round-trip via do_get_tags / do_set_tags. The ``gcp-tag-`` prefix
        # is reserved by the SDK — callers who set ``gcp-tag-*`` keys via
        # ``metadata=`` may have those entries overwritten by subsequent
        # tag operations.
        if metadata or tags:
            merged: dict[str, str] = dict(metadata) if metadata else {}
            for tag_key, tag_value in tags.items():
                merged[f"{_GCP_TAG_PREFIX}{tag_key}"] = tag_value
            blob.metadata = merged
        self._apply_object_lock(blob, object_lock)
        # use_kms_managed_key with no explicit key is a no-op on GCS: objects
        # are always encrypted at rest with Google-managed keys by default.
        if use_kms_managed_key:
            _logger.debug(
                "use_kms_managed_key is a no-op on GCS (objects are always "
                "encrypted at rest) and was ignored"
            )
        upload_kwargs = self._resolve_upload_conditions(
            blob,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
            kms_key_id=kms_key_id,
        )
        checksum_property = None
        if checksum_algorithm is not None:
            checksum_property = _CHECKSUM_PROPERTIES[checksum_algorithm]
            setattr(blob, checksum_property, checksum_value)
            # The library's own checksum option would replace the caller's value.
            upload_kwargs["checksum"] = None
        try:
            self._put_object_conditional(
                lambda: blob.upload_from_string(
                    payload,
                    content_type=content_type or "application/octet-stream",
                    **upload_kwargs,
                ),
                create_if_absent,
            )
        except gcs_exceptions.BadRequest as exc:
            if checksum_algorithm is not None and _CHECKSUM_MISMATCH_MARKER in str(exc):
                raise InvalidArgumentException(
                    CHECKSUM_MISMATCH_MSG.format(checksum_algorithm.value, key, exc)
                ) from exc
            raise
        self._apply_hold_after_upload(blob, object_lock)
        return UploadResponse(
            key=key,
            e_tag=blob.etag,
            version_id=str(blob.generation) if blob.generation is not None else None,
            checksum_algorithm=checksum_algorithm,
            checksum_value=(
                getattr(blob, checksum_property) if checksum_property else None
            ),
        )

    def do_upload_from_path(
        self,
        key: str,
        path: Path,
        *,
        content_type: Optional[str],
        content_length: Optional[int],
        metadata: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration],
        create_if_absent: bool = False,
        if_match: Optional[str] = None,
        if_version_match: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_kms_managed_key: bool = False,
    ) -> UploadResponse:
        """Upload a local file to GCS using the native upload_from_filename API.

        Uses ``Blob.upload_from_filename()`` which automatically handles
        resumable uploads for files larger than 8 MiB, providing better
        performance and reliability for large file uploads without loading
        the entire file into memory.
        """
        blob = self._bucket_handle.blob(key)
        if metadata:
            blob.metadata = dict(metadata)
        self._apply_object_lock(blob, object_lock)
        upload_kwargs = self._resolve_upload_conditions(
            blob,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
            kms_key_id=kms_key_id,
        )

        self._put_object_conditional(
            lambda: blob.upload_from_filename(
                str(path),
                content_type=content_type or "application/octet-stream",
                **upload_kwargs,
            ),
            create_if_absent,
        )
        self._apply_hold_after_upload(blob, object_lock)

        return UploadResponse(
            key=key,
            e_tag=blob.etag,
            version_id=str(blob.generation) if blob.generation is not None else None,
        )

    def _parse_generation(self, version_id: Optional[str]) -> Optional[int]:
        """Convert version_id string to GCS integer generation."""
        if not version_id:
            return None
        try:
            return int(version_id)
        except ValueError as exc:
            raise InvalidArgumentException(
                f"GCP version_id must be an integer generation; got " f"{version_id!r}"
            ) from exc

    def _build_metadata(self, key: str, blob: Any) -> BlobMetadata:
        """Extract BlobMetadata from a GCS blob handle after reload.

        Serves both ``do_download`` and ``do_get_metadata`` — both hold a
        reloaded blob handle, so both get identically-populated metadata,
        including the timestamps. ``blob.updated`` / ``blob.time_created`` are
        already tz-aware datetimes. Custom metadata is returned raw (the
        internal ``gcp-tag-*`` entries are NOT stripped here — see
        ``get_tags`` for the tag-filtered view).

        ``replication_status`` is deliberately left unset (``None``). GCS has no
        per-object replication-status field in the Objects resource — bucket
        replication (cross-bucket / turbo) is configured and reported at the
        bucket level, not stamped on each object like the AWS
        ``x-amz-replication-status`` header. ``None`` here therefore means "no
        per-object signal available", NOT "this object is not replicated";
        callers must not read the absence of a status as unreplicated.
        """
        return BlobMetadata(
            key=key,
            content_length=blob.size,
            content_type=blob.content_type,
            e_tag=blob.etag,
            version_id=str(blob.generation) if blob.generation is not None else None,
            metadata=blob.metadata or {},
            last_modified=blob.updated,
            created_time=blob.time_created,
        )

    @staticmethod
    def _etag_conditions(if_match: Optional[str]) -> dict[str, Any]:
        """Map the cloud-agnostic ETag precondition to GCS download kwargs.

        ``if_match`` → ``if_etag_match`` (read only when the ETag matches).
        A failed condition surfaces from GCS as HTTP 412, translated to
        FailedPreconditionException in the client layer.
        """
        conditions: dict[str, Any] = {}
        if if_match is not None:
            conditions["if_etag_match"] = if_match
        return conditions

    def do_download(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
        if_match: Optional[str] = None,
    ) -> DownloadResponse:
        generation = self._parse_generation(version_id)
        conditions = self._etag_conditions(if_match)

        blob = self._bucket_handle.blob(key, generation=generation)
        try:
            if start is None and end is None:
                body = blob.download_as_bytes(**conditions)
            else:
                body = blob.download_as_bytes(
                    start=start,
                    end=end,
                    **conditions,
                )
            blob.reload()
        except gcs_exceptions.GoogleAPIError as exc:
            raise self.translate_error(exc) from exc

        metadata = self._build_metadata(key, blob)
        return DownloadResponse(key=key, content=body, metadata=metadata)

    def do_download_to_path(
        self,
        key: str,
        destination: Path,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> DownloadResponse:
        """Download to a local file path using GCS download_to_filename.

        Writes to a temp file in the destination's parent directory and atomically
        renames it into place on success, so a mid-stream failure never leaves a
        partial/corrupted file at ``destination``.
        """
        generation = self._parse_generation(version_id)

        blob = self._bucket_handle.blob(key, generation=generation)
        tmp_file = tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        )
        tmp_file.close()  # close the handle; download_to_filename writes by path
        tmp_path = tmp_file.name
        try:
            blob.download_to_filename(
                tmp_path,
                start=start,
                end=end,
            )
            blob.reload()
            os.replace(tmp_path, destination)
        except BaseException as exc:
            # Streaming, reload, or rename failed — remove the temp file so
            # callers never see a partial/corrupted artifact.
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            if isinstance(exc, gcs_exceptions.GoogleAPIError):
                raise self.translate_error(exc) from exc
            raise

        return DownloadResponse(
            key=key, content=b"", metadata=self._build_metadata(key, blob)
        )

    def do_download_to_file(
        self,
        key: str,
        file_obj: BinaryIO,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> DownloadResponse:
        """Download to a file-like object using GCS download_to_file."""
        generation = self._parse_generation(version_id)

        blob = self._bucket_handle.blob(key, generation=generation)
        try:
            blob.download_to_file(
                file_obj,
                start=start,
                end=end,
            )
            blob.reload()
        except gcs_exceptions.GoogleAPIError as exc:
            raise self.translate_error(exc) from exc

        return DownloadResponse(
            key=key, content=b"", metadata=self._build_metadata(key, blob)
        )

    def do_download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> StreamingDownloadResponse:
        """Return a streaming response using GCS blob.open() for lazy reads.

        Both full-object and range downloads stream lazily: bytes are fetched
        from GCS on demand as the caller consumes chunks, never buffered all
        at once. Ranges are realized by seeking the BlobReader to ``start``
        and bounding reads to ``end`` (inclusive), matching the semantics of
        ``download_as_bytes(start=, end=)``.

        The caller must close the response to release the read channel.
        """
        generation = self._parse_generation(version_id)

        blob = self._bucket_handle.blob(key, generation=generation)
        try:
            blob.reload()
        except gcs_exceptions.GoogleAPIError as exc:
            raise self.translate_error(exc) from exc

        metadata = self._build_metadata(key, blob)

        try:
            reader = blob.open("rb", chunk_size=8192)
        except gcs_exceptions.GoogleAPIError as exc:
            raise self.translate_error(exc) from exc

        if start is not None or end is not None:
            try:
                reader.seek(start or 0)
            except gcs_exceptions.GoogleAPIError as exc:
                reader.close()
                raise self.translate_error(exc) from exc

        # ``end`` matches Blob.download_as_bytes semantics: it is INCLUSIVE,
        # so range [start, end] contains end - (start or 0) + 1 bytes. None
        # means "read to EOF".
        remaining: Optional[int] = (end - (start or 0) + 1) if end is not None else None

        def _chunk_iterator() -> Iterator[bytes]:
            """Lazily yield chunks from the GCS BlobReader.

            Reads on demand — no buffering of the full body or range. For
            ranges, stops once ``remaining`` bytes have been emitted.
            """
            nonlocal remaining
            while True:
                if remaining is not None:
                    if remaining <= 0:
                        break
                    to_read = min(8192, remaining)
                else:
                    to_read = 8192
                chunk = reader.read(to_read)
                if not chunk:
                    break
                if remaining is not None:
                    remaining -= len(chunk)
                yield chunk

        return StreamingDownloadResponse(
            key=key,
            metadata=metadata,
            stream=_chunk_iterator(),
            close_func=reader.close,
        )

    def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Delete a single object.

        Without version_id: idempotent — silent no-op when the key does not exist.
        With version_id: targets that specific generation; raises
        ResourceNotFoundException if the generation no longer exists.
        """
        try:
            generation = self._parse_generation(version_id)
            blob = self._bucket_handle.blob(key, generation=generation)
            blob.delete()
        except gcs_exceptions.NotFound:
            if version_id is not None:
                raise  # versioned delete surfaces NotFound
            return  # unversioned: idempotent

    def do_delete_objects(self, objects: List[BlobIdentifier]) -> DeleteObjectsResponse:
        """Batch delete objects from GCS.

        Individual calls are used (rather than the GCS batch transport)
        because the batch context manager does not expose per-object
        error details — only the last exception is raised or all are
        silently swallowed. Per-object error reporting is required by
        the DeleteObjectsResponse contract.

        Chunking is intentionally left to the caller. The driver
        processes the full list in one pass to keep the implementation
        simple and predictable.
        """
        errors: List[DeleteError] = []

        for obj in objects:
            try:
                generation = self._parse_generation(obj.version_id)
                blob = self._bucket_handle.blob(obj.key, generation=generation)
                blob.delete()
            except gcs_exceptions.NotFound:
                # Idempotent — don't report as error
                pass
            except Exception as exc:
                errors.append(
                    DeleteError(
                        key=obj.key,
                        version_id=obj.version_id,
                        code=type(exc).__name__,
                        message=str(exc),
                    )
                )

        return DeleteObjectsResponse(errors=errors)

    def do_copy(
        self,
        src_key: str,
        *,
        dest_bucket: str,
        dest_key: str,
        src_version_id: Optional[str],
    ) -> CopyResponse:
        """Copy from this bucket to another bucket using GCS copy_blob."""
        source_blob = self._bucket_handle.blob(src_key)
        dest_bucket_handle = self._client.bucket(dest_bucket)
        # copy_blob reads the source generation from its own kwarg, not
        # from Blob.generation, so we pass it here explicitly.
        copy_kwargs = {}
        if src_version_id is not None:
            copy_kwargs["source_generation"] = int(src_version_id)
        new_blob = self._bucket_handle.copy_blob(
            source_blob, dest_bucket_handle, new_name=dest_key, **copy_kwargs
        )

        return CopyResponse(
            key=dest_key,
            version_id=(
                str(new_blob.generation) if new_blob.generation is not None else None
            ),
            e_tag=new_blob.etag,
            last_modified=new_blob.updated,
        )

    def do_copy_from(
        self,
        src_bucket: str,
        src_key: str,
        *,
        dest_key: str,
        src_version_id: Optional[str],
    ) -> CopyResponse:
        """Copy from a source bucket into this bucket using GCS copy_blob."""
        src_bucket_handle = self._client.bucket(src_bucket)
        source_blob = src_bucket_handle.blob(src_key)
        # copy_blob reads the source generation from its own kwarg, not
        # from Blob.generation, so we pass it here explicitly.
        copy_kwargs = {}
        if src_version_id is not None:
            copy_kwargs["source_generation"] = int(src_version_id)
        new_blob = src_bucket_handle.copy_blob(
            source_blob, self._bucket_handle, new_name=dest_key, **copy_kwargs
        )

        return CopyResponse(
            key=dest_key,
            version_id=(
                str(new_blob.generation) if new_blob.generation is not None else None
            ),
            e_tag=new_blob.etag,
            last_modified=new_blob.updated,
        )

    def do_get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Retrieve object lock state from GCS object properties.

        Reads retention (mode + expiration) and the temporary / event-based
        holds off the object. Retention counts as a lock only when it carries
        an actual value: LOCKED retention maps to COMPLIANCE, anything else
        maps to GOVERNANCE. A legal hold is reported when either the temporary
        or event-based hold is set. Returns None only when the object has
        neither retention nor a hold.

        Note: the google-cloud-storage client always exposes ``blob.retention``
        as a (possibly empty) ``Retention`` object rather than ``None``, so the
        presence of the object cannot be used to detect configured retention.
        We inspect its ``mode`` / ``retain_until_time`` fields instead.
        """
        generation = self._parse_generation(version_id)
        blob = self._bucket_handle.blob(key, generation=generation)
        blob.reload()

        retention = getattr(blob, "retention", None)
        retention_mode = getattr(retention, "mode", None) if retention else None
        retain_until = (
            getattr(retention, "retain_until_time", None) if retention else None
        )
        has_retention = retention_mode is not None or retain_until is not None

        event_based_hold = bool(blob.event_based_hold)
        temporary_hold = bool(blob.temporary_hold)
        legal_hold = event_based_hold or temporary_hold

        if not has_retention and not legal_hold:
            return None

        mode: Optional[RetentionMode] = None
        if has_retention:
            mode = (
                RetentionMode.COMPLIANCE
                if retention_mode == "Locked"
                else RetentionMode.GOVERNANCE
            )

        return ObjectLockInfo(
            mode=mode,
            retain_until_date=retain_until,
            legal_hold=legal_hold,
            event_based_hold=event_based_hold,
        )

    def do_update_legal_hold(
        self, key: str, enabled: bool, *, version_id: Optional[str] = None
    ) -> None:
        """Set or remove a legal hold via GCS object hold flags.

        GCS has no single "legal hold" property; it exposes two independent
        hold flags -- the temporary hold and the event-based hold -- either of
        which keeps the object undeletable regardless of any retention policy.
        Because either flag alone holds the object, the two directions are
        asymmetric:

        - Placing a hold (``enabled`` True) sets a single flag: the event-based
          hold if one is already active on the object, otherwise the temporary
          hold. One flag is sufficient to hold the object, so there is no need
          to add a second, redundant flag.
        - Releasing a hold (``enabled`` False) clears *both* flags. An object
          can carry both holds at once (e.g. a bucket-default event-based hold
          plus a temporary hold from a prior call); clearing only one would
          leave the object held by the other while reporting success, and would
          disagree with ``get_object_lock``, which reports a legal hold when
          *either* flag is set.

        The operation reads the object's current state first -- to decide which
        flag to set on enable, and to capture the metageneration for the write.
        The read-modify-write is guarded with ``if_metageneration_match`` so a
        concurrent modification between the read and the write surfaces as a
        precondition failure (412) rather than clobbering an unrelated change.

        Retention-intent caveat: GCS exposes no metadata for *why* an
        event-based hold was placed, so releasing a legal hold clears an active
        event-based hold even when it was set for retention rather than
        legal-hold reasons. Making the two concerns fully orthogonal (legal
        hold only ever touching the temporary hold) would be a larger contract
        change.

        A missing object raises an error that propagates to the client layer
        for translation.
        """
        generation = self._parse_generation(version_id)
        blob = self._bucket_handle.blob(key, generation=generation)

        # Read first: on enable the flag to set depends on which hold is already
        # active, and if_metageneration_match then guards the write against a
        # concurrent modification landing between the reload and the patch.
        blob.reload()
        metageneration = blob.metageneration
        if enabled:
            # One flag is enough to hold the object: reuse an active event-based
            # hold, otherwise set the temporary hold rather than adding both.
            if not blob.event_based_hold:
                blob.temporary_hold = True
        else:
            # Clear both: either flag left set would keep the object held and
            # contradict get_object_lock's "held if either flag" read semantics.
            blob.event_based_hold = False
            blob.temporary_hold = False
        blob.patch(if_metageneration_match=metageneration)

    def do_get_tags(self, key: str) -> Mapping[str, str]:
        """Retrieve tags from GCS object metadata (gcp-tag-* entries).

        GCS has no native object tagging API. Tags are stored as custom
        metadata entries with a ``gcp-tag-`` prefix.
        """
        blob = self._bucket_handle.blob(key)
        blob.reload()
        metadata = blob.metadata or {}
        return {
            k[len(_GCP_TAG_PREFIX) :]: v
            for k, v in metadata.items()
            if k.startswith(_GCP_TAG_PREFIX) and v is not None
        }

    def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Store tags as gcp-tag-* metadata entries on a GCS object.

        Replaces all existing tag entries while preserving non-tag
        metadata. Setting old tag keys to ``None`` removes them via
        the GCS metadata PATCH semantics, then new tags are added.

        Uses ``if_metageneration_match`` on the patch call to detect
        concurrent metadata modifications — GCS returns 412 Precondition
        Failed if metadata was changed between reload() and patch().
        """
        blob = self._bucket_handle.blob(key)
        blob.reload()

        # Capture metageneration for optimistic concurrency control
        metageneration = blob.metageneration

        # Copy existing metadata
        existing = blob.metadata or {}
        metadata: dict[str, Optional[str]] = dict(existing)

        # Mark all existing tag keys for deletion (None means remove)
        for k in list(metadata.keys()):
            if k.startswith(_GCP_TAG_PREFIX):
                metadata[k] = None

        # Add new tags (these overwrite the Nones for matching keys)
        for tag_key, tag_value in tags.items():
            metadata[f"{_GCP_TAG_PREFIX}{tag_key}"] = tag_value

        # GCS PATCH semantics: None values delete the corresponding metadata key.
        # The dict contains Optional[str] values but blob.metadata is typed as
        # Mapping[str, str], hence the type: ignore.
        blob.metadata = metadata  # type: ignore[assignment]
        blob.patch(if_metageneration_match=metageneration)

    def do_update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str],
    ) -> None:
        """Update object retention on a GCS object.

        GCS retention modes are ``Locked`` (irreversible, maps to
        COMPLIANCE) and ``Unlocked`` (reducible by privileged callers,
        maps to GOVERNANCE). Reads the current retention off the reloaded
        blob, resolves the effective mode against the shared rules table,
        then patches it back. ``override_unlocked_retention`` must be set
        when shortening an existing Unlocked retention, so it is forwarded
        from ``bypass_governance``.
        """
        generation = self._parse_generation(version_id)
        blob = self._bucket_handle.blob(key, generation=generation)
        blob.reload()

        current = getattr(blob, "retention", None)
        current_mode: Optional[RetentionMode] = None
        current_retain_until: Optional[datetime] = None
        if current is not None and getattr(current, "mode", None) is not None:
            current_mode = (
                RetentionMode.COMPLIANCE
                if current.mode == "Locked"
                else RetentionMode.GOVERNANCE
            )
            current_retain_until = getattr(current, "retain_until_time", None)

        resolved_mode = resolve_and_validate(
            current_mode, current_retain_until, retention
        )

        gcs_mode = "Locked" if resolved_mode == RetentionMode.COMPLIANCE else "Unlocked"
        blob.retention.mode = gcs_mode
        blob.retention.retain_until_time = retention.retain_until_date
        blob.patch(override_unlocked_retention=retention.bypass_governance)

    def do_generate_presigned_url(
        self,
        key: str,
        operation: PresignedOperation,
        *,
        duration_seconds: int,
        content_type: Optional[str],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        kms_key_id: Optional[str],
        content_disposition: Optional[str],
    ) -> PresignedUrlResponse:
        """Generate a V4 signed URL for GCS.

        Note: kms_key_id is silently ignored — GCS does not support
        customer-managed encryption keys in signed URL generation.

        Uses the IAM signBlob API via access token when credentials lack
        a local private key (e.g. impersonated service accounts, ADC).

        A proxy configuration reaches the credential refresh below but not the
        signBlob call, for which the storage library builds its own transport
        and takes no session. Credentials with a local private key sign offline
        and are unaffected.
        """
        blob = self._bucket_handle.blob(key)

        # Ensure credentials are refreshed so we have a valid access token
        # for the IAM signBlob API call. ``self._credentials`` is only None
        # when ``storage_client`` was injected (test-only path); real callers
        # always go through one of the ADC / overrider branches in __init__.
        creds = self._credentials
        assert creds is not None
        if not creds.valid:
            creds.refresh(
                self._auth_request or google.auth.transport.requests.Request()
            )

        # Determine signing credentials: if the credential has a local signer
        # with a real private key (service account key file),
        # generate_signed_url uses it directly — no IAM call needed.
        # Otherwise (impersonated SA, ADC, GCE metadata, or IAM-backed signer),
        # pass access_token + service_account_email so the library signs via
        # the IAM signBlob API.
        sign_kwargs: dict[str, Any] = {}
        has_local_signer = (
            hasattr(creds, "signer")
            and hasattr(creds.signer, "sign")
            and hasattr(creds.signer, "key_id")
            and creds.signer.key_id is not None
        )
        if not has_local_signer:
            # No local private key — use access token signing via IAM signBlob
            sign_kwargs["service_account_email"] = getattr(
                creds, "service_account_email", None
            )
            sign_kwargs["access_token"] = creds.token

        if operation == PresignedOperation.UPLOAD:
            return self._generate_signed_upload_url(
                blob,
                duration_seconds=duration_seconds,
                content_type=content_type,
                metadata=metadata,
                tags=tags,
                content_disposition=content_disposition,
                **sign_kwargs,
            )
        elif operation == PresignedOperation.DOWNLOAD:
            return self._generate_signed_download_url(
                blob,
                duration_seconds=duration_seconds,
                content_disposition=content_disposition,
                **sign_kwargs,
            )
        else:
            raise InvalidArgumentException(
                f"Unsupported presigned URL operation: {operation}"
            )

    def _generate_signed_upload_url(
        self,
        blob: "storage.Blob",
        *,
        duration_seconds: int,
        content_type: Optional[str],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        content_disposition: Optional[str],
        **sign_kwargs: Any,
    ) -> PresignedUrlResponse:
        """Build a V4 signed PUT URL for GCS."""
        headers: dict[str, str] = {}
        required_headers: dict[str, str] = {}

        if content_type:
            required_headers["Content-Type"] = content_type

        if metadata:
            for meta_key, meta_value in metadata.items():
                header_key = f"x-goog-meta-{meta_key}"
                headers[header_key] = meta_value
                required_headers[header_key] = meta_value

        if tags:
            for tag_key, tag_value in tags.items():
                header_key = f"x-goog-meta-gcp-tag-{tag_key}"
                headers[header_key] = tag_value
                required_headers[header_key] = tag_value

        if content_disposition:
            headers["Content-Disposition"] = content_disposition
            required_headers["Content-Disposition"] = content_disposition

        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=duration_seconds),
            method="PUT",
            headers=headers or None,
            content_type=content_type,
            **sign_kwargs,
        )
        return PresignedUrlResponse(
            url=url, http_method="PUT", required_headers=required_headers
        )

    def _generate_signed_download_url(
        self,
        blob: "storage.Blob",
        *,
        duration_seconds: int,
        content_disposition: Optional[str],
        **sign_kwargs: Any,
    ) -> PresignedUrlResponse:
        """Build a V4 signed GET URL for GCS."""
        kwargs: dict = {
            "version": "v4",
            "expiration": timedelta(seconds=duration_seconds),
            "method": "GET",
        }
        if content_disposition:
            kwargs["response_disposition"] = content_disposition
        kwargs.update(sign_kwargs)

        url = blob.generate_signed_url(**kwargs)
        return PresignedUrlResponse(url=url, http_method="GET", required_headers={})

    def do_list_page(
        self,
        *,
        prefix: Optional[str],
        delimiter: Optional[str],
        max_results: int,
        pagination_token: Optional[str],
    ) -> ListBlobsPageResponse:
        """List a single page of objects using GCS list_blobs."""
        kwargs: dict[str, Any] = {"max_results": max_results}
        if prefix is not None:
            kwargs["prefix"] = prefix
        if delimiter is not None:
            kwargs["delimiter"] = delimiter
        if pagination_token is not None:
            kwargs["page_token"] = pagination_token

        try:
            iterator = self._client.list_blobs(self.bucket, **kwargs)
            try:
                page = next(iterator.pages)
            except StopIteration:
                return ListBlobsPageResponse()
        except gcs_exceptions.GoogleAPIError as exc:
            raise self.translate_error(exc) from exc

        blobs = [
            BlobInfo(
                key=blob.name,
                size=blob.size,
                e_tag=blob.etag,
                last_modified=blob.updated,
                storage_class=blob.storage_class,
            )
            for blob in page
        ]
        common_prefixes = list(iterator.prefixes or [])
        return ListBlobsPageResponse(
            blobs=blobs,
            common_prefixes=common_prefixes,
            is_truncated=iterator.next_page_token is not None,
            next_pagination_token=iterator.next_page_token,
        )

    def do_does_object_exist(self, key: str, *, version_id: Optional[str]) -> bool:
        """Check object existence using GCS blob.exists()."""
        generation: Optional[int] = None
        if version_id is not None:
            try:
                generation = int(version_id)
            except ValueError as exc:
                raise InvalidArgumentException(
                    f"GCP version_id must be an integer generation; "
                    f"got {version_id!r}"
                ) from exc
        blob = self._bucket_handle.blob(key, generation=generation)
        # blob.exists() handles 404 internally and returns False, so we
        # only need to translate genuine API errors (auth, server, etc.).
        return bool(blob.exists())

    def do_does_bucket_exist(self) -> bool:
        """Check bucket existence using GCS bucket.exists()."""
        # bucket.exists() handles 404 internally and returns False, so we
        # only need to translate genuine API errors (auth, server, etc.).
        return bool(self._bucket_handle.exists())

    def do_get_metadata(self, key: str, *, version_id: Optional[str]) -> BlobMetadata:
        """Retrieve object metadata (HEAD-style) without downloading the body.

        Uses a metadata-only ``reload()`` on the blob handle, so no object
        body is transferred. Not-found and other API errors are routed
        through ``translate_error`` (NotFound -> ResourceNotFoundException).

        Raises:
            InvalidArgumentException: If version_id is not an integer generation.
        """
        generation = self._parse_generation(version_id)
        blob = self._bucket_handle.blob(key, generation=generation)
        try:
            blob.reload()
        except gcs_exceptions.GoogleAPIError as exc:
            raise self.translate_error(exc) from exc

        # reload() populated the handle, so reuse the shared mapping (keeps
        # download() and get_metadata() returning identical BlobMetadata).
        return self._build_metadata(key, blob)

    def do_list_blob_versions(self, key: str) -> Iterator[BlobInfo]:
        """List all versions of a blob using GCS list_blobs(versions=True).

        Note:
            This is a lazy generator. If a transient error occurs mid-iteration,
            the caller will receive a partial result followed by a translated
            exception. If the caller does not fully consume the iterator, errors
            on subsequent pages will not surface.
        """
        try:
            blobs = self._client.list_blobs(self.bucket, prefix=key, versions=True)
            for blob in blobs:
                # Prefix match could include other keys; filter to exact key
                if blob.name != key:
                    continue
                yield BlobInfo(
                    key=blob.name,
                    size=blob.size,
                    e_tag=blob.etag,
                    last_modified=blob.updated,
                    storage_class=blob.storage_class,
                    version_id=(
                        str(blob.generation) if blob.generation is not None else None
                    ),
                    is_latest=(blob.time_deleted is None),
                )
        except gcs_exceptions.GoogleAPIError as exc:
            raise self.translate_error(exc) from exc

    # multipart upload operations
    # Preview. The public google-cloud-storage Python client does not expose
    # a stepwise multipart upload API today, so we emulate the contract by
    # uploading each part as a temp object and composing them on complete.
    # The upload_id is a client-generated namespace token embedded in those
    # temp object names — not a server-issued multipart session id.
    #
    # Known divergences from AWS native multipart:
    #
    # 1. Orphaned temp objects on crash. If the process dies between
    #    upload_part and complete/abort, temp blobs stay in the bucket and
    #    keep incurring storage cost. GCS has no equivalent of S3's
    #    AbortIncompleteMultipartUpload lifecycle rule.
    # 2. Part-count ceiling of 1024 (vs AWS's 10,000). Enforced via the
    #    pre-flight check in do_upload_part / do_complete_multipart_upload.
    # 3. No cross-process resume. metadata and content_type live only on the
    #    in-memory MultipartUpload dataclass — a separate worker holding only
    #    bucket+key+upload_id will silently lose both on the final object.
    # 4. ETag is GCS's CRC32C-based etag (e.g. CIOT/7D1jJUDEAE=), not S3's
    #    md5-N form (e.g. 4207311c90fcf5d074667f3a1eac7a04-2). Not comparable
    #    across providers.
    # 5. complete_multipart_upload ignores caller-supplied part ETags —
    #    parts are composed in part-number order from whatever bytes are at
    #    the temp keys; a stale or mismatched part cannot be detected.

    def do_initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Mapping[str, str],
        content_type: Optional[str],
        tags: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration] = None,
    ) -> MultipartUpload:
        """Initiate a multipart upload session (preview, compose-based).

        Tags are carried on the in-memory handle and applied to the final
        composed object in do_complete_multipart_upload; GCS has no native
        tagging API, so they land as ``gcp-tag-*`` metadata there. The object
        lock is likewise carried on the handle and applied to the composed
        object at completion — the final object does not exist yet here.
        """
        import uuid

        upload_id = uuid.uuid4().hex
        return MultipartUpload(
            bucket=self.bucket,
            key=key,
            upload_id=upload_id,
            metadata=metadata,
            content_type=content_type,
            tags=tags,
            object_lock=object_lock,
        )

    def do_upload_part(
        self, mpu: MultipartUpload, part: MultipartPart
    ) -> UploadPartResponse:
        """Upload a single part as a temporary object (preview, compose-based)."""
        if part.part_number > _GCS_COMPOSE_MAX_PARTS:
            raise InvalidArgumentException(
                f"GCP multipart upload supports at most {_GCS_COMPOSE_MAX_PARTS} "
                f"parts; got part_number={part.part_number}"
            )
        temp_key = f"{mpu.key}__multipart__{mpu.upload_id}__{part.part_number}"
        blob = self._bucket_handle.blob(temp_key)
        if isinstance(part.content, bytes):
            blob.upload_from_string(
                part.content,
                content_type="application/octet-stream",
            )
        else:
            blob.upload_from_file(
                part.content,
                content_type="application/octet-stream",
                size=part.content_length,
            )
        return UploadPartResponse(
            part_number=part.part_number,
            e_tag=blob.etag or "",
            size_in_bytes=part.content_length,
        )

    _GCS_MAX_COMPOSE = 32

    def do_complete_multipart_upload(
        self, mpu: MultipartUpload, parts: List[UploadPartResponse]
    ) -> MultipartUploadResponse:
        """Compose all uploaded parts into the final object (preview, compose-based)."""
        if len(parts) > _GCS_COMPOSE_MAX_PARTS:
            raise InvalidArgumentException(
                f"GCP multipart upload supports at most {_GCS_COMPOSE_MAX_PARTS} "
                f"parts; got {len(parts)}"
            )
        sorted_parts = sorted(parts, key=lambda p: p.part_number)
        source_blobs = [
            self._bucket_handle.blob(
                f"{mpu.key}__multipart__{mpu.upload_id}__{p.part_number}"
            )
            for p in sorted_parts
        ]
        destination = self._bucket_handle.blob(mpu.key)
        if mpu.content_type:
            destination.content_type = mpu.content_type
        # GCS has no native object tagging API; encode tags as ``gcp-tag-<key>``
        # entries in the object metadata so they round-trip via do_get_tags /
        # do_set_tags — same scheme as do_upload.
        if mpu.metadata or mpu.tags:
            merged: dict[str, str] = dict(mpu.metadata) if mpu.metadata else {}
            for tag_key, tag_value in mpu.tags.items():
                merged[f"{_GCP_TAG_PREFIX}{tag_key}"] = tag_value
            destination.metadata = merged
        # Retention is a writable object field, so set it on the destination
        # before compose to serialize it onto the composed object. Legal holds
        # cannot ride the compose request and are applied after (see below).
        self._apply_object_lock(destination, mpu.object_lock)

        try:
            # GCS compose supports max 32 sources per call; use iterative
            # composition for larger uploads.
            if len(source_blobs) <= self._GCS_MAX_COMPOSE:
                destination.compose(source_blobs)
            else:
                self._iterative_compose(source_blobs, destination, mpu)
        finally:
            # Best-effort cleanup; runs even on compose failure to avoid
            # orphaning temp part objects.
            for blob in source_blobs:
                try:
                    blob.delete()
                except gcs_exceptions.GoogleAPIError:
                    pass

        destination.reload()
        # A legal hold is only writable via objects.patch once the object
        # exists, so it is applied after compose — mirrors do_upload.
        self._apply_hold_after_upload(destination, mpu.object_lock)
        return MultipartUploadResponse(e_tag=destination.etag)

    def _iterative_compose(
        self,
        source_blobs: List["storage.Blob"],
        destination: "storage.Blob",
        mpu: MultipartUpload,
    ) -> None:
        """Compose >32 blobs by batching into intermediates. Best-effort cleanup."""
        intermediate_blobs: List["storage.Blob"] = []
        remaining = list(source_blobs)
        batch_idx = 0

        try:
            while len(remaining) > self._GCS_MAX_COMPOSE:
                batch = remaining[: self._GCS_MAX_COMPOSE]
                remaining = remaining[self._GCS_MAX_COMPOSE :]
                intermediate_key = (
                    f"{mpu.key}__multipart__{mpu.upload_id}__intermediate_{batch_idx}"
                )
                intermediate = self._bucket_handle.blob(intermediate_key)
                # GCS compose() is atomic — a failed compose creates no object,
                # so we don't need to track the intermediate before it succeeds.
                intermediate.compose(batch)
                intermediate_blobs.append(intermediate)
                remaining.insert(0, intermediate)
                batch_idx += 1

            destination.compose(remaining)
        finally:
            for blob in intermediate_blobs:
                try:
                    blob.delete()
                except gcs_exceptions.GoogleAPIError:
                    pass

    def do_list_parts(self, mpu: MultipartUpload) -> List[UploadPartResponse]:
        """List parts uploaded so far (preview, compose-based)."""
        prefix = f"{mpu.key}__multipart__{mpu.upload_id}__"
        results = []
        for blob in self._client.list_blobs(self.bucket, prefix=prefix):
            part_suffix = blob.name[len(prefix) :]
            try:
                part_number = int(part_suffix)
            except ValueError:
                continue
            results.append(
                UploadPartResponse(
                    part_number=part_number,
                    e_tag=blob.etag or "",
                    size_in_bytes=blob.size or 0,
                )
            )
        return sorted(results, key=lambda p: p.part_number)

    def do_abort_multipart_upload(self, mpu: MultipartUpload) -> None:
        """Best-effort cleanup of temp part objects (preview, compose-based)."""
        prefix = f"{mpu.key}__multipart__{mpu.upload_id}__"
        blobs = list(self._client.list_blobs(self.bucket, prefix=prefix))
        for blob in blobs:
            try:
                blob.delete()
            except gcs_exceptions.GoogleAPIError:
                pass

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        if self._auth_session is not None:
            self._auth_session.close()
            self._auth_session = None
