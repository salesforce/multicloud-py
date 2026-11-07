"""AWS S3 implementation of :class:`BaseBlobStore`.

Translates S3 / boto3 errors into the cloud-agnostic exception
hierarchy so callers can write a single ``except`` block that works
across providers.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterator, List, Mapping, Optional
from urllib.parse import quote

from botocore.exceptions import (  # type: ignore[import-not-found]
    BotoCoreError,
    ClientError,
    NoCredentialsError,
)
from multicloud.aws_common.aws_client_config import s3_client_config
from multicloud.aws_common.aws_credentials import get_session
from multicloud.aws_common.error_mapping import AWS_THROTTLING_CODES
from multicloud.blob.blob_aws._aws_common import (
    CHECKSUM_FIELDS,
    apply_kms,
    apply_write_preconditions,
    ensure_etag_quoted,
    is_bad_digest,
    is_conditional_conflict,
    is_missing_object,
    is_precondition_failed,
    map_replication_status,
)
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
from multicloud.common.credentials import CredentialsOverrider
from multicloud.common.exceptions import (
    FailedPreconditionException,
    InvalidArgumentException,
    ResourceConflictException,
    ResourceExhaustedException,
    ResourceExistsException,
    ResourceNotFoundException,
    SubstrateSdkException,
    UnAuthorizedException,
    UnknownException,
)
from multicloud.common.proxy import ProxyConfig

_logger = logging.getLogger(__name__)


@BLOB_REGISTRY.register("aws")
class AwsBlobStore(BaseBlobStore):
    """S3-backed blob store."""

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        *,
        s3_client: Any = None,
        endpoint_url: Optional[str] = None,
        credentials_overrider: Optional[CredentialsOverrider] = None,
        proxy_config: Optional[ProxyConfig] = None,
    ) -> None:
        super().__init__(
            bucket=bucket,
            region=region,
            credentials_overrider=credentials_overrider,
            proxy_config=proxy_config,
        )

        if s3_client is not None:
            # Caller-supplied client takes precedence over credentials_overrider.
            self._s3 = s3_client
            self._session = None
        else:
            # No overrider → get_session falls back to boto3's default chain.
            self._session = get_session(credentials_overrider, region)
            # Pin SigV4: presigned URLs for KMS-encrypted buckets require it,
            # and botocore still downgrades to SigV2 in some legacy regions
            # unless the signature version is set at client construction.
            client_kwargs: dict[str, Any] = {
                "region_name": region,
                "config": s3_client_config(proxy_config),
            }
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url
            self._s3 = self._session.client("s3", **client_kwargs)

    def translate_error(self, exc: Exception) -> SubstrateSdkException:
        """Translate AWS SDK exceptions to common hierarchy.

        Maps boto3/botocore exceptions to cloud-agnostic exception types.
        """
        if isinstance(exc, SubstrateSdkException):
            return exc

        if isinstance(exc, NoCredentialsError):
            return UnAuthorizedException(str(exc))

        if isinstance(exc, ClientError):
            error_info = exc.response.get("Error", {})
            code = error_info.get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            # Checked before the status branches below: AWS reports throttling
            # with HTTP 400 on some services, which would otherwise be read as
            # an invalid argument.
            if code in AWS_THROTTLING_CODES or status == 429:
                return ResourceExhaustedException(str(exc))

            if (
                code in ("NoSuchBucket", "NoSuchKey", "NoSuchUpload", "404")
                or status == 404
            ):
                return ResourceNotFoundException(str(exc))
            if code in ("AccessDenied", "Forbidden", "403") or status == 403:
                return UnAuthorizedException(str(exc))
            # A conditional read/write whose precondition was evaluated and did
            # not hold returns 412 PreconditionFailed -> FailedPreconditionException.
            # (The create-if-absent IfNoneMatch="*" case also returns 412 but is
            # re-labelled to ResourceExistsException locally at the put site
            # before it reaches this generic translation.)
            if is_precondition_failed(exc):
                return FailedPreconditionException(str(exc))
            # A concurrent operation racing a conditional write returns
            # ConditionalRequestConflict (HTTP 409 on S3). Match the error
            # code, not HTTP 409 in general — other AWS APIs reuse 409
            # (for example IAM LimitExceeded). Surface it as
            # ResourceConflictException so callers can retry rather than
            # conclude the precondition failed.
            if is_conditional_conflict(exc):
                return ResourceConflictException(str(exc))
            if (
                code in ("InvalidArgument", "InvalidRequest", "EntityTooSmall", "400")
                or status == 400
            ):
                return InvalidArgumentException(str(exc))

            return UnknownException(str(exc))

        if isinstance(exc, BotoCoreError):
            return InvalidArgumentException(str(exc))

        return UnknownException(str(exc))

    @staticmethod
    def _apply_object_lock(
        put_kwargs: dict[str, Any], object_lock: Optional[ObjectLockConfiguration]
    ) -> None:
        """Add S3 object-lock PutObject parameters from *object_lock*.

        S3 exposes retention as ObjectLockMode + ObjectLockRetainUntilDate,
        and a single legal-hold concept as ObjectLockLegalHoldStatus. S3 has
        no event-based hold, so ``use_event_based_hold`` is ignored here.
        """
        if object_lock is None:
            return
        if object_lock.mode is not None and object_lock.retain_until_date is not None:
            put_kwargs["ObjectLockMode"] = object_lock.mode.value
            put_kwargs["ObjectLockRetainUntilDate"] = object_lock.retain_until_date
        if object_lock.legal_hold:
            put_kwargs["ObjectLockLegalHoldStatus"] = "ON"
        # Log rather than silently drop the unsupported flag.
        if object_lock.use_event_based_hold:
            _logger.debug("use_event_based_hold is not supported on S3 and was ignored")

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
        put_kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": payload,
        }
        if content_type:
            put_kwargs["ContentType"] = content_type
        if metadata:
            put_kwargs["Metadata"] = dict(metadata)
        # Falsy check intentional: skip when caller passes None or {} so an
        # empty Tagging header is never sent. The driver normalizes None to {}
        # via ``tags or {}``; ``{}`` is falsy here, matching the metadata handling.
        if tags:
            # S3 PutObject accepts tags as a URL-encoded query string in
            # the Tagging parameter. Tag limits are enforced by S3, not here.
            put_kwargs["Tagging"] = "&".join(
                f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in tags.items()
            )
        self._apply_object_lock(put_kwargs, object_lock)
        apply_write_preconditions(
            put_kwargs,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
        )
        apply_kms(
            put_kwargs,
            kms_key_id=kms_key_id,
            use_kms_managed_key=use_kms_managed_key,
        )

        # Supplying the digest makes botocore send it as-is instead of
        # computing its own, so S3 validates the caller's value.
        checksum_field = None
        if checksum_algorithm is not None:
            checksum_field = CHECKSUM_FIELDS[checksum_algorithm]
            put_kwargs[checksum_field] = checksum_value

        try:
            resp = self._put_object_conditional(put_kwargs, create_if_absent, if_match)
        except ClientError as exc:
            if checksum_algorithm is not None and is_bad_digest(exc):
                raise InvalidArgumentException(
                    CHECKSUM_MISMATCH_MSG.format(checksum_algorithm.value, key, exc)
                ) from exc
            raise
        return UploadResponse(
            key=key,
            e_tag=_strip_quotes(resp.get("ETag")),
            version_id=resp.get("VersionId"),
            checksum_algorithm=checksum_algorithm,
            checksum_value=resp.get(checksum_field) if checksum_field else None,
        )

    def _put_object_conditional(
        self,
        put_kwargs: dict[str, Any],
        create_if_absent: bool,
        if_match: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call ``put_object``, disambiguating conditional-write outcomes.

        S3 returns HTTP 412 PreconditionFailed for both ``IfNoneMatch="*"``
        (object exists) and ``IfMatch`` (ETag mismatch); only create-if-absent is
        re-labelled to ResourceExistsException here. An ``IfMatch`` overwrite of a
        missing object returns 404, which is re-labelled to
        FailedPreconditionException so an absent target fails the precondition
        rather than surfacing as not-found. Everything else (including the
        retryable ConditionalRequestConflict) propagates for the client layer to
        translate.
        """
        if not create_if_absent and if_match is None:
            return self._s3.put_object(**put_kwargs)  # type: ignore[no-any-return]
        try:
            return self._s3.put_object(**put_kwargs)  # type: ignore[no-any-return]
        except ClientError as exc:
            if create_if_absent and is_precondition_failed(exc):
                raise ResourceExistsException(str(exc)) from exc
            if if_match is not None and is_missing_object(exc):
                raise FailedPreconditionException(str(exc)) from exc
            raise

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
        """Upload a local file to S3 using a single atomic put_object call.

        Opens the file as a streaming binary body and passes it to
        ``put_object()``. Returns ETag/VersionId atomically in the response.
        ContentLength is passed for S3 integrity verification.
        """
        put_kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
        }
        if content_type:
            put_kwargs["ContentType"] = content_type
        else:
            put_kwargs["ContentType"] = "application/octet-stream"
        if content_length is not None:
            put_kwargs["ContentLength"] = content_length
        if metadata:
            put_kwargs["Metadata"] = dict(metadata)
        self._apply_object_lock(put_kwargs, object_lock)
        apply_write_preconditions(
            put_kwargs,
            create_if_absent=create_if_absent,
            if_match=if_match,
            if_version_match=if_version_match,
        )
        apply_kms(
            put_kwargs,
            kms_key_id=kms_key_id,
            use_kms_managed_key=use_kms_managed_key,
        )

        with open(path, "rb") as fh:
            put_kwargs["Body"] = fh
            resp = self._put_object_conditional(put_kwargs, create_if_absent, if_match)

        return UploadResponse(
            key=key,
            e_tag=_strip_quotes(resp.get("ETag")),
            version_id=resp.get("VersionId"),
        )

    def _build_get_kwargs(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
        if_match: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build the common kwargs dict for S3 get_object calls."""
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if version_id is not None:
            kwargs["VersionId"] = version_id
        range_header = _build_range_header(start, end)
        if range_header is not None:
            kwargs["Range"] = range_header
        # Conditional read: IfMatch returns 412 on ETag mismatch. Re-quote so
        # the natural round-trip (download().metadata.e_tag -> if_match) works;
        # download strips the quotes S3's If-Match header requires.
        if if_match is not None:
            kwargs["IfMatch"] = ensure_etag_quoted(if_match)
        return kwargs

    def _build_metadata(self, key: str, resp: dict[str, Any]) -> BlobMetadata:
        """Extract BlobMetadata from an S3 get_object/head_object response.

        ``get_object`` (download) and ``head_object`` (get_metadata) return
        the same metadata keys, including ``LastModified``, so this single
        helper serves both paths — ``download()`` and ``get_metadata()`` return
        an identically-populated ``BlobMetadata``. S3 exposes no distinct
        creation timestamp, so ``created_time`` mirrors ``last_modified``
        (matches the async provider and MulticloudJ).
        """
        last_modified = resp.get("LastModified")
        return BlobMetadata(
            key=key,
            content_length=resp.get("ContentLength"),
            content_type=resp.get("ContentType"),
            e_tag=_strip_quotes(resp.get("ETag")),
            version_id=resp.get("VersionId"),
            metadata=resp.get("Metadata", {}) or {},
            last_modified=last_modified,
            created_time=last_modified,
            replication_status=map_replication_status(resp.get("ReplicationStatus")),
        )

    def do_download(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
        if_match: Optional[str] = None,
    ) -> DownloadResponse:
        get_kwargs = self._build_get_kwargs(
            key,
            version_id=version_id,
            start=start,
            end=end,
            if_match=if_match,
        )

        try:
            resp = self._s3.get_object(**get_kwargs)
            body = resp["Body"].read()
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            raise self.translate_error(exc) from exc

        metadata = self._build_metadata(key, resp)
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
        """Download to a local file path using S3 streaming without memory buffering.

        Writes to a temp file in the destination's parent directory and atomically
        renames it into place on success, so a mid-stream failure never leaves a
        partial/corrupted file at ``destination``.
        """
        get_kwargs = self._build_get_kwargs(
            key,
            version_id=version_id,
            start=start,
            end=end,
        )

        try:
            resp = self._s3.get_object(**get_kwargs)
            body_stream = resp["Body"]
            tmp_file = tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            )
            tmp_path = tmp_file.name
            try:
                try:
                    with tmp_file as f:
                        for chunk in body_stream.iter_chunks(chunk_size=8192):
                            f.write(chunk)
                finally:
                    body_stream.close()
                os.replace(tmp_path, destination)
            except BaseException:
                # Streaming or rename failed — remove the temp file so callers
                # never see a partial/corrupted artifact.
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
                raise
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            raise self.translate_error(exc) from exc

        return DownloadResponse(
            key=key, content=b"", metadata=self._build_metadata(key, resp)
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
        """Download to a file-like object using S3 streaming without memory buffering."""
        get_kwargs = self._build_get_kwargs(
            key,
            version_id=version_id,
            start=start,
            end=end,
        )

        try:
            resp = self._s3.get_object(**get_kwargs)
            body_stream = resp["Body"]
            try:
                for chunk in body_stream.iter_chunks(chunk_size=8192):
                    file_obj.write(chunk)
            finally:
                body_stream.close()
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            raise self.translate_error(exc) from exc

        return DownloadResponse(
            key=key, content=b"", metadata=self._build_metadata(key, resp)
        )

    def do_download_stream(
        self,
        key: str,
        *,
        version_id: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> StreamingDownloadResponse:
        """Return a streaming response wrapping the S3 StreamingBody.

        The caller must close the response to release the HTTP connection.
        """
        get_kwargs = self._build_get_kwargs(
            key,
            version_id=version_id,
            start=start,
            end=end,
        )

        try:
            resp = self._s3.get_object(**get_kwargs)
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            raise self.translate_error(exc) from exc

        body_stream = resp["Body"]
        return StreamingDownloadResponse(
            key=key,
            metadata=self._build_metadata(key, resp),
            stream=body_stream.iter_chunks(chunk_size=8192),
            close_func=body_stream.close,
        )

    def do_delete(self, key: str, *, version_id: Optional[str] = None) -> None:
        """Delete a single object by key. S3 ``delete_object`` is idempotent."""
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if version_id is not None:
            kwargs["VersionId"] = version_id
        self._s3.delete_object(**kwargs)

    def do_delete_objects(self, objects: List[BlobIdentifier]) -> DeleteObjectsResponse:
        """Batch delete using S3 DeleteObjects API.

        Chunking is intentionally left to the caller. The driver sends
        the full list in a single request to keep the implementation
        simple and predictable.
        """
        object_ids: List[dict[str, str]] = []
        for obj in objects:
            entry: dict[str, str] = {"Key": obj.key}
            if obj.version_id is not None:
                entry["VersionId"] = obj.version_id
            object_ids.append(entry)

        resp = self._s3.delete_objects(
            Bucket=self.bucket,
            Delete={"Objects": object_ids, "Quiet": False},
        )

        errors: List[DeleteError] = []
        for err in resp.get("Errors", []):
            errors.append(
                DeleteError(
                    key=err.get("Key", ""),
                    version_id=err.get("VersionId"),
                    code=err.get("Code"),
                    message=err.get("Message"),
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
        """Copy from this bucket to another bucket using S3 CopyObject."""
        copy_source: dict[str, str] = {"Bucket": self.bucket, "Key": src_key}
        if src_version_id is not None:
            copy_source["VersionId"] = src_version_id

        resp = self._s3.copy_object(
            CopySource=copy_source,
            Bucket=dest_bucket,
            Key=dest_key,
        )

        copy_result = resp.get("CopyObjectResult", {})
        return CopyResponse(
            key=dest_key,
            version_id=resp.get("VersionId"),
            e_tag=_strip_quotes(copy_result.get("ETag")),
            last_modified=copy_result.get("LastModified"),
        )

    def do_copy_from(
        self,
        src_bucket: str,
        src_key: str,
        *,
        dest_key: str,
        src_version_id: Optional[str],
    ) -> CopyResponse:
        """Copy from a source bucket into this bucket using S3 CopyObject."""
        copy_source: dict[str, str] = {"Bucket": src_bucket, "Key": src_key}
        if src_version_id is not None:
            copy_source["VersionId"] = src_version_id

        resp = self._s3.copy_object(
            CopySource=copy_source,
            Bucket=self.bucket,
            Key=dest_key,
        )

        copy_result = resp.get("CopyObjectResult", {})
        return CopyResponse(
            key=dest_key,
            version_id=resp.get("VersionId"),
            e_tag=_strip_quotes(copy_result.get("ETag")),
            last_modified=copy_result.get("LastModified"),
        )

    def do_get_object_lock(
        self, key: str, *, version_id: Optional[str] = None
    ) -> Optional[ObjectLockInfo]:
        """Retrieve object lock state via S3 GetObjectRetention + GetObjectLegalHold.

        S3 exposes retention and legal hold as two independent sub-resources,
        each fetched with a separate call. Either call raises
        ``NoSuchObjectLockConfiguration`` when *that* piece is not configured on
        the object, independently of the other. This is expected for a normal
        object that has only one (or neither) of the two set, so each call is
        wrapped to treat that error as "this piece is absent" rather than a
        failure. Any other native error (missing object, bucket without Object
        Lock enabled, access denied) propagates to the client layer for
        translation.

        Returns ``None`` when the object has neither retention nor legal hold.
        """
        version_kwargs = {"VersionId": version_id} if version_id is not None else {}

        mode: Optional[RetentionMode] = None
        retain_until: Optional[datetime] = None
        try:
            retention = self._s3.get_object_retention(
                Bucket=self.bucket, Key=key, **version_kwargs
            ).get("Retention", {})
            if retention.get("Mode"):
                mode = RetentionMode(retention["Mode"])
                retain_until = retention.get("RetainUntilDate")
        except ClientError as exc:
            if not _is_no_object_lock_error(exc):
                raise

        legal_hold = False
        try:
            status = (
                self._s3.get_object_legal_hold(
                    Bucket=self.bucket, Key=key, **version_kwargs
                )
                .get("LegalHold", {})
                .get("Status")
            )
            legal_hold = status == "ON"
        except ClientError as exc:
            if not _is_no_object_lock_error(exc):
                raise

        if mode is None and not legal_hold:
            return None

        return ObjectLockInfo(
            mode=mode,
            retain_until_date=retain_until,
            legal_hold=legal_hold,
            event_based_hold=None,
        )

    def do_update_legal_hold(
        self, key: str, enabled: bool, *, version_id: Optional[str] = None
    ) -> None:
        """Set or remove a legal hold via S3 PutObjectLegalHold.

        S3 models legal hold as an ON/OFF status on the object; setting it to
        OFF is how a hold is removed. Requires the bucket to have Object Lock
        enabled — otherwise S3 raises an error that propagates to the client
        layer for translation.
        """
        version_kwargs = {"VersionId": version_id} if version_id is not None else {}
        self._s3.put_object_legal_hold(
            Bucket=self.bucket,
            Key=key,
            LegalHold={"Status": "ON" if enabled else "OFF"},
            **version_kwargs,
        )

    def do_get_tags(self, key: str) -> Mapping[str, str]:
        """Retrieve object tags using S3 GetObjectTagging API."""
        resp = self._s3.get_object_tagging(Bucket=self.bucket, Key=key)
        return {tag["Key"]: tag["Value"] for tag in resp.get("TagSet", [])}

    def do_set_tags(self, key: str, tags: Mapping[str, str]) -> None:
        """Replace all object tags using S3 PutObjectTagging API."""
        tag_set = [{"Key": k, "Value": v} for k, v in tags.items()]
        self._s3.put_object_tagging(
            Bucket=self.bucket,
            Key=key,
            Tagging={"TagSet": tag_set},
        )

    def do_update_object_retention(
        self,
        key: str,
        retention: ObjectRetentionConfig,
        *,
        version_id: Optional[str],
    ) -> None:
        """Update object retention using the S3 PutObjectRetention API.

        Reads the object's current retention with GetObjectRetention (a 404 /
        NoSuchObjectLockConfiguration means "no current retention"), resolves
        the effective mode against the shared rules table, then writes it back.
        The resolved mode is used so a ``None`` config mode preserves whatever
        is currently on the object.
        """
        version_kwargs = {"VersionId": version_id} if version_id is not None else {}

        current_mode: Optional[RetentionMode] = None
        current_retain_until: Optional[datetime] = None
        try:
            current = self._s3.get_object_retention(
                Bucket=self.bucket, Key=key, **version_kwargs
            ).get("Retention", {})
            if current.get("Mode"):
                current_mode = RetentionMode(current["Mode"])
                current_retain_until = current.get("RetainUntilDate")
        except ClientError as exc:
            if not _is_no_object_lock_error(exc):
                raise

        resolved_mode = resolve_and_validate(
            current_mode, current_retain_until, retention
        )

        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Retention": {
                "Mode": resolved_mode.value,
                "RetainUntilDate": retention.retain_until_date,
            },
        }
        if version_id is not None:
            kwargs["VersionId"] = version_id
        if retention.bypass_governance:
            kwargs["BypassGovernanceRetention"] = True
        self._s3.put_object_retention(**kwargs)

    def do_list_page(
        self,
        *,
        prefix: Optional[str],
        delimiter: Optional[str],
        max_results: int,
        pagination_token: Optional[str],
    ) -> ListBlobsPageResponse:
        """List a single page of objects using S3 list_objects_v2."""
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "MaxKeys": max_results}
        if prefix is not None:
            kwargs["Prefix"] = prefix
        if delimiter is not None:
            kwargs["Delimiter"] = delimiter
        if pagination_token is not None:
            kwargs["ContinuationToken"] = pagination_token

        resp = self._s3.list_objects_v2(**kwargs)
        blobs = [
            BlobInfo(
                key=obj["Key"],
                size=obj.get("Size"),
                e_tag=_strip_quotes(obj.get("ETag")),
                last_modified=obj.get("LastModified"),
                storage_class=obj.get("StorageClass"),
            )
            for obj in resp.get("Contents", [])
        ]
        common_prefixes = [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]
        return ListBlobsPageResponse(
            blobs=blobs,
            common_prefixes=common_prefixes,
            is_truncated=resp.get("IsTruncated", False),
            next_pagination_token=resp.get("NextContinuationToken"),
        )

    def do_does_object_exist(self, key: str, *, version_id: Optional[str]) -> bool:
        """Check object existence using S3 head_object."""
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if version_id is not None:
            kwargs["VersionId"] = version_id
        try:
            self._s3.head_object(**kwargs)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey") or status == 404:
                return False
            raise self.translate_error(exc) from exc

    def do_does_bucket_exist(self) -> bool:
        """Check bucket existence using S3 head_bucket."""
        try:
            self._s3.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchBucket") or status == 404:
                return False
            raise self.translate_error(exc) from exc

    def do_get_metadata(self, key: str, *, version_id: Optional[str]) -> BlobMetadata:
        """Retrieve object metadata using head_object (no body download).

        Uses an S3 HEAD, so no object body is transferred. Not-found and
        other provider errors are routed through ``translate_error``,
        surfacing as ``ResourceNotFoundException`` (404/NoSuchKey/
        NoSuchBucket), ``UnAuthorizedException`` (403), etc.
        """
        head_kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if version_id is not None:
            head_kwargs["VersionId"] = version_id
        try:
            resp = self._s3.head_object(**head_kwargs)
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            raise self.translate_error(exc) from exc

        # head_object returns the same metadata keys as get_object, so reuse
        # the shared mapping (keeps download() and get_metadata() in lockstep).
        return self._build_metadata(key, resp)

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
        """Generate a presigned URL for S3 put_object or get_object.

        The URL is signed with SigV4 (``AWS4-HMAC-SHA256``); the S3 client is
        pinned to SigV4 at construction. This is required for KMS-encrypted
        buckets — SigV2 URLs are rejected with ``SignatureDoesNotMatch`` /
        ``InvalidArgument`` — and SigV4 folds any KMS/metadata/tag headers into
        the signature, so callers must replay every ``required_headers`` entry
        verbatim on the request.
        """
        if operation == PresignedOperation.UPLOAD:
            return self._generate_presigned_upload_url(
                key,
                duration_seconds=duration_seconds,
                content_type=content_type,
                metadata=metadata,
                tags=tags,
                kms_key_id=kms_key_id,
                content_disposition=content_disposition,
            )
        elif operation == PresignedOperation.DOWNLOAD:
            return self._generate_presigned_download_url(
                key,
                duration_seconds=duration_seconds,
                content_disposition=content_disposition,
            )
        else:
            raise InvalidArgumentException(
                f"Unsupported presigned URL operation: {operation}"
            )

    def _generate_presigned_upload_url(
        self,
        key: str,
        *,
        duration_seconds: int,
        content_type: Optional[str],
        metadata: Mapping[str, str],
        tags: Mapping[str, str],
        kms_key_id: Optional[str],
        content_disposition: Optional[str],
    ) -> PresignedUrlResponse:
        """Build a presigned PUT URL with required headers for S3."""
        params: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        required_headers: dict[str, str] = {}

        if content_type:
            params["ContentType"] = content_type
            required_headers["Content-Type"] = content_type

        if metadata:
            params["Metadata"] = dict(metadata)
            for meta_key, meta_value in metadata.items():
                header_key = f"x-amz-meta-{meta_key}"
                required_headers[header_key] = meta_value

        if tags:
            tag_string = "&".join(
                f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in tags.items()
            )
            params["Tagging"] = tag_string
            required_headers["x-amz-tagging"] = tag_string

        if kms_key_id:
            params["ServerSideEncryption"] = "aws:kms"
            params["SSEKMSKeyId"] = kms_key_id
            required_headers["x-amz-server-side-encryption"] = "aws:kms"
            required_headers["x-amz-server-side-encryption-aws-kms-key-id"] = kms_key_id

        if content_disposition:
            params["ContentDisposition"] = content_disposition
            required_headers["Content-Disposition"] = content_disposition

        url = self._s3.generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=duration_seconds,
        )
        return PresignedUrlResponse(
            url=url, http_method="PUT", required_headers=required_headers
        )

    def _generate_presigned_download_url(
        self,
        key: str,
        *,
        duration_seconds: int,
        content_disposition: Optional[str],
    ) -> PresignedUrlResponse:
        """Build a presigned GET URL for S3."""
        params: dict[str, Any] = {"Bucket": self.bucket, "Key": key}

        if content_disposition:
            params["ResponseContentDisposition"] = content_disposition

        url = self._s3.generate_presigned_url(
            ClientMethod="get_object",
            Params=params,
            ExpiresIn=duration_seconds,
        )
        return PresignedUrlResponse(url=url, http_method="GET", required_headers={})

    def do_list_blob_versions(self, key: str) -> Iterator[BlobInfo]:
        """List all versions of a blob using S3 list_object_versions.

        Note:
            This is a lazy generator. If a transient error occurs mid-iteration,
            the caller will receive a partial result followed by a translated
            exception. If the caller does not fully consume the iterator, errors
            on subsequent pages will not surface.
        """
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": key}
        while True:
            try:
                resp = self._s3.list_object_versions(**kwargs)
            except (ClientError, BotoCoreError, NoCredentialsError) as exc:
                raise self.translate_error(exc) from exc

            for version in resp.get("Versions", []):
                if version.get("Key") != key:
                    continue
                yield BlobInfo(
                    key=version["Key"],
                    size=version.get("Size"),
                    e_tag=_strip_quotes(version.get("ETag")),
                    last_modified=version.get("LastModified"),
                    storage_class=version.get("StorageClass"),
                    version_id=version.get("VersionId"),
                    is_latest=version.get("IsLatest"),
                )

            if not resp.get("IsTruncated", False):
                break
            kwargs["KeyMarker"] = resp["NextKeyMarker"]
            kwargs["VersionIdMarker"] = resp["NextVersionIdMarker"]

    # multipart upload operations
    def do_initiate_multipart_upload(
        self,
        key: str,
        *,
        metadata: Mapping[str, str],
        content_type: Optional[str],
        tags: Mapping[str, str],
        object_lock: Optional[ObjectLockConfiguration] = None,
    ) -> MultipartUpload:
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if metadata:
            kwargs["Metadata"] = dict(metadata)
        if content_type:
            kwargs["ContentType"] = content_type
        # S3 accepts tags only on CreateMultipartUpload (not CompleteMultipartUpload),
        # as a URL-encoded query string in the Tagging parameter — same encoding as
        # PutObject. Tag limits are enforced by S3, not here.
        if tags:
            kwargs["Tagging"] = "&".join(
                f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in tags.items()
            )
        # CreateMultipartUpload natively accepts the same object-lock parameters
        # as PutObject; the lock is stamped on the object when the upload
        # completes. Reuse the shared PutObject mapping helper.
        self._apply_object_lock(kwargs, object_lock)
        resp = self._s3.create_multipart_upload(**kwargs)
        return MultipartUpload(
            bucket=self.bucket,
            key=key,
            upload_id=resp["UploadId"],
            metadata=metadata,
            content_type=content_type,
            tags=tags,
            object_lock=object_lock,
        )

    def do_upload_part(
        self, mpu: MultipartUpload, part: MultipartPart
    ) -> UploadPartResponse:
        resp = self._s3.upload_part(
            Bucket=self.bucket,
            Key=mpu.key,
            UploadId=mpu.upload_id,
            PartNumber=part.part_number,
            Body=part.content,
            ContentLength=part.content_length,
        )
        return UploadPartResponse(
            part_number=part.part_number,
            e_tag=_strip_quotes(resp.get("ETag", "")),
            size_in_bytes=part.content_length,
        )

    def do_complete_multipart_upload(
        self, mpu: MultipartUpload, parts: List[UploadPartResponse]
    ) -> MultipartUploadResponse:
        sorted_parts = sorted(parts, key=lambda p: p.part_number)
        multipart_upload = {
            "Parts": [
                {"PartNumber": p.part_number, "ETag": p.e_tag} for p in sorted_parts
            ]
        }
        resp = self._s3.complete_multipart_upload(
            Bucket=self.bucket,
            Key=mpu.key,
            UploadId=mpu.upload_id,
            MultipartUpload=multipart_upload,
        )
        return MultipartUploadResponse(
            e_tag=_strip_quotes(resp.get("ETag")),
        )

    def do_list_parts(self, mpu: MultipartUpload) -> List[UploadPartResponse]:
        all_parts: List[UploadPartResponse] = []
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": mpu.key,
            "UploadId": mpu.upload_id,
        }
        while True:
            resp = self._s3.list_parts(**kwargs)
            all_parts.extend(
                UploadPartResponse(
                    part_number=p["PartNumber"],
                    e_tag=_strip_quotes(p.get("ETag", "")),
                    size_in_bytes=p.get("Size", 0),
                )
                for p in resp.get("Parts", [])
            )
            if not resp.get("IsTruncated", False):
                break
            kwargs["PartNumberMarker"] = resp["NextPartNumberMarker"]
        return sorted(all_parts, key=lambda p: p.part_number)

    def do_abort_multipart_upload(self, mpu: MultipartUpload) -> None:
        self._s3.abort_multipart_upload(
            Bucket=self.bucket,
            Key=mpu.key,
            UploadId=mpu.upload_id,
        )

    def close(self) -> None:
        close = getattr(self._s3, "close", None)
        if callable(close):
            close()


def _is_no_object_lock_error(exc: ClientError) -> bool:
    """Return True when a GetObjectRetention/GetObjectLegalHold call reports that
    the queried piece (retention or legal hold) is simply not configured on the
    object. S3 signals this with the ``NoSuchObjectLockConfiguration`` error
    code, which is expected for objects that have only one (or neither) of the
    two lock sub-resources set."""
    code = exc.response.get("Error", {}).get("Code", "")
    return code == "NoSuchObjectLockConfiguration"


def _strip_quotes(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.strip('"')


def _build_range_header(start: Optional[int], end: Optional[int]) -> Optional[str]:
    if start is None and end is None:
        return None
    if start is None:
        return f"bytes=-{end}"
    if end is None:
        return f"bytes={start}-"
    return f"bytes={start}-{end}"
