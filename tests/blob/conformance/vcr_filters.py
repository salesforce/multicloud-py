"""VCR.py filter module for credential scrubbing and cassette hygiene.

Lives next to the blob conformance tests. If a second service starts
needing the same filters (e.g. ``tests/sts/conformance/``), extract
this into a shared package then; until that day, co-located is
clearer than speculative reuse.

Filters are organized in three tiers:
- SECURITY: Must be filtered to prevent credential leakage
- STABILITY: Should be filtered to prevent replay failures due to timestamps/nonces
- NOISE: May be filtered to reduce cassette diff noise

Usage::

    from tests.blob.conformance.vcr_filters import get_default_vcr_config

    @pytest.fixture(scope="module")
    def vcr_config():
        config = get_default_vcr_config(cassette_dir="path/to/cassettes")
        config["filter_headers"].extend(["X-My-Extra-Header"])
        return config
"""

import base64
import hashlib
import json
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse
from xml.etree import ElementTree

SECURITY_HEADERS = [
    # All providers
    "Authorization",
    "Cookie",
    "Set-Cookie",
    # AWS STS / signing
    "X-Amz-Security-Token",
    "x-amz-security-token",
    "X-Amz-Credential",
    # GCP
    "x-goog-api-key",
    # Ali
    "x-acs-security-token",
    "x-oss-security-token",
]

STABILITY_HEADERS = [
    # AWS signing (timestamps and content hashes change per request)
    "X-Amz-Date",
    "x-amz-date",
    "X-Amz-Content-Sha256",
    "x-amz-content-sha256",
    # GCP (client metadata and routing params)
    "x-goog-api-client",
    "x-goog-request-params",
    # Ali (timestamps, nonces, content hashes)
    "x-acs-date",
    "x-acs-signature-nonce",
    "x-acs-content-sha256",
    "x-oss-date",
    "x-tablestore-date",
    # All providers (client version strings)
    "User-Agent",
    "Accept-Encoding",
]

DEFAULT_IGNORE_HOSTS: Set[str] = {
    # GCP OAuth / IAM token endpoints
    "oauth2.googleapis.com",
    "iamcredentials.googleapis.com",
    "accounts.google.com",
    # AWS STS
    "sts.amazonaws.com",
    # Ali STS
    "sts.aliyuncs.com",
}

# Pattern for regional AWS STS endpoints: sts.<region>.amazonaws.com
_REGIONAL_STS_PATTERN = re.compile(r"^sts\.[a-z0-9-]+\.amazonaws\.com$")

_SENSITIVE_FORM_FIELDS: Set[str] = {
    "client_secret",
    "refresh_token",
    "access_token",
    "assertion",
    "password",
    "client_assertion",
}

_SENSITIVE_JSON_REQUEST_FIELDS: Set[str] = {
    "client_secret",
    "refresh_token",
    "access_token",
    "password",
    "secret",
    "token",
    "assertion",
}

_SENSITIVE_JSON_RESPONSE_FIELDS: Set[str] = {
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
    "token",
    "secret",
    "password",
    "accessToken",
    "refreshToken",
}

# Deliberately NOT scrubbed: the IAM signBlob response's ``signedBlob``. It is a
# signature over one specific string-to-sign, produced by a Google-managed key --
# not a credential, and not replayable to sign anything else. It also has to stay
# intact: the client base64-decodes it, so a redacted value fails to parse.

# GCP service account emails in URIs
_GCP_SA_PATTERN = re.compile(r"(serviceAccounts/)[^/:]+(@[^/:]*\.gserviceaccount\.com)")

# The whole service-account email, project included. The pattern above keeps the
# project part, which is enough for scrubbing but not for matching: a request
# recorded under the real signing identity has to match the same request made
# under the offline mock identity, and those differ in the project too.
_GCP_SA_IDENTITY_PATTERN = re.compile(
    r"(serviceAccounts/)[^/:]+@[^/:]*\.gserviceaccount\.com"
)
_GCP_SA_IDENTITY_PLACEHOLDER = r"\1FILTERED_SA@FILTERED_PROJECT.iam.gserviceaccount.com"

# Identity blocks in object-store XML list responses (e.g. ListParts,
# ListMultipartUploads, ListObjects, ListAllMyBuckets). Their <ID> /
# <DisplayName> children carry the caller's ARN/principal, the role name, and
# the account-scoped canonical user id — none of which any test asserts on, and
# all of which would otherwise be committed verbatim.
_XML_IDENTITY_BLOCK_PATTERN = re.compile(
    r"(<(?:Initiator|Owner)>)(.*?)(</(?:Initiator|Owner)>)", re.DOTALL
)
_XML_ID_PATTERN = re.compile(r"(<ID>).*?(</ID>)", re.DOTALL)
_XML_DISPLAY_NAME_PATTERN = re.compile(r"(<DisplayName>).*?(</DisplayName>)", re.DOTALL)
_XML_IDENTITY_PLACEHOLDER = "REDACTED"

MULTIPART_TAG_MATCHERS = (
    "multipart_tag_header",
    "multipart_metadata_headers",
    "multipart_destination_metadata",
    "patched_metadata",
    "multipart_part_payload",
)

DIRECT_TAG_MATCHERS = ("object_tagging", "patched_metadata")
RANGE_HEADER_MATCHER = "range_header"


def _header_value(request: Any, name: str) -> Optional[str]:
    """Return one normalized request-header value."""
    values = next(
        (
            value
            for key, value in request.headers.items()
            if key.lower() == name.lower()
        ),
        None,
    )
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        values = [values]
    return ",".join(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def range_header(request1: Any, request2: Any) -> None:
    """Match the Range header when either request supplies one."""
    first = _header_value(request1, "Range")
    second = _header_value(request2, "Range")
    assert first == second, f"range headers differ: {first!r} != {second!r}"


def multipart_tag_header(request1: Any, request2: Any) -> None:
    """Match the tag header when either request supplies one."""
    first = _header_value(request1, "x-amz-tagging")
    second = _header_value(request2, "x-amz-tagging")
    assert first == second, f"multipart tag headers differ: {first!r} != {second!r}"


def _metadata_headers(request: Any) -> Dict[str, Optional[str]]:
    """Return normalized custom-metadata request headers."""
    return {
        key.lower(): _header_value(request, key)
        for key in request.headers
        if key.lower().startswith("x-amz-meta-")
    }


def multipart_metadata_headers(request1: Any, request2: Any) -> None:
    """Match custom-metadata headers supplied by either request."""
    first = _metadata_headers(request1)
    second = _metadata_headers(request2)
    assert (
        first == second
    ), f"multipart metadata headers differ: {first!r} != {second!r}"


def _destination_metadata(request: Any) -> Optional[Dict[str, Any]]:
    """Return compose destination metadata from a JSON request."""
    body, _ = _body_to_str(request.body)
    try:
        document = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    destination = document.get("destination") if isinstance(document, dict) else None
    if not isinstance(destination, dict):
        return None
    metadata = destination.get("metadata")
    return metadata if isinstance(metadata, dict) else None


def multipart_destination_metadata(request1: Any, request2: Any) -> None:
    """Match compose destination metadata when either request supplies it."""
    first = _destination_metadata(request1)
    second = _destination_metadata(request2)
    assert (
        first == second
    ), f"multipart destination metadata differs: {first!r} != {second!r}"


def _patched_metadata(request: Any) -> Optional[Dict[str, Any]]:
    """Return metadata from a JSON PATCH request."""
    if getattr(request, "method", "").upper() != "PATCH":
        return None
    body, _ = _body_to_str(request.body)
    try:
        document = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    metadata = document.get("metadata") if isinstance(document, dict) else None
    return metadata if isinstance(metadata, dict) else None


def patched_metadata(request1: Any, request2: Any) -> None:
    """Match metadata carried by JSON PATCH requests."""
    first = _patched_metadata(request1)
    second = _patched_metadata(request2)
    assert first == second, f"patched metadata differs: {first!r} != {second!r}"


def object_tagging(request1: Any, request2: Any) -> None:
    """Match tags carried by object-tagging requests."""

    def tags(request: Any) -> Optional[Dict[str, str]]:
        if getattr(request, "method", "").upper() != "PUT":
            return None
        query = parse_qs(
            urlparse(getattr(request, "uri", "")).query, keep_blank_values=True
        )
        if "tagging" not in query:
            return None
        raw = _request_body_bytes(request)
        if raw is None:
            return None
        try:
            root = ElementTree.fromstring(raw)
        except ElementTree.ParseError:
            return None
        parsed = {}
        for tag in root.iter():
            if tag.tag.rsplit("}", 1)[-1] != "Tag":
                continue
            fields = {child.tag.rsplit("}", 1)[-1]: child.text or "" for child in tag}
            if "Key" in fields and "Value" in fields:
                parsed[fields["Key"]] = fields["Value"]
        return parsed

    first = tags(request1)
    second = tags(request2)
    assert first == second, f"object tags differ: {first!r} != {second!r}"


def _request_body_bytes(request: Any) -> Optional[bytes]:
    body = getattr(request, "body", None)
    if body is None:
        return None
    if isinstance(body, str):
        return body.encode()
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)
    if not hasattr(body, "read"):
        return None
    position = body.tell() if hasattr(body, "tell") else None
    raw = body.read()
    if position is not None and hasattr(body, "seek"):
        body.seek(position)
    return raw


def _multipart_part_digest(request: Any) -> Optional[str]:
    method = getattr(request, "method", "").upper()
    uri = getattr(request, "uri", "")
    query = parse_qs(urlparse(uri).query)
    raw = _request_body_bytes(request)
    if raw is None:
        return None
    if method == "PUT" and "partNumber" in query:
        return hashlib.sha256(raw).hexdigest()
    if method != "POST" or "/upload/storage/v1/" not in uri:
        return None
    boundary = raw.split(b"\r\n", 1)[0]
    for part in raw.split(boundary):
        headers, separator, payload = part.partition(b"\r\n\r\n")
        if separator and b"application/octet-stream" in headers.lower():
            return hashlib.sha256(payload.removesuffix(b"\r\n")).hexdigest()
    return None


def _is_streaming_body_placeholder(request: Any) -> bool:
    # The conftest's streaming body patch records such a body as this text.
    raw = _request_body_bytes(request)
    return raw is not None and raw.startswith(b"<streaming body omitted: ")


def multipart_part_payload(request1: Any, request2: Any) -> None:
    """Match multipart part bytes when either request uploads a part.

    A part recorded as a streaming body placeholder has no bytes to compare.
    Cassettes recorded before vcrpy 8.5.0 hold one for aiobotocore's chunked
    upload body, which vcrpy now reads and records (kevin1024/vcrpy#974).
    """
    if any(_is_streaming_body_placeholder(r) for r in (request1, request2)):
        return
    first = _multipart_part_digest(request1)
    second = _multipart_part_digest(request2)
    assert first == second, f"multipart part payloads differ: {first!r} != {second!r}"


def register_blob_tag_matchers(vcr_instance: Any) -> None:
    """Register request matchers used by tag conformance tests."""
    vcr_instance.register_matcher("multipart_tag_header", multipart_tag_header)
    vcr_instance.register_matcher("object_tagging", object_tagging)
    vcr_instance.register_matcher(
        "multipart_metadata_headers", multipart_metadata_headers
    )
    vcr_instance.register_matcher(
        "multipart_destination_metadata", multipart_destination_metadata
    )
    vcr_instance.register_matcher("patched_metadata", patched_metadata)
    vcr_instance.register_matcher("multipart_part_payload", multipart_part_payload)


def register_range_header_matcher(vcr_instance: Any) -> None:
    """Register the request matcher used by ranged-stream tests."""
    vcr_instance.register_matcher(RANGE_HEADER_MATCHER, range_header)


def _host_from_uri(uri: str) -> str:
    """Extract hostname from a URI string."""
    # Simple extraction: scheme://host[:port]/path
    try:
        after_scheme = uri.split("://", 1)[1] if "://" in uri else uri
        host_port = after_scheme.split("/", 1)[0]
        host = host_port.split(":")[0]
        return host
    except (IndexError, AttributeError):
        return ""


def _should_ignore_host(host: str, ignore_hosts: Set[str]) -> bool:
    """Check if a host should be ignored (never recorded)."""
    if host in ignore_hosts:
        return True
    if _REGIONAL_STS_PATTERN.match(host):
        return True
    return False


def _body_to_str(body: Any) -> tuple:
    """Convert body to string, returning (str_body, was_bytes)."""
    if body is None:
        return "", False
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace"), True
    return str(body), False


def _scrub_form_body(body_str: str) -> str:
    """Scrub sensitive fields from form-encoded body."""
    try:
        params = parse_qs(body_str, keep_blank_values=True)
        changed = False
        for key in _SENSITIVE_FORM_FIELDS:
            if key in params:
                params[key] = [f"FILTERED_{key.upper()}"]
                changed = True
        if changed:
            return urlencode(params, doseq=True)
    except Exception:
        pass
    return body_str


def _scrub_json_fields(body_str: str, fields: Set[str]) -> str:
    """Scrub sensitive fields from a JSON body string."""
    try:
        data = json.loads(body_str)
        if isinstance(data, dict):
            changed = _scrub_dict_recursive(data, fields)
            if changed:
                return json.dumps(data, separators=(",", ":"))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return body_str


def _scrub_dict_recursive(d: dict, fields: Set[str]) -> bool:
    """Recursively scrub sensitive fields in a dict. Returns True if modified."""
    changed = False
    for key in list(d.keys()):
        if key in fields and isinstance(d[key], str):
            d[key] = f"FILTERED_{key.upper()}"
            changed = True
        elif isinstance(d[key], dict):
            if _scrub_dict_recursive(d[key], fields):
                changed = True
    return changed


def _scrub_request_body(body: Any) -> Any:
    """Scrub sensitive data from a request body."""
    if not body:
        return body

    # Only process str or bytes bodies — skip file-like objects (e.g., BytesIO)
    if not isinstance(body, (str, bytes)):
        return body

    body_str, was_bytes = _body_to_str(body)
    if not body_str.strip():
        return body

    # Try JSON first
    original = body_str
    body_str = _scrub_json_fields(body_str, _SENSITIVE_JSON_REQUEST_FIELDS)
    if body_str != original:
        return body_str.encode("utf-8") if was_bytes else body_str

    # Fall back to form-encoded
    body_str = _scrub_form_body(body_str)
    if body_str != original:
        return body_str.encode("utf-8") if was_bytes else body_str

    return body


def _scrub_xml_identity(body_str: str) -> str:
    """Redact <ID>/<DisplayName> inside <Initiator>/<Owner> XML blocks.

    Object-store list responses (ListParts, ListObjects, ...) echo the
    caller's principal ARN, role name, and canonical user id in these
    elements. No conformance test inspects them, so replace their contents
    with a fixed placeholder to keep principals out of committed cassettes.
    """

    def _redact_block(match: "re.Match[str]") -> str:
        inner = match.group(2)
        inner = _XML_ID_PATTERN.sub(rf"\g<1>{_XML_IDENTITY_PLACEHOLDER}\g<2>", inner)
        inner = _XML_DISPLAY_NAME_PATTERN.sub(
            rf"\g<1>{_XML_IDENTITY_PLACEHOLDER}\g<2>", inner
        )
        return f"{match.group(1)}{inner}{match.group(3)}"

    return _XML_IDENTITY_BLOCK_PATTERN.sub(_redact_block, body_str)


# An error response can echo the request back -- S3's SignatureDoesNotMatch
# quotes the whole canonical request, session token included -- so signed-URL
# credentials have to be scrubbed from response bodies, not just from URIs.
_SIGNED_URL_PARAM_IN_TEXT = re.compile(
    r"(X-Amz-(?:Security-Token|Credential|Signature|Algorithm|Date|Expires"
    r"|SignedHeaders)|AWSAccessKeyId)=[^&\s<\"']*",
    re.IGNORECASE,
)
_AWS_ACCESS_KEY_ELEMENT = re.compile(
    r"(<AWSAccessKeyId>)[^<]*(</AWSAccessKeyId>)", re.IGNORECASE
)
_AWS_TEMP_ACCESS_KEY = re.compile(r"\bASIA[A-Z0-9]{12,}\b")


# Placeholder for a recorded signature: valid base64 of the right length, so the
# client can still decode it, but carrying no key-derived bytes.
_SIGNATURE_PLACEHOLDER = base64.b64encode(b"\0" * 256).decode("ascii")
_SIGNED_BLOB_FIELD = re.compile(r'("signedBlob"\s*:\s*")[^"]*(")')


def _scrub_recorded_signature(text: str) -> str:
    """Replace a recorded signature with a decodable placeholder."""
    return _SIGNED_BLOB_FIELD.sub(r"\1" + _SIGNATURE_PLACEHOLDER + r"\2", text)


def _scrub_signed_url_credentials(text: str) -> str:
    """Redact signed-URL credentials echoed back inside a response body."""
    text = _SIGNED_URL_PARAM_IN_TEXT.sub(r"\1=FILTERED", text)
    text = _AWS_ACCESS_KEY_ELEMENT.sub(r"\1FILTERED\2", text)
    return _AWS_TEMP_ACCESS_KEY.sub("FILTERED_ACCESS_KEY_ID", text)


def _scrub_response_body(body: Any) -> Any:
    """Scrub sensitive data from a response body."""
    if not body:
        return body

    body_str, was_bytes = _body_to_str(body)
    if not body_str.strip():
        return body

    scrubbed = _scrub_json_fields(body_str, _SENSITIVE_JSON_RESPONSE_FIELDS)
    scrubbed = _scrub_xml_identity(scrubbed)
    scrubbed = _scrub_signed_url_credentials(scrubbed)
    scrubbed = _scrub_recorded_signature(scrubbed)
    if scrubbed != body_str:
        return scrubbed.encode("utf-8") if was_bytes else scrubbed

    return body


def _scrub_uri(uri: str) -> str:
    """Scrub sensitive information from URIs."""
    # GCP service account emails
    uri = _GCP_SA_PATTERN.sub(r"\1FILTERED_SA\2", uri)
    return uri


# Query parameters carrying the signature, the signing time, or the caller's
# identity. Two reasons to replace them: a presigned URL puts the session token
# in ``x-amz-security-token``, which must never reach a committed fixture, and
# the signature and timestamp change every run, so a recorded request could
# never be matched again. Matched by substring to cover both providers' naming
# and both signature versions.
_SIGNED_URL_PARAM_MARKERS = (
    "signature",
    "credential",
    "accesskeyid",
    "security-token",
    "securitytoken",
    "date",
    "algorithm",
    "signedheaders",
    "expires",
)


def _normalize_signed_url_query(uri: str) -> str:
    """Replace signature and identity query parameters with a fixed placeholder."""
    parsed = urlparse(uri)
    if not parsed.query:
        return uri
    rewritten = []
    changed = False
    for name, value in parse_qsl(parsed.query, keep_blank_values=True):
        if any(marker in name.lower() for marker in _SIGNED_URL_PARAM_MARKERS):
            rewritten.append((name, "FILTERED"))
            changed = True
        else:
            rewritten.append((name, value))
    if not changed:
        return uri
    return urlunparse(parsed._replace(query=urlencode(rewritten)))


def _normalize_service_account_identity(uri: str) -> str:
    """Replace a whole service-account email in a URI with a fixed placeholder.

    Applied to the outgoing request as well as the recorded one, so a request
    recorded under the real signing identity matches the same request replayed
    under the offline mock identity.
    """
    return _GCP_SA_IDENTITY_PATTERN.sub(_GCP_SA_IDENTITY_PLACEHOLDER, uri)


def make_before_record_request(
    ignore_hosts: Optional[Set[str]] = None,
    scrub_bodies: bool = True,
    scrub_uris: bool = True,
    normalize_service_account_identity: bool = False,
    normalize_signed_url_query: bool = False,
) -> Callable:
    """Create a before_record_request callback for VCR.py.

    Args:
        ignore_hosts: Set of hostnames to skip recording entirely.
            Defaults to DEFAULT_IGNORE_HOSTS.
        scrub_bodies: Whether to scrub sensitive fields from request bodies.
        scrub_uris: Whether to scrub sensitive info from URIs.
        normalize_service_account_identity: Whether to replace a whole
            service-account email in the URI with a fixed placeholder. Needed
            only where the identity appears in a *matched* path, so that a
            recording made under real credentials replays under the mock.
        normalize_signed_url_query: Whether to replace signature, timestamp and
            caller-identity query parameters. Required when a test dereferences
            a presigned URL: those values change every run, and one of them is
            the session token.

    Returns:
        A callback function suitable for VCR.py's before_record_request.
        Returns None for requests to ignored hosts (skipping recording).
    """
    if ignore_hosts is None:
        ignore_hosts = DEFAULT_IGNORE_HOSTS

    def _callback(request):
        # Check if this host should be ignored entirely
        host = _host_from_uri(request.uri or "")
        if _should_ignore_host(host, ignore_hosts):
            return None

        # Scrub URI
        if scrub_uris and request.uri:
            request.uri = _scrub_uri(request.uri)
        if normalize_service_account_identity and request.uri:
            request.uri = _normalize_service_account_identity(request.uri)
        if normalize_signed_url_query and request.uri:
            request.uri = _normalize_signed_url_query(request.uri)

        # Scrub request body
        if scrub_bodies and request.body:
            request.body = _scrub_request_body(request.body)

        return request

    return _callback


# Request bodies larger than this are replaced with a placeholder before being
# written to a cassette. Multipart-upload tests send multi-MiB parts on the
# wire (some providers reject a too-small non-final part), which would bloat
# committed cassettes ~2.5x. The request body is not part of ``match_on``, so
# truncating it does not affect replay matching.
MPU_BODY_TRUNCATE_THRESHOLD = 1024 * 1024


def _truncate_large_body(request: Any, threshold: int) -> None:
    """Replace an oversized request body with a small placeholder in place.

    Handles both file-like bodies (boto3 stores ``BytesIO``) and plain
    ``str``/``bytes`` bodies (the multipart MIME payload other providers
    send). Leaves small or unrecognized bodies untouched.
    """
    body = request.body
    if body is None:
        return
    if hasattr(body, "read"):
        raw = body.read()
    elif isinstance(body, str):
        raw = body.encode("utf-8")
    elif isinstance(body, (bytes, bytearray)):
        raw = bytes(body)
    else:
        return
    if len(raw) <= threshold:
        return
    request.body = f"<vcr-truncated {len(raw)} bytes>".encode("utf-8")


def make_mpu_before_record_request(
    ignore_hosts: Optional[Set[str]] = None,
    scrub_bodies: bool = True,
    scrub_uris: bool = True,
    truncate_threshold: int = MPU_BODY_TRUNCATE_THRESHOLD,
    body_scrub_exempt: Optional[Callable[[Any], bool]] = None,
) -> Callable:
    """Create a before_record_request callback that also truncates big bodies.

    Wraps :func:`make_before_record_request` (host ignoring, URI and body
    scrubbing) and, for anything it keeps, replaces request bodies larger
    than ``truncate_threshold`` with a placeholder so multipart-part uploads
    don't bloat committed cassettes.

    Args:
        ignore_hosts: Hostnames to skip recording entirely.
        scrub_bodies: Whether to scrub sensitive fields from request bodies.
        scrub_uris: Whether to scrub sensitive info from URIs.
        truncate_threshold: Byte size above which a request body is replaced.
        body_scrub_exempt: Predicate selecting requests whose body must not be
            scrubbed. Needed where the HTTP layer hands VCR a file-like body
            the scrubber cannot read; scoping it by request keeps every other
            body, such as a metadata patch, scrubbed as usual.

    Returns:
        A callback suitable for VCR.py's before_record_request.
    """
    scrubbing = make_before_record_request(
        ignore_hosts=ignore_hosts,
        scrub_bodies=scrub_bodies,
        scrub_uris=scrub_uris,
    )
    unscrubbed = make_before_record_request(
        ignore_hosts=ignore_hosts,
        scrub_bodies=False,
        scrub_uris=scrub_uris,
    )

    def _callback(request):
        exempt = body_scrub_exempt is not None and body_scrub_exempt(request)
        inner = unscrubbed if exempt else scrubbing
        request = inner(request)
        if request is None:
            return None
        _truncate_large_body(request, truncate_threshold)
        return request

    return _callback


def make_before_record_response(
    scrub_bodies: bool = True,
) -> Callable:
    """Create a before_record_response callback for VCR.py.

    Args:
        scrub_bodies: Whether to scrub sensitive fields from response bodies.

    Returns:
        A callback function suitable for VCR.py's before_record_response.
    """

    def _callback(response):
        if not scrub_bodies:
            return response

        body = response.get("body", {})
        body_content = body.get("string", "")
        if not body_content:
            return response

        scrubbed = _scrub_response_body(body_content)
        if scrubbed is not body_content:
            body["string"] = scrubbed

        return response

    return _callback


def _read_text_body(request: Any) -> str:
    """Decode a request body to text without consuming a file-like body.

    Some clients store the body as ``BytesIO`` (boto3), others as ``str`` or
    ``bytes``. Reading a file-like body is restored to its prior position so
    matching does not disturb the request.
    """
    body = getattr(request, "body", None)
    if body is None:
        return ""
    if hasattr(body, "read"):
        position = body.tell() if hasattr(body, "tell") else None
        try:
            raw: Any = body.read()
        except Exception:  # noqa: BLE001 - unreadable body is simply "no state"
            return ""
        if position is not None and hasattr(body, "seek"):
            body.seek(position)
    else:
        raw = body
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8", errors="replace")
    return str(raw)


class LegalHoldWrite(NamedTuple):
    """The cloud-agnostic intent of one legal-hold-setting request.

    ``enabled`` is ``True`` when the request places a hold and ``False`` when it
    releases one; ``version_id`` is the specific version the write targeted, or
    ``None`` for the latest version.
    """

    enabled: bool
    version_id: Optional[str]


def decode_legal_hold_write(request: Any) -> Optional[LegalHoldWrite]:
    """Decode a legal-hold-setting request into its cloud-agnostic intent.

    Returns ``None`` for every request that does not set a legal hold, so a
    caller scanning replayed traffic sees only hold writes and ignores uploads,
    reads, retention writes, and listing. Two provider shapes are recognized:
    - a PUT of a legal-hold status document keyed by the ``legal-hold`` query
      parameter, carrying ``<Status>ON|OFF</Status>``
    - a metadata PATCH carrying the ``temporaryHold`` / ``eventBasedHold`` flags,
      where a hold is placed when either flag is set and released when both are
      cleared

    Keys on the query parameter, not a URI substring, so an object key that
    merely contains the text ``legal-hold`` in its path is not misread as a
    hold write.
    """
    method = (getattr(request, "method", "") or "").upper()
    uri = getattr(request, "uri", "") or ""
    query = dict(parse_qsl(urlparse(uri).query, keep_blank_values=True))

    if method == "PUT" and "legal-hold" in query:
        body = _read_text_body(request)
        match = re.search(r"<Status>\s*(ON|OFF)\s*</Status>", body, re.IGNORECASE)
        if match is None:
            return None
        return LegalHoldWrite(
            enabled=match.group(1).upper() == "ON",
            version_id=query.get("versionId"),
        )

    if method == "PATCH":
        body = _read_text_body(request)
        if "temporaryHold" not in body and "eventBasedHold" not in body:
            return None
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        if not isinstance(data, dict) or (
            "temporaryHold" not in data and "eventBasedHold" not in data
        ):
            return None
        enabled = bool(data.get("temporaryHold", False)) or bool(
            data.get("eventBasedHold", False)
        )
        return LegalHoldWrite(enabled=enabled, version_id=query.get("generation"))

    return None


def captured_legal_hold_writes(cassette: Any) -> List[LegalHoldWrite]:
    """Return the legal-hold writes captured on a cassette during replay.

    The capture wrapper installed in conftest records one entry per hold write
    that reaches the wire; this returns them in order, or an empty list if none
    were captured (e.g. an implementation that never issues the write).
    """
    return list(getattr(cassette, "_captured_legal_hold_writes", ()))


def get_default_vcr_config(
    cassette_dir: str,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a complete VCR.py configuration dict with sensible defaults.

    Includes all security and stability header filters, host ignoring,
    body scrubbing, and URI scrubbing out of the box.

    Args:
        cassette_dir: Path to the directory where cassettes will be stored.
        **overrides: Additional keys to merge into the config. These override
            any default values (except filter_headers, which should be extended
            after calling this function).

    Returns:
        A configuration dict suitable for returning from a vcr_config fixture.
    """
    config: Dict[str, Any] = {
        "cassette_library_dir": cassette_dir,
        "filter_headers": list(SECURITY_HEADERS + STABILITY_HEADERS),
        "before_record_request": make_before_record_request(),
        "before_record_response": make_before_record_response(),
        "match_on": [
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
        ],
        "decode_compressed_response": True,
    }
    config.update(overrides)
    return config
