import pytest
from vcr.request import Request
from vcr.stubs import aiohttp_stubs

import tests.blob.conformance.conftest  # noqa: F401


@pytest.mark.asyncio
async def test_built_replay_response_has_a_readable_content_stream():
    request = Request("GET", "https://example.invalid/object", None, {})
    recorded = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"Content-Length": ["8"]},
        "body": {"string": b"abcdefgh"},
    }

    response = aiohttp_stubs.build_response(request, recorded, [])

    assert await response.content.read(4) == b"abcd"
    assert await response.content.read(4) == b"efgh"
    assert await response.content.read(4) == b""
