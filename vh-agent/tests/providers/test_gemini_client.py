import base64
import json

import httpx
import pytest

from vh_agent.providers.gemini_client import (
    GeminiClient,
    GeminiClientError,
    inline_video_part,
    text_part,
)


def response(content='{"answer":1}', finish="stop", **updates):
    return {
        "id": "reply",
        "object": "chat.completion",
        "created": 1,
        "model": "gemini-3.8-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "private thought",
                },
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        **updates,
    }


@pytest.mark.parametrize("base_url", ["https://example.test/v1", "https://example.test/gateway/v1"])
def test_video_container_structured_schema_and_high_thinking_are_sent(tmp_path, base_url):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"actual-video-with-audio")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    sent = []

    def respond(request):
        assert str(request.url) == base_url + "/chat/completions"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response())

    with GeminiClient(
        "key",
        base_url,
        "gemini-3.8-flash",
        seed=7,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        result = client.generate(
            "Observe", [inline_video_part(path), text_part("What happens?")], schema=schema
        )
    data = sent[0]["messages"][1]["content"][0]["image_url"]["url"]
    assert data.startswith("data:video/mp4;base64,")
    assert base64.b64decode(data.split(",", 1)[1]) == path.read_bytes()
    assert sent[0]["reasoning_effort"] == "high" and sent[0]["seed"] == 7
    assert sent[0]["response_format"]["json_schema"]["schema"] == schema
    assert result["json"] == {"answer": 1} and result["thinking_returned"]
    assert "private thought" not in json.dumps(result)


@pytest.mark.parametrize("status,retryable", [(401, False), (404, False), (429, True), (500, True)])
def test_http_failures_classify_retries_without_exposing_secrets(status, retryable):
    with (
        GeminiClient(
            "echoed-secret",
            "https://example.test/v1",
            "test",
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(status, text="echoed-secret")
                )
            ),
        ) as client,
        pytest.raises(GeminiClientError) as error,
    ):
        client.generate("", [text_part("hello")], schema={})
    assert str(status) in str(error.value)
    assert "echoed-secret" not in str(error.value)
    assert error.value.retryable is retryable


def test_network_error_is_sanitized():
    def respond(request):
        raise httpx.ReadTimeout("secret reflected by upstream", request=request)

    with (
        GeminiClient(
            "secret",
            "https://example.test/v1",
            "test",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client,
        pytest.raises(GeminiClientError) as error,
    ):
        client.generate("", [text_part("hello")], schema={})
    assert error.value.retryable and "secret" not in str(error.value)


@pytest.mark.parametrize(
    "raw",
    [
        response(finish="length"),
        response(content=None),
        response(content='```json\n{"ok":true}\n```'),
        response(choices=[]),
    ],
)
def test_incomplete_or_invalid_json_is_not_repaired(raw):
    with (
        GeminiClient(
            "secret",
            "https://example.test/v1",
            "test",
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw))
            ),
        ) as client,
        pytest.raises(GeminiClientError),
    ):
        client.generate("", [text_part("hello")], schema={})


@pytest.mark.parametrize(
    "url",
    [
        "https://host.test/v1?key=secret",
        "https://user:secret@host.test/v1",
        "https://host.test/v1#fragment",
    ],
)
def test_endpoint_rejects_embedded_credentials_queries_and_fragments(url):
    with pytest.raises(ValueError):
        GeminiClient("key", url, "test")
