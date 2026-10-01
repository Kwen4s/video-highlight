import base64
import json
from pathlib import Path

import httpx
import pytest

from vh_agent.providers.gemini_client import (
    GeminiClient,
    GeminiClientError,
    function_response_part,
    inline_video_part,
    text_part,
)


def _response(parts, **updates):
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"totalTokenCount": 17},
        "modelVersion": "gemini-test",
        **updates,
    }


def test_native_tool_round_trip_preserves_signature_and_input_history():
    requests = []
    signature = "opaque-signature-must-stay-with-part"
    model_parts = [
        {"text": "private reasoning", "thought": True},
        {
            "functionCall": {"id": "call-1", "name": "watch", "args": {"start_sec": 1}},
            "thoughtSignature": signature,
        },
    ]

    def respond(request):
        requests.append(request)
        parts = model_parts if len(requests) == 1 else [{"text": "Verified."}]
        return httpx.Response(200, json=_response(parts))

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = GeminiClient(
            "test-key", "https://example.test/v1", "gemini-test", http_client=http
        )
        tools = [{"name": "watch", "parametersJsonSchema": {"type": "object"}}]
        prompt = [text_part("Find the event")]
        first = client.generate("Use tools", prompt, tools=tools)
        assert first["text"] == ""
        assert first["content"]["parts"] == model_parts
        assert first["function_calls"] == [model_parts[1]["functionCall"]]
        history = [{"role": "user", "parts": prompt}, first["content"]]
        original_history = json.dumps(history)
        result = client.generate(
            "Use tools",
            [function_response_part("watch", {"color": "blue"}, "call-1")],
            tools=tools,
            history=history,
        )
    assert str(requests[0].url) == "https://example.test/v1beta/models/gemini-test:generateContent"
    assert requests[0].headers["x-goog-api-key"] == "test-key"
    sent = json.loads(requests[1].content)
    assert sent["contents"][1]["parts"] == model_parts
    assert sent["contents"][2]["parts"][0]["functionResponse"]["id"] == "call-1"
    assert json.dumps(history) == original_history
    assert result["text"] == "Verified."
    assert result["usage"] == {"totalTokenCount": 17}


def test_video_bytes_fps_and_json_schema_are_sent_without_text_substitution(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    source = b"test-video-container-bytes"
    video.write_bytes(source)
    schema = {
        "$defs": {"Event": {"type": "object", "properties": {"time": {"type": "number"}}}},
        "type": "object",
        "properties": {"event": {"$ref": "#/$defs/Event"}},
    }
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_response([{"text": '{"event":{"time":2}}'}]))

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = GeminiClient(
            "key", "https://example.test/v1beta", "test", seed=7, http_client=http
        )
        result = client.generate(
            "", [inline_video_part(video, fps=4), text_part("When?")], schema=schema
        )
    part = sent[0]["contents"][0]["parts"][0]
    assert base64.b64decode(part["inlineData"]["data"]) == source
    assert part["inlineData"]["mimeType"] == "video/mp4"
    assert part["videoMetadata"]["fps"] == 4
    assert sent[0]["generationConfig"]["responseJsonSchema"] == schema
    assert sent[0]["generationConfig"]["seed"] == 7
    assert result["json"] == {"event": {"time": 2}}


@pytest.mark.parametrize(
    "status,retryable",
    [(301, False), (401, False), (404, False), (429, True), (500, True), (524, True)],
)
def test_http_failures_classify_retries_without_exposing_secrets(status, retryable):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            status,
            text="echoed-secret should never appear",
            headers={"Location": "https://different-origin.test/"},
        )

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as http:
        client = GeminiClient("echoed-secret", "https://example.test/v1", "test", http_client=http)
        with pytest.raises(GeminiClientError) as error:
            client.generate("", [text_part("hello")])
    assert str(status) in str(error.value)
    assert "echoed-secret" not in str(error.value)
    assert error.value.retryable is retryable
    assert len(calls) == 1


def test_network_error_is_sanitized():
    def respond(request):
        raise httpx.ReadTimeout("secret reflected by upstream", request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = GeminiClient("secret", "https://example.test", "test", http_client=http)
        with pytest.raises(GeminiClientError, match="ReadTimeout") as error:
            client.generate("", [text_part("hello")])
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"candidates": []},
        {"candidates": [{"finishReason": "MAX_TOKENS"}]},
        _response([]),
        _response([{"text": "only hidden reasoning", "thought": True}]),
        _response([{"functionCall": {"name": "watch", "args": "invalid"}}]),
    ],
)
def test_blocked_truncated_empty_and_malformed_results_are_failures(response):
    with pytest.raises(GeminiClientError):
        GeminiClient._normalize(response, parse_json=False)


def test_structured_output_does_not_silently_repair_invalid_json():
    with pytest.raises(GeminiClientError, match="structured JSON"):
        GeminiClient._normalize(_response([{"text": '```json\n{"ok":true}\n```'}]), parse_json=True)


@pytest.mark.parametrize(
    "url",
    [
        "https://host.test/v1?key=secret",
        "https://user:secret@host.test",
        "https://host.test/proxy/v1",
    ],
)
def test_endpoint_rejects_ambiguous_credentials_and_paths(url):
    with pytest.raises(ValueError):
        GeminiClient("key", url, "test")


@pytest.mark.parametrize("fps", [0, -1, 25, float("nan"), float("inf")])
def test_invalid_sampling_rate_is_rejected(tmp_path: Path, fps):
    with pytest.raises(ValueError, match="fps"):
        inline_video_part(tmp_path / "clip.mp4", fps=fps)
