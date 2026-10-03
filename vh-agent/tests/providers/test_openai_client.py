import json

import httpx
import pytest

from vh_agent.providers.openai_client import OpenAIClient, OpenAIClientError


def completed(arguments='{"value":1}'):
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-6.1-sol",
            "output": [
                {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque"},
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "record",
                    "arguments": arguments,
                },
            ],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        },
    }


def stream(events):
    return "\n\n".join("data: " + json.dumps(e) for e in events) + "\n\n"


def test_native_high_stream_returns_only_completed_output_and_keeps_ids():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=stream(
                [
                    {"type": "response.function_call_arguments.delta", "delta": '{"value":'},
                    completed(),
                ]
            ),
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as http,
        OpenAIClient(
            "secret", "https://example.test/v1", "gpt-6.1-sol", http_client=http
        ) as client,
    ):
        result = client.generate("Record", [{"role": "user", "content": "test"}], tools=[])
    assert requests[0]["reasoning"] == {"effort": "high"}
    assert requests[0]["parallel_tool_calls"] is False and requests[0]["store"] is False
    assert result["function_calls"] == [{"name": "record", "args": {"value": 1}, "id": "call_1"}]
    assert result["output"][0]["encrypted_content"] == "opaque"
    assert result["first_event_sec"] is not None


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "response.function_call_arguments.delta", "delta": '{"value":1}'}],
        [completed("{bad json")],
        [{"type": "response.incomplete", "response": {"status": "incomplete"}}],
    ],
)
def test_incomplete_stream_and_invalid_arguments_cannot_execute(events):
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, text=stream(events)
                )
            )
        ) as http,
        OpenAIClient(
            "secret", "https://example.test/v1", "gpt-6.1-sol", http_client=http
        ) as client,
        pytest.raises(OpenAIClientError),
    ):
        client.generate("", [{"role": "user", "content": "test"}], tools=[])


def test_midstream_network_failure_is_retryable_and_sanitized():
    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"type":"response.created"}\n\n'
            raise httpx.RemoteProtocolError("secret server body")

    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=Broken()
                )
            )
        ) as http,
        OpenAIClient(
            "secret", "https://example.test/v1", "gpt-6.1-sol", http_client=http
        ) as client,
        pytest.raises(OpenAIClientError) as error,
    ):
        client.generate("", [{"role": "user", "content": "test"}], tools=[])
    assert error.value.retryable and "secret" not in str(error.value)


@pytest.mark.parametrize(
    "code,retryable",
    [("server_error", True), ("rate_limit_exceeded", True), ("invalid_request", False)],
)
def test_stream_failure_retains_error_code_and_classifies_retry(code, retryable):
    events = [
        {
            "type": "response.failed",
            "response": {
                "id": "resp_failed",
                "status": "failed",
                "error": {"code": code, "message": "secret input echoed by server"},
            },
        }
    ]
    with (
        OpenAIClient(
            "secret",
            "https://example.test/v1",
            "gpt-6.1-sol",
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, text=stream(events)
                    )
                )
            ),
        ) as client,
        pytest.raises(OpenAIClientError) as error,
    ):
        client.generate("", [{"role": "user", "content": "test"}], tools=[])
    assert code in str(error.value) and "secret" not in str(error.value)
    assert error.value.retryable is retryable
