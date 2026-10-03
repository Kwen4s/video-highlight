"""Native streamed Responses requests. Only completed responses can execute tools."""

import json
import re
import time
from typing import Self
from urllib.parse import urlsplit

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


class OpenAIClientError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class OpenAIClient:
    def __init__(self, api_key, base_url, model, *, effort="high", timeout=300, http_client=None):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("OpenAI base URL must be an HTTP URL without credentials or query")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty")
        self.model, self.effort = model, effort
        self.endpoint = base_url.rstrip("/") + "/responses"
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args):
        self._client.close()

    def generate(self, instructions, inputs, *, tools, max_output_tokens=16384):
        started = time.perf_counter()
        first_event_sec = None
        completed = None
        try:
            with self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=inputs,
                reasoning={"effort": self.effort},
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=False,
                max_output_tokens=max_output_tokens,
                store=False,
                stream=True,
                include=["reasoning.encrypted_content"],
            ) as stream:
                for event in stream:
                    if first_event_sec is None:
                        first_event_sec = time.perf_counter() - started
                    if event.type == "response.completed":
                        completed = event.response.model_dump(mode="json", exclude_none=True)
                        break
                    elif event.type in {"response.failed", "response.incomplete", "error"}:
                        raw = event.model_dump(mode="json", exclude_none=True)
                        response = raw.get("response", {})
                        error = response.get("error") or raw
                        code = error.get("code") or (response.get("incomplete_details") or {}).get(
                            "reason", "unknown"
                        )
                        # Keep the machine-readable cause; opaque upstream messages can echo input.
                        code = (
                            code if re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", str(code)) else "unknown"
                        )
                        raise OpenAIClientError(
                            f"Responses {event.type}: {code}",
                            retryable=code in {"server_error", "rate_limit_exceeded", "timeout"},
                        )
        except APIStatusError as exc:
            raise OpenAIClientError(
                f"Responses request returned HTTP {exc.status_code}",
                retryable=exc.status_code in {408, 429} or exc.status_code >= 500,
            ) from None
        except (APIConnectionError, APITimeoutError, httpx.HTTPError):
            raise OpenAIClientError("Responses stream disconnected", retryable=True) from None
        if completed is None or completed.get("status") != "completed":
            raise OpenAIClientError("Responses stream ended before completion", retryable=True)
        calls = []
        for item in completed["output"]:
            if item["type"] == "function_call":
                try:
                    args = json.loads(item["arguments"])
                except (ValueError, TypeError):
                    raise OpenAIClientError("Responses returned invalid tool arguments") from None
                if not isinstance(args, dict) or not item.get("call_id"):
                    raise OpenAIClientError("Responses returned an invalid tool call")
                calls.append({"name": item["name"], "args": args, "id": item["call_id"]})
        usage = completed.get("usage", {})
        return {
            "output": completed["output"],
            "function_calls": calls,
            "usage": {
                k: usage[k]
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "input_tokens_details",
                    "output_tokens_details",
                )
                if k in usage
            },
            "model_version": completed.get("model", ""),
            "response_id": completed["id"],
            "first_event_sec": first_event_sec,
            "elapsed_sec": time.perf_counter() - started,
        }
