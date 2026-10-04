"""Gemini audiovisual requests through the gateway's Chat Completions API."""

import base64
import json
import mimetypes
import socket
import ssl
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, DefaultHttpxClient, OpenAI

MAX_INLINE_REQUEST_BYTES = 20_000_000


class GeminiClientError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def _connection_failure(exc: Exception) -> str:
    """Describe library exception types and OS codes without echoing request data."""
    causes = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        causes.append(current)
        current = current.__cause__ or current.__context__
    category = "connection"
    for error_type, name in (
        (socket.gaierror, "DNS resolution"),
        (ssl.SSLError, "TLS"),
        (httpx.ProxyError, "proxy"),
        (httpx.ConnectTimeout, "connect timeout"),
        (httpx.ReadTimeout, "read timeout"),
        (httpx.WriteTimeout, "write timeout"),
        (httpx.PoolTimeout, "connection pool timeout"),
        (ConnectionRefusedError, "connection refused"),
    ):
        if any(isinstance(cause, error_type) for cause in causes):
            category = name
            break
    details = " -> ".join(type(cause).__name__ for cause in causes)
    for cause in causes:
        if isinstance(cause, OSError) and isinstance(cause.errno, int):
            details += f"; errno={cause.errno}"
        verify_code = getattr(cause, "verify_code", None)
        if isinstance(cause, ssl.SSLCertVerificationError) and isinstance(verify_code, int):
            details += f"; verify_code={verify_code}"
    return f"Gemini {category} failed ({details})"


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _inline_part(path: Path, mime_type: str) -> dict[str, Any]:
    if 4 * ((path.stat().st_size + 2) // 3) >= MAX_INLINE_REQUEST_BYTES:
        raise ValueError("Media exceeds the inline limit; use a shorter range")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64," + base64.b64encode(path.read_bytes()).decode()
        },
    }


def inline_video_part(path: str | Path) -> dict[str, Any]:
    """Send the actual video container, preserving its audio."""
    path = Path(path)
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type is None or not mime_type.startswith("video/"):
        raise ValueError("A recognized video file extension is required")
    return _inline_part(path, mime_type)


def inline_frame_part(path: str | Path) -> dict[str, Any]:
    return _inline_part(Path(path), "image/jpeg")


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 300,
        seed: int | None = None,
        thinking_level: str = "high",
        http_client=None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Gemini base URL must be an HTTP URL without credentials or query")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        if thinking_level not in {"low", "medium", "high"}:
            raise ValueError("Gemini thinking level must be low, medium, or high")
        self.model, self.seed, self.thinking_level = model, seed, thinking_level
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client
            if http_client is not None
            else DefaultHttpxClient(trust_env=False),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args):
        self._client.close()

    def generate(self, system, parts, *, schema, max_output_tokens=8192):
        if not parts or max_output_tokens <= 0:
            raise ValueError("Input parts and a positive output budget are required")
        instructions = (
            f"{system}\n直接返回 JSON 对象，不加 Markdown 代码围栏。JSON 结构：\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": parts},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "reasoning_effort": self.thinking_level,
            "max_tokens": max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "video_result", "strict": True, "schema": schema},
            },
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if len(json.dumps(payload, ensure_ascii=False).encode()) >= MAX_INLINE_REQUEST_BYTES:
            raise ValueError("Request exceeds the 20 MB limit; use a shorter video range")
        try:
            response = self._client.chat.completions.create(**payload)
        except APIStatusError as exc:
            raise GeminiClientError(
                f"Gemini request returned HTTP {exc.status_code}",
                retryable=exc.status_code in {408, 429} or exc.status_code >= 500,
            ) from None
        except (APIConnectionError, APITimeoutError, httpx.HTTPError) as exc:
            raise GeminiClientError(_connection_failure(exc), retryable=True) from None
        if not response.choices:
            raise GeminiClientError("Gemini response has no choices")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise GeminiClientError("Gemini generation did not complete successfully")
        try:
            reading = json.loads(choice.message.content)
        except (ValueError, TypeError):
            raise GeminiClientError("Gemini response is not valid structured JSON") from None
        # Preserve usage and whether reasoning was returned, without persisting thought text.
        return {
            "json": reading,
            "usage": response.usage.model_dump(exclude_none=True) if response.usage else {},
            "model_version": response.model,
            "thinking_returned": bool(getattr(choice.message, "reasoning_content", None)),
        }
