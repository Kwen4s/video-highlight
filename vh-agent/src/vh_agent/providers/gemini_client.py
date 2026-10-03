"""Gemini audiovisual requests through the gateway's Chat Completions API."""

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

MAX_INLINE_REQUEST_BYTES = 20_000_000


class GeminiClientError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


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
            http_client=http_client,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args):
        self._client.close()

    def generate(self, system, parts, *, schema, max_output_tokens=8192):
        if not parts or max_output_tokens <= 0:
            raise ValueError("Input parts and a positive output budget are required")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": parts}]
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
            cause = type(exc.__cause__ or exc).__name__
            raise GeminiClientError(
                f"Gemini request disconnected ({cause})", retryable=True
            ) from None
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
