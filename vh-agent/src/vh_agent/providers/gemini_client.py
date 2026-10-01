"""Small synchronous client for Gemini's native generateContent REST API."""

import base64
import json
import math
import mimetypes
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit, urlunsplit

import httpx

MAX_INLINE_REQUEST_BYTES = 20_000_000


class GeminiClientError(RuntimeError):
    """A native request failed; messages intentionally omit request credentials."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def text_part(text: str) -> dict[str, Any]:
    return {"text": text}


def inline_video_part(path: str | Path, fps: float = 4.0) -> dict[str, Any]:
    """Encode the actual video container, including any existing audio stream."""
    path = Path(path)
    if not math.isfinite(fps) or not 0 < fps <= 24:
        raise ValueError("Video fps must be finite and in the range (0, 24]")
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type is None or not mime_type.startswith("video/"):
        raise ValueError("A recognized video file extension is required")
    if 4 * ((path.stat().st_size + 2) // 3) >= MAX_INLINE_REQUEST_BYTES:
        raise ValueError("Video is too large for an inline request; use a shorter clip")
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        },
        "videoMetadata": {"fps": fps},
    }


def function_response_part(
    name: str, response: dict[str, Any], call_id: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "response": response}
    if call_id is not None:
        result["id"] = call_id
    return {"functionResponse": result}


def _native_endpoint(base_url: str, model: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Gemini base URL must be an HTTP origin without credentials or query")
    if parsed.path.rstrip("/") not in {"", "/v1", "/v1beta"}:
        raise ValueError("Gemini base URL must use the origin root, /v1, or /v1beta")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", model):
        raise ValueError("Gemini model must be a model identifier without a URL or path")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"/v1beta/models/{model}:generateContent", "", "")
    )


class GeminiClient:
    """Use one explicit native endpoint with no protocol or model fallback.

    ``tools`` contains FunctionDeclaration dictionaries, not OpenAI tools.
    ``parts`` and ``history`` use native Gemini Part and Content dictionaries.
    Append returned ``content`` unchanged to history to preserve thought signatures.
    The caller owns tool execution and validates task-specific response schemas.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        seed: int | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        self.endpoint = _native_endpoint(base_url, model)
        self.model = model
        self.seed = seed
        self._api_key = api_key
        self._timeout = timeout
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def generate(
        self,
        system: str,
        parts: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        *,
        history: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate one model turn; ``json`` is parsed only when a schema is supplied."""
        if not parts:
            raise ValueError("At least one input part is required")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        generation: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
        if self.seed is not None:
            generation["seed"] = self.seed
        if schema is not None:
            generation.update(responseMimeType="application/json", responseJsonSchema=schema)
        payload: dict[str, Any] = {
            "contents": [*(history or []), {"role": "user", "parts": parts}],
            "generationConfig": generation,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]
        if tool_config is not None:
            if not tools:
                raise ValueError("tool_config requires function declarations")
            payload["toolConfig"] = tool_config
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) >= MAX_INLINE_REQUEST_BYTES:
            raise ValueError("Request exceeds the 20 MB inline limit; reduce video or history")
        try:
            response = self._http.post(
                self.endpoint,
                headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
                content=body,
                timeout=self._timeout,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise GeminiClientError(
                f"Gemini native request failed ({type(exc).__name__})", retryable=True
            ) from None
        if not response.is_success:
            raise GeminiClientError(
                f"Gemini native request returned HTTP {response.status_code}",
                retryable=response.status_code in {408, 429} or response.is_server_error,
            )
        try:
            raw = response.json()
        except ValueError:
            raise GeminiClientError("Gemini native endpoint returned invalid JSON") from None
        result = self._normalize(raw, parse_json=schema is not None)
        result["text"] = result["text"].replace(self._api_key, "[redacted]")
        return result

    @staticmethod
    def _normalize(raw: Any, *, parse_json: bool) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise GeminiClientError("Gemini native response must be an object")
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise GeminiClientError("Gemini native response has no candidates")
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise GeminiClientError("Gemini native candidate must be an object")
        finish_reason = candidate.get("finishReason")
        if finish_reason not in {None, "STOP"}:
            # Do not interpolate server-controlled text (it may echo credentials).
            raise GeminiClientError("Gemini generation did not complete successfully")
        content = candidate.get("content")
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            raise GeminiClientError("Gemini native candidate has no content parts")
        model_parts = content["parts"]
        if not model_parts or not all(isinstance(part, dict) for part in model_parts):
            raise GeminiClientError("Gemini native candidate has invalid content parts")
        text = "".join(
            part["text"]
            for part in model_parts
            if isinstance(part.get("text"), str) and not part.get("thought", False)
        )
        calls = [deepcopy(part["functionCall"]) for part in model_parts if "functionCall" in part]
        if not all(
            isinstance(call, dict)
            and isinstance(call.get("name"), str)
            and isinstance(call.get("args", {}), dict)
            for call in calls
        ):
            raise GeminiClientError("Gemini returned an invalid function call")
        parsed = None
        if parse_json and not calls:
            try:
                parsed = json.loads(text)
            except ValueError:
                raise GeminiClientError("Gemini response is not valid structured JSON") from None
        if not text and not calls:
            raise GeminiClientError(
                "Gemini returned neither text nor function calls", retryable=True
            )
        return {
            "text": text,
            "json": parsed,
            "function_calls": calls,
            "content": deepcopy(content),
            "usage": deepcopy(raw.get("usageMetadata", {})),
            "finish_reason": finish_reason,
            "model_version": raw.get("modelVersion", ""),
        }
