"""OpenAI-compatible client for DMXAPI with rate-limit handling and retry logic."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from loguru import logger as lg

_RATE_LIMIT_PATTERNS = [
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"requests? per minute", re.IGNORECASE),
    re.compile(r"retry.?after", re.IGNORECASE),
    re.compile(r"一分钟内请求次数过多"),
    re.compile(r"请求过于频繁"),
]


@dataclass
class RequestResult:
    """Outcome of a single API call."""

    model: str
    success: bool
    content: str = ""
    error_type: str = ""
    error_message: str = ""
    retries: int = 0
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)


def _is_rate_limit_error(text: str) -> bool:
    return any(p.search(text) for p in _RATE_LIMIT_PATTERNS)


def _classify_error(exc: Exception, response_text: str = "") -> str:
    """Classify an error into a human-readable category."""
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "network"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            return "rate_limit"
        if code == 401 or code == 403:
            return "auth"
        if code == 404:
            return "model_unavailable"
        if code == 413:
            return "content_too_large"
        if 400 <= code < 500:
            return "format_error"
        if code >= 500:
            return "server_error"
    if _is_rate_limit_error(response_text):
        return "rate_limit"
    return "unknown"


class DMXAPIClient:
    """Async client for DMXAPI (OpenAI-compatible endpoint).

    Supports single-image and multi-image (frame sequence) inputs, with
    automatic rate-limit detection and configurable retry logic.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://www.dmxapi.cn",
        timeout_s: float = 180.0,
        max_retries: int = 3,
        retry_wait_s: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._retry_wait_s = retry_wait_s
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Public API ────────────────────────────────────────────────

    async def chat(
        self,
        *,
        model: str,
        prompt: str,
        images_b64: list[tuple[str, str]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> RequestResult:
        """Send a chat-completion request, optionally with images.

        Args:
            model: Model identifier (e.g. "gpt-5.4").
            prompt: Text prompt.
            images_b64: Optional list of (base64_data, mime_type) tuples.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.

        Returns:
            RequestResult with success/failure info.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images_b64:
            for b64, mime in images_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        return await self._request_with_retry(model, payload)

    async def _request_with_retry(
        self, model: str, payload: dict
    ) -> RequestResult:
        retries = 0
        last_error_type = ""
        last_error_msg = ""

        for attempt in range(1 + self._max_retries):
            t0 = time.monotonic()
            response_text = ""
            try:
                client = await self._ensure_client()
                resp = await client.post("/v1/chat/completions", json=payload)
                response_text = resp.text
                resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                usage = data.get("usage", {})
                latency = time.monotonic() - t0
                return RequestResult(
                    model=model,
                    success=True,
                    content=content,
                    retries=retries,
                    latency_s=latency,
                    usage=usage,
                )

            except Exception as exc:
                latency = time.monotonic() - t0
                error_type = _classify_error(exc, response_text)
                error_msg = str(exc)[:300]
                last_error_type = error_type
                last_error_msg = error_msg

                if error_type == "rate_limit" or _is_rate_limit_error(response_text):
                    retries += 1
                    lg.warning(
                        "[{}] Rate limit hit (attempt {}/{}), waiting {}s ...",
                        model, attempt + 1, 1 + self._max_retries, self._retry_wait_s,
                    )
                    await asyncio.sleep(self._retry_wait_s)
                    continue

                if error_type in ("timeout", "server_error", "network") and attempt < self._max_retries:
                    retries += 1
                    wait = min(self._retry_wait_s, 30.0)
                    lg.warning(
                        "[{}] {} (attempt {}/{}), retrying in {}s ...",
                        model, error_type, attempt + 1, 1 + self._max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                lg.error("[{}] {} - {}", model, error_type, error_msg)
                return RequestResult(
                    model=model,
                    success=False,
                    error_type=error_type,
                    error_message=error_msg,
                    retries=retries,
                    latency_s=latency,
                )

        return RequestResult(
            model=model,
            success=False,
            error_type=last_error_type,
            error_message=f"Max retries ({self._max_retries}) exhausted: {last_error_msg}",
            retries=retries,
        )
