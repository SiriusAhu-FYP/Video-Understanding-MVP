"""pipeline.summarizer 模块的功能测试 (全部使用 mock，无需真实 DeepSeek API)。"""

from __future__ import annotations

import json

import httpx
import pytest

from pipeline.models import FrameDescription
from pipeline.summarizer import Summarizer, _build_frame_text, load_deepseek_prompt


def _make_descriptions(count: int = 3) -> list[FrameDescription]:
    return [
        FrameDescription(
            frame_id=i,
            timestamp_ms=i * 500,
            description=f"第 {i} 帧的描述内容",
        )
        for i in range(count)
    ]


def _mock_deepseek_response(
    summary: str = "这段视频展示了一只猫在公园中追逐蝴蝶的过程。",
) -> httpx.Response:
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": summary},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
    }
    return httpx.Response(
        status_code=200,
        json=body,
        request=httpx.Request("POST", "http://test/v1/chat/completions"),
    )


class TestLoadDeepseekPrompt:
    def test_loads_prompt_file(self) -> None:
        prompt = load_deepseek_prompt()
        assert len(prompt) > 0
        assert "总结" in prompt or "视频" in prompt


class TestBuildFrameText:
    def test_format_with_timestamps(self) -> None:
        descs = _make_descriptions(3)
        text = _build_frame_text(descs)

        assert "[0.0s]" in text
        assert "[0.5s]" in text
        assert "[1.0s]" in text
        assert "帧#0" in text
        assert "帧#2" in text

    def test_empty_descriptions(self) -> None:
        text = _build_frame_text([])
        assert text == ""

    def test_preserves_order(self) -> None:
        descs = _make_descriptions(5)
        text = _build_frame_text(descs)
        lines = text.strip().split("\n")
        assert len(lines) == 5


@pytest.mark.asyncio
class TestSummarizer:
    async def test_summarize_success(self) -> None:
        expected_summary = "视频中一只猫在公园追逐蝴蝶。"
        summarizer = Summarizer(
            api_key="test-key",
            base_url="http://test",
            prompt="测试提示词",
        )
        summarizer._client = httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.MockTransport(
                lambda req: _mock_deepseek_response(expected_summary)
            ),
        )

        descs = _make_descriptions(3)
        result = await summarizer.summarize(descs, duration_s=1.5)

        assert result.summary_text == expected_summary
        assert result.total_keyframes == 3
        assert result.duration_s == 1.5
        assert len(result.frame_descriptions) == 3

        await summarizer.close()

    async def test_empty_descriptions_returns_fallback(self) -> None:
        summarizer = Summarizer(
            api_key="test-key",
            base_url="http://test",
            prompt="测试",
        )

        result = await summarizer.summarize([], duration_s=0.0)
        assert result.total_keyframes == 0
        assert "无有效帧" in result.summary_text

        await summarizer.close()

    async def test_request_includes_auth_header(self) -> None:
        captured_headers: dict = {}

        def capture_handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_headers
            captured_headers = dict(request.headers)
            return _mock_deepseek_response()

        summarizer = Summarizer(
            api_key="sk-test-12345",
            base_url="http://test",
            prompt="测试",
        )
        summarizer._client = httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.MockTransport(capture_handler),
            headers={
                "Authorization": "Bearer sk-test-12345",
                "Content-Type": "application/json",
            },
        )

        await summarizer.summarize(_make_descriptions(1), duration_s=0.5)

        assert "authorization" in captured_headers
        assert "sk-test-12345" in captured_headers["authorization"]

        await summarizer.close()

    async def test_request_payload_structure(self) -> None:
        captured_payload: dict | None = None

        def capture_handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_payload
            captured_payload = json.loads(request.content)
            return _mock_deepseek_response()

        summarizer = Summarizer(
            api_key="test-key",
            base_url="http://test",
            prompt="分析以下帧描述",
        )
        summarizer._client = httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.MockTransport(capture_handler),
        )

        await summarizer.summarize(_make_descriptions(2), duration_s=1.0)

        assert captured_payload is not None
        assert captured_payload["model"] == "deepseek-chat"
        assert len(captured_payload["messages"]) == 1
        assert captured_payload["messages"][0]["role"] == "user"
        assert "帧描述数据" in captured_payload["messages"][0]["content"]

        await summarizer.close()

    async def test_http_error_propagates(self) -> None:
        summarizer = Summarizer(
            api_key="test-key",
            base_url="http://test",
            prompt="测试",
        )
        summarizer._client = httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    status_code=429,
                    text="Rate limit exceeded",
                    request=req,
                )
            ),
        )

        with pytest.raises(httpx.HTTPStatusError):
            await summarizer.summarize(_make_descriptions(1), duration_s=0.5)

        await summarizer.close()

    async def test_close_is_idempotent(self) -> None:
        summarizer = Summarizer(
            api_key="test-key",
            base_url="http://test",
            prompt="测试",
        )
        await summarizer.close()
        await summarizer.close()
