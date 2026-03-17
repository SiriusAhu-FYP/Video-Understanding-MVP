"""pipeline.vlm 模块的功能测试 (全部使用 mock，无需真实 vLLM 服务)。"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from ahu_paimon_toolkit.models import KeyFrame
from ahu_paimon_toolkit.vlm.client import AsyncVLMClient as VLMClient


def _make_keyframe(frame_id: int = 0) -> KeyFrame:
    return KeyFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 500,
        base64_image="dGVzdA==",
        created_at=time.monotonic(),
    )


def _mock_vlm_response(content: str = "一只猫在公园里奔跑。") -> httpx.Response:
    """构造模拟的 vLLM OpenAI 兼容响应。"""
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    return httpx.Response(
        status_code=200,
        json=body,
        request=httpx.Request("POST", "http://test/v1/chat/completions"),
    )


class TestLoadVLMPrompt:
    def test_loads_prompt_file(self) -> None:
        prompt = load_vlm_prompt()
        assert len(prompt) > 0
        assert "视频" in prompt or "描述" in prompt


@pytest.mark.asyncio
class TestVLMClient:
    async def test_describe_frame_success(self) -> None:
        """正常情况下，返回 FrameDescription。"""
        client = VLMClient(prompt="测试提示词")

        transport = httpx.MockTransport(
            lambda request: _mock_vlm_response("画面中有一只橘猫正在草坪上追逐蝴蝶。")
        )
        client._client = httpx.AsyncClient(
            base_url="http://test/v1",
            transport=transport,
        )

        kf = _make_keyframe(frame_id=5)
        result = await client.describe_frame(kf)

        assert result.frame_id == 5
        assert result.timestamp_ms == 2500
        assert "橘猫" in result.description

        await client.close()

    async def test_describe_frame_preserves_frame_metadata(self) -> None:
        client = VLMClient(prompt="测试")

        transport = httpx.MockTransport(lambda request: _mock_vlm_response("测试描述"))
        client._client = httpx.AsyncClient(
            base_url="http://test/v1",
            transport=transport,
        )

        kf = _make_keyframe(frame_id=42)
        result = await client.describe_frame(kf)

        assert result.frame_id == 42
        assert result.timestamp_ms == 42 * 500

        await client.close()

    async def test_request_payload_format(self) -> None:
        """验证发送给 vLLM 的请求格式正确。"""
        captured_request: dict | None = None

        def capture_handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = json.loads(request.content)
            return _mock_vlm_response()

        client = VLMClient(prompt="分析这张图片")
        client._client = httpx.AsyncClient(
            base_url="http://test/v1",
            transport=httpx.MockTransport(capture_handler),
        )

        kf = _make_keyframe(frame_id=0)
        await client.describe_frame(kf)

        assert captured_request is not None
        assert "model" in captured_request
        assert captured_request["messages"][0]["role"] == "user"

        content = captured_request["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

        await client.close()

    async def test_http_error_propagates(self) -> None:
        """vLLM 返回 HTTP 错误时应抛出异常。"""
        client = VLMClient(prompt="测试")
        client._client = httpx.AsyncClient(
            base_url="http://test/v1",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    status_code=500,
                    text="Internal Server Error",
                    request=request,
                )
            ),
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.describe_frame(_make_keyframe())

        await client.close()

    async def test_close_is_idempotent(self) -> None:
        client = VLMClient(prompt="测试")
        await client.close()
        await client.close()
