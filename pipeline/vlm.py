"""vLLM 本地推理客户端：异步调用 OpenAI 兼容接口，对关键帧进行视觉描述。"""

from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger as lg

from pipeline.config import get_settings
from pipeline.models import FrameDescription, KeyFrame

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "vlm_prompt.md"


def load_vlm_prompt() -> str:
    """从 prompts/vlm_prompt.md 读取提示词。"""
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"vLLM 提示词文件不存在: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


class VLMClient:
    """异步 vLLM 推理客户端。

    通过 httpx.AsyncClient 调用本地 vLLM 的 OpenAI 兼容 chat/completions 接口，
    将关键帧图片（Base64）发送给视觉语言模型，获取帧描述。
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        cfg = get_settings().llm
        self._base_url = base_url or cfg.vllm_base_url
        self._model = model or cfg.vllm_model
        self._prompt = prompt or load_vlm_prompt()
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def describe_frame(self, keyframe: KeyFrame) -> FrameDescription:
        """对单个关键帧调用 vLLM，返回结构化描述。"""
        client = await self._ensure_client()

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{keyframe.base64_image}",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }

        lg.debug("调用 vLLM 描述关键帧 #{} (timestamp={}ms)", keyframe.frame_id, keyframe.timestamp_ms)

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]

        lg.info(
            "vLLM 返回帧 #{} 描述 ({} 字符)",
            keyframe.frame_id, len(content),
        )

        return FrameDescription(
            frame_id=keyframe.frame_id,
            timestamp_ms=keyframe.timestamp_ms,
            description=content,
        )

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            lg.debug("vLLM 客户端已关闭")
