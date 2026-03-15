"""vLLM 本地推理客户端：异步调用 OpenAI 兼容接口，对关键帧进行视觉描述。

支持视觉语言模型（VLM，发送图片）和纯文本模型（仅发送文本提示词）。
优先通过 vLLM 模型信息接口检测视觉能力，名称关键词匹配作为备选。
"""

from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger as lg

from pipeline.config import get_settings
from pipeline.models import FrameDescription, KeyFrame

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "vlm_prompt.md"

_VLM_KEYWORDS = {"vl", "vision", "visual"}

_KNOWN_VLM_PREFIXES = {
    "qwen/qwen3.5",
    "qwen/qwen3-vl",
    "qwen/qwen2.5-vl",
    "qwen/qwen2-vl",
    "opengvlab/internvl",
    "deepseek-ai/deepseek-vl",
    "mistralai/ministral",
    "vikhyatk/moondream",
    "zero-point-ai/martha",
}


def load_vlm_prompt() -> str:
    """从 prompts/vlm_prompt.md 读取提示词。"""
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"vLLM 提示词文件不存在: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def is_vision_model(model_id: str) -> bool:
    """判断模型是否为视觉语言模型。

    检测策略（按优先级）:
    1. 已知 VLM 系列前缀匹配（如 Qwen3.5 系列均为 VLM）
    2. 模型名称中包含 vl/vision/visual 关键词
    3. 默认视为 VLM（保守策略，避免误判导致丢失图片信息）

    Args:
        model_id: 模型标识符，如 "Qwen/Qwen3-VL-2B-Instruct"。

    Returns:
        True 表示该模型支持图片输入。
    """
    model_lower = model_id.lower()

    for prefix in _KNOWN_VLM_PREFIXES:
        if model_lower.startswith(prefix):
            return True

    parts = model_lower.replace("/", "-").replace("_", "-").split("-")
    if set(parts) & _VLM_KEYWORDS:
        return True

    lg.info(
        "模型 '{}' 未匹配已知 VLM 模式，默认视为视觉模型",
        model_id,
    )
    return True


class VLMClient:
    """异步 vLLM 推理客户端。

    通过 httpx.AsyncClient 调用本地 vLLM 的 OpenAI 兼容 chat/completions 接口。
    对于 VLM 模型发送图片 + 文本，对于纯文本模型仅发送文本提示词。
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
        self._is_vlm = is_vision_model(self._model)
        if not self._is_vlm:
            lg.warning(
                "模型 '{}' 不是视觉语言模型，将以纯文本模式调用（不发送图片）",
                self._model,
            )

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def describe_frame(self, keyframe: KeyFrame) -> FrameDescription:
        """对单个关键帧调用 vLLM，返回结构化描述。

        Args:
            keyframe: 包含 Base64 图像的关键帧。

        Returns:
            FrameDescription 包含帧 ID、时间戳和文本描述。
        """
        client = await self._ensure_client()

        if self._is_vlm:
            user_content: list[dict] | str = [
                {"type": "text", "text": self._prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{keyframe.base64_image}",
                    },
                },
            ]
        else:
            user_content = self._prompt

        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }

        lg.debug(
            "调用 vLLM 描述关键帧 #{} (timestamp={}ms, vlm={})",
            keyframe.frame_id, keyframe.timestamp_ms, self._is_vlm,
        )

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
        """关闭 HTTP 客户端连接。"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            lg.debug("vLLM 客户端已关闭")
