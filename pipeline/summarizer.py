"""DeepSeek 视频汇总模块：收集所有帧描述，调用云端 DeepSeek API 生成最终总结。"""

from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger as lg

from pipeline.config import get_settings
from pipeline.models import FrameDescription, VideoSummary

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "deepseek_prompt.md"


def load_deepseek_prompt() -> str:
    """从 prompts/deepseek_prompt.md 读取提示词。"""
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"DeepSeek 提示词文件不存在: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _build_frame_text(descriptions: list[FrameDescription]) -> str:
    """将时序帧描述列表拼接为结构化文本，供 DeepSeek 分析。"""
    lines: list[str] = []
    for desc in descriptions:
        ts_s = desc.timestamp_ms / 1000.0
        lines.append(f"[{ts_s:.1f}s] 帧#{desc.frame_id}: {desc.description}")
    return "\n".join(lines)


class Summarizer:
    """DeepSeek 汇总客户端。

    收集一段时间内所有 vLLM 返回的帧描述，打包发送给 DeepSeek API，
    获取最终的视频总结。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        prompt: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        secrets = get_settings().secrets
        self._api_key = api_key or secrets.deepseek_api_key
        self._base_url = base_url or secrets.deepseek_api_base_url
        self._prompt = prompt or load_deepseek_prompt()
        self._timeout = timeout
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

    async def summarize(
        self,
        descriptions: list[FrameDescription],
        duration_s: float,
    ) -> VideoSummary:
        """将所有帧描述发送给 DeepSeek，生成最终视频总结。"""
        if not descriptions:
            lg.warning("没有帧描述可供汇总")
            return VideoSummary(
                frame_descriptions=[],
                summary_text="无有效帧数据，无法生成总结。",
                total_keyframes=0,
                duration_s=duration_s,
            )

        frame_text = _build_frame_text(descriptions)
        user_message = (
            f"{self._prompt}\n\n"
            f"--- 帧描述数据 (共 {len(descriptions)} 帧，视频时长 {duration_s:.1f} 秒) ---\n"
            f"{frame_text}"
        )

        client = await self._ensure_client()

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": user_message},
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
        }

        lg.info("调用 DeepSeek 汇总 {} 个帧描述 (视频时长 {:.1f}s)", len(descriptions), duration_s)

        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        summary_text = data["choices"][0]["message"]["content"]

        lg.info("DeepSeek 汇总完成 ({} 字符)", len(summary_text))

        return VideoSummary(
            frame_descriptions=descriptions,
            summary_text=summary_text,
            total_keyframes=len(descriptions),
            duration_s=duration_s,
        )

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            lg.debug("DeepSeek 客户端已关闭")
