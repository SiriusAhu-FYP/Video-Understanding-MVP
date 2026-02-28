"""Pydantic 数据模型：在流水线各阶段之间流转的结构化数据。"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class KeyFrame(BaseModel):
    """经帧差过滤后保留的关键帧，携带 Base64 图像与生命周期时间戳。"""

    frame_id: int
    timestamp_ms: int = Field(description="相对于录制起点的时间戳 (ms)")
    base64_image: str = Field(description="JPEG Base64 编码字符串")
    created_at: float = Field(
        default_factory=time.monotonic,
        description="monotonic 时钟，用于队列过期判断",
    )


class FrameDescription(BaseModel):
    """vLLM 对单帧画面返回的结构化描述。"""

    frame_id: int
    timestamp_ms: int
    description: str = Field(description="模型对该帧的文本描述")


class VideoSummary(BaseModel):
    """DeepSeek 汇总后生成的最终视频总结。"""

    frame_descriptions: list[FrameDescription]
    summary_text: str
    total_keyframes: int
    duration_s: float
