"""Pydantic 数据模型：在流水线各阶段之间流转的结构化数据。"""

from __future__ import annotations

import time
from typing import Callable

from numpy.typing import NDArray
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


class FrameRecord(BaseModel):
    """单次采样帧的完整记录，包含是否为关键帧及原因。"""

    model_config = {"arbitrary_types_allowed": True}

    frame_idx: int
    timestamp_ms: int
    diff_value: float | None = Field(default=None, description="与上一关键帧的差异值")
    is_keyframe: bool = False
    reason: str = Field(default="", description="关键帧判定原因")
    image_filename: str | None = Field(default=None, description="保存的图片文件名")
    vlm_response: str | None = Field(default=None, description="VLM 返回的描述（仅关键帧）")


class PipelineResult(BaseModel):
    """流水线完整运行结果。"""

    frame_records: list[FrameRecord] = Field(default_factory=list)
    descriptions: list[FrameDescription] = Field(default_factory=list)
    summary: VideoSummary | None = None
    total_sampled: int = 0
    total_keyframes: int = 0
    total_dropped: int = 0
    duration_s: float = 0.0


# 帧采样回调函数签名：
# (frame_idx, timestamp_ms, frame_bgr, diff_value, is_keyframe, reason)
OnFrameSampledCallback = Callable[
    [int, int, "NDArray", float | None, bool, str], None
]
