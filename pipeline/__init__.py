"""视频理解流水线核心模块。

提供帧捕获、帧差过滤、VLM 推理、DeepSeek 汇总等功能，
以及可复用的 PipelineRunner 编排器。
"""

from pipeline.capture import (
    WindowNotFoundError,
    capture_window,
    compute_diff,
    find_window,
    frame_to_base64,
    run_capture_loop,
)
from pipeline.config import Settings, get_settings, setup_logging
from pipeline.models import (
    FrameDescription,
    FrameRecord,
    KeyFrame,
    OnFrameSampledCallback,
    PipelineResult,
    VideoSummary,
)
from pipeline.queue_manager import KeyFrameQueue
from pipeline.runner import PipelineRunner
from pipeline.summarizer import Summarizer
from pipeline.vlm import VLMClient

__all__ = [
    "WindowNotFoundError",
    "capture_window",
    "compute_diff",
    "find_window",
    "frame_to_base64",
    "run_capture_loop",
    "Settings",
    "get_settings",
    "setup_logging",
    "FrameDescription",
    "FrameRecord",
    "KeyFrame",
    "OnFrameSampledCallback",
    "PipelineResult",
    "VideoSummary",
    "KeyFrameQueue",
    "PipelineRunner",
    "Summarizer",
    "VLMClient",
]
