"""Video Understanding MVP 主入口。

使用 PipelineRunner 编排完整流水线：
截图捕获 -> 帧差过滤 -> vLLM 描述 -> DeepSeek 汇总。
"""

from __future__ import annotations

import asyncio

from loguru import logger as lg

from pipeline.runner import PipelineRunner


async def run_pipeline() -> None:
    """运行完整视频理解流水线。"""
    runner = PipelineRunner()
    result = await runner.run()

    lg.info("=" * 60)
    if result.summary:
        lg.info("视频总结:\n{}", result.summary.summary_text)
    lg.info(
        "统计 | 关键帧: {} | 丢弃帧: {} | 录制时长: {:.1f}s",
        result.total_keyframes, result.total_dropped, result.duration_s,
    )
    lg.info("=" * 60)


def main() -> None:
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
