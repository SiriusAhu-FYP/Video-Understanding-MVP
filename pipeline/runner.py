"""可复用的视频理解流水线编排器。

将截图捕获、帧差过滤、VLM 推理、DeepSeek 汇总等步骤编排为一个
可被 main.py 和实验脚本共同调用的 ``PipelineRunner`` 类。
"""

from __future__ import annotations

import asyncio
import signal
import time

from loguru import logger as lg

from pipeline.capture import WindowNotFoundError, run_capture_loop
from pipeline.config import Settings, get_settings, setup_logging
from pipeline.models import (
    FrameDescription,
    OnFrameSampledCallback,
    PipelineResult,
)
from pipeline.queue_manager import KeyFrameQueue
from pipeline.summarizer import Summarizer
from pipeline.vlm import VLMClient


class PipelineRunner:
    """可复用的视频理解流水线编排器。

    Attributes:
        cfg: 全局配置。可传入自定义 Settings 覆盖默认值。
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or get_settings()

    async def run(
        self,
        stop_event: asyncio.Event | None = None,
        on_frame_sampled: OnFrameSampledCallback | None = None,
        skip_summary: bool = False,
    ) -> PipelineResult:
        """运行完整流水线：截图 -> 帧差 -> vLLM -> DeepSeek 汇总。

        Args:
            stop_event: 外部停止信号。为 ``None`` 时内部自动创建。
            on_frame_sampled: 每次采样帧时的回调，用于保存帧图片和元数据。
                签名: ``(frame_idx, timestamp_ms, frame_bgr, diff_value, is_keyframe, reason)``
            skip_summary: 若为 ``True`` 则跳过 DeepSeek 汇总步骤。

        Returns:
            PipelineResult 包含所有帧记录、描述和最终总结。
        """
        cfg = self.cfg
        setup_logging(cfg.log)

        lg.info("=" * 60)
        lg.info("Video Understanding Pipeline 启动")
        lg.info(
            "录制时长: {}s | 截图间隔: {}ms | 帧差算法: {} | 阈值: {}",
            cfg.capture.recording_duration_s,
            cfg.capture.screenshot_interval_ms,
            cfg.algorithm.method,
            cfg.algorithm.diff_threshold,
        )
        lg.info("=" * 60)

        if stop_event is None:
            stop_event = asyncio.Event()

        queue = KeyFrameQueue()
        vlm_client = VLMClient()
        summarizer = Summarizer()
        results: list[FrameDescription] = []

        loop = asyncio.get_running_loop()
        self._register_signal(stop_event, loop)

        # 启动截图线程
        lg.info("正在查找目标窗口...")
        try:
            capture_task = asyncio.ensure_future(
                asyncio.to_thread(
                    run_capture_loop,
                    queue.inner,
                    loop,
                    stop_event,
                    on_frame_sampled,
                )
            )
        except WindowNotFoundError as e:
            lg.error("窗口查找失败: {}", e)
            await vlm_client.close()
            await summarizer.close()
            return PipelineResult()

        consumer_task = asyncio.ensure_future(
            self._consume_frames(queue, vlm_client, results, stop_event)
        )

        timer_task = asyncio.ensure_future(
            self._timer(cfg.capture.recording_duration_s, stop_event)
        )

        await capture_task
        lg.info("截图线程已退出")

        lg.info("等待队列中剩余帧处理完毕 (剩余: {})", queue.qsize)
        try:
            await asyncio.wait_for(consumer_task, timeout=120.0)
        except asyncio.TimeoutError:
            lg.warning("消费者超时，强制结束")
            consumer_task.cancel()

        timer_task.cancel()

        # 汇总阶段
        duration_s = float(cfg.capture.recording_duration_s)
        summary = None

        if not skip_summary and results:
            lg.info(
                "开始 DeepSeek 汇总 | 共 {} 个帧描述 | 队列丢弃帧数: {}",
                len(results), queue.dropped_count,
            )
            try:
                summary = await summarizer.summarize(results, duration_s=duration_s)
                lg.info("视频总结:\n{}", summary.summary_text)
            except Exception:
                lg.exception("DeepSeek 汇总失败")
        elif not results:
            lg.warning("没有帧描述可供汇总")

        await vlm_client.close()
        await summarizer.close()
        lg.info("流水线结束，资源已释放")

        return PipelineResult(
            descriptions=results,
            summary=summary,
            total_sampled=0,
            total_keyframes=len(results),
            total_dropped=queue.dropped_count,
            duration_s=duration_s,
        )

    @staticmethod
    async def _consume_frames(
        queue: KeyFrameQueue,
        vlm_client: VLMClient,
        results: list[FrameDescription],
        stop_event: asyncio.Event,
    ) -> None:
        """vLLM 消费者协程：持续从队列取关键帧并调用 vLLM 获取描述。"""
        while True:
            if stop_event.is_set() and queue.empty():
                break
            try:
                keyframe = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if keyframe is None:
                continue

            try:
                desc = await vlm_client.describe_frame(keyframe)
                results.append(desc)
                lg.info(
                    "帧 #{} 描述完成 | 当前已描述 {} 帧",
                    desc.frame_id, len(results),
                )
            except Exception:
                lg.exception("vLLM 推理失败，跳过帧 #{}", keyframe.frame_id)
            finally:
                queue.task_done()

    @staticmethod
    async def _timer(duration_s: int, stop_event: asyncio.Event) -> None:
        """录制定时器，到时间后通知停止。"""
        lg.info("录制计时器启动，{}s 后自动停止", duration_s)
        start = time.monotonic()
        while not stop_event.is_set():
            elapsed = time.monotonic() - start
            if elapsed >= duration_s:
                lg.info("录制时间已到 ({:.1f}s)，停止截图", elapsed)
                stop_event.set()
                break
            await asyncio.sleep(0.5)

    @staticmethod
    def _register_signal(stop_event: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None:
        """注册 Ctrl+C 信号处理。"""
        def handler() -> None:
            lg.warning("收到中断信号，正在优雅关闭...")
            stop_event.set()

        try:
            loop.add_signal_handler(signal.SIGINT, handler)
        except NotImplementedError:
            signal.signal(signal.SIGINT, lambda *_: handler())
