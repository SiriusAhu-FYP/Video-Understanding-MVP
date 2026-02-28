"""Video Understanding MVP 主入口：编排整条流水线。

流程:
1. 加载配置 + 初始化日志
2. 启动截图线程 (capture) -> 关键帧入队
3. 启动 vLLM 消费者协程 -> 逐帧描述
4. recording_duration_s 到时 -> 停止截图
5. 等待队列排空 -> 调用 DeepSeek 汇总
6. 输出结果并退出
"""

from __future__ import annotations

import asyncio
import signal
import time

from loguru import logger as lg

from pipeline.capture import WindowNotFoundError, run_capture_loop
from pipeline.config import get_settings, setup_logging
from pipeline.models import FrameDescription
from pipeline.queue_manager import KeyFrameQueue
from pipeline.summarizer import Summarizer
from pipeline.vlm import VLMClient


async def consume_frames(
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


async def run_pipeline() -> None:
    """主流水线编排。"""
    cfg = get_settings()
    setup_logging(cfg.log)

    lg.info("=" * 60)
    lg.info("Video Understanding MVP 启动")
    lg.info("录制时长: {}s | 截图间隔: {}ms | 帧差算法: {} | 阈值: {}",
            cfg.capture.recording_duration_s,
            cfg.capture.screenshot_interval_ms,
            cfg.algorithm.method,
            cfg.algorithm.diff_threshold)
    lg.info("=" * 60)

    queue = KeyFrameQueue()
    vlm_client = VLMClient()
    summarizer = Summarizer()
    results: list[FrameDescription] = []

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    # 注册 Ctrl+C 信号处理（Windows 下 SIGINT）
    def _signal_handler() -> None:
        lg.warning("收到中断信号，正在优雅关闭...")
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
    except NotImplementedError:
        # Windows 下 add_signal_handler 不可用，使用 signal 模块替代
        signal.signal(signal.SIGINT, lambda *_: _signal_handler())

    # 启动截图线程
    lg.info("正在查找目标窗口...")
    try:
        capture_task = asyncio.ensure_future(
            asyncio.to_thread(
                run_capture_loop,
                queue.inner,
                loop,
                stop_event,
            )
        )
    except WindowNotFoundError as e:
        lg.error("窗口查找失败: {}", e)
        await vlm_client.close()
        await summarizer.close()
        return

    # 启动 vLLM 消费者
    consumer_task = asyncio.ensure_future(
        consume_frames(queue, vlm_client, results, stop_event)
    )

    # 定时器：到时间后通知停止
    async def _timer() -> None:
        duration = cfg.capture.recording_duration_s
        lg.info("录制计时器启动，{}s 后自动停止", duration)
        start = time.monotonic()
        while not stop_event.is_set():
            elapsed = time.monotonic() - start
            if elapsed >= duration:
                lg.info("录制时间已到 ({:.1f}s)，停止截图", elapsed)
                stop_event.set()
                break
            await asyncio.sleep(0.5)

    timer_task = asyncio.ensure_future(_timer())

    # 等待截图线程结束
    await capture_task
    lg.info("截图线程已退出")

    # 等待队列排空
    lg.info("等待队列中剩余帧处理完毕 (剩余: {})", queue.qsize)
    try:
        await asyncio.wait_for(consumer_task, timeout=120.0)
    except asyncio.TimeoutError:
        lg.warning("消费者超时，强制结束")
        consumer_task.cancel()

    timer_task.cancel()

    # 汇总阶段
    lg.info("=" * 60)
    lg.info("开始 DeepSeek 汇总 | 共 {} 个帧描述 | 队列丢弃帧数: {}",
            len(results), queue.dropped_count)

    duration_s = cfg.capture.recording_duration_s
    try:
        summary = await summarizer.summarize(results, duration_s=float(duration_s))
        lg.info("=" * 60)
        lg.info("视频总结:\n{}", summary.summary_text)
        lg.info("=" * 60)
        lg.info(
            "统计 | 关键帧: {} | 丢弃帧: {} | 录制时长: {:.1f}s",
            summary.total_keyframes, queue.dropped_count, summary.duration_s,
        )
    except Exception:
        lg.exception("DeepSeek 汇总失败")

    # 清理资源
    await vlm_client.close()
    await summarizer.close()
    lg.info("流水线结束，资源已释放")


def main() -> None:
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
