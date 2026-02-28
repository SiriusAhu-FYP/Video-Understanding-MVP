"""带过期淘汰机制的异步队列管理器。

核心原则: 宁可丢帧也绝不处理陈旧数据。
消费者从队列取帧时，自动检测并丢弃超过 expiry_time_ms 的过期帧。
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger as lg

from pipeline.config import get_settings
from pipeline.models import KeyFrame


class KeyFrameQueue:
    """封装 asyncio.Queue，增加过期淘汰逻辑。

    参数:
        max_size: 队列最大容量，防止内存溢出
        expiry_time_ms: 关键帧的有效期 (毫秒)，超过则丢弃
    """

    def __init__(self, max_size: int | None = None, expiry_time_ms: int | None = None) -> None:
        cfg = get_settings().queue
        self._max_size = max_size if max_size is not None else cfg.max_size
        self._expiry_time_ms = expiry_time_ms if expiry_time_ms is not None else cfg.expiry_time_ms
        self._queue: asyncio.Queue[KeyFrame] = asyncio.Queue(maxsize=self._max_size)
        self._dropped_count = 0

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def inner(self) -> asyncio.Queue[KeyFrame]:
        """暴露底层队列，供 capture 线程通过 run_coroutine_threadsafe 入队。"""
        return self._queue

    def put_nowait(self, item: KeyFrame) -> None:
        """非阻塞入队，队列满时抛出 asyncio.QueueFull。"""
        self._queue.put_nowait(item)

    async def put(self, item: KeyFrame) -> None:
        """阻塞入队。"""
        await self._queue.put(item)

    async def get(self) -> KeyFrame | None:
        """从队列取出一个关键帧，自动跳过已过期的帧。

        返回 None 仅在 get 被取消时发生（正常情况下会阻塞等待）。
        过期帧被丢弃时会记录日志。
        """
        while True:
            item = await self._queue.get()

            age_ms = (time.monotonic() - item.created_at) * 1000
            if age_ms > self._expiry_time_ms:
                self._dropped_count += 1
                lg.warning(
                    "队列积压，丢弃过期关键帧 #{} | 年龄={:.0f}ms > 有效期={}ms | 累计丢弃: {}",
                    item.frame_id, age_ms, self._expiry_time_ms, self._dropped_count,
                )
                self._queue.task_done()
                continue

            lg.debug(
                "取出关键帧 #{} | 年龄={:.0f}ms | 队列剩余: {}",
                item.frame_id, age_ms, self._queue.qsize(),
            )
            return item

    def task_done(self) -> None:
        """通知队列当前任务已处理完毕。"""
        self._queue.task_done()

    async def join(self) -> None:
        """等待队列中所有任务被处理完毕。"""
        await self._queue.join()

    def empty(self) -> bool:
        return self._queue.empty()
