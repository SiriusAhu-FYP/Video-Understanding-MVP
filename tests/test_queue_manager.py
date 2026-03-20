"""pipeline.queue_manager 模块的功能测试。"""

from __future__ import annotations

import asyncio
import time

import pytest

from ahu_paimon_toolkit.models import KeyFrame
from ahu_paimon_toolkit.pipeline.queue_manager import KeyFrameQueue


def _make_keyframe(frame_id: int = 0, age_s: float = 0.0) -> KeyFrame:
    """创建测试用关键帧，age_s 指定其"已存活时间"。"""
    return KeyFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 500,
        base64_image="dGVzdA==",
        created_at=time.monotonic() - age_s,
    )


@pytest.mark.asyncio
class TestKeyFrameQueue:
    async def test_put_and_get(self) -> None:
        q = KeyFrameQueue(max_size=10, expiry_time_ms=5000)
        kf = _make_keyframe(frame_id=1)
        q.put_nowait(kf)
        result = await q.get()
        assert result is not None
        assert result.frame_id == 1

    async def test_fifo_order(self) -> None:
        q = KeyFrameQueue(max_size=10, expiry_time_ms=5000)
        for i in range(5):
            q.put_nowait(_make_keyframe(frame_id=i))

        ids = []
        for _ in range(5):
            item = await q.get()
            assert item is not None
            ids.append(item.frame_id)
            q.task_done()

        assert ids == [0, 1, 2, 3, 4]

    async def test_drops_expired_frames(self) -> None:
        """过期帧应被跳过，消费者拿到的是新鲜的帧。"""
        q = KeyFrameQueue(max_size=10, expiry_time_ms=1000)

        # 放入 3 个过期帧（年龄 2 秒 > 有效期 1 秒）
        for i in range(3):
            q.put_nowait(_make_keyframe(frame_id=i, age_s=2.0))

        # 放入 1 个新鲜帧
        fresh = _make_keyframe(frame_id=99, age_s=0.0)
        q.put_nowait(fresh)

        result = await q.get()
        assert result is not None
        assert result.frame_id == 99
        assert q.dropped_count == 3

    async def test_maxsize_enforced(self) -> None:
        q = KeyFrameQueue(max_size=2, expiry_time_ms=5000)
        q.put_nowait(_make_keyframe(frame_id=0))
        q.put_nowait(_make_keyframe(frame_id=1))

        with pytest.raises(asyncio.QueueFull):
            q.put_nowait(_make_keyframe(frame_id=2))

    async def test_qsize_tracking(self) -> None:
        q = KeyFrameQueue(max_size=10, expiry_time_ms=5000)
        assert q.qsize == 0

        q.put_nowait(_make_keyframe(frame_id=0))
        assert q.qsize == 1

        await q.get()
        assert q.qsize == 0

    async def test_empty(self) -> None:
        q = KeyFrameQueue(max_size=10, expiry_time_ms=5000)
        assert q.empty()

        q.put_nowait(_make_keyframe(frame_id=0))
        assert not q.empty()

    async def test_task_done_and_join(self) -> None:
        q = KeyFrameQueue(max_size=10, expiry_time_ms=5000)
        q.put_nowait(_make_keyframe(frame_id=0))

        item = await q.get()
        assert item is not None
        q.task_done()
        await asyncio.wait_for(q.join(), timeout=1.0)

    async def test_dropped_count_increments(self) -> None:
        q = KeyFrameQueue(max_size=10, expiry_time_ms=100)

        for i in range(5):
            q.put_nowait(_make_keyframe(frame_id=i, age_s=1.0))

        q.put_nowait(_make_keyframe(frame_id=99, age_s=0.0))

        await q.get()
        assert q.dropped_count == 5
