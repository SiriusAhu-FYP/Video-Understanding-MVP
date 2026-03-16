"""画面捕获与帧差过滤模块。

职责:
1. 通过 win32gui 模糊匹配目标窗口
2. 使用 mss 对窗口区域极速截图
3. 用 OpenCV 计算当前帧与上一关键帧的差异 (MSE / SSIM)
4. 超过阈值的帧压缩、编码为 Base64 并返回
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import TYPE_CHECKING

import cv2
import mss
import numpy as np
import win32gui
from loguru import logger as lg

from pipeline.config import get_settings
from pipeline.models import KeyFrame, OnFrameSampledCallback

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ── 窗口发现 ──────────────────────────────────────────────────────

def _enum_windows() -> list[tuple[int, str]]:
    """枚举所有可见窗口，返回 (hwnd, title) 列表。"""
    results: list[tuple[int, str]] = []

    def callback(hwnd: int, _: object) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                results.append((hwnd, title))
        return True

    win32gui.EnumWindows(callback, None)
    return results


def find_window(keyword: str) -> tuple[int, str]:
    """根据关键词模糊匹配窗口标题，返回最佳匹配的 (hwnd, title)。

    匹配策略：优先选择标题中包含 keyword 且长度最短的窗口（最精确匹配）。
    """
    windows = _enum_windows()
    keyword_lower = keyword.lower()

    candidates = [
        (hwnd, title)
        for hwnd, title in windows
        if keyword_lower in title.lower()
    ]

    if not candidates:
        raise WindowNotFoundError(
            f"未找到标题包含 '{keyword}' 的窗口。"
            f"当前可见窗口: {[t for _, t in windows[:10]]}"
        )

    # 标题越短说明匹配越精确
    best = min(candidates, key=lambda x: len(x[1]))
    lg.info("窗口匹配成功: hwnd={}, title='{}'", best[0], best[1])
    return best


class WindowNotFoundError(Exception):
    pass


# ── 窗口截图 ──────────────────────────────────────────────────────

def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """获取窗口的屏幕坐标 (left, top, right, bottom)。"""
    rect = win32gui.GetWindowRect(hwnd)
    return rect


def capture_window(hwnd: int, max_size: int) -> NDArray[np.uint8]:
    """对指定窗口截图并缩放到 max_size 长边上限，返回 BGR numpy 数组。"""
    left, top, right, bottom = get_window_rect(hwnd)
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        raise ValueError(f"窗口尺寸无效: {width}x{height}")

    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        # mss 返回 BGRA，转为 BGR
        frame = np.array(screenshot, dtype=np.uint8)[:, :, :3]

    frame = _resize_frame(frame, max_size)
    return frame


def _resize_frame(frame: NDArray[np.uint8], max_size: int) -> NDArray[np.uint8]:
    """等比缩放，使长边不超过 max_size。如果已经小于则不缩放。"""
    h, w = frame.shape[:2]
    long_edge = max(h, w)

    if long_edge <= max_size:
        return frame

    scale = max_size / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ── 帧差计算 ──────────────────────────────────────────────────────

def compute_mse(frame_a: NDArray[np.uint8], frame_b: NDArray[np.uint8]) -> float:
    """计算两帧之间的均方误差 (Mean Squared Error)。

    值越大说明差异越大。完全相同的两帧 MSE = 0。
    """
    diff = frame_a.astype(np.float64) - frame_b.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_ssim(frame_a: NDArray[np.uint8], frame_b: NDArray[np.uint8]) -> float:
    """计算两帧之间的结构相似性指数 (SSIM)。

    范围 [0, 1]，值越小差异越大。完全相同的两帧 SSIM = 1。
    为了与 MSE 保持"越大 = 差异越大"的语义一致，返回 1 - SSIM。
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(gray_a.astype(np.float64), (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b.astype(np.float64), (11, 11), 1.5)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a.astype(np.float64) ** 2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b.astype(np.float64) ** 2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(
        gray_a.astype(np.float64) * gray_b.astype(np.float64), (11, 11), 1.5
    ) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / denominator

    # 返回 1 - SSIM，使得"值越大 = 差异越大"
    return float(1.0 - np.mean(ssim_map))


def compute_diff(
    frame_a: NDArray[np.uint8],
    frame_b: NDArray[np.uint8],
    method: str = "mse",
) -> float:
    """根据配置的算法计算帧差异值。"""
    if frame_a.shape != frame_b.shape:
        h, w = frame_a.shape[:2]
        frame_b = cv2.resize(frame_b, (w, h))
    if method == "mse":
        return compute_mse(frame_a, frame_b)
    elif method == "ssim":
        return compute_ssim(frame_a, frame_b)
    else:
        raise ValueError(f"未知的帧差算法: {method}，支持 'mse' 或 'ssim'")


# ── Base64 编码 ───────────────────────────────────────────────────

def frame_to_base64(frame: NDArray[np.uint8], quality: int = 85) -> str:
    """将 BGR numpy 数组编码为 JPEG Base64 字符串。"""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


# ── 截图循环 (同步，运行在线程中) ──────────────────────────────────

def run_capture_loop(
    queue: asyncio.Queue[KeyFrame],
    loop: asyncio.AbstractEventLoop,
    stop_event: asyncio.Event,
    on_frame_sampled: OnFrameSampledCallback | None = None,
) -> None:
    """截图主循环：在独立线程中运行，将关键帧推入异步队列。

    Args:
        queue: 异步队列，用于向消费者传递关键帧。
        loop: 主事件循环引用，用于线程安全地操作异步队列。
        stop_event: 外部通知停止的信号。
        on_frame_sampled: 可选回调，每次采样时调用（无论是否为关键帧），
            签名: (frame_idx, timestamp_ms, frame_bgr, diff_value, is_keyframe, reason)。
    """
    cfg = get_settings()
    interval_s = cfg.capture.screenshot_interval_ms / 1000.0
    method = cfg.algorithm.method
    threshold = cfg.algorithm.diff_threshold

    hwnd, title = find_window(cfg.capture.window_title_keyword)
    lg.info(
        "截图循环启动 | 窗口='{}' | 间隔={}ms | 算法={} | 阈值={}",
        title, cfg.capture.screenshot_interval_ms, method, threshold,
    )

    last_keyframe: NDArray[np.uint8] | None = None
    frame_id = 0
    sample_idx = 0
    start_time = time.monotonic()

    while not stop_event.is_set():
        iter_start = time.perf_counter()

        try:
            frame = capture_window(hwnd, cfg.capture.max_size)
        except Exception:
            lg.warning("截图失败，窗口可能已关闭或最小化，等待重试...")
            time.sleep(interval_s)
            continue

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if last_keyframe is not None:
            diff = compute_diff(last_keyframe, frame, method)
            if diff < threshold:
                reason = f"差异值 {diff:.2f} < 阈值 {threshold:.2f}"
                if on_frame_sampled is not None:
                    on_frame_sampled(sample_idx, elapsed_ms, frame, diff, False, reason)
                sample_idx += 1
                elapsed = time.perf_counter() - iter_start
                sleep_time = max(0, interval_s - elapsed)
                time.sleep(sleep_time)
                continue
            reason = f"差异值 {diff:.2f} >= 阈值 {threshold:.2f}"
            lg.debug("帧差异值: {:.2f} (超过阈值 {:.2f})，提取为关键帧 #{}", diff, threshold, frame_id)
        else:
            diff = None
            reason = "首帧，自动标记为关键帧"

        # 当前帧确认为关键帧
        if on_frame_sampled is not None:
            on_frame_sampled(sample_idx, elapsed_ms, frame, diff, True, reason)
        sample_idx += 1

        last_keyframe = frame.copy()
        b64 = frame_to_base64(frame)

        keyframe = KeyFrame(
            frame_id=frame_id,
            timestamp_ms=elapsed_ms,
            base64_image=b64,
        )
        frame_id += 1

        future = asyncio.run_coroutine_threadsafe(
            _safe_put(queue, keyframe), loop
        )
        try:
            future.result(timeout=2.0)
        except Exception:
            lg.warning("队列已满，丢弃关键帧 #{}", keyframe.frame_id)

        elapsed = time.perf_counter() - iter_start
        sleep_time = max(0, interval_s - elapsed)
        time.sleep(sleep_time)

    lg.info("截图循环结束 | 共采样 {} 帧，提取 {} 个关键帧", sample_idx, frame_id)


async def _safe_put(queue: asyncio.Queue[KeyFrame], item: KeyFrame) -> None:
    """非阻塞入队，队列满时立即放弃。"""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        raise
