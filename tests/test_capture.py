"""pipeline.capture 模块的功能测试。

注意: 窗口发现 (find_window) 和实际截图 (capture_window) 依赖 Windows 桌面环境，
这里主要测试帧差计算、Base64 编码、图片缩放等纯计算逻辑。
窗口相关测试使用 mock。
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ahu_paimon_toolkit.capture import (
    WindowNotFoundError,
    compute_diff,
    find_window,
    frame_to_base64,
)
from ahu_paimon_toolkit.capture.frame_diff import compute_mse, compute_ssim


# ── 帧差计算测试 ──────────────────────────────────────────────────

class TestComputeMSE:
    def test_identical_frames_return_zero(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert compute_mse(frame, frame) == 0.0

    def test_different_frames_return_positive(self) -> None:
        frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 128, dtype=np.uint8)
        mse = compute_mse(frame_a, frame_b)
        assert mse > 0

    def test_mse_is_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        frame_a = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
        frame_b = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
        assert compute_mse(frame_a, frame_b) == pytest.approx(
            compute_mse(frame_b, frame_a)
        )

    def test_mse_known_value(self) -> None:
        """白色帧 vs 黑色帧，MSE = 255^2 = 65025。"""
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        assert compute_mse(black, white) == pytest.approx(65025.0)


class TestComputeSSIM:
    def test_identical_frames_return_zero_diff(self) -> None:
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        diff = compute_ssim(frame, frame)
        assert diff == pytest.approx(0.0, abs=1e-4)

    def test_different_frames_return_positive_diff(self) -> None:
        frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 255, dtype=np.uint8)
        diff = compute_ssim(frame_a, frame_b)
        assert diff > 0


class TestComputeDiff:
    def test_dispatch_mse(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = compute_diff(frame, frame, method="mse")
        assert result == 0.0

    def test_dispatch_ssim(self) -> None:
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = compute_diff(frame, frame, method="ssim")
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_invalid_method_raises(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown diff method"):
            compute_diff(frame, frame, method="invalid")


# ── 图片缩放测试 ──────────────────────────────────────────────────

@pytest.mark.skip(reason="_resize_frame was removed from capture module")
class TestResizeFrame:
    def test_no_resize_if_small_enough(self) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        result = _resize_frame(frame, max_size=512)  # noqa: F821
        assert result.shape == (200, 300, 3)

    def test_resize_landscape(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = _resize_frame(frame, max_size=512)  # noqa: F821
        assert result.shape[1] == 512
        assert result.shape[0] == 288

    def test_resize_portrait(self) -> None:
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        result = _resize_frame(frame, max_size=512)  # noqa: F821
        assert result.shape[0] == 512
        assert result.shape[1] == 288

    def test_resize_preserves_aspect_ratio(self) -> None:
        frame = np.zeros((768, 1024, 3), dtype=np.uint8)
        result = _resize_frame(frame, max_size=256)  # noqa: F821
        ratio_before = 1024 / 768
        ratio_after = result.shape[1] / result.shape[0]
        assert ratio_before == pytest.approx(ratio_after, abs=0.05)


# ── Base64 编码测试 ───────────────────────────────────────────────

class TestFrameToBase64:
    def test_returns_valid_base64(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        b64 = frame_to_base64(frame)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_decodable_as_jpeg(self) -> None:
        frame = np.full((50, 50, 3), 128, dtype=np.uint8)
        b64 = frame_to_base64(frame)
        decoded = base64.b64decode(b64)
        img_array = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape[0] == 50
        assert img.shape[1] == 50


# ── 窗口发现测试 (mock) ──────────────────────────────────────────

class TestFindWindow:
    @patch("ahu_paimon_toolkit.capture.window_capture._enum_windows")
    def test_finds_matching_window(self, mock_enum: MagicMock) -> None:
        mock_enum.return_value = [
            (100, "记事本 - 无标题"),
            (200, "PotPlayer - video.mp4"),
            (300, "Windows Terminal"),
        ]
        hwnd, title = find_window("PotPlayer")
        assert hwnd == 200
        assert "PotPlayer" in title

    @patch("ahu_paimon_toolkit.capture.window_capture._enum_windows")
    def test_case_insensitive(self, mock_enum: MagicMock) -> None:
        mock_enum.return_value = [(100, "POTPLAYER - video.mp4")]
        hwnd, title = find_window("potplayer")
        assert hwnd == 100

    @patch("ahu_paimon_toolkit.capture.window_capture._enum_windows")
    def test_prefers_shortest_title(self, mock_enum: MagicMock) -> None:
        mock_enum.return_value = [
            (100, "PotPlayer - 很长的视频名称.mp4 [1920x1080]"),
            (200, "PotPlayer"),
        ]
        hwnd, _ = find_window("PotPlayer")
        assert hwnd == 200

    @patch("ahu_paimon_toolkit.capture.window_capture._enum_windows")
    def test_raises_when_not_found(self, mock_enum: MagicMock) -> None:
        mock_enum.return_value = [
            (100, "记事本"),
            (200, "Chrome"),
        ]
        with pytest.raises(WindowNotFoundError):
            find_window("PotPlayer")
