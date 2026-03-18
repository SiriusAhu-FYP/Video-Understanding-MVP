"""视频理解专项实验脚本。

使用完整流水线（截图 -> 帧差 -> vLLM -> DeepSeek）对视频进行理解测试。
通过 PotPlayer 播放视频并从窗口截图，测试不同模型的效果。

用法:
    uv run experiments/video_understanding/run_experiment.py
    uv run experiments/video_understanding/run_experiment.py --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean

import cv2
import win32gui
from loguru import logger as lg
from openai import OpenAI

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXPERIMENT_DIR.parent.parent
_REPORTS_DIR = _EXPERIMENT_DIR / "reports"

sys.path.insert(0, str(_PROJECT_ROOT))

from ahu_paimon_toolkit.vlm.model_utils import detect_model_from_url as detect_model, model_short_name  # noqa: E501
from experiments.utils.logging import setup_experiment_log
from ahu_paimon_toolkit.capture import (
    WindowNotFoundError,
    capture_window,
    compute_diff,
    find_window,
    frame_to_base64,
    run_capture_loop,
)
from ahu_paimon_toolkit.config import ToolkitSettings as Settings, setup_logging
from ahu_paimon_toolkit.models import (
    FrameDescription,
    FrameRecord,
    KeyFrame,
    PipelineResult,
)
from ahu_paimon_toolkit.pipeline.queue_manager import KeyFrameQueue
from ahu_paimon_toolkit.pipeline.summarizer import Summarizer
from ahu_paimon_toolkit.vlm.client import AsyncVLMClient as VLMClient

ASSETS_VIDEOS_DIR = _PROJECT_ROOT / "assets" / "videos"

_VIDEOS_DIR = ASSETS_VIDEOS_DIR


# ── 工具函数 ──────────────────────────────────────────────────────

def get_video_duration_s(video_path: Path) -> float:
    """用 OpenCV 获取视频时长（秒）。"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise ValueError(f"无法读取视频 FPS: {video_path}")
    return frame_count / fps


def _kill_all_players() -> None:
    """强制终止所有 PotPlayer 进程，确保干净状态。"""
    import psutil
    killed = 0
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info["name"] or "").lower()
            if "potplayer" in name:
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed:
        lg.info("已强制终止 {} 个 PotPlayer 进程", killed)
        time.sleep(1.0)


def launch_player(video_path: Path) -> subprocess.Popen:
    """启动播放器播放视频，返回进程句柄用于后续清理。"""
    import os
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
    player_exe = os.getenv("PLAYER_EXE_PATH", "")
    if not player_exe:
        raise RuntimeError(
            "PLAYER_EXE_PATH 未在 .env 中配置，无法启动播放器"
        )
    if not Path(player_exe).exists():
        raise FileNotFoundError(f"播放器不存在: {player_exe}")

    _kill_all_players()
    lg.info("启动播放器: {} -> {}", Path(player_exe).name, video_path.name)
    return subprocess.Popen([player_exe, str(video_path)])


def wait_for_window(keyword: str, timeout_s: float = 15.0) -> tuple[int, str]:
    """轮询等待窗口出现。"""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            return find_window(keyword)
        except WindowNotFoundError:
            time.sleep(0.5)
    raise WindowNotFoundError(f"等待 {timeout_s}s 后仍未找到窗口: {keyword}")


def close_player(player_proc: subprocess.Popen | None, keyword: str) -> None:
    """关闭播放器：先礼后兵，WM_CLOSE → terminate → kill。"""
    WM_CLOSE = 0x0010
    try:
        hwnd, title = find_window(keyword)
        win32gui.PostMessage(hwnd, WM_CLOSE, 0, 0)
        lg.info("已发送关闭信号: '{}'", title)
    except WindowNotFoundError:
        pass

    if player_proc is not None:
        try:
            player_proc.wait(timeout=3.0)
            lg.info("播放器进程已正常退出")
            return
        except subprocess.TimeoutExpired:
            lg.warning("播放器未响应 WM_CLOSE，强制终止")
            player_proc.kill()
            player_proc.wait(timeout=5.0)

    _kill_all_players()


# ── 单次运行 ──────────────────────────────────────────────────────

async def run_single_pipeline(
    video_path: Path,
    run_dir: Path,
    cfg: Settings,
) -> PipelineResult:
    """对单个视频执行一次完整流水线，保存所有帧和日志。

    Args:
        video_path: 视频文件路径。
        run_dir: 本次运行的输出目录（含 frames/、run_log.json 等）。
        cfg: 流水线配置。

    Returns:
        PipelineResult 包含所有帧记录和总结。
    """
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records: list[FrameRecord] = []

    # 帧采样回调：保存每帧图片 + 记录元数据
    def on_frame_sampled(
        frame_idx: int,
        timestamp_ms: int,
        frame_bgr: object,
        diff_value: float | None,
        is_keyframe: bool,
        reason: str,
    ) -> None:
        tag = "key" if is_keyframe else "skip"
        filename = f"frame_{frame_idx:04d}_{tag}.png"
        cv2.imwrite(str(frames_dir / filename), frame_bgr)

        record = FrameRecord(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            diff_value=diff_value,
            is_keyframe=is_keyframe,
            reason=reason,
            image_filename=filename,
        )
        frame_records.append(record)

    # 启动播放器（跟踪进程句柄用于可靠清理）
    player_proc = launch_player(video_path)
    time.sleep(2.0)

    # 等待窗口
    keyword = cfg.capture.window_title_keyword
    wait_for_window(keyword)

    # 运行流水线：使用 cfg 中的实际模型名（自动检测的，非 config.toml 默认值）
    import os
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")

    queue = KeyFrameQueue()
    vlm_client = VLMClient(base_url=cfg.llm.vllm_base_url, model=cfg.llm.vllm_model)
    summarizer = Summarizer(api_key=deepseek_key, api_base_url=deepseek_base)
    results: list[FrameDescription] = []
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    try:
        capture_task = asyncio.ensure_future(
            asyncio.to_thread(
                run_capture_loop,
                queue.inner,
                loop,
                stop_event,
                window_keyword=cfg.capture.window_title_keyword,
                interval_ms=cfg.capture.screenshot_interval_ms,
                max_size=cfg.capture.max_size,
                diff_method=cfg.algorithm.method,
                diff_threshold=cfg.algorithm.diff_threshold,
                on_frame_sampled=on_frame_sampled,
            )
        )

        # 消费者协程
        async def consume() -> None:
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
                    for rec in frame_records:
                        if rec.is_keyframe and rec.vlm_response is None:
                            ts_diff = abs(rec.timestamp_ms - desc.timestamp_ms)
                            if ts_diff < cfg.capture.screenshot_interval_ms:
                                rec.vlm_response = desc.description
                                break
                    lg.info("帧 #{} 描述完成 | 已描述 {} 帧", desc.frame_id, len(results))
                except Exception:
                    lg.exception("vLLM 推理失败，跳过帧 #{}", keyframe.frame_id)
                finally:
                    queue.task_done()

        consumer_task = asyncio.ensure_future(consume())

        # 定时器
        duration_s = cfg.capture.recording_duration_s
        async def timer() -> None:
            lg.info("录制计时器启动，{}s 后停止", duration_s)
            start = time.monotonic()
            while not stop_event.is_set():
                if time.monotonic() - start >= duration_s:
                    lg.info("录制时间已到，停止截图")
                    stop_event.set()
                    break
                await asyncio.sleep(0.5)

        timer_task = asyncio.ensure_future(timer())

        await capture_task
        try:
            await asyncio.wait_for(consumer_task, timeout=120.0)
        except asyncio.TimeoutError:
            lg.warning("消费者超时，强制结束")
            consumer_task.cancel()
        timer_task.cancel()

        # DeepSeek 汇总
        summary = None
        if results:
            try:
                summary = await summarizer.summarize(results, duration_s=float(duration_s))
                lg.info("DeepSeek 汇总完成")
            except Exception:
                lg.exception("DeepSeek 汇总失败")

        await vlm_client.close()
        await summarizer.close()
    finally:
        # 无论成功或异常，都确保播放器被彻底关闭
        close_player(player_proc, keyword)

    pipeline_result = PipelineResult(
        frame_records=frame_records,
        descriptions=results,
        summary=summary,
        total_sampled=len(frame_records),
        total_keyframes=sum(1 for r in frame_records if r.is_keyframe),
        total_dropped=queue.dropped_count,
        duration_s=float(duration_s),
    )

    # 保存结构化日志
    run_log = {
        "video": video_path.name,
        "model": cfg.llm.vllm_model,
        "config": {
            "screenshot_interval_ms": cfg.capture.screenshot_interval_ms,
            "max_size": cfg.capture.max_size,
            "recording_duration_s": cfg.capture.recording_duration_s,
            "algorithm": cfg.algorithm.method,
            "diff_threshold": cfg.algorithm.diff_threshold,
        },
        "stats": {
            "total_sampled": pipeline_result.total_sampled,
            "total_keyframes": pipeline_result.total_keyframes,
            "total_dropped": pipeline_result.total_dropped,
            "duration_s": pipeline_result.duration_s,
        },
        "frames": [r.model_dump() for r in frame_records],
        "summary": summary.summary_text if summary else None,
    }
    log_path = run_dir / "run_log.json"
    log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")

    if summary:
        (run_dir / "summary.txt").write_text(summary.summary_text, encoding="utf-8")

    return pipeline_result


# ── 报告生成 ──────────────────────────────────────────────────────

def generate_model_report(
    model_name: str,
    all_results: dict[str, list[PipelineResult]],
    report_dir: Path,
) -> None:
    """生成单模型实验报告。

    Args:
        model_name: 模型名称。
        all_results: {video_name: [PipelineResult per run]}
        report_dir: 报告输出目录。
    """
    lines: list[str] = [
        f"# 视频理解实验报告: {model_name}",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for video_name, run_results in all_results.items():
        lines.append(f"## 视频: {video_name}")
        lines.append("")
        lines.append("| 运行 | 总采样帧 | 关键帧 | 丢弃帧 | VLM 描述数 |")
        lines.append("|------|---------|--------|--------|-----------|")
        for i, result in enumerate(run_results, 1):
            lines.append(
                f"| run_{i} | {result.total_sampled} | {result.total_keyframes} "
                f"| {result.total_dropped} | {len(result.descriptions)} |"
            )

        lines.append("")
        for i, result in enumerate(run_results, 1):
            lines.append(f"### run_{i} DeepSeek 总结")
            if result.summary:
                lines.append(f"\n```\n{result.summary.summary_text}\n```\n")
            else:
                lines.append("\n*总结生成失败*\n")

            lines.append(f"**关键帧描述（前 5 帧）:**\n")
            for desc in result.descriptions[:5]:
                ts = desc.timestamp_ms / 1000.0
                lines.append(f"- [{ts:.1f}s] 帧#{desc.frame_id}: {desc.description[:200]}")
            lines.append("")

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    lg.info("报告已保存: {}", report_path)


# ── 主流程 ────────────────────────────────────────────────────────

async def run_experiment(
    num_runs: int = 3,
    output_dir: Path | None = None,
    base_url: str | None = None,
) -> dict[str, list[PipelineResult]]:
    """运行完整实验：对所有视频执行多次流水线。

    Args:
        output_dir: If provided, use this as the parent directory.
                    Report will be placed in output_dir/{model_short_name}/.
        base_url: Override vLLM base URL (e.g. for WSL2 IP connectivity).

    Returns:
        {video_name: [PipelineResult per run]}
    """
    effective_url = base_url or "http://localhost:8000/v1"

    # 自动检测模型
    model_id = detect_model(effective_url)
    short_name = model_short_name(model_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is not None:
        report_dir = output_dir / short_name
    else:
        report_dir = _REPORTS_DIR / f"{short_name}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    log_sink_id = setup_experiment_log(report_dir / "experiment.log")

    lg.info("=" * 60)
    lg.info("视频理解专项实验启动")
    lg.info("模型: {} | 重复次数: {}", model_id, num_runs)
    lg.info("=" * 60)

    # 发现视频文件
    videos = sorted(_VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        lg.error("未找到视频文件: {}", _VIDEOS_DIR)
        sys.exit(1)
    lg.info("视频文件: {}", [v.name for v in videos])

    # 加载基础配置并覆盖模型名 + base_url
    import tomllib
    config_path = _PROJECT_ROOT / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            base_cfg_dict = tomllib.load(f)
        cfg = Settings(**base_cfg_dict)
    else:
        cfg = Settings()

    cfg_dict = cfg.model_dump()
    cfg_dict["llm"]["vllm_model"] = model_id
    cfg_dict["llm"]["vllm_base_url"] = effective_url

    all_results: dict[str, list[PipelineResult]] = {}

    for video_path in videos:
        video_name = video_path.stem
        duration = get_video_duration_s(video_path)
        # 录制时长 = 视频时长 + 3s 缓冲
        rec_duration = int(duration) + 3

        lg.info("─" * 40)
        lg.info("视频: {} ({:.1f}s) | 录制时长: {}s", video_name, duration, rec_duration)

        run_results: list[PipelineResult] = []

        for run_idx in range(1, num_runs + 1):
            lg.info("[run {}/{}] {} 开始", run_idx, num_runs, video_name)

            run_dir = report_dir / f"{video_name}_run{run_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # 为每次运行创建独立的 Settings
            run_cfg_dict = cfg_dict.copy()
            run_cfg_dict["capture"] = {**cfg_dict["capture"], "recording_duration_s": rec_duration}
            run_cfg = Settings(**run_cfg_dict)

            try:
                result = await run_single_pipeline(video_path, run_dir, run_cfg)
                run_results.append(result)
                lg.info(
                    "[run {}/{}] {} 完成 | 采样={} 关键帧={} 描述={}",
                    run_idx, num_runs, video_name,
                    result.total_sampled, result.total_keyframes, len(result.descriptions),
                )
            except Exception:
                lg.exception("[run {}/{}] {} 失败", run_idx, num_runs, video_name)

            # 运行间等待，让资源释放
            time.sleep(3.0)

        all_results[video_name] = run_results

    generate_model_report(model_id, all_results, report_dir)

    lg.info("=" * 60)
    lg.info("实验完成 | 模型: {} | 报告: {}", model_id, report_dir)
    return all_results


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="视频理解专项实验")
    parser.add_argument("--runs", type=int, default=3, help="每视频重复次数 (默认 3)")
    args = parser.parse_args()

    asyncio.run(run_experiment(num_runs=args.runs))


if __name__ == "__main__":
    main()
