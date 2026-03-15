"""Multi-model comprehensive benchmark orchestrator.

Reads config_benchmark.toml, iterates through models, manages vLLM services
via WSL, runs benchmark_speed and video_understanding experiments, and
generates comparison reports.

Usage:
    uv run run_benchmark.py
    uv run run_benchmark.py --config my_config.toml
    uv run run_benchmark.py --models "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-0.8B"
    uv run run_benchmark.py --skip-speed   # only run video_understanding
    uv run run_benchmark.py --skip-video   # only run benchmark_speed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import tomllib
from loguru import logger as lg

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

from toolkit.common import (
    get_gpu_info,
    model_short_name,
    wait_for_vllm_ready,
)
from toolkit.vllm_manager import (
    create_launch_script,
    start_vllm,
    stop_vllm,
)


def load_config(config_path: Path) -> dict:
    """Load and return the benchmark config from a TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def create_session_dir(config: dict) -> Path:
    """Create the timestamped session directory and save metadata."""
    ts_format = config.get("general", {}).get("timestamp_format", "%Y-%m-%d_%H-%M-%S")
    timestamp = datetime.now().strftime(ts_format)
    session_dir = PROJECT_ROOT / "experiments" / "multi_model_benchmark" / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_meta(session_dir: Path, config: dict, config_path: Path) -> None:
    """Save config snapshot and environment metadata."""
    shutil.copy2(config_path, session_dir / "config.toml")

    meta = {
        "git_hash": get_git_hash(),
        "gpu": get_gpu_info(),
        "python_version": sys.version,
        "start_time": datetime.now().isoformat(),
        "models": config.get("models", {}).get("list", []),
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_benchmark_speed_for_model(
    base_url: str,
    output_dir: Path,
    config: dict,
) -> bool:
    """Run benchmark_speed experiment for the currently loaded model."""
    bs_cfg = config.get("benchmark_speed", {})
    if not bs_cfg.get("enabled", True):
        lg.info("benchmark_speed 已禁用，跳过")
        return True

    try:
        from experiments.benchmark_speed.benchmark import run_benchmark

        scenario_ids = None
        scenarios_val = bs_cfg.get("scenarios", "all")
        if isinstance(scenarios_val, list):
            scenario_ids = scenarios_val

        results, report_dir = run_benchmark(
            base_url=base_url,
            num_runs=bs_cfg.get("runs", 3),
            warmup_runs=bs_cfg.get("warmup", 1),
            scenario_ids=scenario_ids,
            output_dir=output_dir,
        )
        lg.info("benchmark_speed 完成: {} 条结果 -> {}", len(results), report_dir)
        return True
    except Exception:
        lg.exception("benchmark_speed 执行失败")
        return False


def run_video_understanding_for_model(
    output_dir: Path,
    config: dict,
) -> bool:
    """Run video_understanding experiment for the currently loaded model."""
    vu_cfg = config.get("video_understanding", {})
    if not vu_cfg.get("enabled", True):
        lg.info("video_understanding 已禁用，跳过")
        return True

    try:
        from experiments.video_understanding.run_experiment import run_experiment

        results = asyncio.run(
            run_experiment(
                num_runs=vu_cfg.get("runs", 3),
                output_dir=output_dir,
            )
        )
        total_runs = sum(len(v) for v in results.values())
        lg.info("video_understanding 完成: {} 个视频, {} 次运行", len(results), total_runs)
        return True
    except Exception:
        lg.exception("video_understanding 执行失败")
        return False


def generate_speed_comparison(session_dir: Path, models: list[str]) -> None:
    """Generate a comparison report across models for benchmark_speed."""
    import csv

    speed_dir = session_dir / "benchmark_speed"
    if not speed_dir.exists():
        return

    lines = [
        "# Benchmark Speed - 多模型对比报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 模型性能总览",
        "",
        "| 模型 | 平均 TTFT (s) | 平均吞吐量 (tok/s) | 平均总耗时 (s) | 平均输出 Tokens | 测试次数 |",
        "|------|---------------|--------------------:|---------------:|----------------:|---------:|",
    ]

    for model_id in models:
        short = model_short_name(model_id)
        csv_path = speed_dir / short / "raw_data.csv"
        if not csv_path.exists():
            lines.append(f"| {model_id} | - | - | - | - | 0 |")
            continue

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            lines.append(f"| {model_id} | - | - | - | - | 0 |")
            continue

        ttfts = [float(r["ttft_s"]) for r in rows]
        tps = [float(r["throughput_tps"]) for r in rows]
        totals = [float(r["total_time_s"]) for r in rows]
        tokens = [float(r["output_tokens"]) for r in rows]

        lines.append(
            f"| {model_id} | {mean(ttfts):.3f} | {mean(tps):.1f} "
            f"| {mean(totals):.3f} | {mean(tokens):.0f} | {len(rows)} |"
        )

    lines.append("")
    lines.append("## 理论帧率分析")
    lines.append("")
    lines.append("基于平均 TTFT（首 token 延迟），各模型理论最大支持帧率：")
    lines.append("")
    lines.append("| 模型 | 平均 TTFT (s) | 理论最大 FPS | 安全范围 FPS (70%) |")
    lines.append("|------|---------------|-------------:|-------------------:|")

    for model_id in models:
        short = model_short_name(model_id)
        csv_path = speed_dir / short / "raw_data.csv"
        if not csv_path.exists():
            continue

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        avg_ttft = mean(float(r["ttft_s"]) for r in rows)
        max_fps = 1.0 / avg_ttft if avg_ttft > 0 else 0
        safe_fps = max_fps * 0.7

        lines.append(
            f"| {model_id} | {avg_ttft:.3f} | {max_fps:.1f} | {safe_fps:.1f} |"
        )

    lines.append("")
    lines.append("> **注意**: 理论最大 FPS = 1 / TTFT。安全范围取 70% 以留出处理余量。")
    lines.append("> 实际部署时还需考虑帧采样策略、网络延迟和后处理开销。")

    report_path = speed_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    lg.info("benchmark_speed 对比报告: {}", report_path)


def generate_video_comparison(session_dir: Path, models: list[str]) -> None:
    """Generate a comparison report across models for video_understanding."""
    vu_dir = session_dir / "video_understanding"
    if not vu_dir.exists():
        return

    lines = [
        "# Video Understanding - 多模型对比报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 模型表现总览",
        "",
        "| 模型 | 视频数 | 总运行数 | 平均关键帧数 | 平均描述数 | 平均丢弃帧 |",
        "|------|-------:|---------:|------------:|-----------:|-----------:|",
    ]

    for model_id in models:
        short = model_short_name(model_id)
        model_dir = vu_dir / short
        if not model_dir.exists():
            lines.append(f"| {model_id} | 0 | 0 | - | - | - |")
            continue

        video_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        total_runs = 0
        keyframes_list = []
        descs_list = []
        dropped_list = []

        for vdir in video_dirs:
            run_dirs = [r for r in vdir.iterdir() if r.is_dir() and r.name.startswith("run")]
            for rdir in run_dirs:
                total_runs += 1
                log_path = rdir / "run_log.json"
                if log_path.exists():
                    try:
                        log_data = json.loads(log_path.read_text(encoding="utf-8"))
                        stats = log_data.get("stats", {})
                        keyframes_list.append(stats.get("total_keyframes", 0))
                        descs_list.append(
                            len([f for f in log_data.get("frames", []) if f.get("vlm_response")])
                        )
                        dropped_list.append(stats.get("total_dropped", 0))
                    except Exception:
                        pass

        avg_kf = mean(keyframes_list) if keyframes_list else 0
        avg_desc = mean(descs_list) if descs_list else 0
        avg_drop = mean(dropped_list) if dropped_list else 0

        lines.append(
            f"| {model_id} | {len(video_dirs)} | {total_runs} "
            f"| {avg_kf:.1f} | {avg_desc:.1f} | {avg_drop:.1f} |"
        )

    lines.append("")
    lines.append("## 帧处理延迟分析")
    lines.append("")
    lines.append("未来得及处理的帧被丢弃（KeyFrameQueue expiry 机制），不会导致")
    lines.append("推理进程阻塞。但高丢弃率意味着模型处理速度无法跟上帧采样速度，")
    lines.append("建议降低采样频率或使用更快的模型。")

    report_path = vu_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    lg.info("video_understanding 对比报告: {}", report_path)


def generate_final_report(session_dir: Path, config: dict, models: list[str]) -> None:
    """Generate the comprehensive multi-model benchmark report."""
    lines = [
        "# 多模型综合测试报告 (Multi-Model Comprehensive Benchmark)",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gpu_info = meta.get("gpu", {})
        lines.extend([
            "## 实验环境",
            "",
            f"- **GPU**: {gpu_info.get('name', 'N/A')} ({gpu_info.get('memory_total_mb', 0)} MB)",
            f"- **Python**: {meta.get('python_version', 'N/A').split()[0]}",
            f"- **Git**: {meta.get('git_hash', 'N/A')}",
            f"- **开始时间**: {meta.get('start_time', 'N/A')}",
            "",
        ])

    lines.extend([
        "## 模型清单",
        "",
        "| # | 模型 ID | 简称 | 类型 |",
        "|---|---------|------|------|",
    ])
    for i, model_id in enumerate(models, 1):
        short = model_short_name(model_id)
        lines.append(f"| {i} | {model_id} | {short} | VLM |")

    lines.extend([
        "",
        "> **注意**: 本次测试未涉及社区制作的量化模型（如 unsloth GGUF 等）。",
        "> 后续可考虑纳入 GGUF/AWQ/GPTQ 等量化版本进行对比。",
        "",
    ])

    speed_comparison = session_dir / "benchmark_speed" / "comparison_report.md"
    if speed_comparison.exists():
        lines.extend([
            "## Benchmark Speed 结果",
            "",
            f"详见 [comparison_report.md](benchmark_speed/comparison_report.md)",
            "",
        ])

    vu_comparison = session_dir / "video_understanding" / "comparison_report.md"
    if vu_comparison.exists():
        lines.extend([
            "## Video Understanding 结果",
            "",
            f"详见 [comparison_report.md](video_understanding/comparison_report.md)",
            "",
        ])

    lines.extend([
        "## 研究问题",
        "",
        "### 理论最快支持每秒多少帧？",
        "",
        "取决于模型的 TTFT（Time To First Token）。理论最大 FPS = 1 / TTFT。",
        "详见 benchmark_speed 对比报告中的帧率分析表。",
        "",
        "### 安全范围大概是每秒截取多少帧？",
        "",
        "建议取理论最大 FPS 的 70% 作为安全上限。",
        "这为网络延迟、帧差计算、队列管理等开销预留了 30% 的缓冲。",
        "",
        "### 未来得及处理的帧如何处理？",
        "",
        "当前实现使用 KeyFrameQueue 的过期淘汰机制：",
        "- 超过 `expiry_time_ms`（默认 10s）未被消费的帧会被自动丢弃",
        "- 丢弃的帧不会阻塞推理进程",
        "- 但高丢弃率意味着信息遗漏，可能导致对游戏状态的理解不连贯",
        "- 建议方案：降低采样频率 + 使用更快的模型 + 优化帧差阈值",
        "",
        "## 综合建议",
        "",
        "*（待实验完成后根据数据填写）*",
    ])

    report_path = session_dir / "final_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    lg.info("综合报告: {}", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="多模型综合测试")
    parser.add_argument(
        "--config",
        default="config_benchmark.toml",
        help="配置文件路径 (默认: config_benchmark.toml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="指定模型列表（覆盖配置文件）",
    )
    parser.add_argument("--skip-speed", action="store_true", help="跳过 benchmark_speed")
    parser.add_argument("--skip-video", action="store_true", help="跳过 video_understanding")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        lg.error("配置文件不存在: {}", config_path)
        sys.exit(1)

    config = load_config(config_path)

    if args.skip_speed:
        config.setdefault("benchmark_speed", {})["enabled"] = False
    if args.skip_video:
        config.setdefault("video_understanding", {})["enabled"] = False

    models = args.models or config.get("models", {}).get("list", [])
    if not models:
        lg.error("未指定任何模型")
        sys.exit(1)

    base_url = config.get("vllm", {}).get("base_url", "http://localhost:8000/v1")
    startup_timeout = config.get("vllm", {}).get("startup_timeout_s", 300)
    poll_interval = config.get("vllm", {}).get("poll_interval_s", 15)

    # Create session directory
    session_dir = create_session_dir(config)
    save_meta(session_dir, config, config_path)

    # Setup logging
    log_path = session_dir / "orchestrator.log"
    lg.add(
        str(log_path),
        level="DEBUG",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    )

    lg.info("=" * 70)
    lg.info("多模型综合测试启动")
    lg.info("会话目录: {}", session_dir)
    lg.info("模型列表 ({}): {}", len(models), models)
    lg.info("=" * 70)

    speed_dir = session_dir / "benchmark_speed"
    video_dir = session_dir / "video_understanding"
    speed_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    completed_models: list[str] = []
    failed_models: list[str] = []

    for idx, model_id in enumerate(models, 1):
        lg.info("━" * 70)
        lg.info("[{}/{}] 模型: {}", idx, len(models), model_id)
        lg.info("━" * 70)

        vllm_proc = None
        try:
            # Create launch script and start vLLM
            create_launch_script(model_id)
            vllm_proc = start_vllm(model_id)

            # Wait for readiness
            lg.info("等待 vLLM 就绪 (超时: {}s) ...", startup_timeout)
            detected_model = wait_for_vllm_ready(
                base_url=base_url,
                timeout_s=startup_timeout,
                poll_interval_s=poll_interval,
            )
            lg.info("vLLM 已就绪: {}", detected_model)

            # Run benchmark_speed
            if config.get("benchmark_speed", {}).get("enabled", True):
                lg.info("── 运行 benchmark_speed ──")
                run_benchmark_speed_for_model(base_url, speed_dir, config)

            # Run video_understanding
            if config.get("video_understanding", {}).get("enabled", True):
                lg.info("── 运行 video_understanding ──")
                run_video_understanding_for_model(video_dir, config)

            completed_models.append(model_id)
            lg.info("模型 {} 测试完成", model_id)

        except Exception:
            lg.exception("模型 {} 测试失败", model_id)
            failed_models.append(model_id)

        finally:
            # Always stop vLLM
            lg.info("停止 vLLM 服务 ...")
            stop_vllm(vllm_proc)
            time.sleep(5)

    # Generate comparison reports
    lg.info("=" * 70)
    lg.info("生成对比报告 ...")

    generate_speed_comparison(session_dir, models)
    generate_video_comparison(session_dir, models)
    generate_final_report(session_dir, config, models)

    # Update meta with end time
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["end_time"] = datetime.now().isoformat()
        meta["completed_models"] = completed_models
        meta["failed_models"] = failed_models
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    lg.info("=" * 70)
    lg.info("多模型综合测试完成")
    lg.info("成功: {}/{} | 失败: {}", len(completed_models), len(models), len(failed_models))
    lg.info("会话目录: {}", session_dir)
    if failed_models:
        lg.warning("失败的模型: {}", failed_models)
    lg.info("=" * 70)


if __name__ == "__main__":
    main()
