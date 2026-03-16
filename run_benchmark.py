"""多模型综合基准测试编排器。

读取 config_benchmark.toml，遍历模型，通过 WSL 管理 vLLM 服务，
运行 benchmark_speed 和 video_understanding 实验，并生成对比报告和可视化图表。

用法:
    uv run run_benchmark.py
    uv run run_benchmark.py --config my_config.toml
    uv run run_benchmark.py --models "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-0.8B"
    uv run run_benchmark.py --skip-speed   # 仅运行 video_understanding
    uv run run_benchmark.py --skip-video   # 仅运行 benchmark_speed
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

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
    get_vllm_base_url,
    start_vllm,
    stop_vllm,
)


# ── Config & Metadata ────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def create_session_dir(config: dict) -> Path:
    ts_format = config.get("general", {}).get(
        "timestamp_format", "%Y-%m-%d_%H-%M-%S"
    )
    timestamp = datetime.now().strftime(ts_format)
    session_dir = (
        PROJECT_ROOT / "experiments" / "multi_model_benchmark" / timestamp
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_meta(session_dir: Path, config: dict, config_path: Path) -> None:
    shutil.copy2(config_path, session_dir / "config.toml")
    meta = {
        "git_hash": get_git_hash(),
        "gpu": get_gpu_info(),
        "python_version": sys.version,
        "start_time": datetime.now().isoformat(),
        "models": config.get("models", {}).get("list", []),
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def update_meta(session_dir: Path, **kwargs: object) -> None:
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(kwargs)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )


# ── Pre-flight Validation ────────────────────────────────────────


def preflight_health_check(base_url: str, model_id: str) -> bool:
    """发送最小请求验证模型能正常响应（带重试）。"""
    import httpx

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }

    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(10)
                lg.info("健康检查重试 ({}/3) ...", attempt + 1)
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content and len(content) > 1:
                    lg.info("健康检查通过: '{}'", content[:60])
                    return True
                lg.warning("健康检查: 空响应")
        except Exception as e:
            lg.warning("健康检查尝试 {}/3 失败: {}", attempt + 1, str(e)[:120])
    return False


def preflight_vision_check(base_url: str, model_id: str) -> bool:
    """发送一张图片验证视觉功能正常（带重试）。"""
    import httpx
    from toolkit.common import load_images

    images = load_images()
    if not images:
        lg.warning("无测试图片，跳过视觉检查")
        return True

    fname, b64, mime, _ = images[0]
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image briefly."},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                }},
            ],
        }],
        "max_tokens": 64,
        "temperature": 0.0,
    }

    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(10)
                lg.info("视觉检查重试 ({}/3) ...", attempt + 1)
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if text and len(text) > 5:
                    lg.info("视觉检查通过: '{}'", text[:80])
                    return True
                lg.warning("视觉检查: 响应过短 '{}'", text)
        except Exception as e:
            lg.warning("视觉检查尝试 {}/3 失败: {}", attempt + 1, str(e)[:120])
    return False


def probe_gpu_memory(
    model_id: str,
    base_url: str,
    startup_timeout: float,
    poll_interval: float,
) -> float | None:
    """探测模型所需的最小 gpu_memory_utilization。

    从 0.35 开始递增，使用 enforce-eager 跳过 CUDA graph 编译以加速探测。
    返回成功的值，或 None 表示全部失败。
    """
    for util in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        lg.info("  尝试 gpu_memory_utilization={:.2f} ...", util)
        proc = None
        try:
            proc = start_vllm(
                model_id,
                extra_args={"enforce-eager": True},
                gpu_memory_utilization=util,
            )
            detected = wait_for_vllm_ready(
                base_url=base_url,
                timeout_s=min(startup_timeout, 120),
                poll_interval_s=poll_interval,
            )
            if detected:
                lg.info("  成功: gpu_memory_utilization={:.2f}", util)
                return util
        except Exception as e:
            lg.info("  gpu_memory_utilization={:.2f} 失败: {}", util, str(e)[:120])
        finally:
            stop_vllm(proc)
            time.sleep(3)
    return None


def run_preflight(
    models: list[str],
    base_url: str,
    startup_timeout: float,
    poll_interval: float,
    *,
    probe_memory: bool = True,
) -> tuple[list[str], list[dict], dict[str, float]]:
    """验证每个模型能否在 vLLM 上正常启动并进行视觉推理。

    Returns: (valid_models, incidents, gpu_util_map)
    """
    valid: list[str] = []
    incidents: list[dict] = []
    gpu_util_map: dict[str, float] = {}

    for idx, model_id in enumerate(models, 1):
        lg.info("━" * 60)
        lg.info("[预飞行 {}/{}] {}", idx, len(models), model_id)

        # GPU 显存探测
        optimal_util = 0.5
        if probe_memory:
            lg.info("  开始 GPU 显存探测 ...")
            result = probe_gpu_memory(
                model_id, base_url, startup_timeout, poll_interval,
            )
            if result is None:
                incidents.append({
                    "model": model_id,
                    "phase": "gpu_probe",
                    "error": "所有 gpu_memory_utilization (0.3-0.6) 均无法启动",
                    "resolution": "从实验中排除",
                })
                lg.error("[排除] {} - GPU 显存探测全部失败", model_id)
                continue
            optimal_util = result
            gpu_util_map[model_id] = optimal_util
            lg.info("  最优 gpu_memory_utilization: {:.2f}", optimal_util)

        proc = None
        try:
            proc = start_vllm(model_id, gpu_memory_utilization=optimal_util)
            lg.info("等待 vLLM 就绪 ...")
            detected = wait_for_vllm_ready(
                base_url=base_url,
                timeout_s=startup_timeout,
                poll_interval_s=poll_interval,
            )
            lg.info("vLLM 已就绪: {}", detected)
            time.sleep(5)

            ok_text = preflight_health_check(base_url, detected)
            ok_vision = preflight_vision_check(base_url, detected)

            if ok_text and ok_vision:
                valid.append(model_id)
                lg.info("[通过] {}", model_id)
            else:
                failed_parts = []
                if not ok_text:
                    failed_parts.append("文本")
                if not ok_vision:
                    failed_parts.append("视觉")
                incidents.append({
                    "model": model_id,
                    "phase": "health_check",
                    "error": f"{'和'.join(failed_parts)}检查失败",
                    "resolution": "从实验中排除",
                })
                lg.warning("[未通过] {} - {}检查失败", model_id, "和".join(failed_parts))

        except Exception as e:
            error_msg = str(e)
            from toolkit.vllm_manager import read_vllm_log
            stderr_snippet = read_vllm_log(tail=40)

            incidents.append({
                "model": model_id,
                "phase": "startup",
                "error": error_msg,
                "stderr_snippet": stderr_snippet,
                "resolution": "从实验中排除",
            })
            lg.error("[未通过] {} - {}", model_id, error_msg)

        finally:
            lg.info("停止 vLLM ...")
            stop_vllm(proc)
            time.sleep(5)

    return valid, incidents, gpu_util_map


# ── Experiment Runners ────────────────────────────────────────────


def run_benchmark_speed_for_model(
    base_url: str, output_dir: Path, config: dict,
) -> bool:
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
    output_dir: Path, config: dict, base_url: str | None = None,
) -> bool:
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
                base_url=base_url,
            )
        )
        total_runs = sum(len(v) for v in results.values())
        lg.info(
            "video_understanding 完成: {} 个视频, {} 次运行",
            len(results), total_runs,
        )
        return True
    except Exception:
        lg.exception("video_understanding 执行失败")
        return False


# ── Report Generation ─────────────────────────────────────────────


def _collect_speed_data(speed_dir: Path, models: list[str]) -> dict:
    data: dict = {}
    for model_id in models:
        short = model_short_name(model_id)
        csv_path = speed_dir / short / "raw_data.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        ttfts = [float(r["ttft_s"]) for r in rows]
        tps_vals = [float(r["throughput_tps"]) for r in rows]
        data[model_id] = {
            "ttft": mean(ttfts),
            "ttft_std": stdev(ttfts) if len(ttfts) > 1 else 0,
            "tps": mean(tps_vals),
            "tps_std": stdev(tps_vals) if len(tps_vals) > 1 else 0,
            "total": mean(float(r["total_time_s"]) for r in rows),
            "tokens": mean(float(r["output_tokens"]) for r in rows),
            "vram": int(rows[0].get("vram_mb", 0)),
            "runs": len(rows),
        }
    return data


def generate_speed_comparison(session_dir: Path, models: list[str]) -> None:
    speed_dir = session_dir / "benchmark_speed"
    if not speed_dir.exists():
        return

    speed_data = _collect_speed_data(speed_dir, models)

    lines = [
        "# 推理速度基准测试 - 多模型对比报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 性能总览",
        "",
        "| 模型 | 平均TTFT (s) | 平均吞吐量 (tok/s) | 平均总耗时 (s) | 平均Token数 | 运行次数 |",
        "|------|:---:|:---:|:---:|:---:|:---:|",
    ]

    for model_id in models:
        short = model_short_name(model_id)
        d = speed_data.get(model_id)
        if not d:
            lines.append(f"| {model_id} | - | - | - | - | 0 |")
            continue
        lines.append(
            f"| {model_id} | {d['ttft']:.3f}±{d['ttft_std']:.3f} | {d['tps']:.1f}±{d['tps_std']:.1f} "
            f"| {d['total']:.3f} | {d['tokens']:.0f} | {d['runs']} |"
        )

    lines.extend([
        "",
        "## 帧率分析",
        "",
        "| 模型 | 平均TTFT (s) | 理论最大FPS | 安全FPS (70%) |",
        "|------|:---:|:---:|:---:|",
    ])

    for model_id in models:
        d = speed_data.get(model_id)
        if not d:
            continue
        max_fps = 1.0 / d["ttft"] if d["ttft"] > 0 else 0
        lines.append(
            f"| {model_id} | {d['ttft']:.3f} | {max_fps:.1f} | {max_fps * 0.7:.1f} |"
        )

    if speed_data:
        lines.extend(["", "## 分析", ""])
        fastest = min(speed_data.items(), key=lambda x: x[1]["ttft"])
        highest_tp = max(speed_data.items(), key=lambda x: x[1]["tps"])
        lines.extend([
            f"**延迟最低**: {fastest[0]} (TTFT {fastest[1]['ttft']:.3f}s，理论最大帧率 "
            f"{1.0/fastest[1]['ttft']:.1f} FPS)",
            "",
            f"**吞吐量最高**: {highest_tp[0]} ({highest_tp[1]['tps']:.1f} tok/s)",
            "",
            "> 理论最大 FPS = 1 / TTFT。安全帧率取 70% 以预留网络、队列和帧差计算的开销。",
            "> 实际使用中帧差过滤会显著降低 VLM 负载，因此安全帧率通常足够。",
        ])

    (speed_dir / "comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    lg.info("benchmark_speed 对比报告已生成")


def generate_video_comparison(session_dir: Path, models: list[str]) -> None:
    vu_dir = session_dir / "video_understanding"
    if not vu_dir.exists():
        return

    lines = [
        "# 视频理解 - 多模型对比报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 模型 | 视频数 | 运行次数 | 平均关键帧 | 平均描述数 | 平均丢弃帧 |",
        "|------|:---:|:---:|:---:|:---:|:---:|",
    ]

    model_stats: dict[str, dict] = {}

    for model_id in models:
        short = model_short_name(model_id)
        model_dir = vu_dir / short
        if not model_dir.exists():
            lines.append(f"| {model_id} | 0 | 0 | - | - | - |")
            continue

        total_runs = 0
        kf_list, desc_list, drop_list = [], [], []

        for item in model_dir.iterdir():
            if not item.is_dir():
                continue
            log_path = item / "run_log.json"
            if log_path.exists():
                try:
                    ld = json.loads(log_path.read_text(encoding="utf-8"))
                    stats = ld.get("stats", {})
                    kf_list.append(stats.get("total_keyframes", 0))
                    desc_list.append(
                        len([f for f in ld.get("frames", [])
                             if f.get("vlm_response")])
                    )
                    drop_list.append(stats.get("total_dropped", 0))
                    total_runs += 1
                except Exception:
                    pass

        avg_kf = mean(kf_list) if kf_list else 0
        avg_desc = mean(desc_list) if desc_list else 0
        avg_drop = mean(drop_list) if drop_list else 0
        lines.append(
            f"| {model_id} | - | {total_runs} "
            f"| {avg_kf:.1f} | {avg_desc:.1f} | {avg_drop:.1f} |"
        )
        model_stats[model_id] = {
            "avg_kf": avg_kf, "avg_desc": avg_desc, "avg_drop": avg_drop,
            "runs": total_runs,
        }

    if model_stats:
        lines.extend(["", "## 分析", ""])
        best_desc = max(model_stats.items(), key=lambda x: x[1]["avg_desc"])
        least_drop = min(model_stats.items(), key=lambda x: x[1]["avg_drop"])
        lines.extend([
            f"**描述生成最多**: {best_desc[0]} (平均 {best_desc[1]['avg_desc']:.1f} 个描述/运行)",
            "",
            f"**丢帧最少**: {least_drop[0]} (平均 {least_drop[1]['avg_drop']:.1f} 帧丢弃/运行)",
            "",
            "> KeyFrameQueue 中超过 expiry_time_ms 的帧会被静默丢弃。",
            "> 高丢帧率意味着模型推理速度跟不上截图频率，建议降低采集帧率或换用更快的模型。",
        ])

    (vu_dir / "comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    lg.info("video_understanding 对比报告已生成")


def generate_charts(
    session_dir: Path, models: list[str],
    gpu_util_map: dict[str, float] | None = None,
) -> None:
    from toolkit.visualization import (
        plot_gpu_memory_comparison,
        plot_metric_ranking,
        plot_scenario_comparison,
        plot_summary_table,
        plot_video_comparison,
    )

    charts_dir = session_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    speed_dir = session_dir / "benchmark_speed"

    speed_data = _collect_speed_data(speed_dir, models) if speed_dir.exists() else {}

    tested_models = [m for m in models if m in speed_data]
    if not tested_models:
        lg.warning("无速度数据，跳过图表生成")
        return

    model_ttfts = [speed_data[m]["ttft"] for m in tested_models]
    model_tps = [speed_data[m]["tps"] for m in tested_models]
    model_totals = [speed_data[m]["total"] for m in tested_models]
    model_tokens = [speed_data[m]["tokens"] for m in tested_models]
    model_vram = [speed_data[m]["vram"] for m in tested_models]

    plot_metric_ranking(
        tested_models, model_ttfts, "TTFT (s)",
        "平均 TTFT 排名（越低越好）",
        charts_dir / "ttft_ranking.png",
        higher_is_better=False, unit="s",
    )
    plot_metric_ranking(
        tested_models, model_tps, "吞吐量 (tok/s)",
        "平均吞吐量排名（越高越好）",
        charts_dir / "throughput_ranking.png",
        higher_is_better=True, unit=" tok/s",
    )

    scenario_names: list[str] = []
    scenario_matrix: list[list[float]] = [[] for _ in tested_models]
    for i, model_id in enumerate(tested_models):
        short = model_short_name(model_id)
        csv_path = speed_dir / short / "raw_data.csv"
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_scenario: dict[str, list[float]] = {}
        for r in rows:
            by_scenario.setdefault(r["scenario_id"], []).append(
                float(r["ttft_s"])
            )
        if i == 0:
            scenario_names = sorted(by_scenario.keys())
        for sname in scenario_names:
            vals = by_scenario.get(sname, [])
            scenario_matrix[i].append(mean(vals) if vals else 0)

    if scenario_names:
        plot_scenario_comparison(
            scenario_names, tested_models, scenario_matrix,
            "TTFT (s)", "各场景 TTFT 对比",
            charts_dir / "scenario_comparison.png",
        )

    headers = ["模型", "TTFT (s)", "吞吐量", "总耗时 (s)", "Token数", "VRAM (MB)"]
    table_rows = []
    for i, m in enumerate(tested_models):
        short = m.split("/")[-1] if "/" in m else m
        table_rows.append([
            short, f"{model_ttfts[i]:.3f}", f"{model_tps[i]:.1f}",
            f"{model_totals[i]:.3f}", f"{model_tokens[i]:.0f}",
            str(model_vram[i]),
        ])
    plot_summary_table(
        headers, table_rows, "推理速度基准测试汇总",
        charts_dir / "summary_table.png",
        highlight_col=1, highlight_best="min",
    )

    # 视频理解图表
    vu_dir = session_dir / "video_understanding"
    vu_models, vu_kf, vu_dropped, vu_descs = [], [], [], []
    for model_id in models:
        short = model_short_name(model_id)
        model_dir = vu_dir / short
        if not model_dir.exists():
            continue
        kf_l, desc_l, drop_l = [], [], []
        for vdir in model_dir.iterdir():
            if not vdir.is_dir():
                continue
            lp = vdir / "run_log.json"
            if lp.exists():
                try:
                    ld = json.loads(lp.read_text(encoding="utf-8"))
                    stats = ld.get("stats", {})
                    kf_l.append(stats.get("total_keyframes", 0))
                    desc_l.append(
                        len([f for f in ld.get("frames", [])
                             if f.get("vlm_response")])
                    )
                    drop_l.append(stats.get("total_dropped", 0))
                except Exception:
                    pass
        if kf_l:
            vu_models.append(model_id)
            vu_kf.append(mean(kf_l))
            vu_descs.append(mean(desc_l))
            vu_dropped.append(mean(drop_l))

    if vu_models:
        plot_video_comparison(
            vu_models, vu_kf, vu_dropped, vu_descs,
            "视频理解多模型对比",
            charts_dir / "video_comparison.png",
        )

    # GPU 显存图表
    if gpu_util_map:
        gpu_models = [m for m in tested_models if m in gpu_util_map]
        if gpu_models:
            plot_gpu_memory_comparison(
                gpu_models,
                [gpu_util_map[m] for m in gpu_models],
                [speed_data[m]["vram"] for m in gpu_models],
                "各模型 GPU 显存利用率与 VRAM 占用",
                charts_dir / "gpu_memory_comparison.png",
            )

    lg.info("图表已生成: {}", charts_dir)


def generate_final_report(
    session_dir: Path, config: dict, models: list[str],
    incidents: list[dict],
    gpu_util_map: dict[str, float] | None = None,
) -> None:
    lines = [
        "# 多模型综合基准测试报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验目标",
        "",
        "本实验旨在系统评估 8 个小型视觉语言模型（VLM, 0.8B-4B 参数）在游戏场景 AI 伴侣应用中的表现。",
        "评估维度包括：",
        "",
        "1. **推理速度**：首Token延迟（TTFT）、吞吐量（tok/s）、理论最大帧率",
        "2. **视频理解**：实时截图→帧差检测→VLM描述→DeepSeek汇总的完整流水线效果",
        "3. **资源效率**：各模型最小可用 GPU 显存配置",
        "",
    ]

    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gpu = meta.get("gpu", {})
        completed = meta.get("completed_models", [])
        failed = meta.get("failed_models", [])
        lines.extend([
            "## 实验环境",
            "",
            f"- **GPU**: {gpu.get('name', 'N/A')} ({gpu.get('memory_total_mb', 0)} MB)",
            f"- **Python**: {meta.get('python_version', 'N/A').split()[0]}",
            f"- **Git**: {meta.get('git_hash', 'N/A')}",
            f"- **开始时间**: {meta.get('start_time', 'N/A')}",
            f"- **结束时间**: {meta.get('end_time', 'N/A')}",
            "",
        ])
    else:
        completed, failed = [], []

    lines.extend([
        "## 模型清单",
        "",
        "| # | 模型ID | 简称 | 参数量级 | 状态 |",
        "|---|--------|------|---------|------|",
    ])

    model_sizes = {
        "Qwen/Qwen3.5-2B": "2B", "Qwen/Qwen3.5-0.8B": "0.8B",
        "Qwen/Qwen3-VL-2B-Instruct": "2B", "OpenGVLab/InternVL2_5-2B": "2B",
        "microsoft/Phi-3.5-vision-instruct": "4.2B",
        "mistralai/Ministral-3-3B-Instruct-2512": "3B",
        "deepseek-ai/deepseek-vl2-tiny": "~3B",
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct": "2.2B",
    }

    for i, m in enumerate(models, 1):
        short = model_short_name(m)
        size = model_sizes.get(m, "?")
        if m in completed:
            status = "已完成"
        elif m in failed:
            status = "实验中失败"
        else:
            status = "预飞行排除"
        lines.append(f"| {i} | {m} | {short} | {size} | {status} |")

    # GPU 显存优化结果
    if gpu_util_map:
        lines.extend([
            "",
            "## GPU 显存优化",
            "",
            "通过逐步探测，为每个模型找到了在 RTX 4080 SUPER (16GB) 上可用的最小 "
            "`gpu_memory_utilization` 值：",
            "",
            "| 模型 | 最小 gpu_memory_utilization | 等效分配显存 |",
            "|------|:---:|:---:|",
        ])
        for m, util in gpu_util_map.items():
            alloc_mb = int(16376 * util)
            lines.append(f"| {m} | {util:.2f} | ~{alloc_mb} MB |")
        lines.extend([
            "",
            "> 更低的 gpu_memory_utilization 意味着模型可与其他任务共享 GPU 资源，"
            "这对实际部署非常重要。",
            "",
        ])

    # 图表 + 分析
    charts_dir = session_dir / "charts"
    speed_dir = session_dir / "benchmark_speed"
    speed_data = _collect_speed_data(speed_dir, completed) if speed_dir.exists() else {}

    if speed_data:
        lines.extend([
            "## 推理速度测试结果",
            "",
            "| 模型 | TTFT (s) | 吞吐量 (tok/s) | 总耗时 (s) | 最大FPS | 安全FPS |",
            "|------|:---:|:---:|:---:|:---:|:---:|",
        ])
        for m, d in speed_data.items():
            max_fps = 1.0 / d["ttft"] if d["ttft"] > 0 else 0
            lines.append(
                f"| {m.split('/')[-1]} | {d['ttft']:.3f} | {d['tps']:.1f} "
                f"| {d['total']:.3f} | {max_fps:.2f} | {max_fps * 0.7:.2f} |"
            )
        lines.append("")

    if charts_dir.exists():
        for fname, caption in [
            ("summary_table.png", "汇总表"),
            ("ttft_ranking.png", "TTFT 排名"),
            ("throughput_ranking.png", "吞吐量排名"),
            ("scenario_comparison.png", "各场景 TTFT 对比"),
        ]:
            if (charts_dir / fname).exists():
                lines.extend([f"### {caption}", "", f"![{caption}](charts/{fname})", ""])

    if speed_data:
        fastest = min(speed_data.items(), key=lambda x: x[1]["ttft"])
        highest_tp = max(speed_data.items(), key=lambda x: x[1]["tps"])
        lines.extend([
            "### 速度测试分析",
            "",
            f"在 5 个游戏场景（短描述、详细描述、物体检测、动作分析、UI识别）的测试中，"
            f"**{fastest[0].split('/')[-1]}** 以 {fastest[1]['ttft']:.3f}s 的 TTFT 取得最低延迟，"
            f"对应理论最大帧率 {1.0/fastest[1]['ttft']:.1f} FPS。",
            "",
            f"吞吐量方面，**{highest_tp[0].split('/')[-1]}** 以 {highest_tp[1]['tps']:.1f} tok/s "
            f"领先。吞吐量影响单帧描述的生成速度——高吞吐量模型更适合需要详细描述的离线分析场景。",
            "",
            "对于实时游戏伴侣应用，TTFT 是关键指标：",
            "",
            "- 要达到 1 FPS 实时描述，需要 TTFT < 1s",
            "- 考虑帧差过滤后实际 VLM 调用频率远低于帧率，多数模型均能满足需求",
            "- 安全帧率取理论值的 70%，为网络传输和队列调度预留余量",
            "",
        ])

    # 视频理解结果
    vu_cmp = session_dir / "video_understanding" / "comparison_report.md"
    if vu_cmp.exists():
        lines.extend([
            "## 视频理解测试结果",
            "",
        ])
        if (charts_dir / "video_comparison.png").exists():
            lines.extend(["![视频理解对比](charts/video_comparison.png)", ""])
        lines.extend([
            f"详细数据见 [comparison_report.md](video_understanding/comparison_report.md)。",
            "",
        ])

    if gpu_util_map and (charts_dir / "gpu_memory_comparison.png").exists():
        lines.extend([
            "## GPU 资源效率",
            "",
            "![GPU 显存对比](charts/gpu_memory_comparison.png)",
            "",
        ])

    # 研究问题
    lines.extend([
        "## 研究问题回答",
        "",
        "### 1. 理论最大帧率是多少？",
        "",
    ])
    if speed_data:
        fastest = min(speed_data.items(), key=lambda x: x[1]["ttft"])
        lines.append(
            f"最快模型 **{fastest[0].split('/')[-1]}** 的理论最大帧率为 "
            f"{1.0/fastest[1]['ttft']:.1f} FPS（TTFT = {fastest[1]['ttft']:.3f}s）。"
        )
    lines.extend([
        "",
        "### 2. 安全的截图采集帧率是多少？",
        "",
        "建议取理论最大帧率的 70% 作为安全值，以预留网络传输、队列调度和帧差计算的开销。",
        "在实际使用中，帧差过滤会大幅降低 VLM 调用频率——仅当画面发生显著变化时才触发推理。",
        "",
        "### 3. 未处理的帧如何处理？",
        "",
        "KeyFrameQueue 使用过期机制（默认 10 秒），超时的帧被静默丢弃，不阻塞流水线。",
        "丢帧率高说明模型推理跟不上采集速度，应降低采集帧率或换用更快的模型。",
        "",
        "## 应用建议",
        "",
    ])
    if speed_data:
        fastest = min(speed_data.items(), key=lambda x: x[1]["ttft"])
        highest_tp = max(speed_data.items(), key=lambda x: x[1]["tps"])
        lines.extend([
            f"- **实时伴侣**：优先选择低延迟模型 **{fastest[0].split('/')[-1]}**，"
            f"采样频率 ~1 帧/2-3s",
            f"- **离线分析**：使用高吞吐模型 **{highest_tp[0].split('/')[-1]}** 批量处理",
            "- **资源受限环境**：选择 GPU 显存需求最低的模型，可与游戏进程共享 GPU",
            "",
        ])

    if incidents:
        lines.extend(["## 事件记录", ""])
        for inc in incidents:
            lines.append(
                f"- **{inc['model']}**（{inc.get('phase', '未知')}阶段）："
                f"{inc.get('error', '未知错误')} → {inc.get('resolution', '')}"
            )
        lines.append("")

    (session_dir / "final_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    lg.info("最终报告已生成")


def generate_incident_report(
    session_dir: Path, incidents: list[dict],
) -> None:
    lines = [
        "# 事件报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if not incidents:
        lines.append("本次实验过程中未发生任何意外事件。")
    else:
        lines.append(f"共记录 {len(incidents)} 个事件。")
        lines.append("")
        for i, inc in enumerate(incidents, 1):
            lines.extend([
                f"## 事件 {i}: {inc['model']}",
                "",
                f"- **阶段**: {inc.get('phase', '未知')}",
                f"- **错误**: {inc.get('error', 'N/A')}",
                f"- **处置**: {inc.get('resolution', 'N/A')}",
            ])
            if inc.get("stderr_snippet"):
                lines.extend([
                    "",
                    "**vLLM 输出（末尾片段）**:",
                    f"```\n{inc['stderr_snippet']}\n```",
                ])
            lines.append("")

    (session_dir / "incident_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    lg.info("事件报告已生成")


def generate_readme(session_dir: Path, config: dict, models: list[str]) -> None:
    from experiments.benchmark_speed.benchmark import SCENARIOS

    lines = [
        "# 多模型综合基准测试",
        "",
        "本目录包含一次完整的多模型基准测试会话。",
        "",
        "## 目的",
        "",
        "系统评估多个 VLM 模型在游戏场景 AI 伴侣应用中的推理速度和视频理解能力。",
        "",
        "## 测试模型",
        "",
    ]
    for i, m in enumerate(models, 1):
        lines.append(f"{i}. `{m}`")

    lines.extend([
        "",
        "## 实验内容",
        "",
        "### 1. 推理速度基准测试 (benchmark_speed)",
        "",
        "对游戏截图进行多场景推理，测量 TTFT、吞吐量和总推理时间。",
        "",
        "**场景列表**:",
        "",
    ])
    for s in SCENARIOS:
        lines.append(f"- `{s.id}` ({s.name}): max_tokens={s.max_tokens}")

    lines.extend([
        "",
        "### 2. 视频理解 (video_understanding)",
        "",
        "完整流水线: PotPlayer 播放 → 窗口截图 → 帧差检测 → vLLM 描述 → DeepSeek 汇总",
        "",
        "## 复现步骤",
        "",
        "### 前置条件",
        "",
        "- Windows 10/11 + WSL2",
        "- NVIDIA GPU (>= 12GB VRAM 推荐)",
        "- Python >= 3.12, uv 包管理器",
        "- WSL 中安装 vLLM",
        "- PotPlayer（用于 video_understanding）",
        "",
        "### 步骤",
        "",
        "```bash",
        "# 1. 安装依赖",
        "uv sync",
        "",
        "# 2. 在项目根目录创建 .env",
        "# DEEPSEEK_API_KEY=your_key",
        "# DEEPSEEK_API_BASE_URL=https://api.deepseek.com",
        "# PLAYER_EXE_PATH=D:\\Programs\\PotPlayer\\PotPlayerMini64.exe",
        "",
        "# 3. 确保模型已缓存在 WSL 中 (~/.cache/huggingface/hub/)",
        "",
        "# 4. 运行基准测试",
        "uv run run_benchmark.py",
        "",
        "# 可选参数:",
        "# --config my_config.toml",
        "# --models Qwen/Qwen3.5-2B Qwen/Qwen3.5-0.8B",
        "# --skip-speed   (仅运行 video_understanding)",
        "# --skip-video   (仅运行 benchmark_speed)",
        "```",
        "",
        "## 目录结构",
        "",
        "```",
        ".",
        "├── config.toml               # 配置快照",
        "├── meta.json                  # 环境元数据",
        "├── orchestrator.log           # 完整日志",
        "├── README.md                  # 本文件",
        "├── final_report.md            # 综合报告",
        "├── incident_report.md         # 事件记录",
        "├── charts/                    # 可视化图表",
        "├── benchmark_speed/",
        "│   ├── comparison_report.md",
        "│   └── {model}/",
        "│       ├── raw_data.csv",
        "│       ├── report.md",
        "│       └── benchmark.log",
        "└── video_understanding/",
        "    ├── comparison_report.md",
        "    └── {model}/",
        "        ├── report.md",
        "        └── {video}_run{N}/",
        "            ├── frames/",
        "            ├── run_log.json",
        "            └── summary.txt",
        "```",
        "",
    ])

    (session_dir / "README.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    lg.info("README 已生成")


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="多模型基准测试")
    parser.add_argument(
        "--config", default="config_benchmark.toml",
        help="配置文件 (默认: config_benchmark.toml)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="覆盖配置中的模型列表",
    )
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="跳过预飞行验证（假设所有模型可用）",
    )
    parser.add_argument(
        "--skip-memory-probe", action="store_true",
        help="跳过 GPU 显存探测（使用默认 0.5）",
    )
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
        lg.error("未指定模型")
        sys.exit(1)

    base_url = get_vllm_base_url()
    startup_timeout = config.get("vllm", {}).get("startup_timeout_s", 300)
    poll_interval = config.get("vllm", {}).get("poll_interval_s", 15)

    session_dir = create_session_dir(config)
    save_meta(session_dir, config, config_path)

    log_path = session_dir / "orchestrator.log"
    lg.add(
        str(log_path), level="DEBUG", encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    )

    lg.info("=" * 70)
    lg.info("多模型基准测试启动")
    lg.info("会话目录: {}", session_dir)
    lg.info("模型列表 ({}): {}", len(models), models)
    lg.info("=" * 70)

    # ── Phase 0: 预飞行 ──
    incidents: list[dict] = []
    gpu_util_map: dict[str, float] = {}

    if not args.skip_preflight:
        lg.info("=" * 70)
        lg.info("Phase 0: 预飞行模型验证")
        lg.info("=" * 70)
        valid_models, incidents, gpu_util_map = run_preflight(
            models, base_url, startup_timeout, poll_interval,
            probe_memory=not args.skip_memory_probe,
        )
        lg.info(
            "预飞行完成: {}/{} 个模型通过",
            len(valid_models), len(models),
        )
        if incidents:
            lg.warning("失败模型: {}", [i["model"] for i in incidents])
        models = valid_models
    else:
        lg.info("预飞行已跳过 (--skip-preflight)")

    if not models:
        lg.error("预飞行后无可用模型，中止")
        generate_incident_report(session_dir, incidents)
        sys.exit(1)

    # ── Phase 1: 实验 ──
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

        proc = None
        try:
            gpu_util = gpu_util_map.get(model_id, config["vllm"]["gpu_memory_utilization"])
            proc = start_vllm(model_id, gpu_memory_utilization=gpu_util)

            lg.info("等待 vLLM 就绪 (超时: {}s) ...", startup_timeout)
            detected = wait_for_vllm_ready(
                base_url=base_url,
                timeout_s=startup_timeout,
                poll_interval_s=poll_interval,
            )
            lg.info("vLLM 已就绪: {}", detected)

            if config.get("benchmark_speed", {}).get("enabled", True):
                lg.info("── 运行 benchmark_speed ──")
                run_benchmark_speed_for_model(base_url, speed_dir, config)

            if config.get("video_understanding", {}).get("enabled", True):
                lg.info("── 运行 video_understanding ──")
                run_video_understanding_for_model(video_dir, config, base_url)

            completed_models.append(model_id)
            lg.info("模型 {} 已完成", model_id)

        except Exception:
            lg.exception("模型 {} 失败", model_id)
            failed_models.append(model_id)
            incidents.append({
                "model": model_id,
                "phase": "experiment",
                "error": "详见 orchestrator.log",
                "resolution": "已跳过",
            })

        finally:
            lg.info("停止 vLLM ...")
            stop_vllm(proc)
            time.sleep(5)

    # ── Phase 2: 报告 ──
    update_meta(
        session_dir,
        end_time=datetime.now().isoformat(),
        completed_models=completed_models,
        failed_models=failed_models,
        gpu_util_map=gpu_util_map,
    )

    lg.info("=" * 70)
    lg.info("生成报告和图表 ...")

    generate_speed_comparison(session_dir, completed_models)
    generate_video_comparison(session_dir, completed_models)
    generate_charts(session_dir, completed_models, gpu_util_map)
    generate_final_report(session_dir, config, models, incidents, gpu_util_map)
    generate_incident_report(session_dir, incidents)
    generate_readme(session_dir, config, models)

    lg.info("=" * 70)
    lg.info("基准测试完成")
    lg.info(
        "通过: {}/{} | 失败: {}",
        len(completed_models), len(models), len(failed_models),
    )
    lg.info("会话目录: {}", session_dir)
    if failed_models:
        lg.warning("失败模型: {}", failed_models)
    lg.info("=" * 70)


if __name__ == "__main__":
    main()
