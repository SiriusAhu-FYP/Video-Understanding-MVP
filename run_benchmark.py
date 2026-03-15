"""Multi-model comprehensive benchmark orchestrator.

Reads config_benchmark.toml, iterates through models, manages vLLM services
via WSL, runs benchmark_speed and video_understanding experiments, and
generates comparison reports with visualizations.

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
import csv
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
    """Send a minimal request to verify the model responds."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=8,
            temperature=0.0,
        )
        if resp.choices and resp.choices[0].message.content:
            lg.info("Health check OK: '{}'", resp.choices[0].message.content[:60])
            return True
        lg.warning("Health check: empty response")
        return False
    except Exception as e:
        lg.error("Health check failed: {}", e)
        return False


def run_preflight(
    models: list[str],
    base_url: str,
    startup_timeout: float,
    poll_interval: float,
) -> tuple[list[str], list[dict]]:
    """Validate each model can start on vLLM. Returns (valid_models, incidents)."""
    valid: list[str] = []
    incidents: list[dict] = []

    for idx, model_id in enumerate(models, 1):
        lg.info("━" * 60)
        lg.info("[Pre-flight {}/{}] {}", idx, len(models), model_id)

        proc = None
        try:
            proc = start_vllm(model_id)
            lg.info("等待 vLLM 就绪 ...")
            detected = wait_for_vllm_ready(
                base_url=base_url,
                timeout_s=startup_timeout,
                poll_interval_s=poll_interval,
            )
            lg.info("vLLM 已就绪: {}", detected)

            ok = preflight_health_check(base_url, detected)
            if ok:
                valid.append(model_id)
                lg.info("[PASS] {}", model_id)
            else:
                incidents.append({
                    "model": model_id,
                    "phase": "health_check",
                    "error": "empty or invalid response",
                    "resolution": "excluded from experiment",
                })
                lg.warning("[FAIL] {} - health check failed", model_id)

        except Exception as e:
            error_msg = str(e)
            # Capture vLLM stderr for debugging
            stderr_snippet = ""
            if proc and proc.stdout:
                try:
                    proc.stdout.flush()
                    import io
                    raw = proc.stdout.read1(8192) if hasattr(proc.stdout, "read1") else b""
                    stderr_snippet = raw.decode("utf-8", errors="replace")[-2000:]
                except Exception:
                    pass

            incidents.append({
                "model": model_id,
                "phase": "startup",
                "error": error_msg,
                "stderr_snippet": stderr_snippet,
                "resolution": "excluded from experiment",
            })
            lg.error("[FAIL] {} - {}", model_id, error_msg)

        finally:
            lg.info("停止 vLLM ...")
            stop_vllm(proc)
            time.sleep(5)

    return valid, incidents


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
    output_dir: Path, config: dict,
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
        data[model_id] = {
            "ttft": mean(float(r["ttft_s"]) for r in rows),
            "tps": mean(float(r["throughput_tps"]) for r in rows),
            "total": mean(float(r["total_time_s"]) for r in rows),
            "tokens": mean(float(r["output_tokens"]) for r in rows),
            "vram": int(rows[0].get("vram_mb", 0)),
        }
    return data


def generate_speed_comparison(session_dir: Path, models: list[str]) -> None:
    speed_dir = session_dir / "benchmark_speed"
    if not speed_dir.exists():
        return

    lines = [
        "# Benchmark Speed - Multi-Model Comparison",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Performance Overview",
        "",
        "| Model | Avg TTFT (s) | Avg Throughput (tok/s) | Avg Total (s) | Avg Tokens | Runs |",
        "|-------|:---:|:---:|:---:|:---:|:---:|",
    ]

    for model_id in models:
        short = model_short_name(model_id)
        csv_path = speed_dir / short / "raw_data.csv"
        if not csv_path.exists():
            lines.append(f"| {model_id} | - | - | - | - | 0 |")
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
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

    lines.extend([
        "",
        "## Frame Rate Analysis",
        "",
        "| Model | Avg TTFT (s) | Max FPS | Safe FPS (70%) |",
        "|-------|:---:|:---:|:---:|",
    ])

    for model_id in models:
        short = model_short_name(model_id)
        csv_path = speed_dir / short / "raw_data.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        avg_ttft = mean(float(r["ttft_s"]) for r in rows)
        max_fps = 1.0 / avg_ttft if avg_ttft > 0 else 0
        lines.append(
            f"| {model_id} | {avg_ttft:.3f} | {max_fps:.1f} | {max_fps * 0.7:.1f} |"
        )

    lines.extend([
        "",
        "> Max FPS = 1 / TTFT. Safe range = 70% of max to allow for overhead.",
    ])

    (speed_dir / "comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    lg.info("benchmark_speed comparison report written")


def generate_video_comparison(session_dir: Path, models: list[str]) -> None:
    vu_dir = session_dir / "video_understanding"
    if not vu_dir.exists():
        return

    lines = [
        "# Video Understanding - Multi-Model Comparison",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Model | Videos | Runs | Avg Keyframes | Avg Descriptions | Avg Dropped |",
        "|-------|:---:|:---:|:---:|:---:|:---:|",
    ]

    for model_id in models:
        short = model_short_name(model_id)
        model_dir = vu_dir / short
        if not model_dir.exists():
            lines.append(f"| {model_id} | 0 | 0 | - | - | - |")
            continue

        video_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        total_runs = 0
        kf_list, desc_list, drop_list = [], [], []

        for vdir in video_dirs:
            run_dirs = sorted(
                r for r in vdir.iterdir()
                if r.is_dir() and r.name.startswith("run")
            )
            for rdir in run_dirs:
                total_runs += 1
                log_path = rdir / "run_log.json"
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
                    except Exception:
                        pass

        avg_kf = mean(kf_list) if kf_list else 0
        avg_desc = mean(desc_list) if desc_list else 0
        avg_drop = mean(drop_list) if drop_list else 0
        lines.append(
            f"| {model_id} | {len(video_dirs)} | {total_runs} "
            f"| {avg_kf:.1f} | {avg_desc:.1f} | {avg_drop:.1f} |"
        )

    lines.extend([
        "",
        "Frames exceeding the `expiry_time_ms` in KeyFrameQueue are discarded.",
        "High drop rates suggest the model cannot keep up with the capture rate.",
    ])

    (vu_dir / "comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    lg.info("video_understanding comparison report written")


def generate_charts(session_dir: Path, models: list[str]) -> None:
    from toolkit.visualization import (
        plot_metric_ranking,
        plot_scenario_comparison,
        plot_summary_table,
        plot_video_comparison,
    )

    charts_dir = session_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    speed_dir = session_dir / "benchmark_speed"

    tested_models: list[str] = []
    model_ttfts: list[float] = []
    model_tps: list[float] = []
    model_totals: list[float] = []
    model_tokens: list[float] = []
    model_vram: list[int] = []

    for model_id in models:
        short = model_short_name(model_id)
        csv_path = speed_dir / short / "raw_data.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        tested_models.append(model_id)
        model_ttfts.append(mean(float(r["ttft_s"]) for r in rows))
        model_tps.append(mean(float(r["throughput_tps"]) for r in rows))
        model_totals.append(mean(float(r["total_time_s"]) for r in rows))
        model_tokens.append(mean(float(r["output_tokens"]) for r in rows))
        model_vram.append(int(rows[0].get("vram_mb", 0)))

    if not tested_models:
        lg.warning("No speed data found, skipping charts")
        return

    plot_metric_ranking(
        tested_models, model_ttfts, "TTFT (s)",
        "Average TTFT Ranking (lower is better)",
        charts_dir / "ttft_ranking.png",
        higher_is_better=False, unit="s",
    )
    plot_metric_ranking(
        tested_models, model_tps, "Throughput (tok/s)",
        "Average Throughput Ranking (higher is better)",
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
            "TTFT (s)", "Per-Scenario TTFT Comparison",
            charts_dir / "scenario_comparison.png",
        )

    headers = [
        "Model", "TTFT (s)", "Throughput", "Total (s)", "Tokens", "VRAM (MB)",
    ]
    table_rows = []
    for i, m in enumerate(tested_models):
        short = m.split("/")[-1] if "/" in m else m
        table_rows.append([
            short, f"{model_ttfts[i]:.3f}", f"{model_tps[i]:.1f}",
            f"{model_totals[i]:.3f}", f"{model_tokens[i]:.0f}",
            str(model_vram[i]),
        ])
    plot_summary_table(
        headers, table_rows, "Benchmark Speed Summary",
        charts_dir / "summary_table.png",
        highlight_col=1, highlight_best="min",
    )

    # Video understanding charts
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
            for rdir in vdir.iterdir():
                if not rdir.is_dir():
                    continue
                lp = rdir / "run_log.json"
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
            "Video Understanding Comparison",
            charts_dir / "video_comparison.png",
        )

    lg.info("Charts generated: {}", charts_dir)


def generate_final_report(
    session_dir: Path, config: dict, models: list[str],
    incidents: list[dict],
) -> None:
    lines = [
        "# Multi-Model Comprehensive Benchmark Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gpu = meta.get("gpu", {})
        completed = meta.get("completed_models", [])
        failed = meta.get("failed_models", [])
        lines.extend([
            "## Environment",
            "",
            f"- **GPU**: {gpu.get('name', 'N/A')} ({gpu.get('memory_total_mb', 0)} MB)",
            f"- **Python**: {meta.get('python_version', 'N/A').split()[0]}",
            f"- **Git**: {meta.get('git_hash', 'N/A')}",
            f"- **Start**: {meta.get('start_time', 'N/A')}",
            f"- **End**: {meta.get('end_time', 'N/A')}",
            "",
        ])
    else:
        completed, failed = [], []

    lines.extend([
        "## Model Inventory",
        "",
        "| # | Model ID | Short Name | Status |",
        "|---|----------|------------|--------|",
    ])
    for i, m in enumerate(models, 1):
        short = model_short_name(m)
        if m in completed:
            status = "Completed"
        elif m in failed:
            status = "Failed"
        else:
            status = "Excluded (pre-flight)"
        lines.append(f"| {i} | {m} | {short} | {status} |")

    lines.extend([
        "",
        "> This benchmark does not cover community quantized models (unsloth GGUF/AWQ/GPTQ).",
        "",
    ])

    # Charts
    charts_dir = session_dir / "charts"
    if charts_dir.exists():
        lines.extend(["## Visualizations", ""])
        for fname, caption in [
            ("summary_table.png", "Summary Table"),
            ("ttft_ranking.png", "TTFT Ranking"),
            ("throughput_ranking.png", "Throughput Ranking"),
            ("scenario_comparison.png", "Per-Scenario TTFT"),
            ("video_comparison.png", "Video Understanding"),
        ]:
            if (charts_dir / fname).exists():
                lines.extend([f"### {caption}", "", f"![{caption}](charts/{fname})", ""])

    # Speed data
    speed_dir = session_dir / "benchmark_speed"
    speed_data = _collect_speed_data(speed_dir, completed) if speed_dir.exists() else {}
    if speed_data:
        lines.extend([
            "## Benchmark Speed Results",
            "",
            "| Model | TTFT (s) | Throughput (tok/s) | Total (s) | Max FPS | Safe FPS |",
            "|-------|:---:|:---:|:---:|:---:|:---:|",
        ])
        for m, d in speed_data.items():
            max_fps = 1.0 / d["ttft"] if d["ttft"] > 0 else 0
            lines.append(
                f"| {m.split('/')[-1]} | {d['ttft']:.3f} | {d['tps']:.1f} "
                f"| {d['total']:.3f} | {max_fps:.2f} | {max_fps * 0.7:.2f} |"
            )
        lines.extend([
            "",
            f"See [comparison_report.md](benchmark_speed/comparison_report.md) for details.",
            "",
        ])

    # Video understanding
    vu_cmp = session_dir / "video_understanding" / "comparison_report.md"
    if vu_cmp.exists():
        lines.extend([
            "## Video Understanding Results",
            "",
            f"See [comparison_report.md](video_understanding/comparison_report.md) for details.",
            "",
        ])

    # Research questions
    lines.extend([
        "## Research Questions",
        "",
        "### 1. Maximum theoretical FPS?",
        "",
        "Max FPS = 1 / TTFT. See the table above.",
    ])
    if speed_data:
        fastest = min(speed_data.items(), key=lambda x: x[1]["ttft"])
        lines.append(
            f"Best: **{fastest[0].split('/')[-1]}** with TTFT {fastest[1]['ttft']:.3f}s "
            f"-> {1.0 / fastest[1]['ttft']:.2f} FPS."
        )
    lines.extend([
        "",
        "### 2. Safe capture rate?",
        "",
        "Recommended: 70% of max FPS to account for network, queue, and frame-diff overhead.",
        "In practice, frame-diff filtering reduces VLM load significantly.",
        "",
        "### 3. How are unprocessed frames handled?",
        "",
        "KeyFrameQueue uses an expiry mechanism (default 10s). Expired frames are silently",
        "discarded without blocking the pipeline. High drop rates indicate the model is too",
        "slow -- reduce capture rate or use a faster model.",
        "",
        "## Recommendations",
        "",
    ])
    if speed_data:
        fastest = min(speed_data.items(), key=lambda x: x[1]["ttft"])
        highest_tp = max(speed_data.items(), key=lambda x: x[1]["tps"])
        lines.extend([
            f"- **Lowest latency**: {fastest[0].split('/')[-1]} (TTFT {fastest[1]['ttft']:.3f}s)",
            f"- **Highest throughput**: {highest_tp[0].split('/')[-1]} ({highest_tp[1]['tps']:.1f} tok/s)",
            "- **Real-time companion**: prioritize low-latency models, sample ~1 frame every 2-3s",
            "- **Offline analysis**: use high-throughput models for batch processing",
            "",
        ])

    if incidents:
        lines.extend(["## Incidents", ""])
        for inc in incidents:
            lines.append(f"- **{inc['model']}**: {inc.get('error', 'unknown')} ({inc.get('resolution', '')})")
        lines.append("")

    (session_dir / "final_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    lg.info("Final report written")


def generate_incident_report(
    session_dir: Path, incidents: list[dict],
) -> None:
    lines = [
        "# Incident Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if not incidents:
        lines.append("No incidents occurred during this experiment session.")
    else:
        lines.append(f"Total incidents: {len(incidents)}")
        lines.append("")
        for i, inc in enumerate(incidents, 1):
            lines.extend([
                f"## Incident {i}: {inc['model']}",
                "",
                f"- **Phase**: {inc.get('phase', 'unknown')}",
                f"- **Error**: {inc.get('error', 'N/A')}",
                f"- **Resolution**: {inc.get('resolution', 'N/A')}",
            ])
            if inc.get("stderr_snippet"):
                lines.extend([
                    "",
                    "**vLLM output (last 2000 chars)**:",
                    f"```\n{inc['stderr_snippet']}\n```",
                ])
            lines.append("")

    (session_dir / "incident_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    lg.info("Incident report written")


def generate_readme(session_dir: Path, config: dict, models: list[str]) -> None:
    from experiments.benchmark_speed.benchmark import SCENARIOS

    lines = [
        "# Multi-Model Comprehensive Benchmark",
        "",
        "This directory contains a complete multi-model benchmark session.",
        "",
        "## Purpose",
        "",
        "Systematically evaluate multiple VLM/LLM models for inference speed",
        "and video understanding capability in a game-scene AI companion context.",
        "",
        "## Models Tested",
        "",
    ]
    for i, m in enumerate(models, 1):
        lines.append(f"{i}. `{m}`")

    lines.extend([
        "",
        "## Experiments",
        "",
        "### 1. Benchmark Speed",
        "",
        "Measures TTFT, throughput, and total inference time across game screenshots.",
        "",
        "**Scenarios**:",
        "",
    ])
    for s in SCENARIOS:
        lines.append(f"- `{s.id}` ({s.name}): max_tokens={s.max_tokens}")

    lines.extend([
        "",
        "### 2. Video Understanding",
        "",
        "Full pipeline: PotPlayer -> window capture -> frame-diff -> vLLM -> DeepSeek summary.",
        "",
        "## Reproduction",
        "",
        "### Prerequisites",
        "",
        "- Windows 10/11 with WSL2",
        "- NVIDIA GPU (>= 12GB VRAM recommended)",
        "- Python >= 3.12, uv package manager",
        "- vLLM installed in WSL",
        "- PotPlayer (for video_understanding)",
        "",
        "### Steps",
        "",
        "```bash",
        "# 1. Install dependencies",
        "uv sync",
        "",
        "# 2. Create .env in project root",
        "# DEEPSEEK_API_KEY=your_key",
        "# DEEPSEEK_API_BASE_URL=https://api.deepseek.com",
        "# PLAYER_EXE_PATH=D:\\Programs\\PotPlayer\\PotPlayerMini64.exe",
        "",
        "# 3. Ensure models are cached in WSL (~/.cache/huggingface/hub/)",
        "",
        "# 4. Run the benchmark",
        "uv run run_benchmark.py",
        "",
        "# Optional flags:",
        "# --config my_config.toml",
        "# --models Qwen/Qwen3.5-2B Qwen/Qwen3.5-0.8B",
        "# --skip-speed   (only video_understanding)",
        "# --skip-video   (only benchmark_speed)",
        "```",
        "",
        "## Directory Structure",
        "",
        "```",
        ".",
        "├── config.toml               # Config snapshot",
        "├── meta.json                  # Environment metadata",
        "├── orchestrator.log           # Full log",
        "├── README.md                  # This file",
        "├── final_report.md            # Comprehensive report",
        "├── incident_report.md         # Startup issues & solutions",
        "├── charts/                    # Visualization PNGs",
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
    lg.info("README written")


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model benchmark")
    parser.add_argument(
        "--config", default="config_benchmark.toml",
        help="Config file (default: config_benchmark.toml)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Override model list from config",
    )
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip pre-flight validation (assume all models work)",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        lg.error("Config not found: {}", config_path)
        sys.exit(1)

    config = load_config(config_path)
    if args.skip_speed:
        config.setdefault("benchmark_speed", {})["enabled"] = False
    if args.skip_video:
        config.setdefault("video_understanding", {})["enabled"] = False

    models = args.models or config.get("models", {}).get("list", [])
    if not models:
        lg.error("No models specified")
        sys.exit(1)

    base_url = config.get("vllm", {}).get("base_url", "http://localhost:8000/v1")
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
    lg.info("Multi-Model Benchmark Starting")
    lg.info("Session: {}", session_dir)
    lg.info("Models ({}): {}", len(models), models)
    lg.info("=" * 70)

    # ── Phase 0: Pre-flight ──
    incidents: list[dict] = []
    if not args.skip_preflight:
        lg.info("=" * 70)
        lg.info("Phase 0: Pre-flight Model Validation")
        lg.info("=" * 70)
        valid_models, incidents = run_preflight(
            models, base_url, startup_timeout, poll_interval,
        )
        lg.info(
            "Pre-flight complete: {}/{} models passed",
            len(valid_models), len(models),
        )
        if incidents:
            lg.warning("Failed models: {}", [i["model"] for i in incidents])
        models = valid_models
    else:
        lg.info("Pre-flight skipped (--skip-preflight)")

    if not models:
        lg.error("No valid models after pre-flight, aborting")
        generate_incident_report(session_dir, incidents)
        sys.exit(1)

    # ── Phase 1: Experiments ──
    speed_dir = session_dir / "benchmark_speed"
    video_dir = session_dir / "video_understanding"
    speed_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    completed_models: list[str] = []
    failed_models: list[str] = []

    for idx, model_id in enumerate(models, 1):
        lg.info("━" * 70)
        lg.info("[{}/{}] Model: {}", idx, len(models), model_id)
        lg.info("━" * 70)

        proc = None
        try:
            proc = start_vllm(model_id)

            lg.info("Waiting for vLLM (timeout: {}s) ...", startup_timeout)
            detected = wait_for_vllm_ready(
                base_url=base_url,
                timeout_s=startup_timeout,
                poll_interval_s=poll_interval,
            )
            lg.info("vLLM ready: {}", detected)

            if config.get("benchmark_speed", {}).get("enabled", True):
                lg.info("── Running benchmark_speed ──")
                run_benchmark_speed_for_model(base_url, speed_dir, config)

            if config.get("video_understanding", {}).get("enabled", True):
                lg.info("── Running video_understanding ──")
                run_video_understanding_for_model(video_dir, config)

            completed_models.append(model_id)
            lg.info("Model {} completed", model_id)

        except Exception:
            lg.exception("Model {} failed", model_id)
            failed_models.append(model_id)
            incidents.append({
                "model": model_id,
                "phase": "experiment",
                "error": "see orchestrator.log for traceback",
                "resolution": "skipped",
            })

        finally:
            lg.info("Stopping vLLM ...")
            stop_vllm(proc)
            time.sleep(5)

    # ── Phase 2: Reports ──
    update_meta(
        session_dir,
        end_time=datetime.now().isoformat(),
        completed_models=completed_models,
        failed_models=failed_models,
    )

    lg.info("=" * 70)
    lg.info("Generating reports and charts ...")

    generate_speed_comparison(session_dir, completed_models)
    generate_video_comparison(session_dir, completed_models)
    generate_charts(session_dir, completed_models)
    generate_final_report(session_dir, config, models, incidents)
    generate_incident_report(session_dir, incidents)
    generate_readme(session_dir, config, models)

    lg.info("=" * 70)
    lg.info("Benchmark complete")
    lg.info(
        "Passed: {}/{} | Failed: {}",
        len(completed_models), len(models), len(failed_models),
    )
    lg.info("Session: {}", session_dir)
    if failed_models:
        lg.warning("Failed: {}", failed_models)
    lg.info("=" * 70)


if __name__ == "__main__":
    main()
