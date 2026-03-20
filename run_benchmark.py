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
import tomllib
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from loguru import logger as lg

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_IMAGES_DIR = PROJECT_ROOT / "assets" / "images"

sys.path.insert(0, str(PROJECT_ROOT))

from ahu_paimon_toolkit.utils.gpu import get_gpu_info
from ahu_paimon_toolkit.vlm.model_utils import model_short_name, wait_for_vllm_ready
from utils.reporting import generate_report_from_template
from utils.vllm_manager import (
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
    general = config.get("general", {})
    ts_format = general.get("timestamp_format", "%Y-%m-%d_%H-%M-%S")
    timestamp = datetime.now().strftime(ts_format)
    output_dir = general.get("output_dir", "results/v3")
    session_dir = PROJECT_ROOT / output_dir / timestamp
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
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_meta(session_dir: Path, **kwargs: object) -> None:
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(kwargs)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
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

    from ahu_paimon_toolkit.utils.image import load_images

    images = load_images(ASSETS_IMAGES_DIR)
    if not images:
        lg.warning("无测试图片，跳过视觉检查")
        return True

    fname, b64, mime, _ = images[0]
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                        },
                    },
                ],
            }
        ],
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
                text = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
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
    *,
    candidates: list[float] | None = None,
    enforce_eager: bool = True,
) -> float | None:
    """探测模型所需的最小 gpu_memory_utilization。

    从最小值开始递增，使用 enforce-eager 跳过 CUDA graph 编译以加速探测。
    Returns the first successful value, or None if all fail.
    """
    if candidates is None:
        candidates = [0.45, 0.50, 0.55, 0.60]

    extra: dict = {}
    if enforce_eager:
        extra["enforce-eager"] = True

    for util in candidates:
        lg.info("  尝试 gpu_memory_utilization={:.2f} ...", util)
        proc = None
        try:
            proc = start_vllm(
                model_id,
                extra_args=extra or None,
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
    preflight_cfg: dict | None = None,
) -> tuple[list[str], list[dict], dict[str, float]]:
    """验证每个模型能否在 vLLM 上正常启动并进行视觉推理。

    Returns: (valid_models, incidents, gpu_util_map)
    """
    pf = preflight_cfg or {}
    do_health = pf.get("health_check", True)
    do_vision = pf.get("vision_check", True)
    gpu_probe_cfg = pf.get("gpu_probe", {})
    do_probe = gpu_probe_cfg.get("enabled", True)
    probe_candidates = gpu_probe_cfg.get("candidates", [0.45, 0.50, 0.55, 0.60])
    probe_timeout = gpu_probe_cfg.get("timeout_s", 120)
    probe_eager = gpu_probe_cfg.get("enforce_eager", True)

    valid: list[str] = []
    incidents: list[dict] = []
    gpu_util_map: dict[str, float] = {}

    for idx, model_id in enumerate(models, 1):
        lg.info("━" * 60)
        lg.info("[预飞行 {}/{}] {}", idx, len(models), model_id)

        optimal_util = 0.5
        if do_probe:
            lg.info("  开始 GPU 显存探测 (candidates={}) ...", probe_candidates)
            result = probe_gpu_memory(
                model_id,
                base_url,
                min(startup_timeout, probe_timeout),
                poll_interval,
                candidates=probe_candidates,
                enforce_eager=probe_eager,
            )
            if result is None:
                incidents.append({
                    "model": model_id,
                    "phase": "gpu_probe",
                    "error": f"所有 gpu_memory_utilization {probe_candidates} 均无法启动",
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

            ok_text = preflight_health_check(base_url, detected) if do_health else True
            ok_vision = preflight_vision_check(base_url, detected) if do_vision else True

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
                lg.warning(
                    "[未通过] {} - {}检查失败", model_id, "和".join(failed_parts)
                )

        except Exception as e:
            error_msg = str(e)
            from utils.vllm_manager import read_vllm_log

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


def _get_experiment_cfg(config: dict, experiment_name: str) -> dict:
    """Get per-experiment config from ``[experiments.<name>]``."""
    return config.get("experiments", {}).get(experiment_name, {})


def _is_experiment_enabled(config: dict, experiment_name: str) -> bool:
    """Check if an experiment is enabled via its ``enabled`` field."""
    exp_cfg = config.get("experiments", {}).get(experiment_name, {})
    return exp_cfg.get("enabled", False)


def run_benchmark_speed_for_model(
    base_url: str,
    output_dir: Path,
    config: dict,
) -> bool:
    if not _is_experiment_enabled(config, "benchmark_speed"):
        lg.info("benchmark_speed 不在 experiments.run 列表中，跳过")
        return True

    bs_cfg = _get_experiment_cfg(config, "benchmark_speed")

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
    base_url: str | None = None,
) -> bool:
    if not _is_experiment_enabled(config, "video_understanding"):
        lg.info("video_understanding 不在 experiments.run 列表中，跳过")
        return True

    vu_cfg = _get_experiment_cfg(config, "video_understanding")

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
            len(results),
            total_runs,
        )
        return True
    except Exception:
        lg.exception("video_understanding 执行失败")
        return False


def run_benchmark_quality_for_model(
    base_url: str,
    output_dir: Path,
    config: dict,
    model_id: str,
) -> bool:
    if not _is_experiment_enabled(config, "benchmark_quality"):
        lg.info("benchmark_quality 不在 experiments.run 列表中，跳过")
        return True

    bq_cfg = _get_experiment_cfg(config, "benchmark_quality")

    try:
        import os

        from dotenv import load_dotenv

        from ahu_paimon_toolkit.evaluation.judge import LLMJudge
        from experiments.benchmark_quality.benchmark import (
            load_asset_jsons,
            run_benchmark_quality,
        )

        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        api_base = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")

        if not api_key:
            lg.warning("DEEPSEEK_API_KEY 未设置，跳过 benchmark_quality")
            return False

        judge = LLMJudge(api_key=api_key, api_base_url=api_base)
        assets = load_asset_jsons(ASSETS_IMAGES_DIR)
        runs = bq_cfg.get("runs", 5)

        scores = asyncio.run(
            run_benchmark_quality(
                vlm_base_url=base_url,
                vlm_model=model_id,
                judge=judge,
                assets=assets,
                output_dir=output_dir,
                runs=runs,
                judge_delay_s=bq_cfg.get("judge_delay_s", 1.0),
            )
        )
        lg.info("benchmark_quality 完成: {} 条评分", len(scores))
        return True
    except Exception:
        lg.exception("benchmark_quality 执行失败")
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


def _collect_quality_summary(quality_dir: Path, models: list[str]) -> list[str]:
    """Collect quality scores from benchmark_quality results."""
    if not quality_dir.exists():
        return []

    rows = [
        "| 模型 | 均分 (0-10) | 标准差 | 评估次数 |",
        "|------|:---:|:---:|:---:|",
    ]

    for model_id in models:
        short = model_short_name(model_id)
        scores_csv = quality_dir / short / "scores.csv"
        if not scores_csv.exists():
            rows.append(f"| {model_id} | - | - | 0 |")
            continue
        try:
            with open(scores_csv, encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            if not csv_rows:
                rows.append(f"| {model_id} | - | - | 0 |")
                continue
            scores = [float(r["total_score"]) for r in csv_rows]
            avg = mean(scores)
            sd = stdev(scores) if len(scores) > 1 else 0
            rows.append(f"| {model_id} | {avg:.2f} | {sd:.2f} | {len(scores)} |")
        except Exception:
            rows.append(f"| {model_id} | - | - | 0 |")

    return rows


def generate_speed_comparison(session_dir: Path, models: list[str]) -> None:
    speed_dir = session_dir / "benchmark_speed"
    if not speed_dir.exists():
        return

    speed_data = _collect_speed_data(speed_dir, models)

    speed_rows = []
    speed_rows.append("| 模型 | 平均TTFT (s) | 平均吞吐量 (tok/s) | 平均总耗时 (s) | 平均Token数 | 运行次数 |")
    speed_rows.append("|------|:---:|:---:|:---:|:---:|:---:|")
    for model_id in models:
        d = speed_data.get(model_id)
        if not d:
            speed_rows.append(f"| {model_id} | - | - | - | - | 0 |")
            continue
        speed_rows.append(
            f"| {model_id} | {d['ttft']:.3f}±{d['ttft_std']:.3f} | {d['tps']:.1f}±{d['tps_std']:.1f} "
            f"| {d['total']:.3f} | {d['tokens']:.0f} | {d['runs']} |"
        )

    fps_rows = []
    fps_rows.append("| 模型 | 平均TTFT (s) | 理论最大FPS | 安全FPS (70%) |")
    fps_rows.append("|------|:---:|:---:|:---:|")
    for model_id in models:
        d = speed_data.get(model_id)
        if not d:
            continue
        max_fps = 1.0 / d["ttft"] if d["ttft"] > 0 else 0
        fps_rows.append(f"| {model_id} | {d['ttft']:.3f} | {max_fps:.1f} | {max_fps * 0.7:.1f} |")

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "speed_table": "\n".join(speed_rows),
        "fps_table": "\n".join(fps_rows),
    }

    generate_report_from_template(
        "speed_comparison.md",
        data,
        speed_dir / "comparison_report.md",
        use_deepseek=bool(speed_data),
    )
    lg.info("benchmark_speed 对比报告已生成")


def _collect_video_stats(
    vu_dir: Path, models: list[str],
) -> tuple[list[str], dict[str, dict]]:
    """Collect per-model video understanding stats. Returns (table_lines, model_stats)."""
    rows = []
    rows.append("| 模型 | 视频数 | 运行次数 | 平均关键帧 | 平均描述数 | 平均丢弃帧 |")
    rows.append("|------|:---:|:---:|:---:|:---:|:---:|")

    model_stats: dict[str, dict] = {}

    for model_id in models:
        short = model_short_name(model_id)
        model_dir = vu_dir / short
        if not model_dir.exists():
            rows.append(f"| {model_id} | 0 | 0 | - | - | - |")
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
                        len([f for f in ld.get("frames", []) if f.get("vlm_response")])
                    )
                    drop_list.append(stats.get("total_dropped", 0))
                    total_runs += 1
                except Exception:
                    pass

        avg_kf = mean(kf_list) if kf_list else 0
        avg_desc = mean(desc_list) if desc_list else 0
        avg_drop = mean(drop_list) if drop_list else 0
        rows.append(
            f"| {model_id} | - | {total_runs} "
            f"| {avg_kf:.1f} | {avg_desc:.1f} | {avg_drop:.1f} |"
        )
        model_stats[model_id] = {
            "avg_kf": avg_kf,
            "avg_desc": avg_desc,
            "avg_drop": avg_drop,
            "runs": total_runs,
        }

    return rows, model_stats


def generate_video_comparison(session_dir: Path, models: list[str]) -> None:
    vu_dir = session_dir / "video_understanding"
    if not vu_dir.exists():
        return

    rows, model_stats = _collect_video_stats(vu_dir, models)

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_table": "\n".join(rows),
    }

    generate_report_from_template(
        "video_comparison.md",
        data,
        vu_dir / "comparison_report.md",
        use_deepseek=bool(model_stats),
    )
    lg.info("video_understanding 对比报告已生成")


def generate_quality_comparison(session_dir: Path, models: list[str]) -> None:
    quality_dir = session_dir / "benchmark_quality"
    if not quality_dir.exists():
        return

    overview_rows = _collect_quality_summary(quality_dir, models)
    if len(overview_rows) <= 2:
        return

    dimension_rows = [
        "| 模型 | 核心理解 | 关键信息覆盖 | 任务完成度 | 助手价值 | 幻觉控制 |",
        "|------|:---:|:---:|:---:|:---:|:---:|",
    ]
    prompt_rows = [
        "| 模型 | A_description | B_assistant |",
        "|------|:---:|:---:|",
    ]

    for model_id in models:
        short = model_short_name(model_id)
        scores_csv = quality_dir / short / "scores.csv"
        if not scores_csv.exists():
            continue
        try:
            with open(scores_csv, encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            if not csv_rows:
                continue

            dim_names = [
                k for k in csv_rows[0].keys()
                if k not in ("asset_id", "model_id", "prompt_mode", "total_score", "max_score")
            ]
            dim_avgs = []
            for d in dim_names:
                vals = [float(r[d]) for r in csv_rows if r.get(d)]
                dim_avgs.append(f"{mean(vals):.2f}" if vals else "-")

            if dim_avgs:
                dimension_rows.append(f"| {short} | {' | '.join(dim_avgs)} |")

            for mode in ("A_description", "B_assistant"):
                mode_scores = [float(r["total_score"]) for r in csv_rows if r.get("prompt_mode") == mode]
                if mode == "A_description":
                    a_avg = f"{mean(mode_scores):.2f}" if mode_scores else "-"
                else:
                    b_avg = f"{mean(mode_scores):.2f}" if mode_scores else "-"
            prompt_rows.append(f"| {short} | {a_avg} | {b_avg} |")
        except Exception:
            pass

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quality_table": "\n".join(overview_rows),
        "dimension_scores": "\n".join(dimension_rows) if len(dimension_rows) > 2 else "",
        "prompt_mode_scores": "\n".join(prompt_rows) if len(prompt_rows) > 2 else "",
    }

    generate_report_from_template(
        "quality_comparison.md",
        data,
        quality_dir / "comparison_report.md",
        use_deepseek=True,
    )
    lg.info("benchmark_quality 对比报告已生成")


def generate_charts(
    session_dir: Path,
    models: list[str],
    gpu_util_map: dict[str, float] | None = None,
) -> None:
    from ahu_paimon_toolkit.utils.visualization import (
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
        tested_models,
        model_ttfts,
        "TTFT (s)",
        "平均 TTFT 排名（越低越好）",
        charts_dir / "ttft_ranking.png",
        higher_is_better=False,
        unit="s",
    )
    plot_metric_ranking(
        tested_models,
        model_tps,
        "吞吐量 (tok/s)",
        "平均吞吐量排名（越高越好）",
        charts_dir / "throughput_ranking.png",
        higher_is_better=True,
        unit=" tok/s",
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
            by_scenario.setdefault(r["scenario_id"], []).append(float(r["ttft_s"]))
        if i == 0:
            scenario_names = sorted(by_scenario.keys())
        for sname in scenario_names:
            vals = by_scenario.get(sname, [])
            scenario_matrix[i].append(mean(vals) if vals else 0)

    if scenario_names:
        plot_scenario_comparison(
            scenario_names,
            tested_models,
            scenario_matrix,
            "TTFT (s)",
            "各场景 TTFT 对比",
            charts_dir / "scenario_comparison.png",
        )

    headers = ["模型", "TTFT (s)", "吞吐量", "总耗时 (s)", "Token数", "VRAM (MB)"]
    table_rows = []
    for i, m in enumerate(tested_models):
        short = m.split("/")[-1] if "/" in m else m
        table_rows.append([
            short,
            f"{model_ttfts[i]:.3f}",
            f"{model_tps[i]:.1f}",
            f"{model_totals[i]:.3f}",
            f"{model_tokens[i]:.0f}",
            str(model_vram[i]),
        ])
    plot_summary_table(
        headers,
        table_rows,
        "推理速度基准测试汇总",
        charts_dir / "summary_table.png",
        highlight_col=1,
        highlight_best="min",
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
                        len([f for f in ld.get("frames", []) if f.get("vlm_response")])
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
            vu_models,
            vu_kf,
            vu_dropped,
            vu_descs,
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
    session_dir: Path,
    config: dict,
    models: list[str],
    incidents: list[dict],
    gpu_util_map: dict[str, float] | None = None,
) -> None:
    meta_path = session_dir / "meta.json"
    completed: list[str] = []
    failed: list[str] = []
    env_lines: list[str] = []

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gpu = meta.get("gpu", {})
        completed = meta.get("completed_models", [])
        failed = meta.get("failed_models", [])
        env_lines = [
            f"- **GPU**: {gpu.get('name', 'N/A')} ({gpu.get('memory_total_mb', 0)} MB)",
            f"- **Python**: {meta.get('python_version', 'N/A').split()[0]}",
            f"- **Git**: {meta.get('git_hash', 'N/A')}",
            f"- **开始时间**: {meta.get('start_time', 'N/A')}",
            f"- **结束时间**: {meta.get('end_time', 'N/A')}",
        ]

    model_table_rows = [
        "| # | 模型ID | 简称 | 状态 |",
        "|---|--------|------|------|",
    ]
    for i, m in enumerate(models, 1):
        short = model_short_name(m)
        if m in completed:
            status = "已完成"
        elif m in failed:
            status = "实验中失败"
        else:
            status = "预飞行排除"
        model_table_rows.append(f"| {i} | {m} | {short} | {status} |")

    gpu_rows: list[str] = []
    if gpu_util_map:
        gpu_rows.append("| 模型 | 最小 gpu_memory_utilization | 等效分配显存 |")
        gpu_rows.append("|------|:---:|:---:|")
        for m, util in gpu_util_map.items():
            alloc_mb = int(16376 * util)
            gpu_rows.append(f"| {m} | {util:.2f} | ~{alloc_mb} MB |")

    speed_dir = session_dir / "benchmark_speed"
    speed_data = _collect_speed_data(speed_dir, completed) if speed_dir.exists() else {}
    speed_rows: list[str] = []
    speed_chart_lines: list[str] = []
    charts_dir = session_dir / "charts"

    if speed_data:
        speed_rows.append("| 模型 | TTFT (s) | 吞吐量 (tok/s) | 总耗时 (s) | 最大FPS | 安全FPS |")
        speed_rows.append("|------|:---:|:---:|:---:|:---:|:---:|")
        for m, d in speed_data.items():
            max_fps = 1.0 / d["ttft"] if d["ttft"] > 0 else 0
            speed_rows.append(
                f"| {m.split('/')[-1]} | {d['ttft']:.3f} | {d['tps']:.1f} "
                f"| {d['total']:.3f} | {max_fps:.2f} | {max_fps * 0.7:.2f} |"
            )

    if charts_dir.exists():
        for fname, caption in [
            ("summary_table.png", "汇总表"),
            ("ttft_ranking.png", "TTFT 排名"),
            ("throughput_ranking.png", "吞吐量排名"),
            ("scenario_comparison.png", "各场景 TTFT 对比"),
        ]:
            if (charts_dir / fname).exists():
                speed_chart_lines.extend([f"### {caption}", "", f"![{caption}](charts/{fname})", ""])

    quality_dir = session_dir / "benchmark_quality"
    quality_rows = _collect_quality_summary(quality_dir, completed)

    vu_dir = session_dir / "video_understanding"
    video_rows: list[str] = []
    video_chart_lines: list[str] = []
    if vu_dir.exists():
        vr, _ = _collect_video_stats(vu_dir, completed)
        video_rows = vr
        if (charts_dir / "video_comparison.png").exists():
            video_chart_lines.append("![视频理解对比](charts/video_comparison.png)")

    incident_lines: list[str] = []
    for inc in incidents:
        incident_lines.append(
            f"- **{inc['model']}**（{inc.get('phase', '未知')}阶段）："
            f"{inc.get('error', '未知错误')} → {inc.get('resolution', '')}"
        )

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": "\n".join(env_lines),
        "model_table": "\n".join(model_table_rows),
        "gpu_memory_table": "\n".join(gpu_rows),
        "speed_results": "\n".join(speed_rows),
        "speed_charts": "\n".join(speed_chart_lines),
        "quality_results": "\n".join(quality_rows),
        "video_results": "\n".join(video_rows),
        "video_charts": "\n".join(video_chart_lines),
        "incidents": "\n".join(incident_lines) if incident_lines else "本次实验过程中未发生任何意外事件。",
    }

    generate_report_from_template(
        "final_report.md",
        data,
        session_dir / "final_report.md",
        use_deepseek=bool(speed_data),
    )
    lg.info("最终报告已生成")


def generate_incident_report(
    session_dir: Path,
    incidents: list[dict],
) -> None:
    if not incidents:
        summary = "本次实验过程中未发生任何意外事件。"
        details = ""
    else:
        summary = f"共记录 {len(incidents)} 个事件。"
        detail_lines: list[str] = []
        for i, inc in enumerate(incidents, 1):
            detail_lines.extend([
                f"## 事件 {i}: {inc['model']}",
                "",
                f"- **阶段**: {inc.get('phase', '未知')}",
                f"- **错误**: {inc.get('error', 'N/A')}",
                f"- **处置**: {inc.get('resolution', 'N/A')}",
            ])
            if inc.get("stderr_snippet"):
                detail_lines.extend([
                    "",
                    "**vLLM 输出（末尾片段）**:",
                    f"```\n{inc['stderr_snippet']}\n```",
                ])
            detail_lines.append("")
        details = "\n".join(detail_lines)

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "incident_summary": summary,
        "incident_details": details,
    }

    generate_report_from_template(
        "incident_report.md",
        data,
        session_dir / "incident_report.md",
        use_deepseek=bool(incidents),
    )
    lg.info("事件报告已生成")


def generate_readme(session_dir: Path, config: dict, models: list[str]) -> None:
    from experiments.benchmark_speed.benchmark import SCENARIOS

    model_list = "\n".join(f"{i}. `{m}`" for i, m in enumerate(models, 1))
    scenario_list = "\n".join(
        f"- `{s.id}` ({s.name}): max_tokens={s.max_tokens}" for s in SCENARIOS
    )

    data = {
        "model_list": model_list,
        "scenario_list": scenario_list,
    }

    generate_report_from_template(
        "readme.md",
        data,
        session_dir / "README.md",
        use_deepseek=False,
    )
    lg.info("README 已生成")


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="多模型基准测试")
    parser.add_argument(
        "--config",
        default="config_benchmark.toml",
        help="配置文件 (默认: config_benchmark.toml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="覆盖配置中的模型列表",
    )
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过预飞行验证（假设所有模型可用）",
    )
    parser.add_argument(
        "--skip-memory-probe",
        action="store_true",
        help="跳过 GPU 显存探测（使用默认 0.5）",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        lg.error("配置文件不存在: {}", config_path)
        sys.exit(1)

    config = load_config(config_path)

    all_experiments = ["benchmark_speed", "benchmark_quality", "video_understanding"]
    run_list: list[str] = [
        name for name in all_experiments
        if _is_experiment_enabled(config, name)
    ]
    if args.skip_speed and "benchmark_speed" in run_list:
        run_list.remove("benchmark_speed")
    if args.skip_quality and "benchmark_quality" in run_list:
        run_list.remove("benchmark_quality")
    if args.skip_video and "video_understanding" in run_list:
        run_list.remove("video_understanding")

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
        str(log_path),
        level="DEBUG",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    )

    lg.info("=" * 70)
    lg.info("多模型基准测试启动")
    lg.info("会话目录: {}", session_dir)
    lg.info("模型列表 ({}): {}", len(models), models)
    lg.info("实验列表: {}", run_list)
    lg.info("=" * 70)

    # ── Phase 0: 预飞行 ──
    incidents: list[dict] = []
    gpu_util_map: dict[str, float] = {}

    preflight_cfg = config.get("preflight", {})
    do_preflight = preflight_cfg.get("enabled", True) and not args.skip_preflight

    if do_preflight:
        if args.skip_memory_probe:
            preflight_cfg.setdefault("gpu_probe", {})["enabled"] = False

        lg.info("=" * 70)
        lg.info("Phase 0: 预飞行模型验证")
        lg.info("=" * 70)
        valid_models, incidents, gpu_util_map = run_preflight(
            models,
            base_url,
            startup_timeout,
            poll_interval,
            preflight_cfg=preflight_cfg,
        )
        lg.info(
            "预飞行完成: {}/{} 个模型通过",
            len(valid_models),
            len(models),
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
    quality_dir = session_dir / "benchmark_quality"
    video_dir = session_dir / "video_understanding"
    speed_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    completed_models: list[str] = []
    failed_models: list[str] = []

    for idx, model_id in enumerate(models, 1):
        lg.info("━" * 70)
        lg.info("[{}/{}] 模型: {}", idx, len(models), model_id)
        lg.info("━" * 70)

        proc = None
        try:
            gpu_util = gpu_util_map.get(
                model_id, config["vllm"]["gpu_memory_utilization"]
            )
            proc = start_vllm(model_id, gpu_memory_utilization=gpu_util)

            lg.info("等待 vLLM 就绪 (超时: {}s) ...", startup_timeout)
            detected = wait_for_vllm_ready(
                base_url=base_url,
                timeout_s=startup_timeout,
                poll_interval_s=poll_interval,
            )
            lg.info("vLLM 已就绪: {}", detected)

            for experiment in run_list:
                lg.info("── 运行 {} ──", experiment)
                if experiment == "benchmark_speed":
                    run_benchmark_speed_for_model(base_url, speed_dir, config)
                elif experiment == "benchmark_quality":
                    run_benchmark_quality_for_model(base_url, quality_dir, config, detected)
                elif experiment == "video_understanding":
                    run_video_understanding_for_model(video_dir, config, base_url)
                else:
                    lg.warning("未知实验类型: {}", experiment)

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
    generate_quality_comparison(session_dir, completed_models)
    generate_video_comparison(session_dir, completed_models)
    generate_charts(session_dir, completed_models, gpu_util_map)
    generate_final_report(session_dir, config, models, incidents, gpu_util_map)
    generate_incident_report(session_dir, incidents)
    generate_readme(session_dir, config, models)

    lg.info("=" * 70)
    lg.info("基准测试完成")
    lg.info(
        "通过: {}/{} | 失败: {}",
        len(completed_models),
        len(models),
        len(failed_models),
    )
    lg.info("会话目录: {}", session_dir)
    if failed_models:
        lg.warning("失败模型: {}", failed_models)
    lg.info("=" * 70)


if __name__ == "__main__":
    main()
