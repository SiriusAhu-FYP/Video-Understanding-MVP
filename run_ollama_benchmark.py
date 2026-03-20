"""Ollama Qwen3-VL-2B 基准测试编排器。

与 run_benchmark.py 的 vLLM 编排器功能对等，但面向已在本地运行的 Ollama 服务。
不需要模型生命周期管理 — Ollama 自行管理模型的加载/卸载。

用法:
    uv run run_ollama_benchmark.py
    uv run run_ollama_benchmark.py --runs 5
    uv run run_ollama_benchmark.py --skip-video
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from loguru import logger as lg

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_IMAGES_DIR = PROJECT_ROOT / "assets" / "images"

sys.path.insert(0, str(PROJECT_ROOT))

from ahu_paimon_toolkit.utils.gpu import get_gpu_info
from ahu_paimon_toolkit.vlm.model_utils import model_short_name

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_NATIVE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-vl:2b"
OLLAMA_SHORT = "qwen3-vl_2b"

# Qwen3 models use "thinking" mode by default in Ollama.
# Thinking tokens consume the max_tokens budget. The OpenAI-compatible API
# does NOT support `think: false`, but the native Ollama API does.
# Strategy: use native API with think=false for quality/video benchmarks,
# and document the thinking overhead impact on speed benchmarks.


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


def preflight_check() -> bool:
    """Verify Ollama is reachable and the model responds to vision queries."""
    import httpx

    lg.info("预飞行检查: 验证 Ollama 服务 ({}) ...", OLLAMA_BASE_URL)

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/models")
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            if OLLAMA_MODEL not in models:
                lg.error("Ollama 中未找到模型 {}，可用模型: {}", OLLAMA_MODEL, models)
                return False
            lg.info("模型 {} 可用 ✓", OLLAMA_MODEL)
    except Exception as e:
        lg.error("无法连接 Ollama: {}. 请确保 ollama serve 正在运行", e)
        return False

    lg.info("预飞行: 文本推理检查 (native API, think=false) ...")
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{OLLAMA_NATIVE_URL}/api/chat", json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "stream": False,
                "think": False,
            })
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
            lg.info("文本检查通过: '{}'", text[:60])
    except Exception as e:
        lg.error("文本推理失败: {}", e)
        return False

    lg.info("预飞行: 视觉推理检查 (native API, think=false) ...")
    try:
        from ahu_paimon_toolkit.utils.image import load_images
        all_images = load_images(ASSETS_IMAGES_DIR)
        if not all_images:
            lg.warning("无测试图片，跳过视觉检查")
            return True
        _, b64, _, _ = all_images[0]
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{OLLAMA_NATIVE_URL}/api/chat", json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Describe this image briefly.", "images": [b64]}],
                "stream": False,
                "think": False,
            })
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
            if len(text) > 5:
                lg.info("视觉检查通过: '{}'", text[:80])
                return True
            lg.warning("视觉检查: 响应过短 '{}'", text)
            return False
    except Exception as e:
        lg.error("视觉推理失败: {}", e)
        return False


def run_speed(output_dir: Path, runs: int, warmup: int) -> bool:
    """Run speed benchmark with inflated max_tokens for Ollama's thinking overhead."""
    try:
        from experiments.benchmark_speed import benchmark as speed_mod

        THINKING_BUDGET = 1536
        original_scenarios = speed_mod.SCENARIOS
        patched = []
        for s in original_scenarios:
            patched.append(speed_mod.Scenario(
                id=s.id,
                name=s.name,
                prompt=s.prompt,
                max_tokens=s.max_tokens + THINKING_BUDGET,
            ))
        speed_mod.SCENARIOS = patched
        speed_mod.SCENARIO_MAP = {s.id: s for s in patched}

        try:
            results, report_dir = speed_mod.run_benchmark(
                base_url=OLLAMA_BASE_URL,
                num_runs=runs,
                warmup_runs=warmup,
                output_dir=output_dir,
            )
            lg.info("benchmark_speed 完成: {} 条结果 -> {}", len(results), report_dir)
            return True
        finally:
            speed_mod.SCENARIOS = original_scenarios
            speed_mod.SCENARIO_MAP = {s.id: s for s in original_scenarios}
    except Exception:
        lg.exception("benchmark_speed 执行失败")
        return False


def run_quality(output_dir: Path, runs: int) -> bool:
    """Run quality benchmark. Patches max_tokens for Ollama thinking overhead."""
    import os
    from dotenv import load_dotenv
    from ahu_paimon_toolkit.evaluation.judge import LLMJudge
    from experiments.benchmark_quality import benchmark as qual_mod

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_base = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        lg.warning("DEEPSEEK_API_KEY 未设置，跳过 benchmark_quality")
        return False

    THINKING_BUDGET = 1536

    original_fn = qual_mod.run_single_evaluation

    async def patched_eval(**kwargs):
        kwargs["max_tokens"] = kwargs.get("max_tokens", 512) + THINKING_BUDGET
        return await original_fn(**kwargs)

    qual_mod.run_single_evaluation = patched_eval

    try:
        judge = LLMJudge(api_key=api_key, api_base_url=api_base)
        assets = qual_mod.load_asset_jsons(ASSETS_IMAGES_DIR)
        scores = asyncio.run(qual_mod.run_benchmark_quality(
            vlm_base_url=OLLAMA_BASE_URL,
            vlm_model=OLLAMA_MODEL,
            judge=judge,
            assets=assets,
            output_dir=output_dir,
            runs=runs,
            judge_delay_s=1.0,
        ))
        lg.info("benchmark_quality 完成: {} 条评分", len(scores))
        return True
    except Exception:
        lg.exception("benchmark_quality 执行失败")
        return False
    finally:
        qual_mod.run_single_evaluation = original_fn


def run_video(output_dir: Path, runs: int) -> bool:
    try:
        from experiments.video_understanding.run_experiment import run_experiment
        results = asyncio.run(run_experiment(
            num_runs=runs,
            output_dir=output_dir,
            base_url=OLLAMA_BASE_URL,
        ))
        total = sum(len(v) for v in results.values())
        lg.info("video_understanding 完成: {} 个视频, {} 次运行", len(results), total)
        return True
    except Exception:
        lg.exception("video_understanding 执行失败")
        return False


def detect_outliers(values: list[float], threshold: float = 2.0) -> list[int]:
    """Return indices of values that deviate more than `threshold` stdevs from mean."""
    if len(values) < 3:
        return []
    m = mean(values)
    s = stdev(values)
    if s < 1e-9:
        return []
    return [i for i, v in enumerate(values) if abs(v - m) > threshold * s]


def check_speed_outliers(speed_dir: Path) -> list[dict]:
    """Check speed raw_data.csv for outlier runs. Returns list of flagged entries."""
    csv_path = speed_dir / OLLAMA_SHORT / "raw_data.csv"
    if not csv_path.exists():
        return []

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        return []

    ttfts = [float(r["ttft_s"]) for r in rows]
    outlier_indices = detect_outliers(ttfts)

    flagged = []
    for idx in outlier_indices:
        flagged.append({
            "type": "speed",
            "run_idx": rows[idx].get("run_idx", idx),
            "scenario": rows[idx].get("scenario_id", "?"),
            "metric": "ttft_s",
            "value": ttfts[idx],
            "mean": mean(ttfts),
            "stdev": stdev(ttfts),
        })

    return flagged


def check_quality_outliers(quality_dir: Path) -> list[dict]:
    """Check quality scores.csv for outlier scores."""
    csv_path = quality_dir / OLLAMA_SHORT / "scores.csv"
    if not csv_path.exists():
        return []

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        return []

    scores = [float(r["total_score"]) for r in rows]
    outlier_indices = detect_outliers(scores, threshold=1.5)

    flagged = []
    for idx in outlier_indices:
        flagged.append({
            "type": "quality",
            "asset": rows[idx].get("asset_id", "?"),
            "prompt_mode": rows[idx].get("prompt_mode", "?"),
            "metric": "total_score",
            "value": scores[idx],
            "mean": mean(scores),
            "stdev": stdev(scores),
        })

    return flagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama Qwen3-VL-2B 基准测试")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    output_base = PROJECT_ROOT / "results" / "v3-ollama"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = output_base / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    log_path = session_dir / "orchestrator.log"
    lg.add(str(log_path), level="DEBUG", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}")

    meta = {
        "backend": "ollama",
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "git_hash": get_git_hash(),
        "gpu": get_gpu_info(),
        "python_version": sys.version,
        "start_time": datetime.now().isoformat(),
        "runs": args.runs,
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    lg.info("=" * 70)
    lg.info("Ollama 基准测试启动")
    lg.info("模型: {} (via {})", OLLAMA_MODEL, OLLAMA_BASE_URL)
    lg.info("会话目录: {}", session_dir)
    lg.info("运行次数: {}", args.runs)
    lg.info("=" * 70)

    if not args.skip_preflight:
        if not preflight_check():
            lg.error("预飞行检查失败，中止")
            sys.exit(1)
    else:
        lg.info("预飞行已跳过")

    speed_dir = session_dir / "benchmark_speed"
    quality_dir = session_dir / "benchmark_quality"
    video_dir = session_dir / "video_understanding"
    speed_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    ok_speed = ok_quality = ok_video = False

    if not args.skip_speed:
        lg.info("━" * 60)
        lg.info("实验 1/3: benchmark_speed")
        lg.info("━" * 60)
        ok_speed = run_speed(speed_dir, args.runs, args.warmup)

    if not args.skip_quality:
        lg.info("━" * 60)
        lg.info("实验 2/3: benchmark_quality")
        lg.info("━" * 60)
        ok_quality = run_quality(quality_dir, args.runs)

    if not args.skip_video:
        lg.info("━" * 60)
        lg.info("实验 3/3: video_understanding")
        lg.info("━" * 60)
        ok_video = run_video(video_dir, args.runs)

    # Outlier detection
    lg.info("=" * 70)
    lg.info("异常值检测")
    lg.info("=" * 70)

    outliers = []
    if ok_speed:
        outliers.extend(check_speed_outliers(speed_dir))
    if ok_quality:
        outliers.extend(check_quality_outliers(quality_dir))

    if outliers:
        lg.warning("检测到 {} 个异常值:", len(outliers))
        for o in outliers:
            lg.warning(
                "  [{type}] {metric}={value:.4f} (mean={mean:.4f}, stdev={stdev:.4f})",
                **o,
            )
        outlier_path = session_dir / "outliers.json"
        outlier_path.write_text(
            json.dumps(outliers, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        lg.info("异常值详情已保存: {}", outlier_path)
    else:
        lg.info("未检测到明显异常值 ✓")

    # Update meta
    meta["end_time"] = datetime.now().isoformat()
    meta["experiments_completed"] = {
        "benchmark_speed": ok_speed,
        "benchmark_quality": ok_quality,
        "video_understanding": ok_video,
    }
    meta["outlier_count"] = len(outliers)
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    lg.info("=" * 70)
    lg.info("Ollama 基准测试完成")
    lg.info("速度: {} | 质量: {} | 视频: {}",
            "✓" if ok_speed else "✗",
            "✓" if ok_quality else "✗",
            "✓" if ok_video else "✗")
    lg.info("异常值: {}", len(outliers))
    lg.info("会话目录: {}", session_dir)
    lg.info("=" * 70)


if __name__ == "__main__":
    main()
