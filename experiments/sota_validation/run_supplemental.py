"""SOTA Supplemental Experiment Runner.

Analyzes previous SOTA experiment results to identify incomplete, unstable,
or poor-performance tasks. Runs only the needed supplemental/redo tasks,
merges old good data with new data into complete_data/, and produces an
updated QA report.

Usage:
    uv run experiments/sota_validation/run_supplemental.py
    uv run experiments/sota_validation/run_supplemental.py --old-session 2026-03-19_01-39-43
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev as _stdev
from typing import Any

from loguru import logger as lg

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.sota_validation.aggregate import (
    aggregate_results,
    generate_comparison_report,
    identify_problematic_samples,
)
from experiments.sota_validation.dmxapi_client import DMXAPIClient
from experiments.sota_validation.frame_extractor import extract_all_videos, load_frames_as_b64
from experiments.sota_validation.preflight import run_preflight
from experiments.sota_validation.run_image_experiment import (
    ASSETS_IMAGES_DIR,
    build_judge_prompt,
    load_asset_jsons,
    load_image_b64,
    parse_judge_response,
)
from experiments.sota_validation.visualization import (
    plot_dimension_heatmap,
    plot_latency_comparison,
    plot_per_sample_comparison,
    plot_score_ranking,
    plot_stability,
)

ASSETS_VIDEOS_DIR = _PROJECT_ROOT / "assets" / "videos"

STDEV_THRESHOLD = 1.2
MEAN_THRESHOLD = 6.0
TARGET_RUNS = 5


# ── Utilities ────────────────────────────────────────────────────


def _safe_stdev(vals: list[float]) -> float:
    return _stdev(vals) if len(vals) >= 2 else 0.0


def _read_model_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_video_jsons() -> list[dict]:
    jsons: list[dict] = []
    candidates: list[Path] = []
    for entry in sorted(ASSETS_VIDEOS_DIR.iterdir()):
        if entry.is_dir():
            candidates.extend(sorted(entry.glob("*.json")))
        elif entry.suffix == ".json":
            candidates.append(entry)
    for p in candidates:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if data:
            data["_json_path"] = str(p)
            jsons.append(data)
    return jsons


# ── Phase 1: Triage ──────────────────────────────────────────────


def _compute_cell_stats(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Group rows by (asset_id, prompt_mode); compute stats on valid scores."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    max_runs: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        key = (row["asset_id"], row["prompt_mode"])
        run_num = int(row.get("run", 0))
        max_runs[key] = max(max_runs[key], run_num)
        score = float(row.get("total_score", -1))
        if score >= 0:
            cells[key].append(score)

    result: dict[tuple[str, str], dict] = {}
    for key in set(cells.keys()) | set(max_runs.keys()):
        scores = cells.get(key, [])
        result[key] = {
            "count": len(scores),
            "mean": round(mean(scores), 2) if scores else 0.0,
            "stdev": round(_safe_stdev(scores), 2) if scores else 0.0,
            "max_run": max_runs.get(key, 0),
        }
    return result


def triage_old_session(
    old_session_dir: Path,
    experiment_type: str,
    all_task_keys: list[tuple[str, str]],
    new_models: list[str],
    old_models: list[str],
) -> dict[str, dict[tuple[str, str], dict]]:
    """Classify every (model, asset_id, prompt_mode) as KEEP / REDO_5 / SUPPLEMENT_N.

    Returns: {model_id: {(asset_id, prompt_mode): {action, runs_needed, start_run, old_stats}}}
    """
    exp_dir = old_session_dir / experiment_type
    triage: dict[str, dict[tuple[str, str], dict]] = {}

    for model_id in old_models:
        model_safe = model_id.replace("/", "_")
        rows = _read_model_csv(exp_dir / model_safe / "results.csv")
        cell_stats = _compute_cell_stats(rows)

        model_triage: dict[tuple[str, str], dict] = {}
        for task_key in all_task_keys:
            stats = cell_stats.get(task_key)
            if stats is None:
                model_triage[task_key] = {
                    "action": "REDO_5", "runs_needed": TARGET_RUNS,
                    "start_run": 1, "old_stats": None,
                }
            elif stats["stdev"] >= STDEV_THRESHOLD or stats["mean"] < MEAN_THRESHOLD:
                model_triage[task_key] = {
                    "action": "REDO_5", "runs_needed": TARGET_RUNS,
                    "start_run": 1, "old_stats": stats,
                }
            elif stats["count"] < TARGET_RUNS:
                needed = TARGET_RUNS - stats["count"]
                model_triage[task_key] = {
                    "action": f"SUPPLEMENT_{needed}",
                    "runs_needed": needed,
                    "start_run": stats["max_run"] + 1,
                    "old_stats": stats,
                }
            else:
                model_triage[task_key] = {
                    "action": "KEEP", "runs_needed": 0,
                    "start_run": 0, "old_stats": stats,
                }
        triage[model_id] = model_triage

    for model_id in new_models:
        triage[model_id] = {
            tk: {"action": "REDO_5", "runs_needed": TARGET_RUNS,
                 "start_run": 1, "old_stats": None}
            for tk in all_task_keys
        }

    return triage


def extract_kept_rows(
    old_session_dir: Path,
    experiment_type: str,
    triage: dict[str, dict[tuple[str, str], dict]],
    workspace_dir: Path,
) -> None:
    """Copy rows for KEEP / SUPPLEMENT cells from old CSVs into workspace."""
    exp_dir = old_session_dir / experiment_type
    short_type = "image" if "image" in experiment_type else "video"

    for model_id, cells in triage.items():
        model_safe = model_id.replace("/", "_")
        old_rows = _read_model_csv(exp_dir / model_safe / "results.csv")
        if not old_rows:
            continue

        kept: list[dict] = []
        for row in old_rows:
            task_key = (row["asset_id"], row["prompt_mode"])
            info = cells.get(task_key)
            if info and (info["action"] == "KEEP" or info["action"].startswith("SUPPLEMENT")):
                kept.append(row)

        if kept:
            out = workspace_dir / f"kept_{short_type}_{model_safe}.csv"
            _write_csv(kept, out)
            lg.info("Kept {} rows: {} ({}) -> {}", len(kept), model_id, short_type, out.name)


# ── Phase 3: Targeted Experiment Runners ─────────────────────────


async def _run_targeted_image_model(
    client: DMXAPIClient,
    judge_client: DMXAPIClient,
    judge_model: str,
    model_id: str,
    tasks: list[tuple[str, str, int, int]],
    assets_by_id: dict[str, dict],
    output_dir: Path,
) -> tuple[str, list[dict]]:
    """Run only the specified (asset_id, prompt_mode, runs_needed, start_run) image tasks."""
    lg.info("[Image] {} – {} tasks", model_id, len(tasks))
    results: list[dict] = []
    model_dir = output_dir / model_id.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    for asset_id, prompt_mode, runs_needed, start_run in tasks:
        asset = assets_by_id.get(asset_id)
        if not asset:
            lg.warning("[{}] asset {} not found", model_id, asset_id)
            continue
        img_data = load_image_b64(asset)
        if img_data is None:
            lg.warning("[{}] image file missing for {}", model_id, asset_id)
            continue
        prompt_text = asset["prompts"][prompt_mode]

        for run_idx in range(start_run, start_run + runs_needed):
            lg.info("  {} | {} | {} | run {}", asset_id, model_id.split("/")[-1], prompt_mode, run_idx)

            vlm = await client.chat(
                model=model_id, prompt=prompt_text,
                images_b64=[img_data], max_tokens=1024, temperature=0.1,
            )
            rec: dict[str, Any] = {
                "model": model_id, "asset_id": asset_id, "prompt_mode": prompt_mode,
                "run": run_idx, "timestamp": datetime.now().isoformat(),
                "vlm_success": vlm.success,
                "vlm_response": vlm.content[:2000] if vlm.success else "",
                "vlm_latency_s": round(vlm.latency_s, 3),
                "vlm_retries": vlm.retries,
                "vlm_error_type": vlm.error_type,
                "vlm_error": vlm.error_message[:300] if not vlm.success else "",
            }

            if not vlm.success:
                lg.warning("    [{}] VLM failed: {}", model_id, vlm.error_type)
                rec.update({"judge_success": False, "total_score": -1, "max_score": 10})
                results.append(rec)
                continue

            judge_prompt = build_judge_prompt(asset, prompt_mode, vlm.content)
            jr = await judge_client.chat(
                model=judge_model, prompt=judge_prompt, max_tokens=1024, temperature=0.0,
            )
            if jr.success:
                parsed = parse_judge_response(jr.content)
                if parsed:
                    rec.update({
                        "judge_success": True,
                        "total_score": parsed.get("total_score", -1),
                        "max_score": parsed.get("max_score", 10),
                        "dimension_scores": json.dumps(parsed.get("dimension_scores", {})),
                        "strengths": json.dumps(parsed.get("strengths", []), ensure_ascii=False),
                        "weaknesses": json.dumps(parsed.get("weaknesses", []), ensure_ascii=False),
                        "missing_points": json.dumps(parsed.get("missing_points", []), ensure_ascii=False),
                        "hallucinations": json.dumps(parsed.get("hallucinations", []), ensure_ascii=False),
                        "judge_latency_s": round(jr.latency_s, 3),
                        "judge_retries": jr.retries,
                    })
                else:
                    lg.warning("    [{}] judge JSON parse failed", model_id)
                    rec.update({"judge_success": False, "total_score": -1, "max_score": 10,
                                "judge_raw": jr.content[:500]})
            else:
                lg.warning("    [{}] judge call failed: {}", model_id, jr.error_type)
                rec.update({"judge_success": False, "total_score": -1, "max_score": 10})

            results.append(rec)

    if results:
        _write_csv(results, model_dir / "results.csv")
    lg.info("[Image] {} done – {} rows", model_id, len(results))
    return model_id, results


async def _run_targeted_video_model(
    client: DMXAPIClient,
    judge_client: DMXAPIClient,
    judge_model: str,
    model_id: str,
    tasks: list[tuple[str, str, int, int]],
    video_assets_by_id: dict[str, dict],
    cached_frames: dict[str, list[Path]],
    output_dir: Path,
) -> tuple[str, list[dict]]:
    """Run only the specified (asset_id, prompt_mode, runs_needed, start_run) video tasks."""
    lg.info("[Video] {} – {} tasks", model_id, len(tasks))
    results: list[dict] = []
    model_dir = output_dir / model_id.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    for asset_id, prompt_mode, runs_needed, start_run in tasks:
        asset = video_assets_by_id.get(asset_id)
        if not asset:
            lg.warning("[{}] video asset {} not found", model_id, asset_id)
            continue
        frames = cached_frames.get(asset_id)
        if not frames:
            lg.warning("[{}] no cached frames for {}", model_id, asset_id)
            continue

        frame_images = load_frames_as_b64(frames)
        prompt_text = asset["prompts"][prompt_mode]
        frame_context = (
            f"The following {len(frame_images)} images are sequential frames "
            f"extracted from a game video at regular intervals. "
            f"Analyze them as a video sequence.\n\n{prompt_text}"
        )

        for run_idx in range(start_run, start_run + runs_needed):
            lg.info("  {} | {} | {} | run {}", asset_id, model_id.split("/")[-1], prompt_mode, run_idx)

            vlm = await client.chat(
                model=model_id, prompt=frame_context,
                images_b64=frame_images, max_tokens=1024, temperature=0.1,
            )
            rec: dict[str, Any] = {
                "model": model_id, "asset_id": asset_id, "asset_type": "video",
                "prompt_mode": prompt_mode, "run": run_idx,
                "num_frames": len(frame_images),
                "timestamp": datetime.now().isoformat(),
                "vlm_success": vlm.success,
                "vlm_response": vlm.content[:2000] if vlm.success else "",
                "vlm_latency_s": round(vlm.latency_s, 3),
                "vlm_retries": vlm.retries,
                "vlm_error_type": vlm.error_type,
                "vlm_error": vlm.error_message[:300] if not vlm.success else "",
            }

            if not vlm.success:
                lg.warning("    [{}] VLM failed: {}", model_id, vlm.error_type)
                rec.update({"judge_success": False, "total_score": -1, "max_score": 10})
                results.append(rec)
                continue

            judge_prompt = build_judge_prompt(asset, prompt_mode, vlm.content)
            jr = await judge_client.chat(
                model=judge_model, prompt=judge_prompt, max_tokens=1024, temperature=0.0,
            )
            if jr.success:
                parsed = parse_judge_response(jr.content)
                if parsed:
                    rec.update({
                        "judge_success": True,
                        "total_score": parsed.get("total_score", -1),
                        "max_score": parsed.get("max_score", 10),
                        "dimension_scores": json.dumps(parsed.get("dimension_scores", {})),
                        "strengths": json.dumps(parsed.get("strengths", []), ensure_ascii=False),
                        "weaknesses": json.dumps(parsed.get("weaknesses", []), ensure_ascii=False),
                        "missing_points": json.dumps(parsed.get("missing_points", []), ensure_ascii=False),
                        "hallucinations": json.dumps(parsed.get("hallucinations", []), ensure_ascii=False),
                        "judge_latency_s": round(jr.latency_s, 3),
                        "judge_retries": jr.retries,
                    })
                else:
                    lg.warning("    [{}] judge JSON parse failed", model_id)
                    rec.update({"judge_success": False, "total_score": -1, "max_score": 10,
                                "judge_raw": jr.content[:500]})
            else:
                lg.warning("    [{}] judge call failed: {}", model_id, jr.error_type)
                rec.update({"judge_success": False, "total_score": -1, "max_score": 10})

            results.append(rec)

    if results:
        _write_csv(results, model_dir / "results.csv")
    lg.info("[Video] {} done – {} rows", model_id, len(results))
    return model_id, results


# ── Phase 4: Merge ───────────────────────────────────────────────


def merge_complete_data(
    workspace_dir: Path,
    new_session_dir: Path,
    complete_dir: Path,
    experiment_type: str,
    all_models: list[str],
) -> None:
    """Concatenate kept CSVs + new result CSVs → complete_data/{exp}_quality/{model}/results.csv."""
    exp_type_dir = f"{experiment_type}_quality"

    for model_id in all_models:
        model_safe = model_id.replace("/", "_")
        all_rows: list[dict] = []

        kept_path = workspace_dir / f"kept_{experiment_type}_{model_safe}.csv"
        if kept_path.exists():
            all_rows.extend(_read_model_csv(kept_path))

        new_path = new_session_dir / exp_type_dir / model_safe / "results.csv"
        if new_path.exists():
            all_rows.extend(_read_model_csv(new_path))

        if not all_rows:
            continue

        out_dir = complete_dir / exp_type_dir / model_safe
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(all_rows, out_dir / "results.csv")
        lg.info("Complete: {} ({}) – {} rows", model_id, experiment_type, len(all_rows))


# ── Phase 5: QA Report ──────────────────────────────────────────


def _generate_supplemental_report(
    session_dir: Path,
    complete_data_dir: Path,
    config: dict,
    all_models: list[str],
    image_triage: dict[str, dict[tuple[str, str], dict]],
    video_triage: dict[str, dict[tuple[str, str], dict]],
    image_agg: dict,
    video_agg: dict,
    problems: list[dict],
    old_session_name: str,
) -> Path:
    qa_dir = _PROJECT_ROOT / "QA" / "v3" / "sota_validation"
    qa_dir.mkdir(parents=True, exist_ok=True)
    judge_model = config.get("judge", {}).get("model", "deepseek-chat")

    lines = [
        "# SOTA 模型校准实验 - 补充实验评估报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**完整数据**: `{complete_data_dir.relative_to(_PROJECT_ROOT)}`",
        f"**新实验数据**: `{session_dir.relative_to(_PROJECT_ROOT)}`",
        f"**旧实验数据**: `results/v3-sota/{old_session_name}`",
        f"**裁判模型**: {judge_model}",
        "",
        "---",
        "",
        "## 1. 实验目的",
        "",
        "本次补充实验针对上一轮 SOTA 校准实验中发现的以下问题进行定点重跑：",
        f"- 不完整的实验（运行次数 < {TARGET_RUNS}）→ 补齐缺失轮次",
        f"- 高方差实验（标准差 ≥ {STDEV_THRESHOLD}）→ 全部 {TARGET_RUNS} 轮重跑",
        f"- 低分实验（均分 < {MEAN_THRESHOLD}）→ 全部 {TARGET_RUNS} 轮重跑",
        "- 新增模型 kimi-k2.5 的全部实验",
        "",
        "## 2. 补充分析摘要",
        "",
    ]

    for label, triage in [("图片", image_triage), ("视频", video_triage)]:
        if not triage:
            continue
        lines.append(f"### {label}实验分析")
        lines.append("")
        lines.append("| 模型 | KEEP | REDO | SUPPLEMENT | 新运行数 |")
        lines.append("|------|:---:|:---:|:---:|:---:|")
        for model_id in sorted(triage.keys()):
            cells = triage[model_id]
            keep = sum(1 for v in cells.values() if v["action"] == "KEEP")
            redo = sum(1 for v in cells.values() if v["action"] == "REDO_5")
            supp = sum(1 for v in cells.values() if v["action"].startswith("SUPPLEMENT"))
            new_runs = sum(v["runs_needed"] for v in cells.values())
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            lines.append(f"| {short} | {keep} | {redo} | {supp} | {new_runs} |")
        lines.append("")

    lines.extend([
        "## 3. 实验方法",
        "",
        "- **被测模型**通过 DMXAPI 完成任务回答",
        f"- **{judge_model}** 统一作为裁判，根据 rubric 评分",
        "- 所有模型**并行执行**，各自独立写入结果",
        "- 旧实验中合格的数据直接保留，仅重跑/补充不合格部分",
        "- 最终结果合并到 `complete_data/` 中",
        "",
    ])

    if image_agg:
        lines.extend(["## 4. 图片实验完整结果", ""])
        for model_id, assets in sorted(image_agg.items()):
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            lines.append(f"### {short}")
            lines.append("")
            lines.append("| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |")
            lines.append("|------|--------|:---:|:---:|:---:|:---:|:---:|")
            for asset_id, modes in sorted(assets.items()):
                for pm, stats in sorted(modes.items()):
                    lines.append(
                        f"| {asset_id} | {pm} | {stats['mean']} | "
                        f"{stats['stdev']} | {stats['min']} | "
                        f"{stats['max']} | {stats['count']} |"
                    )
            lines.append("")
    else:
        lines.extend(["## 4. 图片实验完整结果", "", "无数据。", ""])

    if video_agg:
        lines.extend(["## 5. 视频实验完整结果", ""])
        for model_id, assets in sorted(video_agg.items()):
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            lines.append(f"### {short}")
            lines.append("")
            lines.append("| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |")
            lines.append("|------|--------|:---:|:---:|:---:|:---:|:---:|")
            for asset_id, modes in sorted(assets.items()):
                for pm, stats in sorted(modes.items()):
                    lines.append(
                        f"| {asset_id} | {pm} | {stats['mean']} | "
                        f"{stats['stdev']} | {stats['min']} | "
                        f"{stats['max']} | {stats['count']} |"
                    )
            lines.append("")
    else:
        lines.extend(["## 5. 视频实验完整结果", "", "无数据。", ""])

    charts_dir = session_dir / "charts"
    if charts_dir.exists() and any(charts_dir.glob("*.png")):
        lines.extend(["## 6. 可视化", ""])
        for png in sorted(charts_dir.glob("*.png")):
            rel = png.relative_to(_PROJECT_ROOT)
            lines.append(f"![{png.stem}]({rel})")
            lines.append("")

    lines.extend(["## 7. 争议样本分析", ""])
    if problems:
        lines.append("以下样本中超过半数 SOTA 模型得分低于 7.0，应优先检查设计：")
        lines.append("")
        for p in problems:
            lines.append(
                f"- **{p['asset_id']}** ({p['prompt_mode']}): "
                f"均分 {p['avg_across_models']}, "
                f"{p['models_below_threshold']}/{p['total_models']} 个模型低于阈值"
            )
    else:
        lines.append("未发现争议样本。所有样本在多数 SOTA 模型上得分 >= 7.0。")

    lines.extend(["", "## 8. 结论", "", "<!-- 此部分将在审阅后填写 -->", ""])

    report_path = qa_dir / "评估报告_supplemental.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    lg.info("QA report: {}", report_path)
    return report_path


# ── Main Orchestrator ────────────────────────────────────────────


async def _main(args: argparse.Namespace) -> None:
    config_path = _PROJECT_ROOT / args.config
    if not config_path.exists():
        lg.error("Config not found: {}", config_path)
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")

    # ── Clients ──────────────────────────────────────────────────
    api_cfg = config.get("api", {})
    api_key = os.getenv(api_cfg.get("api_key_env", "DMX_API_KEY"), "")
    base_url = os.getenv(api_cfg.get("base_url_env", "DMX_API_BASE_URL"), "https://www.dmxapi.cn")
    if not api_key:
        lg.error("DMXAPI key not set"); sys.exit(1)

    client = DMXAPIClient(
        api_key=api_key, base_url=base_url,
        timeout_s=api_cfg.get("timeout_s", 180),
        max_retries=api_cfg.get("max_retries", 3),
        retry_wait_s=api_cfg.get("retry_wait_s", 60),
    )

    judge_cfg = config.get("judge", {})
    judge_model = judge_cfg.get("model", "deepseek-chat")
    judge_key = os.getenv(judge_cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    judge_url = os.getenv(
        judge_cfg.get("base_url_env", "DEEPSEEK_API_BASE_URL"), "https://api.deepseek.com"
    )
    if not judge_key:
        lg.error("DeepSeek key not set"); sys.exit(1)

    judge_client = DMXAPIClient(
        api_key=judge_key, base_url=judge_url,
        timeout_s=judge_cfg.get("timeout_s", 120),
        max_retries=judge_cfg.get("max_retries", 3),
        retry_wait_s=judge_cfg.get("retry_wait_s", 30),
    )

    # ── Session directories ──────────────────────────────────────
    old_session_name = args.old_session
    old_session_dir = _PROJECT_ROOT / "results" / "v3-sota" / old_session_name
    if not old_session_dir.exists():
        lg.error("Old session not found: {}", old_session_dir)
        sys.exit(1)

    ts = datetime.now().strftime(
        config.get("general", {}).get("timestamp_format", "%Y-%m-%d_%H-%M-%S")
    )
    output_base = config.get("general", {}).get("output_dir", "results/v3-sota")
    session_dir = _PROJECT_ROOT / output_base / ts
    session_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = session_dir / "_workspace"
    workspace_dir.mkdir(exist_ok=True)
    complete_data_dir = session_dir / "complete_data"

    shutil.copy2(config_path, session_dir / "config.toml")
    meta: dict[str, Any] = {
        "experiment": "sota_supplemental",
        "start_time": datetime.now().isoformat(),
        "old_session": old_session_name,
        "judge_model": judge_model,
        "python_version": sys.version,
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log_path = session_dir / "experiment.log"
    lg.add(str(log_path), level="DEBUG", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}")

    lg.info("=" * 70)
    lg.info("SOTA Supplemental Experiment")
    lg.info("Session: {}", session_dir)
    lg.info("Old session: {}", old_session_dir)
    lg.info("=" * 70)

    # ── Load assets ──────────────────────────────────────────────
    image_assets = load_asset_jsons(ASSETS_IMAGES_DIR)
    image_assets_by_id = {a["id"]: a for a in image_assets}
    image_task_keys = [(a["id"], pm) for a in image_assets for pm in ["A_description", "B_assistant"]]

    video_assets = _load_video_jsons()
    video_assets_by_id = {a["id"]: a for a in video_assets}
    video_task_keys = [(a["id"], pm) for a in video_assets for pm in ["A_description", "B_assistant"]]

    lg.info("Image assets: {} ({} tasks)  Video assets: {} ({} tasks)",
            len(image_assets), len(image_task_keys), len(video_assets), len(video_task_keys))

    # ── Discover old models from CSV files ───────────────────────
    old_models: list[str] = []
    old_image_dir = old_session_dir / "image_quality"
    if old_image_dir.exists():
        for d in sorted(old_image_dir.iterdir()):
            if d.is_dir() and (d / "results.csv").exists():
                rows = _read_model_csv(d / "results.csv")
                if rows:
                    old_models.append(rows[0]["model"])

    config_models = config.get("models", {}).get("list", [])
    # Only keep old models that are still in the config (drop removed models)
    old_models = [m for m in old_models if m in config_models]
    new_models = [m for m in config_models if m not in old_models]
    all_models = list(dict.fromkeys(old_models + new_models))

    lg.info("Old models ({}): {}", len(old_models), old_models)
    lg.info("New models ({}): {}", len(new_models), new_models)
    lg.info("All models ({}): {}", len(all_models), all_models)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Triage
    # ═══════════════════════════════════════════════════════════════
    lg.info("=" * 70)
    lg.info("Phase 1: Triage")

    image_triage = triage_old_session(old_session_dir, "image_quality", image_task_keys, new_models, old_models)
    video_triage = triage_old_session(old_session_dir, "video_quality", video_task_keys, new_models, old_models)

    triage_json: dict[str, Any] = {"image": {}, "video": {}}
    for model_id, cells in image_triage.items():
        triage_json["image"][model_id] = {
            f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items() if kk != "old_stats"}
            for k, v in cells.items()
        }
    for model_id, cells in video_triage.items():
        triage_json["video"][model_id] = {
            f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items() if kk != "old_stats"}
            for k, v in cells.items()
        }
    (workspace_dir / "triage.json").write_text(
        json.dumps(triage_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for label, triage in [("image", image_triage), ("video", video_triage)]:
        for model_id, cells in sorted(triage.items()):
            actions = [v["action"] for v in cells.values()]
            new_runs = sum(v["runs_needed"] for v in cells.values())
            lg.info("  [{}] {} → KEEP={} REDO={} SUPP={} runs={}",
                    label, model_id,
                    sum(1 for a in actions if a == "KEEP"),
                    sum(1 for a in actions if a == "REDO_5"),
                    sum(1 for a in actions if a.startswith("SUPPLEMENT")),
                    new_runs)

    extract_kept_rows(old_session_dir, "image_quality", image_triage, workspace_dir)
    extract_kept_rows(old_session_dir, "video_quality", video_triage, workspace_dir)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Preflight new models
    # ═══════════════════════════════════════════════════════════════
    incidents: list[dict] = []
    if new_models:
        lg.info("=" * 70)
        lg.info("Phase 2: Preflight new models – {}", new_models)
        passed_new, pf_incidents = await run_preflight(client, new_models)
        incidents.extend(pf_incidents)
        failed_new = [m for m in new_models if m not in passed_new]
        for m in failed_new:
            image_triage.pop(m, None)
            video_triage.pop(m, None)
        new_models = passed_new
        if not passed_new:
            lg.warning("All new models failed preflight")
    else:
        lg.info("No new models to preflight")

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Run supplemental experiments
    # ═══════════════════════════════════════════════════════════════
    lg.info("=" * 70)
    lg.info("Phase 3: Supplemental experiments")

    image_model_tasks: dict[str, list[tuple[str, str, int, int]]] = {}
    for model_id, cells in image_triage.items():
        tasks = [
            (aid, pm, info["runs_needed"], info["start_run"])
            for (aid, pm), info in cells.items() if info["runs_needed"] > 0
        ]
        if tasks:
            image_model_tasks[model_id] = tasks

    video_model_tasks: dict[str, list[tuple[str, str, int, int]]] = {}
    for model_id, cells in video_triage.items():
        tasks = [
            (aid, pm, info["runs_needed"], info["start_run"])
            for (aid, pm), info in cells.items() if info["runs_needed"] > 0
        ]
        if tasks:
            video_model_tasks[model_id] = tasks

    total_image_runs = sum(t[2] for ts in image_model_tasks.values() for t in ts)
    total_video_runs = sum(t[2] for ts in video_model_tasks.values() for t in ts)
    lg.info("Image: {} models, {} runs  |  Video: {} models, {} runs",
            len(image_model_tasks), total_image_runs, len(video_model_tasks), total_video_runs)

    if image_model_tasks:
        image_out_dir = session_dir / "image_quality"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        coros = [
            _run_targeted_image_model(
                client, judge_client, judge_model, mid, ts, image_assets_by_id, image_out_dir,
            )
            for mid, ts in image_model_tasks.items()
        ]
        for r in await asyncio.gather(*coros, return_exceptions=True):
            if isinstance(r, Exception):
                lg.error("Image task exception: {}", r)

    if video_model_tasks:
        video_cfg = config.get("video", {})
        cache_dir = _PROJECT_ROOT / video_cfg.get("cache_dir", "tmp/sota_frames")
        cached_frames = extract_all_videos(
            cache_dir,
            interval_s=video_cfg.get("frame_interval_s", 1.0),
            max_frames=video_cfg.get("max_frames", 30),
        )
        video_out_dir = session_dir / "video_quality"
        video_out_dir.mkdir(parents=True, exist_ok=True)
        coros = [
            _run_targeted_video_model(
                client, judge_client, judge_model, mid, ts,
                video_assets_by_id, cached_frames, video_out_dir,
            )
            for mid, ts in video_model_tasks.items()
        ]
        for r in await asyncio.gather(*coros, return_exceptions=True):
            if isinstance(r, Exception):
                lg.error("Video task exception: {}", r)

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Merge → complete_data
    # ═══════════════════════════════════════════════════════════════
    lg.info("=" * 70)
    lg.info("Phase 4: Merge → complete_data")

    merge_complete_data(workspace_dir, session_dir, complete_data_dir, "image", all_models)
    merge_complete_data(workspace_dir, session_dir, complete_data_dir, "video", all_models)

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Aggregate + Charts + Report
    # ═══════════════════════════════════════════════════════════════
    lg.info("=" * 70)
    lg.info("Phase 5: Aggregate, Charts & Report")

    image_agg: dict = {}
    complete_image_dir = complete_data_dir / "image_quality"
    if complete_image_dir.exists():
        image_agg = aggregate_results(complete_image_dir, complete_image_dir / "aggregated.csv", "image")

    video_agg: dict = {}
    complete_video_dir = complete_data_dir / "video_quality"
    if complete_video_dir.exists():
        video_agg = aggregate_results(complete_video_dir, complete_video_dir / "aggregated.csv", "video")

    charts_dir = session_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    combined_agg: dict = {}
    combined_agg.update(image_agg)
    for m, a in video_agg.items():
        combined_agg.setdefault(m, {}).update(a)

    if combined_agg:
        plot_score_ranking(combined_agg, charts_dir / "score_ranking.png")
        plot_stability(combined_agg, charts_dir / "stability.png")
        plot_per_sample_comparison(combined_agg, charts_dir / "per_sample_comparison.png")

    if complete_image_dir.exists():
        plot_dimension_heatmap(complete_image_dir, charts_dir / "dimension_heatmap_image.png",
                               "Image – Dimension Scores")
        plot_latency_comparison(complete_image_dir, charts_dir / "latency_image.png",
                                "Image – VLM Latency")

    if complete_video_dir.exists():
        plot_dimension_heatmap(complete_video_dir, charts_dir / "dimension_heatmap_video.png",
                               "Video – Dimension Scores")
        plot_latency_comparison(complete_video_dir, charts_dir / "latency_video.png",
                                "Video – VLM Latency")

    problems = identify_problematic_samples({**image_agg, **video_agg})
    generate_comparison_report(
        image_agg, video_agg, problems, session_dir / "comparison_report.md",
        incidents=incidents,
    )

    _generate_supplemental_report(
        session_dir, complete_data_dir, config, all_models,
        image_triage, video_triage,
        image_agg, video_agg, problems, old_session_name,
    )

    meta.update({
        "end_time": datetime.now().isoformat(),
        "all_models": all_models,
        "new_models": new_models,
        "image_runs_executed": total_image_runs,
        "video_runs_executed": total_video_runs,
        "problematic_samples": len(problems),
    })
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    await client.close()
    await judge_client.close()

    lg.info("=" * 70)
    lg.info("Supplemental experiment complete")
    lg.info("Session: {}", session_dir)
    lg.info("Complete data: {}", complete_data_dir)
    lg.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="SOTA Supplemental Experiment")
    parser.add_argument("--config", default="config_sota_validation.toml",
                        help="Config file (default: config_sota_validation.toml)")
    parser.add_argument("--old-session", default="2026-03-19_01-39-43",
                        help="Old session timestamp directory name")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
