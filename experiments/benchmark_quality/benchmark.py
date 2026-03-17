"""LLM-as-Judge quality benchmark for VLM game-scene understanding.

Loads asset JSONs with structured prompts and rubrics, sends images to the
tested VLM, then evaluates the responses via DeepSeek (LLM-as-Judge).

Usage:
    uv run experiments/benchmark_quality/benchmark.py
    uv run experiments/benchmark_quality/benchmark.py --runs 3
    uv run experiments/benchmark_quality/benchmark.py --base-url http://172.x.x.x:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from loguru import logger as lg

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXPERIMENT_DIR.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from ahu_paimon_toolkit.evaluation.judge import LLMJudge
from ahu_paimon_toolkit.evaluation.scoring import ScoreAggregator
from ahu_paimon_toolkit.models import JudgeScore
from ahu_paimon_toolkit.utils.image import encode_image, get_image_mime
from ahu_paimon_toolkit.vlm.model_utils import detect_model_from_url, model_short_name


ASSETS_IMAGES_DIR = _PROJECT_ROOT / "assets" / "images"
ASSETS_VIDEOS_DIR = _PROJECT_ROOT / "assets" / "videos"


def load_asset_jsons(assets_dir: Path) -> list[dict]:
    """Load all JSON task-description files from an assets directory."""
    jsons = []
    for p in sorted(assets_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if data:
            data["_json_path"] = str(p)
            jsons.append(data)
    return jsons


async def run_single_evaluation(
    *,
    vlm_base_url: str,
    vlm_model: str,
    asset: dict,
    prompt_mode: str,
    image_b64: str,
    image_mime: str,
    judge: LLMJudge,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> JudgeScore:
    """Run one VLM inference + one Judge evaluation for a single asset/prompt_mode."""
    import httpx

    prompt_text = asset["prompts"][prompt_mode]

    async with httpx.AsyncClient(
        base_url=vlm_base_url,
        timeout=httpx.Timeout(60.0),
    ) as client:
        payload = {
            "model": vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_mime};base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        vlm_response = data["choices"][0]["message"]["content"]

    lg.info(
        "VLM response for {} / {} ({} chars)",
        asset["id"], prompt_mode, len(vlm_response),
    )

    score = await judge.evaluate(
        asset_id=asset["id"],
        model_id=vlm_model,
        prompt_mode=prompt_mode,
        vlm_response=vlm_response,
        task_definition=asset["task_definition"],
        reference_answer=asset["reference_answer"],
        scoring_rubric=asset["scoring"],
        grading_prompt=asset["grading_prompt_for_judge_model"],
    )

    lg.info(
        "Judge score for {} / {} / {}: {}/{}",
        asset["id"], vlm_model, prompt_mode,
        score.total_score, score.max_score,
    )

    return score


async def run_benchmark_quality(
    *,
    vlm_base_url: str,
    vlm_model: str,
    judge: LLMJudge,
    assets: list[dict],
    output_dir: Path,
    runs: int = 5,
    judge_delay_s: float = 1.0,
) -> list[JudgeScore]:
    """Run the full quality benchmark for one model.

    Returns all JudgeScore results.
    """
    model_dir = output_dir / model_short_name(vlm_model)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[JudgeScore] = []

    for asset in assets:
        asset_type = asset.get("type", "image")
        if asset_type != "image":
            continue

        image_file = asset.get("image_file", "")
        image_path = ASSETS_IMAGES_DIR / image_file
        if not image_path.exists():
            lg.warning("Image not found: {}, skipping", image_path)
            continue

        image_b64 = encode_image(image_path)
        image_mime = get_image_mime(image_path)

        for prompt_mode in ["A_description", "B_assistant"]:
            for run_idx in range(runs):
                lg.info(
                    "=== {} | {} | {} | run {}/{} ===",
                    asset["id"], vlm_model, prompt_mode, run_idx + 1, runs,
                )
                try:
                    score = await run_single_evaluation(
                        vlm_base_url=vlm_base_url,
                        vlm_model=vlm_model,
                        asset=asset,
                        prompt_mode=prompt_mode,
                        image_b64=image_b64,
                        image_mime=image_mime,
                        judge=judge,
                    )
                    all_scores.append(score)
                except Exception:
                    lg.exception(
                        "Evaluation failed: {} / {} / run {}",
                        asset["id"], prompt_mode, run_idx + 1,
                    )

                await asyncio.sleep(judge_delay_s)

    _save_scores_csv(all_scores, model_dir / "scores.csv")
    _generate_model_report(all_scores, vlm_model, model_dir / "report.md")

    return all_scores


def _save_scores_csv(scores: list[JudgeScore], path: Path) -> None:
    """Save all scores to a CSV file."""
    if not scores:
        return

    all_dims = set()
    for s in scores:
        all_dims.update(s.dimension_scores.keys())
    dim_names = sorted(all_dims)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["asset_id", "model_id", "prompt_mode", "total_score", "max_score"] + dim_names
        writer.writerow(header)
        for s in scores:
            row = [s.asset_id, s.model_id, s.prompt_mode, s.total_score, s.max_score]
            for d in dim_names:
                row.append(s.dimension_scores.get(d, ""))
            writer.writerow(row)

    lg.info("Saved {} scores to {}", len(scores), path)


def _generate_model_report(scores: list[JudgeScore], model_id: str, path: Path) -> None:
    """Generate a markdown report for a single model's quality evaluation."""
    agg = ScoreAggregator(scores)
    summary = agg.summary_dict()

    lines = [
        f"# Quality Benchmark Report: {model_id}",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Evaluations**: {summary['count']}",
        "",
        "## Overall Score",
        "",
        f"- **Mean**: {summary['mean_total']:.2f} / 10",
        f"- **Stdev**: {summary['stdev_total']:.2f}",
        f"- **95% CI**: [{summary['ci_95_lower']:.2f}, {summary['ci_95_upper']:.2f}]",
        "",
        "## Per-Dimension Scores",
        "",
        "| Dimension | Mean Score (0-2) |",
        "|-----------|:----------------:|",
    ]
    for dim_name, dim_mean in summary["mean_per_dimension"].items():
        lines.append(f"| {dim_name} | {dim_mean:.2f} |")

    lines.extend([
        "",
        "## By Prompt Mode",
        "",
    ])

    for mode in ["A_description", "B_assistant"]:
        mode_agg = agg.filter_by_prompt_mode(mode)
        if mode_agg.count > 0:
            mode_summary = mode_agg.summary_dict()
            lines.append(f"### {mode}")
            lines.append(f"- Mean: {mode_summary['mean_total']:.2f} / 10")
            lines.append(f"- 95% CI: [{mode_summary['ci_95_lower']:.2f}, {mode_summary['ci_95_upper']:.2f}]")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    lg.info("Report generated: {}", path)


def generate_comparison_report(
    all_model_scores: dict[str, list[JudgeScore]],
    output_dir: Path,
) -> None:
    """Generate a cross-model comparison report."""
    lines = [
        "# Quality Benchmark Comparison Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model Rankings (by mean total score)",
        "",
        "| Rank | Model | Mean Score | Stdev | 95% CI |",
        "|:----:|-------|:---------:|:-----:|:------:|",
    ]

    rankings = []
    for model_id, scores in all_model_scores.items():
        agg = ScoreAggregator(scores)
        ci = agg.ci_95_total()
        rankings.append((model_id, agg.mean_total(), agg.stdev_total(), ci))

    rankings.sort(key=lambda x: x[1], reverse=True)

    for rank, (model_id, m, s, ci) in enumerate(rankings, 1):
        short = model_id.split("/")[-1] if "/" in model_id else model_id
        lines.append(f"| {rank} | {short} | {m:.2f} | {s:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")

    report_path = output_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    _save_all_scores_csv(all_model_scores, output_dir / "all_scores.csv")

    lg.info("Comparison report generated: {}", report_path)


def _save_all_scores_csv(
    all_model_scores: dict[str, list[JudgeScore]],
    path: Path,
) -> None:
    """Save all scores from all models to a single CSV."""
    all_scores = []
    for scores in all_model_scores.values():
        all_scores.extend(scores)
    _save_scores_csv(all_scores, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge Quality Benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--judge-delay", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv
    import os

    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_base = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        lg.error("DEEPSEEK_API_KEY not set in .env")
        sys.exit(1)

    vlm_model = detect_model_from_url(args.base_url)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = _EXPERIMENT_DIR / "reports" / ts

    output_dir.mkdir(parents=True, exist_ok=True)

    judge = LLMJudge(api_key=api_key, api_base_url=api_base)
    assets = load_asset_jsons(ASSETS_IMAGES_DIR)

    lg.info("Starting quality benchmark: model={}, assets={}, runs={}", vlm_model, len(assets), args.runs)

    scores = asyncio.run(
        run_benchmark_quality(
            vlm_base_url=args.base_url,
            vlm_model=vlm_model,
            judge=judge,
            assets=assets,
            output_dir=output_dir,
            runs=args.runs,
            judge_delay_s=args.judge_delay,
        )
    )

    lg.info("Benchmark complete: {} total evaluations", len(scores))


if __name__ == "__main__":
    main()
