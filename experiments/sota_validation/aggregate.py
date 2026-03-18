"""Result aggregation for SOTA validation experiments.

Reads per-model CSV results and produces:
- Per model x sample: mean, stdev, min, max, per-dimension means
- Cross-model comparison: identifies problematic samples
- Output: aggregated CSV + comparison markdown
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from loguru import logger as lg


def _safe_stdev(vals: list[float]) -> float:
    return stdev(vals) if len(vals) >= 2 else 0.0


def aggregate_results(
    results_dir: Path,
    output_path: Path,
    experiment_type: str = "image",
) -> dict:
    """Aggregate results from per-model CSV files.

    Args:
        results_dir: Directory containing per-model subdirectories with results.csv.
        output_path: Path for the output aggregated CSV.
        experiment_type: "image" or "video" for report labeling.

    Returns:
        Nested dict: {model: {asset_id: {prompt_mode: stats_dict}}}
    """
    aggregated: dict = {}
    all_rows: list[dict] = []

    for model_dir in sorted(results_dir.iterdir()):
        csv_path = model_dir / "results.csv"
        if not csv_path.exists():
            continue

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        model_id = rows[0]["model"] if rows else model_dir.name
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for row in rows:
            score = float(row.get("total_score", -1))
            if score < 0:
                continue
            asset_id = row["asset_id"]
            prompt_mode = row["prompt_mode"]
            grouped[asset_id][prompt_mode].append(score)

        model_stats: dict = {}
        for asset_id, modes in grouped.items():
            asset_stats: dict = {}
            for pm, scores in modes.items():
                asset_stats[pm] = {
                    "mean": round(mean(scores), 2),
                    "stdev": round(_safe_stdev(scores), 2),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }
                all_rows.append({
                    "model": model_id,
                    "asset_id": asset_id,
                    "prompt_mode": pm,
                    "mean_score": round(mean(scores), 2),
                    "stdev": round(_safe_stdev(scores), 2),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "count": len(scores),
                    "failures": sum(1 for r in rows if r["asset_id"] == asset_id and r["prompt_mode"] == pm and float(r.get("total_score", -1)) < 0),
                })
            model_stats[asset_id] = asset_stats

        aggregated[model_id] = model_stats

    # Save aggregated CSV
    if all_rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        lg.info("Aggregated CSV: {}", output_path)

    return aggregated


def identify_problematic_samples(
    aggregated: dict,
    threshold: float = 7.0,
) -> list[dict]:
    """Identify samples where most SOTA models score below threshold.

    Returns list of {asset_id, prompt_mode, models_below, avg_across_models}.
    """
    sample_scores: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)

    for model_id, assets in aggregated.items():
        for asset_id, modes in assets.items():
            for pm, stats in modes.items():
                sample_scores[(asset_id, pm)].append((model_id, stats["mean"]))

    problems = []
    for (asset_id, pm), model_vals in sample_scores.items():
        scores = [s for _, s in model_vals]
        avg = mean(scores)
        below = [(m, s) for m, s in model_vals if s < threshold]
        if len(below) > len(model_vals) / 2:
            problems.append({
                "asset_id": asset_id,
                "prompt_mode": pm,
                "avg_across_models": round(avg, 2),
                "models_below_threshold": len(below),
                "total_models": len(model_vals),
                "details": [(m, round(s, 2)) for m, s in below],
            })

    return sorted(problems, key=lambda x: x["avg_across_models"])


def generate_comparison_report(
    image_agg: dict,
    video_agg: dict,
    problems: list[dict],
    output_path: Path,
    incidents: list[dict] | None = None,
) -> None:
    """Generate the cross-model comparison markdown report."""
    lines = [
        "# SOTA Validation - Cross-Model Comparison Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Image results
    if image_agg:
        lines.extend(["## Image Quality Results", ""])
        lines.extend(_build_summary_table(image_agg))

    # Video results
    if video_agg:
        lines.extend(["", "## Video Quality Results", ""])
        lines.extend(_build_summary_table(video_agg))

    # Problematic samples
    lines.extend(["", "## Problematic Samples", ""])
    if problems:
        lines.append("Samples where >50% of SOTA models scored below 7.0:")
        lines.append("")
        lines.append("| Sample | Prompt Mode | Avg Score | Models Below / Total |")
        lines.append("|--------|-------------|:---------:|:--------------------:|")
        for p in problems:
            lines.append(
                f"| {p['asset_id']} | {p['prompt_mode']} | "
                f"{p['avg_across_models']} | "
                f"{p['models_below_threshold']}/{p['total_models']} |"
            )
        lines.append("")
        lines.append("These samples may have design issues (ambiguous prompts, "
                      "overly strict rubrics, or insufficient information).")
    else:
        lines.append("No problematic samples identified (all samples have "
                      "majority SOTA models scoring >= 7.0).")

    # Incidents
    if incidents:
        lines.extend(["", "## Incidents", ""])
        lines.append("| Model | Phase | Error |")
        lines.append("|-------|-------|-------|")
        for inc in incidents:
            lines.append(f"| {inc['model']} | {inc['phase']} | {inc.get('error', '')[:80]} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    lg.info("Comparison report: {}", output_path)


def _build_summary_table(aggregated: dict) -> list[str]:
    """Build a per-model summary table."""
    lines = []
    lines.append("| Model | Samples | Mean Score | Stdev | Min | Max |")
    lines.append("|-------|:-------:|:---------:|:-----:|:---:|:---:|")

    for model_id, assets in sorted(aggregated.items()):
        all_means = []
        for modes in assets.values():
            for stats in modes.values():
                all_means.append(stats["mean"])
        if not all_means:
            continue
        short = model_id.split("/")[-1] if "/" in model_id else model_id
        lines.append(
            f"| {short} | {len(all_means)} | "
            f"{mean(all_means):.2f} | {_safe_stdev(all_means):.2f} | "
            f"{min(all_means):.1f} | {max(all_means):.1f} |"
        )

    return lines
