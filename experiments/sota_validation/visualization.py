"""Chart generation for SOTA validation experiments.

Reuses the style setup from ahu_paimon_toolkit.utils.visualization.
Generates 5 chart types from aggregated results.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from loguru import logger as lg

_PALETTE = "muted"
_FIG_DPI = 200
_FONT_SIZES = {"title": 14, "label": 11, "tick": 9, "legend": 9}
_CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", palette=_PALETTE, font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": _FIG_DPI,
        "savefig.dpi": _FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
        "axes.titlesize": _FONT_SIZES["title"],
        "axes.labelsize": _FONT_SIZES["label"],
        "xtick.labelsize": _FONT_SIZES["tick"],
        "ytick.labelsize": _FONT_SIZES["tick"],
        "legend.fontsize": _FONT_SIZES["legend"],
        "font.sans-serif": _CJK_FONTS,
        "axes.unicode_minus": False,
    })


def _short(model_id: str, max_len: int = 24) -> str:
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    return name[:max_len - 2] + ".." if len(name) > max_len else name


def _safe_stdev(vals: list[float]) -> float:
    from statistics import stdev
    return stdev(vals) if len(vals) >= 2 else 0.0


# ── Chart 1: Score Ranking ───────────────────────────────────────


def plot_score_ranking(
    aggregated: dict,
    output_path: Path,
    title: str = "SOTA Validation - Mean Score Ranking",
) -> Path:
    """Horizontal bar chart: mean total score per model."""
    _apply_style()

    model_means: list[tuple[str, float]] = []
    for model_id, assets in aggregated.items():
        scores = [s["mean"] for a in assets.values() for s in a.values()]
        if scores:
            model_means.append((model_id, mean(scores)))

    model_means.sort(key=lambda x: x[1], reverse=True)
    names = [_short(m) for m, _ in model_means]
    values = [v for _, v in model_means]
    colors = sns.color_palette(_PALETTE, len(names))

    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(names) + 1.2)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=_FONT_SIZES["tick"])

    ax.set_xlabel("Mean Score (0-10)")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlim(0, 10.5)
    ax.invert_yaxis()
    sns.despine(left=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    lg.info("Chart: {}", output_path)
    return output_path


# ── Chart 2: Dimension Heatmap ───────────────────────────────────


def plot_dimension_heatmap(
    results_dir: Path,
    output_path: Path,
    title: str = "SOTA Validation - Dimension Scores Heatmap",
) -> Path | None:
    """Heatmap: model x scoring dimension mean scores."""
    _apply_style()

    model_dims: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for model_dir in sorted(results_dir.iterdir()):
        csv_path = model_dir / "results.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ds_raw = row.get("dimension_scores", "")
                if not ds_raw:
                    continue
                try:
                    dims = json.loads(ds_raw)
                except json.JSONDecodeError:
                    continue
                model_id = row["model"]
                for dim_name, score in dims.items():
                    try:
                        model_dims[model_id][dim_name].append(float(score))
                    except (ValueError, TypeError):
                        pass

    if not model_dims:
        return None

    all_dims = sorted({d for dims in model_dims.values() for d in dims})
    models = sorted(model_dims.keys())
    names = [_short(m) for m in models]

    data = np.zeros((len(models), len(all_dims)))
    for i, m in enumerate(models):
        for j, d in enumerate(all_dims):
            vals = model_dims[m].get(d, [])
            data[i, j] = mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(max(8, len(all_dims) * 1.5), max(4, len(names) * 0.6 + 1.5)))
    sns.heatmap(
        data, annot=True, fmt=".2f",
        xticklabels=all_dims, yticklabels=names,
        cmap="YlOrRd", vmin=0, vmax=2, linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontweight="bold", pad=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    lg.info("Chart: {}", output_path)
    return output_path


# ── Chart 3: Stability (Stdev) ───────────────────────────────────


def plot_stability(
    aggregated: dict,
    output_path: Path,
    title: str = "SOTA Validation - Score Stability (lower = more stable)",
) -> Path:
    """Bar chart: mean stdev per model across all samples."""
    _apply_style()

    model_stdevs: list[tuple[str, float]] = []
    for model_id, assets in aggregated.items():
        stdevs = [s["stdev"] for a in assets.values() for s in a.values()]
        if stdevs:
            model_stdevs.append((model_id, mean(stdevs)))

    model_stdevs.sort(key=lambda x: x[1])
    names = [_short(m) for m, _ in model_stdevs]
    values = [v for _, v in model_stdevs]
    colors = sns.color_palette(_PALETTE, len(names))

    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(names) + 1.2)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=_FONT_SIZES["tick"])

    ax.set_xlabel("Mean Stdev")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.invert_yaxis()
    sns.despine(left=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    lg.info("Chart: {}", output_path)
    return output_path


# ── Chart 4: Per-Sample Comparison ────────────────────────────────


def plot_per_sample_comparison(
    aggregated: dict,
    output_path: Path,
    title: str = "SOTA Validation - Per-Sample Cross-Model Comparison",
) -> Path:
    """Grouped bar chart: each sample's mean across all models."""
    _apply_style()

    all_samples: set[str] = set()
    for assets in aggregated.values():
        all_samples.update(assets.keys())
    samples = sorted(all_samples)

    models = sorted(aggregated.keys())
    n_samples = len(samples)
    n_models = len(models)
    x = np.arange(n_samples)
    width = 0.8 / max(n_models, 1)
    colors = sns.color_palette(_PALETTE, n_models)
    names = [_short(m, 18) for m in models]

    fig, ax = plt.subplots(figsize=(max(10, n_samples * 2), 6))
    for i, (model_id, name) in enumerate(zip(models, names)):
        vals = []
        for s in samples:
            modes = aggregated.get(model_id, {}).get(s, {})
            means = [st["mean"] for st in modes.values()]
            vals.append(mean(means) if means else 0)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=name, color=colors[i],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=25, ha="right")
    ax.set_ylabel("Mean Score (0-10)")
    ax.set_ylim(0, 10.5)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    sns.despine()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    lg.info("Chart: {}", output_path)
    return output_path


# ── Chart 5: Latency Comparison ───────────────────────────────────


def plot_latency_comparison(
    results_dir: Path,
    output_path: Path,
    title: str = "SOTA Validation - Mean VLM Latency (seconds)",
) -> Path:
    """Bar chart: mean VLM latency per model."""
    _apply_style()

    model_latencies: dict[str, list[float]] = defaultdict(list)

    for model_dir in sorted(results_dir.iterdir()):
        csv_path = model_dir / "results.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lat = row.get("vlm_latency_s", "")
                if lat and row.get("vlm_success", "").lower() == "true":
                    try:
                        model_latencies[row["model"]].append(float(lat))
                    except ValueError:
                        pass

    if not model_latencies:
        lg.warning("No latency data for chart")
        return output_path

    items = sorted(model_latencies.items(), key=lambda x: mean(x[1]))
    names = [_short(m) for m, _ in items]
    values = [mean(v) for _, v in items]
    colors = sns.color_palette(_PALETTE, len(names))

    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(names) + 1.2)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=_FONT_SIZES["tick"])

    ax.set_xlabel("Mean Latency (s)")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.invert_yaxis()
    sns.despine(left=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    lg.info("Chart: {}", output_path)
    return output_path


# ── Generate All Charts ──────────────────────────────────────────


def generate_all_charts(
    session_dir: Path,
    image_agg: dict,
    video_agg: dict,
) -> list[Path]:
    """Generate all SOTA validation charts. Returns list of output paths."""
    charts_dir = session_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    combined = {}
    combined.update(image_agg)
    for model_id, assets in video_agg.items():
        if model_id in combined:
            combined[model_id].update(assets)
        else:
            combined[model_id] = assets

    if combined:
        generated.append(plot_score_ranking(combined, charts_dir / "score_ranking.png"))
        generated.append(plot_stability(combined, charts_dir / "stability.png"))
        generated.append(plot_per_sample_comparison(combined, charts_dir / "per_sample_comparison.png"))

    image_dir = session_dir / "image_quality"
    if image_dir.exists():
        p = plot_dimension_heatmap(image_dir, charts_dir / "dimension_heatmap_image.png",
                                   "Image Experiment - Dimension Scores")
        if p:
            generated.append(p)
        generated.append(plot_latency_comparison(image_dir, charts_dir / "latency_image.png",
                                                 "Image Experiment - VLM Latency"))

    video_dir = session_dir / "video_quality"
    if video_dir.exists():
        p = plot_dimension_heatmap(video_dir, charts_dir / "dimension_heatmap_video.png",
                                   "Video Experiment - Dimension Scores")
        if p:
            generated.append(p)
        generated.append(plot_latency_comparison(video_dir, charts_dir / "latency_video.png",
                                                 "Video Experiment - VLM Latency"))

    lg.info("Generated {} charts in {}", len(generated), charts_dir)
    return generated
