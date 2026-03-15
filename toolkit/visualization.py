"""Reusable scientific visualization module for benchmark results.

Generates publication-quality charts using matplotlib + seaborn.
All functions accept plain dicts/lists and output PNG files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np


_PALETTE = "muted"
_FIG_DPI = 180
_FONT_SIZES = {"title": 13, "label": 11, "tick": 9, "legend": 9}


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", palette=_PALETTE, font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": _FIG_DPI,
        "savefig.dpi": _FIG_DPI,
        "savefig.bbox": "tight",
        "axes.titlesize": _FONT_SIZES["title"],
        "axes.labelsize": _FONT_SIZES["label"],
        "xtick.labelsize": _FONT_SIZES["tick"],
        "ytick.labelsize": _FONT_SIZES["tick"],
        "legend.fontsize": _FONT_SIZES["legend"],
        "font.family": "sans-serif",
    })


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def plot_metric_ranking(
    model_ids: list[str],
    values: list[float],
    metric_label: str,
    title: str,
    output_path: Path,
    *,
    higher_is_better: bool = True,
    unit: str = "",
) -> Path:
    """Horizontal bar chart ranking models by a single metric."""
    _apply_style()

    names = [_short_name(m) for m in model_ids]
    order = sorted(range(len(values)), key=lambda i: values[i],
                   reverse=higher_is_better)
    sorted_names = [names[i] for i in order]
    sorted_vals = [values[i] for i in order]

    colors = sns.color_palette(_PALETTE, len(sorted_names))

    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(sorted_names) + 1)))
    bars = ax.barh(sorted_names, sorted_vals, color=colors, edgecolor="white",
                   linewidth=0.5)

    for bar, val in zip(bars, sorted_vals):
        fmt = f"{val:.1f}{unit}" if val >= 1 else f"{val:.3f}{unit}"
        ax.text(bar.get_width() + max(sorted_vals) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                fmt, va="center", fontsize=_FONT_SIZES["tick"])

    ax.set_xlabel(metric_label)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    sns.despine(left=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_scenario_comparison(
    scenario_names: list[str],
    model_ids: list[str],
    values_matrix: list[list[float]],
    metric_label: str,
    title: str,
    output_path: Path,
) -> Path:
    """Grouped bar chart comparing models across scenarios."""
    _apply_style()

    n_scenarios = len(scenario_names)
    n_models = len(model_ids)
    x = np.arange(n_scenarios)
    width = 0.8 / n_models

    colors = sns.color_palette(_PALETTE, n_models)
    short_names = [_short_name(m) for m in model_ids]

    fig, ax = plt.subplots(figsize=(max(8, n_scenarios * 1.5), 5))
    for i, (name, vals) in enumerate(zip(short_names, values_matrix)):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=name, color=colors[i],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=25, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    sns.despine()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_summary_table(
    headers: list[str],
    rows: list[list[str]],
    title: str,
    output_path: Path,
    *,
    highlight_col: int | None = None,
    highlight_best: str = "min",
) -> Path:
    """Render a publication-quality metrics table as a PNG image."""
    _apply_style()

    n_rows = len(rows)
    n_cols = len(headers)
    fig_h = max(2.0, 0.4 * n_rows + 1.2)
    fig_w = max(8, 1.2 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(_FONT_SIZES["tick"])
    table.scale(1, 1.5)

    header_color = sns.color_palette(_PALETTE)[0]
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(n_rows):
        bg = "#f9f9f9" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i + 1, j].set_facecolor(bg)

    if highlight_col is not None and rows:
        try:
            numeric = [float(rows[i][highlight_col]) for i in range(n_rows)]
            best_idx = numeric.index(
                min(numeric) if highlight_best == "min" else max(numeric)
            )
            cell = table[best_idx + 1, highlight_col]
            cell.set_text_props(fontweight="bold", color="#d32f2f")
        except (ValueError, IndexError):
            pass

    ax.set_title(title, fontweight="bold",
                 fontsize=_FONT_SIZES["title"], pad=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_video_comparison(
    model_ids: list[str],
    keyframes: list[float],
    dropped: list[float],
    descriptions: list[float],
    title: str,
    output_path: Path,
) -> Path:
    """Grouped bar chart for video understanding metrics per model."""
    _apply_style()

    names = [_short_name(m) for m in model_ids]
    x = np.arange(len(names))
    width = 0.25

    colors = sns.color_palette(_PALETTE, 3)

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    ax.bar(x - width, keyframes, width, label="Avg Keyframes", color=colors[0])
    ax.bar(x, descriptions, width, label="Avg Descriptions", color=colors[1])
    ax.bar(x + width, dropped, width, label="Avg Dropped", color=colors[2])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    sns.despine()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
