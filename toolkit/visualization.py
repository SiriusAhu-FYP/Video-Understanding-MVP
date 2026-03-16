"""可复用的科研可视化模块。

使用 matplotlib + seaborn 生成出版质量的图表。
所有函数接受纯 dict/list 输入，输出 PNG 文件。
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


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _truncate(name: str, max_len: int = 28) -> str:
    return name[:max_len - 2] + ".." if len(name) > max_len else name


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
    """水平条形图：按单一指标对模型排名。"""
    _apply_style()

    names = [_truncate(_short_name(m)) for m in model_ids]
    order = sorted(range(len(values)), key=lambda i: values[i],
                   reverse=higher_is_better)
    sorted_names = [names[i] for i in order]
    sorted_vals = [values[i] for i in order]

    colors = sns.color_palette(_PALETTE, len(sorted_names))

    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(sorted_names) + 1.2)))
    bars = ax.barh(sorted_names, sorted_vals, color=colors, edgecolor="white",
                   linewidth=0.5)

    for bar, val in zip(bars, sorted_vals):
        fmt = f"{val:.1f}{unit}" if val >= 1 else f"{val:.3f}{unit}"
        ax.text(bar.get_width() + max(sorted_vals) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                fmt, va="center", fontsize=_FONT_SIZES["tick"])

    ax.set_xlabel(metric_label)
    ax.set_title(title, fontweight="bold", pad=14)
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
    """分组柱状图：跨场景对比多个模型。"""
    _apply_style()

    n_scenarios = len(scenario_names)
    n_models = len(model_ids)
    x = np.arange(n_scenarios)
    width = 0.8 / n_models

    colors = sns.color_palette(_PALETTE, n_models)
    short_names = [_truncate(_short_name(m), 22) for m in model_ids]

    fig, ax = plt.subplots(figsize=(max(9, n_scenarios * 1.8), 5.5))
    for i, (name, vals) in enumerate(zip(short_names, values_matrix)):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=name, color=colors[i],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=25, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
              fontsize=8)
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
    """将指标汇总表渲染为 PNG 图片（自动列宽，防止文字溢出）。"""
    _apply_style()

    n_rows = len(rows)
    n_cols = len(headers)

    max_content_len = max(
        max((len(str(c)) for c in row), default=0) for row in rows
    ) if rows else 6
    col_w = max(1.4, min(2.5, max_content_len * 0.14))
    fig_w = max(10, col_w * n_cols + 1.5)
    fig_h = max(2.5, 0.45 * n_rows + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(n_cols)))
    table.scale(1, 1.6)

    header_color = sns.color_palette(_PALETTE)[0]
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    for i in range(n_rows):
        bg = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i + 1, j].set_facecolor(bg)
            table[i + 1, j].set_text_props(fontsize=8.5)

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
                 fontsize=_FONT_SIZES["title"], pad=18)

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
    """分组柱状图：视频理解各模型关键帧/描述/丢弃帧对比。"""
    _apply_style()

    names = [_truncate(_short_name(m), 22) for m in model_ids]
    x = np.arange(len(names))
    width = 0.25

    colors = sns.color_palette(_PALETTE, 3)

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.4), 5.5))
    ax.bar(x - width, keyframes, width, label="平均关键帧", color=colors[0])
    ax.bar(x, descriptions, width, label="平均描述数", color=colors[1])
    ax.bar(x + width, dropped, width, label="平均丢弃帧", color=colors[2])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("数量")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.legend(frameon=False)
    sns.despine()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_gpu_memory_comparison(
    model_ids: list[str],
    gpu_util: list[float],
    vram_mb: list[int],
    title: str,
    output_path: Path,
) -> Path:
    """双轴柱状图：各模型 GPU 利用率与显存占用对比。"""
    _apply_style()

    names = [_truncate(_short_name(m), 22) for m in model_ids]
    x = np.arange(len(names))
    colors = sns.color_palette(_PALETTE, 2)

    fig, ax1 = plt.subplots(figsize=(max(9, len(names) * 1.4), 5.5))
    bars1 = ax1.bar(x - 0.2, gpu_util, 0.35, label="GPU 利用率",
                    color=colors[0], edgecolor="white")
    ax1.set_ylabel("GPU Memory Utilization")
    ax1.set_ylim(0, max(gpu_util) * 1.3 if gpu_util else 1)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, vram_mb, 0.35, label="VRAM (MB)",
                    color=colors[1], edgecolor="white")
    ax2.set_ylabel("VRAM (MB)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_title(title, fontweight="bold", pad=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False,
               loc="upper right")
    sns.despine(right=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
