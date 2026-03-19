"""Generate dissertation-quality charts for model selection & validation report.

Reads aggregated CSVs from ./data/ and produces publication-ready PNGs in ./charts/.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHARTS = ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

_DPI = 250
_CJK = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]

SOTA_TIER_1 = ["gpt-5.4", "gpt-4.1", "claude-sonnet-4-6", "claude-opus-4-6"]
SOTA_TIER_2 = ["kimi-k2.5", "Qwen3-VL-235B-A22B-Instruct", "qwen3.5-397b-a17b",
               "qwen2.5-vl-72b-instruct"]
SOTA_TIER_3 = ["gemini-3.1-pro-preview", "glm-5"]

LOCAL_MODELS = {
    "Qwen3-VL-2B": "local_Qwen3-VL-2B_scores.csv",
    "InternVL2.5-2B": "local_InternVL2_5-2B_scores.csv",
    "DeepSeek-VL2-tiny": "local_DeepSeek-VL2-tiny_scores.csv",
    "SmolVLM2-2.2B": "local_SmolVLM2-2.2B_scores.csv",
}

LOCAL_SPEED = {
    "Qwen3-VL-2B": "local_speed_Qwen_Qwen3-VL-2B-Instruct.csv",
    "InternVL2.5-2B": "local_speed_OpenGVLab_InternVL2_5-2B.csv",
    "DeepSeek-VL2-tiny": "local_speed_deepseek-ai_deepseek-vl2-tiny.csv",
    "SmolVLM2-2.2B": "local_speed_HuggingFaceTB_SmolVLM2-2.2B-Instruct.csv",
}

MODEL_ORDER = (SOTA_TIER_1 + SOTA_TIER_2 + SOTA_TIER_3)


def _apply_style():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": _DPI, "savefig.dpi": _DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
        "axes.titlesize": 14, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.sans-serif": _CJK, "axes.unicode_minus": False,
    })


def _short(name: str, mx: int = 20) -> str:
    n = name.split("/")[-1] if "/" in name else name
    return n[:mx-2] + ".." if len(n) > mx else n


def _read_agg(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _model_means(rows: list[dict]) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        scores[r["model"]].append(float(r["mean_score"]))
    return {m: mean(v) for m, v in scores.items()}


def _model_stdevs(rows: list[dict]) -> dict[str, float]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(float(r["mean_score"]))
    result = {}
    for m, v in by_model.items():
        result[m] = stdev(v) if len(v) > 1 else 0.0
    return result


# ── Chart 1: SOTA Model Ranking (Image + Video combined) ──────────────────

def chart_sota_ranking():
    img_rows = _read_agg(DATA / "sota_image_aggregated.csv")
    vid_rows = _read_agg(DATA / "sota_video_aggregated.csv")

    img_means = _model_means(img_rows)
    vid_means = _model_means(vid_rows)

    models = [m for m in MODEL_ORDER if m in img_means]
    img_vals = [img_means.get(m, 0) for m in models]
    vid_vals = [vid_means.get(m, 0) for m in models]

    idx = np.argsort([-((iv + vv) / 2) for iv, vv in zip(img_vals, vid_vals)])
    models = [models[i] for i in idx]
    img_vals = [img_vals[i] for i in idx]
    vid_vals = [vid_vals[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x - w/2, img_vals, w, label="Image Quality", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + w/2, vid_vals, w, label="Video Quality", color="#DD8452", edgecolor="white")

    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Mean Score (0-10)")
    ax.set_title("SOTA Model Quality Ranking — Image vs Video Tasks")
    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in models], rotation=30, ha="right")
    ax.set_ylim(0, 11)
    ax.legend(loc="upper right")
    ax.axhline(y=6.0, color="red", linestyle="--", alpha=0.5, label="Threshold (6.0)")

    fig.tight_layout()
    fig.savefig(CHARTS / "fig1_sota_model_ranking.png")
    plt.close(fig)
    print("  ✓ fig1_sota_model_ranking.png")


# ── Chart 2: Local Model Quality Comparison ───────────────────────────────

def chart_local_quality():
    model_task_scores: dict[str, dict[str, list[float]]] = {}

    for display_name, fname in LOCAL_MODELS.items():
        path = DATA / fname
        if not path.exists():
            continue
        scores_by_task: dict[str, list[float]] = defaultdict(list)
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = f"{r['asset_id']}\n{r['prompt_mode']}"
                scores_by_task[key].append(float(r["total_score"]))
        model_task_scores[display_name] = {k: mean(v) for k, v in scores_by_task.items()}

    all_tasks = sorted(set().union(*(d.keys() for d in model_task_scores.values())))
    models = list(model_task_scores.keys())
    n_models = len(models)
    n_tasks = len(all_tasks)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_tasks)
    w = 0.8 / n_models
    colors = sns.color_palette("muted", n_models)

    for i, model in enumerate(models):
        vals = [model_task_scores[model].get(t, 0) for t in all_tasks]
        offset = (i - n_models/2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=model, color=colors[i], edgecolor="white")

    ax.set_ylabel("Mean Score (0-10)")
    ax.set_title("Local Model Quality — Per-Task Breakdown (DeepSeek Judge)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 11)
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(y=6.0, color="red", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(CHARTS / "fig2_local_model_quality.png")
    plt.close(fig)
    print("  ✓ fig2_local_model_quality.png")


# ── Chart 3: Local Model Speed Comparison ─────────────────────────────────

def chart_local_speed():
    model_ttft: dict[str, list[float]] = {}
    model_tps: dict[str, list[float]] = {}
    model_vram: dict[str, float] = {}

    for display_name, fname in LOCAL_SPEED.items():
        path = DATA / fname
        if not path.exists():
            continue
        ttfts, tpss = [], []
        vram_val = 0
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ttfts.append(float(r["ttft_s"]))
                tpss.append(float(r["throughput_tps"]))
                vram_val = float(r["vram_mb"])
        model_ttft[display_name] = ttfts
        model_tps[display_name] = tpss
        model_vram[display_name] = vram_val

    models = list(model_ttft.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # TTFT
    ax = axes[0]
    means = [mean(model_ttft[m]) for m in models]
    colors = sns.color_palette("muted", len(models))
    bars = ax.bar(models, means, color=colors, edgecolor="white")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v*1000:.0f}ms", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("TTFT (seconds)")
    ax.set_title("Time to First Token")
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)

    # Throughput
    ax = axes[1]
    means = [mean(model_tps[m]) for m in models]
    bars = ax.bar(models, means, color=colors, edgecolor="white")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Generation Throughput")
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)

    # VRAM
    ax = axes[2]
    vrams = [model_vram[m] for m in models]
    bars = ax.bar(models, vrams, color=colors, edgecolor="white")
    for bar, v in zip(bars, vrams):
        ax.text(bar.get_x() + bar.get_width()/2, v + 50,
                f"{v/1024:.1f}GB", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("VRAM (MB)")
    ax.set_title("GPU Memory Usage")
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)

    fig.suptitle("Local Model Inference Performance (RTX 4080 SUPER, vLLM)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS / "fig3_local_model_speed.png")
    plt.close(fig)
    print("  ✓ fig3_local_model_speed.png")


# ── Chart 4: SOTA vs Local Gap Analysis ───────────────────────────────────

def chart_sota_vs_local():
    img_rows = _read_agg(DATA / "sota_image_aggregated.csv")
    sota_means = _model_means(img_rows)

    local_means: dict[str, float] = {}
    for display_name, fname in LOCAL_MODELS.items():
        path = DATA / fname
        if not path.exists():
            continue
        all_scores = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                all_scores.append(float(r["total_score"]))
        local_means[display_name] = mean(all_scores)

    top_sota = sorted(sota_means.items(), key=lambda x: -x[1])[:5]
    all_items = top_sota + list(local_means.items())
    names = [_short(n) for n, _ in all_items]
    values = [v for _, v in all_items]

    colors = []
    for i in range(len(all_items)):
        if i < len(top_sota):
            colors.append("#4C72B0")
        else:
            colors.append("#C44E52")

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, values):
        ax.text(v + 0.15, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Score (0-10)")
    ax.set_title("SOTA (Top-5) vs Local 2B Models — Image Quality Gap")
    ax.set_xlim(0, 11)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4C72B0", label="SOTA (Cloud API)"),
                       Patch(facecolor="#C44E52", label="Local 2B Models")]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    fig.savefig(CHARTS / "fig4_sota_vs_local_gap.png")
    plt.close(fig)
    print("  ✓ fig4_sota_vs_local_gap.png")


# ── Chart 5: Dimension Radar — Best SOTA vs Best Local ────────────────────

def chart_radar_comparison():
    dims = ["core_understanding", "key_information_coverage",
            "task_completion", "assistant_value", "hallucination_control"]
    dim_labels = ["Core\nUnderstanding", "Key Info\nCoverage",
                  "Task\nCompletion", "Assistant\nValue", "Hallucination\nControl"]

    def _load_dim_means(path: Path) -> dict[str, float]:
        dm: dict[str, list[float]] = defaultdict(list)
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for d in dims:
                    if d in r:
                        dm[d].append(float(r[d]))
        return {d: mean(v) if v else 0 for d, v in dm.items()}

    local_best = _load_dim_means(DATA / "local_Qwen3-VL-2B_scores.csv")

    sota_files = list(DATA.glob("sota_image_*_results.csv"))
    best_sota_name, best_sota_dims = "", {}
    best_total = -1
    for sf in sota_files:
        dm: dict[str, list[float]] = defaultdict(list)
        with open(sf, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ds = r.get("dimension_scores", "{}")
                import json
                try:
                    parsed = json.loads(ds.replace("'", '"'))
                except Exception:
                    continue
                for d in dims:
                    if d in parsed:
                        dm[d].append(float(parsed[d]))
        if not dm:
            continue
        means = {d: mean(v) if v else 0 for d, v in dm.items()}
        total = sum(means.values())
        if total > best_total:
            best_total = total
            best_sota_dims = means
            best_sota_name = sf.stem.replace("sota_image_", "").replace("_results", "")

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    local_vals = [local_best.get(d, 0) for d in dims] + [local_best.get(dims[0], 0)]
    sota_vals = [best_sota_dims.get(d, 0) for d in dims] + [best_sota_dims.get(dims[0], 0)]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, sota_vals, "o-", linewidth=2, label=f"SOTA: {_short(best_sota_name)}", color="#4C72B0")
    ax.fill(angles, sota_vals, alpha=0.15, color="#4C72B0")
    ax.plot(angles, local_vals, "s-", linewidth=2, label="Local: Qwen3-VL-2B", color="#C44E52")
    ax.fill(angles, local_vals, alpha=0.15, color="#C44E52")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=9)
    ax.set_ylim(0, 2.1)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0", "0.5", "1.0", "1.5", "2.0"], fontsize=7)
    ax.set_title("Scoring Dimension Radar — Best SOTA vs Best Local 2B", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    fig.savefig(CHARTS / "fig5_radar_comparison.png")
    plt.close(fig)
    print("  ✓ fig5_radar_comparison.png")


# ── Chart 6: SOTA Image Heatmap (all models × all tasks) ─────────────────

def chart_sota_heatmap():
    rows = _read_agg(DATA / "sota_image_aggregated.csv")

    models_set = set()
    tasks_set = set()
    data_map: dict[tuple[str, str], float] = {}
    for r in rows:
        m = r["model"]
        t = f"{r['asset_id']} / {r['prompt_mode']}"
        models_set.add(m)
        tasks_set.add(t)
        data_map[(m, t)] = float(r["mean_score"])

    ordered_models = [m for m in MODEL_ORDER if m in models_set]
    ordered_tasks = sorted(tasks_set)

    matrix = np.zeros((len(ordered_models), len(ordered_tasks)))
    for i, m in enumerate(ordered_models):
        for j, t in enumerate(ordered_tasks):
            matrix[i, j] = data_map.get((m, t), np.nan)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=ordered_tasks,
                yticklabels=[_short(m) for m in ordered_models],
                vmin=0, vmax=10, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Mean Score"})
    ax.set_title("SOTA Model × Task Heatmap — Image Quality (5 runs each)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    fig.tight_layout()
    fig.savefig(CHARTS / "fig6_sota_heatmap.png")
    plt.close(fig)
    print("  ✓ fig6_sota_heatmap.png")


# ── Chart 7: Combined Score Summary Table ─────────────────────────────────

def chart_summary_table():
    img_rows = _read_agg(DATA / "sota_image_aggregated.csv")
    vid_rows = _read_agg(DATA / "sota_video_aggregated.csv")
    img_means = _model_means(img_rows)
    vid_means = _model_means(vid_rows)
    img_stds = _model_stdevs(img_rows)
    vid_stds = _model_stdevs(vid_rows)

    models = [m for m in MODEL_ORDER if m in img_means]
    table_data = []
    for m in models:
        im, ist = img_means.get(m, 0), img_stds.get(m, 0)
        vm, vst = vid_means.get(m, 0), vid_stds.get(m, 0)
        combined = (im + vm) / 2
        table_data.append([_short(m, 26), f"{im:.2f}", f"{ist:.2f}",
                           f"{vm:.2f}", f"{vst:.2f}", f"{combined:.2f}"])

    table_data.sort(key=lambda x: -float(x[5]))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    col_labels = ["Model", "Image\nMean", "Image\nStdev", "Video\nMean", "Video\nStdev", "Combined"]
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row <= 3:
            cell.set_facecolor("#e8f0fe")
        cell.set_edgecolor("#cccccc")

    ax.set_title("SOTA Model Score Summary (Image + Video)", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(CHARTS / "fig7_summary_table.png")
    plt.close(fig)
    print("  ✓ fig7_summary_table.png")


def main():
    _apply_style()
    print("Generating dissertation charts...")
    chart_sota_ranking()
    chart_local_quality()
    chart_local_speed()
    chart_sota_vs_local()
    chart_radar_comparison()
    chart_sota_heatmap()
    chart_summary_table()
    print(f"\nAll charts saved to {CHARTS}")


if __name__ == "__main__":
    main()
