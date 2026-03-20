"""SOTA Model Validation Orchestrator.

Validates experiment design by running top-tier cloud models through DMXAPI.
All models run in parallel; DeepSeek is used as the unified judge.

Usage:
    uv run experiments/sota_validation/run_sota_validation.py
    uv run experiments/sota_validation/run_sota_validation.py --config my_config.toml
    uv run experiments/sota_validation/run_sota_validation.py --skip-video
    uv run experiments/sota_validation/run_sota_validation.py --skip-image
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from loguru import logger as lg

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.sota_validation.aggregate import (
    aggregate_results,
    generate_comparison_report,
    identify_problematic_samples,
)
from experiments.sota_validation.dmxapi_client import DMXAPIClient
from experiments.sota_validation.frame_extractor import extract_all_videos
from experiments.sota_validation.preflight import run_preflight
from experiments.sota_validation.run_image_experiment import run_image_experiment
from experiments.sota_validation.run_video_experiment import run_video_experiment
from experiments.sota_validation.visualization import generate_all_charts


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _create_session_dir(config: dict) -> Path:
    general = config.get("general", {})
    ts_fmt = general.get("timestamp_format", "%Y-%m-%d_%H-%M-%S")
    output_dir = general.get("output_dir", "results/v3-sota")
    timestamp = datetime.now().strftime(ts_fmt)
    session_dir = _PROJECT_ROOT / output_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _save_meta(session_dir: Path, config: dict, config_path: Path) -> None:
    shutil.copy2(config_path, session_dir / "config.toml")
    meta = {
        "experiment": "sota_validation",
        "start_time": datetime.now().isoformat(),
        "models": config.get("models", {}).get("list", []),
        "judge_model": config.get("judge", {}).get("model", "deepseek-chat"),
        "parallel": True,
        "python_version": sys.version,
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _update_meta(session_dir: Path, **kwargs: object) -> None:
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(kwargs)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_qa_report(
    session_dir: Path,
    config: dict,
    models: list[str],
    passed_models: list[str],
    incidents: list[dict],
    image_agg: dict,
    video_agg: dict,
    problems: list[dict],
) -> None:
    """Generate the QA report for the SOTA validation experiment."""
    qa_dir = _PROJECT_ROOT / "QA" / "v3" / "sota_validation"
    qa_dir.mkdir(parents=True, exist_ok=True)
    judge_model = config.get("judge", {}).get("model", "deepseek-chat")

    lines = [
        "# SOTA 模型校准实验 - 评估报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据来源**: `{session_dir.relative_to(_PROJECT_ROOT)}`",
        f"**裁判模型**: {judge_model}",
        "",
        "---",
        "",
        "## 1. 实验目的",
        "",
        "本实验验证当前评估设计（样本、提示词、评分规则）是否合理。",
        "如果 SOTA 模型普遍能获得接近满分，说明设计合理；",
        "如果某个样本让大部分 SOTA 模型失分，则优先怀疑设计问题。",
        "",
        "## 2. 实验方法",
        "",
        "- **被测模型**通过 DMXAPI 完成任务回答",
        f"- **{judge_model}** 统一作为裁判，根据 rubric 评分",
        "- 所有模型**并行执行**，各自独立写入结果",
        "- 每个样本 × 2 prompt 模式 × 5 次 = 10 次评估/模型",
        "",
        "## 3. 模型清单",
        "",
        f"计划测试 {len(models)} 个模型，通过预检 {len(passed_models)} 个：",
        "",
        "| # | 模型 | 预检 |",
        "|:---:|------|:---:|",
    ]
    for i, m in enumerate(models, 1):
        status = "PASS" if m in passed_models else "FAIL"
        lines.append(f"| {i} | {m} | {status} |")

    lines.extend([
        "",
        "## 4. 预检方法",
        "",
        "使用一张真实游戏截图（最短边 512px，JPEG 压缩），并行发送给所有模型。",
        "只有返回有意义响应的模型才进入正式实验。",
        "",
        "## 5. 视频处理方式",
        "",
        "- 每个视频预先抽取固定帧序列并缓存到 `tmp/sota_frames/`",
        "- 所有模型、所有轮次复用相同的帧序列",
        "- 帧序列作为多图输入发送给 API",
        "",
    ])

    # Image results
    if image_agg:
        lines.extend(["## 6. 图片实验结果", ""])
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
        lines.extend(["## 6. 图片实验结果", "", "未执行或无数据。", ""])

    # Video results
    if video_agg:
        lines.extend(["## 7. 视频实验结果", ""])
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
        lines.extend(["## 7. 视频实验结果", "", "未执行或无数据。", ""])

    # Charts
    charts_dir = session_dir / "charts"
    if charts_dir.exists():
        lines.extend(["## 8. 可视化", ""])
        for png in sorted(charts_dir.glob("*.png")):
            rel = png.relative_to(_PROJECT_ROOT)
            lines.append(f"![{png.stem}]({rel})")
            lines.append("")

    # Problems
    lines.extend(["## 9. 争议样本分析", ""])
    if problems:
        lines.append("以下样本中超过半数 SOTA 模型得分低于 7.0，应优先检查设计：")
        lines.append("")
        for p in problems:
            lines.append(f"- **{p['asset_id']}** ({p['prompt_mode']}): "
                         f"均分 {p['avg_across_models']}, "
                         f"{p['models_below_threshold']}/{p['total_models']} 个模型低于阈值")
    else:
        lines.append("未发现争议样本。所有样本在多数 SOTA 模型上得分 >= 7.0。")

    # Conclusions
    lines.extend([
        "",
        "## 10. 结论",
        "",
        "<!-- 此部分将在实验完成后根据实际数据填写 -->",
        "",
        "- 哪些 SOTA 模型能够稳定高分？",
        "- 哪些样本对 SOTA 模型仍不稳定？",
        "- 图片与视频样本中，哪个最容易造成分歧？",
        "- 当前 4 图 + 2 视频是否适合作为下一步小模型实验基准？",
        "- 如果不适合，应改哪些样本或改哪里？",
    ])

    if incidents:
        lines.extend(["", "## 附录：事件记录", ""])
        for inc in incidents:
            lines.append(f"- [{inc['model']}] {inc.get('phase', '')}: {inc.get('error', '')[:100]}")

    report_path = qa_dir / "评估报告.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    lg.info("QA report: {}", report_path)


async def _main(args: argparse.Namespace) -> None:
    config_path = _PROJECT_ROOT / args.config
    if not config_path.exists():
        lg.error("Config file not found: {}", config_path)
        sys.exit(1)

    config = load_config(config_path)
    models = config.get("models", {}).get("list", [])
    if not models:
        lg.error("No models specified in config")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")

    # DMXAPI client for VLM calls
    api_cfg = config.get("api", {})
    api_key = os.getenv(api_cfg.get("api_key_env", "DMX_API_KEY"), "")
    base_url = os.getenv(api_cfg.get("base_url_env", "DMX_API_BASE_URL"), "https://www.dmxapi.cn")
    if not api_key:
        lg.error("DMXAPI key not set (env: {})", api_cfg.get("api_key_env"))
        sys.exit(1)

    client = DMXAPIClient(
        api_key=api_key,
        base_url=base_url,
        timeout_s=api_cfg.get("timeout_s", 180),
        max_retries=api_cfg.get("max_retries", 3),
        retry_wait_s=api_cfg.get("retry_wait_s", 60),
    )

    # DeepSeek client for judge calls
    judge_cfg = config.get("judge", {})
    judge_model = judge_cfg.get("model", "deepseek-chat")
    judge_key = os.getenv(judge_cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    judge_url = os.getenv(judge_cfg.get("base_url_env", "DEEPSEEK_API_BASE_URL"), "https://api.deepseek.com")
    if not judge_key:
        lg.error("DeepSeek API key not set (env: {})", judge_cfg.get("api_key_env"))
        sys.exit(1)

    judge_client = DMXAPIClient(
        api_key=judge_key,
        base_url=judge_url,
        timeout_s=judge_cfg.get("timeout_s", 120),
        max_retries=judge_cfg.get("max_retries", 3),
        retry_wait_s=judge_cfg.get("retry_wait_s", 30),
    )

    session_dir = _create_session_dir(config)
    _save_meta(session_dir, config, config_path)

    log_path = session_dir / "experiment.log"
    lg.add(str(log_path), level="DEBUG", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}")

    lg.info("=" * 70)
    lg.info("SOTA Validation Experiment (parallel, DeepSeek judge)")
    lg.info("Session: {}", session_dir)
    lg.info("Models: {} | Judge: {}", len(models), judge_model)
    lg.info("=" * 70)

    # Phase 0: Preflight (parallel)
    lg.info("Phase 0: Preflight (parallel)")
    passed_models, incidents = await run_preflight(client, models)

    if not passed_models:
        lg.error("No models passed preflight, aborting")
        _update_meta(session_dir, end_time=datetime.now().isoformat(),
                     passed_models=[], incidents=incidents)
        await client.close()
        await judge_client.close()
        sys.exit(1)

    # Phase 1: Image experiment (all models in parallel)
    image_agg: dict = {}
    do_image = config.get("experiments", {}).get("image_quality", {}).get("enabled", True)
    if do_image and not args.skip_image:
        lg.info("=" * 70)
        lg.info("Phase 1: Image Quality Experiment (parallel)")
        image_dir = session_dir / "image_quality"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_runs = config.get("experiments", {}).get("image_quality", {}).get("runs", 5)
        await run_image_experiment(
            client, judge_client, judge_model, passed_models, image_dir, runs=image_runs
        )
        image_agg = aggregate_results(image_dir, image_dir / "aggregated.csv", "image")
    else:
        lg.info("Skipping image experiment")

    # Phase 2: Video experiment (all models in parallel)
    video_agg: dict = {}
    do_video = config.get("experiments", {}).get("video_quality", {}).get("enabled", True)
    if do_video and not args.skip_video:
        lg.info("=" * 70)
        lg.info("Phase 2: Video Frame Extraction")
        video_cfg = config.get("video", {})
        cache_dir = _PROJECT_ROOT / video_cfg.get("cache_dir", "tmp/sota_frames")
        cached_frames = extract_all_videos(
            cache_dir,
            interval_s=video_cfg.get("frame_interval_s", 1.0),
            max_frames=video_cfg.get("max_frames", 30),
        )

        lg.info("Phase 2b: Video Quality Experiment (parallel)")
        video_dir = session_dir / "video_quality"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_runs = config.get("experiments", {}).get("video_quality", {}).get("runs", 5)
        await run_video_experiment(
            client, judge_client, judge_model, passed_models, cached_frames, video_dir, runs=video_runs
        )
        video_agg = aggregate_results(video_dir, video_dir / "aggregated.csv", "video")
    else:
        lg.info("Skipping video experiment")

    # Phase 3: Aggregation + Reports + Charts
    lg.info("=" * 70)
    lg.info("Phase 3: Aggregation, Charts & Reporting")
    all_agg = {}
    all_agg.update(image_agg)
    all_agg.update(video_agg)
    problems = identify_problematic_samples(all_agg)

    generate_comparison_report(
        image_agg, video_agg, problems,
        session_dir / "comparison_report.md",
        incidents=incidents,
    )

    generate_all_charts(session_dir, image_agg, video_agg)

    _generate_qa_report(
        session_dir, config, models, passed_models,
        incidents, image_agg, video_agg, problems,
    )

    _update_meta(
        session_dir,
        end_time=datetime.now().isoformat(),
        passed_models=passed_models,
        failed_models=[m for m in models if m not in passed_models],
        incidents=incidents,
        problematic_samples=len(problems),
    )

    await client.close()
    await judge_client.close()

    lg.info("=" * 70)
    lg.info("SOTA Validation complete")
    lg.info("Passed: {}/{}", len(passed_models), len(models))
    lg.info("Session: {}", session_dir)
    lg.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="SOTA Model Validation Experiment")
    parser.add_argument(
        "--config",
        default="config_sota_validation.toml",
        help="Config file (default: config_sota_validation.toml)",
    )
    parser.add_argument("--skip-image", action="store_true", help="Skip image experiment")
    parser.add_argument("--skip-video", action="store_true", help="Skip video experiment")
    args = parser.parse_args()

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
