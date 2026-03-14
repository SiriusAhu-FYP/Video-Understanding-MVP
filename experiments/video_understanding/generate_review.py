"""实验日志审阅文档生成器。

读取单次实验运行的日志文件夹（含 frames/ 和 run_log.json），
生成包含帧时间线、关键帧判定原因、VLM 输出等信息的 Markdown 审阅文档。

用法:
    uv run experiments/video_understanding/generate_review.py <run_dir>
    uv run experiments/video_understanding/generate_review.py reports/Qwen_xxx/video_MC_01_run1

也可作为模块导入:
    from generate_review import generate_review
    generate_review(Path("path/to/run_dir"))
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def generate_review(run_dir: Path, output_path: Path | None = None) -> Path:
    """读取单次实验运行的日志，生成人类可读的审阅文档。

    Args:
        run_dir: 包含 ``frames/`` 和 ``run_log.json`` 的运行目录。
        output_path: 输出 md 文件路径，默认为 ``run_dir/review.md``。

    Returns:
        生成的 review.md 文件路径。

    Raises:
        FileNotFoundError: 当 run_log.json 不存在时抛出。
    """
    log_path = run_dir / "run_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"run_log.json 不存在: {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        run_log = json.load(f)

    if output_path is None:
        output_path = run_dir / "review.md"

    video_name = run_log.get("video", "未知视频")
    model_name = run_log.get("model", "未知模型")
    config = run_log.get("config", {})
    stats = run_log.get("stats", {})
    frames = run_log.get("frames", [])
    summary_text = run_log.get("summary", "")

    lines: list[str] = []

    # 标题与元信息
    lines.append(f"# 实验审阅: {video_name}")
    lines.append("")
    lines.append("## 运行元信息")
    lines.append("")
    lines.append(f"- **模型**: `{model_name}`")
    lines.append(f"- **视频**: `{video_name}`")
    lines.append(f"- **运行目录**: `{run_dir.name}`")
    lines.append("")
    lines.append("### 配置参数")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")
    for key, value in config.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # 统计摘要
    lines.append("## 统计摘要")
    lines.append("")
    lines.append(f"- **总采样帧数**: {stats.get('total_sampled', 0)}")
    lines.append(f"- **关键帧数**: {stats.get('total_keyframes', 0)}")
    lines.append(f"- **丢弃帧数**: {stats.get('total_dropped', 0)}")
    lines.append(f"- **录制时长**: {stats.get('duration_s', 0):.1f}s")

    total = stats.get("total_sampled", 0)
    keyframes = stats.get("total_keyframes", 0)
    if total > 0:
        lines.append(f"- **关键帧率**: {keyframes / total * 100:.1f}%")
    lines.append("")

    # 帧时间线表格
    lines.append("## 帧时间线")
    lines.append("")
    lines.append(
        "| 帧序号 | 时间戳 | 差异值 | 关键帧 | 判定原因 | 图片 | VLM 描述 |"
    )
    lines.append(
        "|--------|--------|--------|--------|----------|------|----------|"
    )

    for frame in frames:
        idx = frame.get("frame_idx", "?")
        ts_ms = frame.get("timestamp_ms", 0)
        ts_s = f"{ts_ms / 1000:.1f}s"
        diff = frame.get("diff_value")
        diff_str = f"{diff:.2f}" if diff is not None else "-"
        is_key = frame.get("is_keyframe", False)
        key_mark = "**是**" if is_key else "否"
        reason = frame.get("reason", "")
        img_file = frame.get("image_filename")
        img_link = f"[{img_file}](frames/{img_file})" if img_file else "-"
        vlm = frame.get("vlm_response")
        vlm_str = _truncate(vlm, 120) if vlm else "-"

        lines.append(
            f"| {idx} | {ts_s} | {diff_str} | {key_mark} | {reason} | {img_link} | {vlm_str} |"
        )

    lines.append("")

    # DeepSeek 总结
    lines.append("## DeepSeek 最终总结")
    lines.append("")
    if summary_text:
        lines.append(f"```\n{summary_text}\n```")
    else:
        lines.append("*总结不可用*")
    lines.append("")

    # 关键帧详细描述
    key_frames = [f for f in frames if f.get("is_keyframe") and f.get("vlm_response")]
    if key_frames:
        lines.append("## 关键帧详细描述")
        lines.append("")
        for frame in key_frames:
            idx = frame["frame_idx"]
            ts_s = frame["timestamp_ms"] / 1000
            img_file = frame.get("image_filename", "")
            vlm = frame.get("vlm_response", "")
            lines.append(f"### 帧 #{idx} ({ts_s:.1f}s)")
            lines.append("")
            if img_file:
                lines.append(f"![帧 {idx}](frames/{img_file})")
            lines.append("")
            lines.append(f"> {vlm}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _truncate(text: str, max_len: int) -> str:
    """截断文本并添加省略号。"""
    text = text.replace("\n", " ").replace("|", "\\|")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="实验日志审阅文档生成器")
    parser.add_argument("run_dir", type=Path, help="包含 frames/ 和 run_log.json 的运行目录")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出文件路径")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir

    try:
        out = generate_review(run_dir, args.output)
        print(f"审阅文档已生成: {out}")
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
