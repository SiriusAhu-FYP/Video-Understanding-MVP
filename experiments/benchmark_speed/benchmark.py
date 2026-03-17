"""VLM 多场景基准测试脚本。

用法:
    uv run benchmark.py                                   # 自动检测模型，运行全部场景
    uv run benchmark.py --runs 5                           # 每场景每图跑 5 次
    uv run benchmark.py --scenarios short_desc detailed_desc  # 只跑指定场景
    uv run benchmark.py --base-url http://localhost:8000/v1   # 指定 vLLM 地址
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from loguru import logger as lg
from openai import OpenAI

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXPERIMENT_DIR.parent.parent
_REPORTS_DIR = _EXPERIMENT_DIR / "reports"

sys.path.insert(0, str(_PROJECT_ROOT))

from ahu_paimon_toolkit.utils.image import encode_image, get_image_mime, load_images as _load_images_full
from ahu_paimon_toolkit.utils.gpu import get_gpu_memory_mb
from ahu_paimon_toolkit.vlm.model_utils import detect_model, model_short_name

ASSETS_IMAGES_DIR = _PROJECT_ROOT / "assets" / "images"


# ── 场景定义 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    prompt: str
    max_tokens: int


SCENARIOS: list[Scenario] = [
    Scenario(
        id="short_desc",
        name="短文本描述",
        prompt="用一句话简要概括这张游戏截图的画面内容。",
        max_tokens=64,
    ),
    Scenario(
        id="detailed_desc",
        name="详细描述",
        prompt=(
            "请用中文详细描述这张游戏截图中的所有元素，"
            "包括角色、环境、颜色、光照、构图、"
            "以及任何可以观察到的细节。尽可能全面。"
        ),
        max_tokens=512,
    ),
    Scenario(
        id="object_detection",
        name="JSON 物体检测",
        prompt=(
            "识别这张游戏截图中的所有主要物体和角色，"
            "输出 JSON 格式，每个对象包含 bbox_2d (左上角x, 左上角y, 右下角x, 右下角y) 和 label。"
            "格式示例：{\"objects\": [{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"名称\"}, ...]}"
        ),
        max_tokens=512,
    ),
    Scenario(
        id="action_analysis",
        name="角色动作分析",
        prompt=(
            "分析这张游戏截图中角色正在执行什么动作或行为。"
            "描述角色的姿态、朝向、与环境的交互，以及可能正在进行的游戏操作。"
        ),
        max_tokens=256,
    ),
    Scenario(
        id="ui_recognition",
        name="游戏 UI 识别",
        prompt=(
            "识别这张游戏截图中所有可见的 UI/HUD 元素，"
            "例如：血条、蓝条、小地图、技能图标、物品栏、计分板、对话框等。"
            "逐一列出每个 UI 元素的位置（画面的哪个区域）和当前显示的状态/数值。"
        ),
        max_tokens=256,
    ),
]

SCENARIO_MAP: dict[str, Scenario] = {s.id: s for s in SCENARIOS}


# ── 工具函数 ──────────────────────────────────────────────────────

def load_images(assets_dir: Path | None = None) -> list[tuple[str, str, str]]:
    """Load images returning (filename, base64, mime) tuples."""
    full = _load_images_full(assets_dir)
    return [(name, b64, mime) for name, b64, mime, _ in full]


# ── 单次推理 ──────────────────────────────────────────────────────

@dataclass
class RunResult:
    model: str
    scenario_id: str
    scenario_name: str
    image_name: str
    run_idx: int
    ttft_s: float
    throughput_tps: float
    total_time_s: float
    output_tokens: int
    vram_mb: int
    response_text: str


def run_single_inference(
    client: OpenAI,
    model: str,
    scenario: Scenario,
    image_name: str,
    image_b64: str,
    image_mime: str,
    run_idx: int,
) -> RunResult:
    """执行单次推理并测量性能指标。"""
    start_time = time.perf_counter()
    vram_before = get_gpu_memory_mb()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": scenario.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
                ],
            },
        ],
        stream=True,
        temperature=0.0,
        max_tokens=scenario.max_tokens,
        stream_options={"include_usage": True},
    )

    ttft: float | None = None
    output_tokens = 0
    chunks: list[str] = []

    for chunk in response:
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.perf_counter() - start_time
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage is not None:
            output_tokens = chunk.usage.completion_tokens

    total_time = time.perf_counter() - start_time
    vram_after = get_gpu_memory_mb()

    if ttft is None:
        ttft = total_time

    generation_time = total_time - ttft
    throughput = output_tokens / generation_time if generation_time > 0 else 0.0

    response_text = "".join(chunks)

    return RunResult(
        model=model,
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        image_name=image_name,
        run_idx=run_idx,
        ttft_s=round(ttft, 4),
        throughput_tps=round(throughput, 2),
        total_time_s=round(total_time, 4),
        output_tokens=output_tokens,
        vram_mb=vram_after,
        response_text=response_text,
    )


# ── CSV 写入 ──────────────────────────────────────────────────────

CSV_HEADERS = [
    "model", "scenario_id", "scenario_name", "image_name", "run_idx",
    "ttft_s", "throughput_tps", "total_time_s", "output_tokens", "vram_mb",
    "response_preview",
]


def init_csv(csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)


def append_csv(csv_path: Path, result: RunResult) -> None:
    preview = result.response_text[:200].replace("\n", " ")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            result.model, result.scenario_id, result.scenario_name,
            result.image_name, result.run_idx,
            result.ttft_s, result.throughput_tps, result.total_time_s,
            result.output_tokens, result.vram_mb,
            preview,
        ])


# ── 报告生成 ──────────────────────────────────────────────────────

def generate_report(results: list[RunResult], model: str, report_dir: Path) -> str:
    """生成单模型的 Markdown 报告。"""
    lines: list[str] = []
    lines.append(f"# VLM 基准测试报告: {model}")
    lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**GPU 显存**: {results[0].vram_mb} MB (运行时)" if results else "")
    lines.append(f"**总推理次数**: {len(results)}")
    lines.append("")

    # 按场景汇总
    lines.append("## 各场景性能汇总")
    lines.append("")
    lines.append("| 场景 | 平均 TTFT (s) | 平均吞吐量 (tok/s) | 平均总耗时 (s) | 平均输出 Tokens | 测试次数 |")
    lines.append("|------|---------------|--------------------:|---------------:|----------------:|---------:|")

    for scenario in SCENARIOS:
        s_results = [r for r in results if r.scenario_id == scenario.id]
        if not s_results:
            continue
        avg_ttft = mean(r.ttft_s for r in s_results)
        avg_tp = mean(r.throughput_tps for r in s_results)
        avg_total = mean(r.total_time_s for r in s_results)
        avg_tokens = mean(r.output_tokens for r in s_results)
        lines.append(
            f"| {scenario.name} | {avg_ttft:.3f} | {avg_tp:.1f} | "
            f"{avg_total:.3f} | {avg_tokens:.0f} | {len(s_results)} |"
        )

    # 按图片汇总
    lines.append("")
    lines.append("## 各图片性能汇总")
    lines.append("")
    lines.append("| 图片 | 平均 TTFT (s) | 平均吞吐量 (tok/s) | 平均总耗时 (s) | 测试次数 |")
    lines.append("|------|---------------|--------------------:|---------------:|---------:|")

    image_names = sorted(set(r.image_name for r in results))
    for img in image_names:
        i_results = [r for r in results if r.image_name == img]
        avg_ttft = mean(r.ttft_s for r in i_results)
        avg_tp = mean(r.throughput_tps for r in i_results)
        avg_total = mean(r.total_time_s for r in i_results)
        lines.append(
            f"| {img} | {avg_ttft:.3f} | {avg_tp:.1f} | "
            f"{avg_total:.3f} | {len(i_results)} |"
        )

    # 全局汇总
    lines.append("")
    lines.append("## 全局统计")
    lines.append("")
    all_ttft = [r.ttft_s for r in results]
    all_tp = [r.throughput_tps for r in results]
    all_total = [r.total_time_s for r in results]
    all_tokens = [r.output_tokens for r in results]

    lines.append(f"- **平均 TTFT**: {mean(all_ttft):.3f}s (std: {stdev(all_ttft):.3f}s)" if len(all_ttft) > 1 else f"- **平均 TTFT**: {mean(all_ttft):.3f}s")
    lines.append(f"- **平均吞吐量**: {mean(all_tp):.1f} tokens/s (std: {stdev(all_tp):.1f})" if len(all_tp) > 1 else f"- **平均吞吐量**: {mean(all_tp):.1f} tokens/s")
    lines.append(f"- **平均总耗时**: {mean(all_total):.3f}s")
    lines.append(f"- **平均输出 Tokens**: {mean(all_tokens):.0f}")
    lines.append(f"- **总推理次数**: {len(results)}")
    lines.append("")

    # 输出示例
    lines.append("## 输出示例（每场景首次运行）")
    lines.append("")
    for scenario in SCENARIOS:
        s_results = [r for r in results if r.scenario_id == scenario.id]
        if not s_results:
            continue
        first = s_results[0]
        lines.append(f"### {scenario.name} (`{scenario.id}`)")
        lines.append(f"\n**图片**: {first.image_name} | **Tokens**: {first.output_tokens} | **耗时**: {first.total_time_s}s")
        lines.append(f"\n**Prompt**: {scenario.prompt[:100]}...")
        lines.append(f"\n**输出**:")
        lines.append(f"```\n{first.response_text[:500]}\n```")
        lines.append("")

    report_text = "\n".join(lines)
    report_path = report_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    lg.info("报告已保存至: {}", report_path)
    return report_text


# ── 主流程 ────────────────────────────────────────────────────────

def run_benchmark(
    base_url: str = "http://localhost:8000/v1",
    num_runs: int = 3,
    warmup_runs: int = 1,
    scenario_ids: list[str] | None = None,
    output_dir: Path | None = None,
) -> tuple[list[RunResult], Path]:
    """执行完整的基准测试流程。

    Args:
        output_dir: If provided, use this as the parent directory.
                    Report will be placed in output_dir/{model_short_name}/.

    返回: (所有结果列表, 报告目录路径)
    """
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    # 自动检测模型
    model = detect_model(client)
    short_name = model_short_name(model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is not None:
        report_dir = output_dir / short_name
    else:
        report_dir = _REPORTS_DIR / f"{short_name}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 配置日志到文件
    log_path = report_dir / "benchmark.log"
    lg.add(str(log_path), level="DEBUG", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}")

    lg.info("=" * 60)
    lg.info("VLM 基准测试启动")
    lg.info("模型: {} | 重复次数: {} | 预热次数: {}", model, num_runs, warmup_runs)
    lg.info("报告目录: {}", report_dir)
    lg.info("=" * 60)

    # 加载图片
    images = load_images(ASSETS_IMAGES_DIR)

    # 筛选场景
    scenarios = SCENARIOS
    if scenario_ids:
        scenarios = [SCENARIO_MAP[sid] for sid in scenario_ids if sid in SCENARIO_MAP]
        if not scenarios:
            lg.error("未找到匹配的场景 ID: {}", scenario_ids)
            sys.exit(1)
    lg.info("测试场景: {}", [s.id for s in scenarios])

    # 初始化 CSV
    csv_path = report_dir / "raw_data.csv"
    init_csv(csv_path)

    all_results: list[RunResult] = []
    total_tasks = len(scenarios) * len(images) * num_runs
    completed = 0

    for scenario in scenarios:
        lg.info("─" * 40)
        lg.info("场景: {} ({})", scenario.name, scenario.id)

        for img_name, img_b64, img_mime in images:
            # Warmup
            for w in range(warmup_runs):
                lg.debug("预热 {}/{} | {} | {}", w + 1, warmup_runs, scenario.id, img_name)
                try:
                    run_single_inference(client, model, scenario, img_name, img_b64, img_mime, -1)
                except Exception:
                    lg.exception("预热失败: {} | {}", scenario.id, img_name)

            # 正式测试
            for run_idx in range(num_runs):
                completed += 1
                lg.info(
                    "[{}/{}] {} | {} | run {}",
                    completed, total_tasks, scenario.id, img_name, run_idx + 1,
                )
                try:
                    result = run_single_inference(
                        client, model, scenario, img_name, img_b64, img_mime, run_idx + 1,
                    )
                    all_results.append(result)
                    append_csv(csv_path, result)
                    lg.info(
                        "  TTFT={:.3f}s | 吞吐={:.1f} tok/s | 总耗时={:.3f}s | Tokens={}",
                        result.ttft_s, result.throughput_tps,
                        result.total_time_s, result.output_tokens,
                    )
                except Exception:
                    lg.exception("推理失败: {} | {} | run {}", scenario.id, img_name, run_idx + 1)

    # 生成报告
    lg.info("=" * 60)
    if all_results:
        generate_report(all_results, model, report_dir)
        lg.info("基准测试完成 | 成功: {}/{} | 报告: {}", len(all_results), total_tasks, report_dir)
    else:
        lg.error("没有成功的推理结果，无法生成报告")

    lg.info("CSV 数据: {}", csv_path)
    return all_results, report_dir


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VLM 多场景基准测试")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM 服务地址")
    parser.add_argument("--runs", type=int, default=3, help="每场景每图重复次数 (默认 3)")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数 (默认 1)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help=f"指定场景 ID (可选: {', '.join(s.id for s in SCENARIOS)})")
    args = parser.parse_args()

    run_benchmark(
        base_url=args.base_url,
        num_runs=args.runs,
        warmup_runs=args.warmup,
        scenario_ids=args.scenarios,
    )


if __name__ == "__main__":
    main()
