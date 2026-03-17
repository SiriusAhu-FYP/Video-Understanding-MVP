"""游戏画面分析专项实验脚本。

针对项目实际需求（陪玩 AI），测试 VLM 在以下 4 个任务上的表现：
1. UI 元素检测 (Bounding Box)
2. 画面描述
3. 玩家意图推测
4. 情况评价（情感支持 + 客观评价）

用法:
    uv run experiments/gameplay_analysis/run_experiment.py
    uv run experiments/gameplay_analysis/run_experiment.py --runs 5
    uv run experiments/gameplay_analysis/run_experiment.py --tasks ui_bbox scene_desc
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np
from loguru import logger as lg
from openai import OpenAI

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXPERIMENT_DIR.parent.parent
_REPORTS_DIR = _EXPERIMENT_DIR / "reports"
_LOGS_DIR = _EXPERIMENT_DIR / "logs"

sys.path.insert(0, str(_PROJECT_ROOT))

from ahu_paimon_toolkit.utils.image import encode_image, get_image_mime, load_images
from ahu_paimon_toolkit.utils.gpu import get_gpu_memory_mb
from ahu_paimon_toolkit.vlm.model_utils import detect_model, model_short_name

ASSETS_IMAGES_DIR = _PROJECT_ROOT / "assets" / "images"

# ── 游戏信息 ──────────────────────────────────────────────────────
# 文件名前缀 -> (中文名, 简介)

GAME_INFO: dict[str, tuple[str, str]] = {
    "Arknights": (
        "明日方舟",
        "一款策略塔防手游。玩家扮演博士，在地图上部署拥有不同技能的干员来抵御敌人的进攻。"
        "画面通常包含俯视角地图、干员站位、敌人行进路线、技能 CD 等 UI 元素。",
    ),
    "Cyberpunk-2077": (
        "赛博朋克2077",
        "一款第一人称开放世界动作 RPG。玩家扮演雇佣兵 V 在夜之城中接任务、驾车、"
        "战斗和对话。画面包含小地图、任务提示、对话字幕、驾驶 HUD 等元素。",
    ),
    "Genshin-Impact": (
        "原神",
        "一款开放世界冒险 RPG。玩家操控旅行者和不同元素的角色组队探索提瓦特大陆，"
        "进行元素反应战斗。画面包含角色血条、元素技能图标、小地图、体力条等。",
    ),
    "Minecraft": (
        "我的世界",
        "一款沙盒生存建造游戏。玩家以第一人称视角在方块世界中采集资源、建造建筑、"
        "对抗怪物。画面包含物品栏、生命值、饥饿值、经验条、十字准星等 UI。",
    ),
    "P5R": (
        "女神异闻录5 皇家版",
        "一款日式回合制 RPG。玩家扮演怪盗团团长潜入异世界宫殿战斗。"
        "战斗界面包含角色 HP/SP、技能菜单（攻击/技能/道具/换人）、敌人信息等。",
    ),
    "SuperMarioBros": (
        "超级马里奥兄弟",
        "一款经典 2D 横版平台跳跃游戏。玩家操控马里奥跳跃、踩敌人、吃蘑菇和金币，"
        "最终到达关卡终点旗杆。画面包含得分、金币数、剩余时间、生命数等信息。",
    ),
}


def _get_game_info(image_name: str) -> tuple[str, str, str]:
    """从文件名提取游戏前缀并返回 (前缀, 中文名, 简介)。"""
    stem = Path(image_name).stem
    prefix = re.split(r"_\d", stem)[0]
    if prefix in GAME_INFO:
        cn_name, desc = GAME_INFO[prefix]
        return prefix, cn_name, desc
    return prefix, prefix, ""


# ── 任务定义 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Task:
    id: str
    name: str
    user_prompt: str
    max_tokens: int


TASKS: list[Task] = [
    Task(
        id="ui_bbox",
        name="UI 元素检测 (Bounding Box)",
        user_prompt=(
            "请识别这张游戏截图中所有可见的 UI/HUD 元素（如血条、技能图标、小地图、物品栏、"
            "得分/金币显示、对话框、菜单按钮等），以 JSON 格式输出。\n"
            "每个元素包含 bbox_2d（归一化到 [0,1000] 范围的 [x1,y1,x2,y2] 坐标）和 label（中文标签）。\n"
            '输出格式：[{"bbox_2d": [x1,y1,x2,y2], "label": "元素名称"}, ...]\n'
            "只输出 JSON，不要其他文字。"
        ),
        max_tokens=1024,
    ),
    Task(
        id="scene_desc",
        name="画面描述",
        user_prompt=(
            "请用中文详细描述这张游戏截图中的画面内容。包括：\n"
            "1. 场景环境（室内/室外、地形、天气、光照等）\n"
            "2. 角色状态（位置、动作、装备等）\n"
            "3. 正在发生的事件（战斗/探索/对话/建造等）\n"
            "用 2-3 段话描述，语言自然流畅。"
        ),
        max_tokens=512,
    ),
    Task(
        id="player_intent",
        name="玩家意图推测",
        user_prompt=(
            "根据这张游戏截图中的画面，推测玩家当前最可能的意图和接下来的操作。\n"
            "请分析：\n"
            "1. 玩家当前正在做什么？\n"
            "2. 玩家接下来最可能想做什么？（2-3 个可能性）\n"
            "3. 你作为陪玩 AI 会给出什么操作建议？\n"
            "用简洁的中文回答。"
        ),
        max_tokens=256,
    ),
    Task(
        id="situation_eval",
        name="情况评价",
        user_prompt=(
            "请以游戏陪玩 AI 的身份，对这张游戏截图中的情况进行评价。分两部分输出：\n\n"
            "【情感支持】\n"
            "像一个陪玩朋友一样，根据画面情况给出情感性的回应。"
            "如果是好的情况就鼓励（如「打得不错！」「这波操作很秀」），"
            "如果是困难/失败就共情（如「别灰心，这关确实难」「下次一定能过」），"
            "如果是普通情况就友好陪伴（如「继续加油」「看起来进展顺利」）。\n\n"
            "【客观评价】\n"
            "用客观的游戏分析视角评价当前局势：\n"
            "- 优势/劣势判断\n"
            "- 潜在风险或机会\n"
            "- 关键的注意事项\n"
            "用中文输出，两部分都要有。"
        ),
        max_tokens=512,
    ),
]

TASK_MAP: dict[str, Task] = {t.id: t for t in TASKS}


# ── 工具函数 ──────────────────────────────────────────────────────


# ── Bounding Box 绘制 ─────────────────────────────────────────────

def _parse_bbox_json(text: str) -> list[dict] | None:
    """尝试从模型输出中提取 bbox JSON 数组，支持被截断的 JSON。"""
    # 剥离 ```json ... ``` 代码块（可能不完整）
    m = re.search(r"```(?:json)?\s*\n?(.*?)(?:```|$)", text, re.DOTALL)
    raw = m.group(1).strip() if m else text.strip()

    # 尝试直接解析
    for candidate in [raw]:
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "objects" in data:
                return data["objects"]
        except json.JSONDecodeError:
            pass

    # 提取以 [ 开头的部分（可能被截断）
    idx = raw.find("[")
    if idx == -1:
        return None
    fragment = raw[idx:]

    # 完整数组
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        pass

    # 处理被截断的 JSON 数组：找到最后一个完整的 } 并闭合数组
    last_brace = fragment.rfind("}")
    if last_brace > 0:
        truncated = fragment[: last_brace + 1] + "]"
        try:
            result = json.loads(truncated)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def draw_bboxes(
    image_path: Path,
    bbox_data: list[dict],
    output_path: Path,
    coord_scale: int = 1000,
) -> bool:
    """在图片上绘制 bounding box 并保存。坐标假设归一化到 [0, coord_scale]。"""
    img = cv2.imread(str(image_path))
    if img is None:
        lg.warning("无法读取图片: {}", image_path)
        return False

    h, w = img.shape[:2]

    # 加载中文字体（如果可用就用 PIL，否则用 OpenCV 内置字体）
    use_pil = False
    try:
        from PIL import Image, ImageDraw, ImageFont
        font_path = "C:/Windows/Fonts/msyh.ttc"
        if Path(font_path).exists():
            font = ImageFont.truetype(font_path, max(16, h // 40))
            use_pil = True
    except ImportError:
        pass

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]

    if use_pil:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

    for i, item in enumerate(bbox_data):
        bbox = item.get("bbox_2d")
        label = item.get("label", "?")
        if not bbox or len(bbox) != 4:
            continue

        x1 = int(bbox[0] * w / coord_scale)
        y1 = int(bbox[1] * h / coord_scale)
        x2 = int(bbox[2] * w / coord_scale)
        y2 = int(bbox[3] * h / coord_scale)

        color = colors[i % len(colors)]
        thickness = max(2, min(w, h) // 300)

        if use_pil:
            rgb_color = color
            draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=thickness)
            text_y = max(0, y1 - font.size - 4)
            draw.text((x1, text_y), label, fill=rgb_color, font=font)
        else:
            bgr_color = (color[2], color[1], color[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, thickness)
            font_scale = max(0.5, min(w, h) / 1000)
            cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr_color, thickness)

    if use_pil:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    lg.debug("bbox 标注图已保存: {}", output_path)
    return True


# ── 单次推理 ──────────────────────────────────────────────────────

@dataclass
class RunResult:
    model: str
    task_id: str
    task_name: str
    game: str
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
    task: Task,
    game_cn: str,
    game_desc: str,
    image_name: str,
    image_b64: str,
    image_mime: str,
    run_idx: int,
) -> RunResult:
    """执行单次推理并测量性能指标。"""
    system_msg = (
        f"你是一名专业的游戏陪玩 AI 助手。\n"
        f"当前游戏：{game_cn}\n"
        f"游戏简介：{game_desc}"
    )

    start_time = time.perf_counter()
    vram_before = get_gpu_memory_mb()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task.user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
                ],
            },
        ],
        stream=True,
        temperature=0.0,
        max_tokens=task.max_tokens,
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

    _, game_cn_name, _ = _get_game_info(image_name)

    return RunResult(
        model=model,
        task_id=task.id,
        task_name=task.name,
        game=game_cn_name,
        image_name=image_name,
        run_idx=run_idx,
        ttft_s=round(ttft, 4),
        throughput_tps=round(throughput, 2),
        total_time_s=round(total_time, 4),
        output_tokens=output_tokens,
        vram_mb=vram_after,
        response_text="".join(chunks),
    )


# ── CSV 写入 ──────────────────────────────────────────────────────

CSV_HEADERS = [
    "model", "task_id", "task_name", "game", "image_name", "run_idx",
    "ttft_s", "throughput_tps", "total_time_s", "output_tokens", "vram_mb",
    "response_text",
]


def init_csv(csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADERS)


def append_csv(csv_path: Path, result: RunResult) -> None:
    preview = result.response_text.replace("\n", " ")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            result.model, result.task_id, result.task_name,
            result.game, result.image_name, result.run_idx,
            result.ttft_s, result.throughput_tps, result.total_time_s,
            result.output_tokens, result.vram_mb, preview,
        ])


# ── 报告生成 ──────────────────────────────────────────────────────

def generate_report(results: list[RunResult], model: str, report_dir: Path) -> None:
    lines: list[str] = []
    lines.append(f"# 游戏画面分析实验报告: {model}")
    lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**GPU 显存**: {results[0].vram_mb} MB (运行时)" if results else "")
    lines.append(f"**总推理次数**: {len(results)}")
    lines.append("")

    # 按任务汇总
    lines.append("## 各任务性能汇总")
    lines.append("")
    lines.append("| 任务 | 平均 TTFT (s) | 平均吞吐量 (tok/s) | 平均总耗时 (s) | 平均输出 Tokens | 测试次数 |")
    lines.append("|------|---------------|--------------------:|---------------:|----------------:|---------:|")

    for task in TASKS:
        t_results = [r for r in results if r.task_id == task.id]
        if not t_results:
            continue
        lines.append(
            f"| {task.name} | {mean(r.ttft_s for r in t_results):.3f} "
            f"| {mean(r.throughput_tps for r in t_results):.1f} "
            f"| {mean(r.total_time_s for r in t_results):.3f} "
            f"| {mean(r.output_tokens for r in t_results):.0f} "
            f"| {len(t_results)} |"
        )

    # 按游戏汇总
    lines.append("")
    lines.append("## 各游戏性能汇总")
    lines.append("")
    lines.append("| 游戏 | 平均 TTFT (s) | 平均吞吐量 (tok/s) | 平均总耗时 (s) | 测试次数 |")
    lines.append("|------|---------------|--------------------:|---------------:|---------:|")

    games = sorted(set(r.game for r in results))
    for game in games:
        g_results = [r for r in results if r.game == game]
        lines.append(
            f"| {game} | {mean(r.ttft_s for r in g_results):.3f} "
            f"| {mean(r.throughput_tps for r in g_results):.1f} "
            f"| {mean(r.total_time_s for r in g_results):.3f} "
            f"| {len(g_results)} |"
        )

    # 全局统计
    lines.append("")
    lines.append("## 全局统计")
    lines.append("")
    all_ttft = [r.ttft_s for r in results]
    all_tp = [r.throughput_tps for r in results]
    lines.append(f"- **平均 TTFT**: {mean(all_ttft):.3f}s" + (f" (std: {stdev(all_ttft):.3f}s)" if len(all_ttft) > 1 else ""))
    lines.append(f"- **平均吞吐量**: {mean(all_tp):.1f} tokens/s" + (f" (std: {stdev(all_tp):.1f})" if len(all_tp) > 1 else ""))
    lines.append(f"- **平均总耗时**: {mean(r.total_time_s for r in results):.3f}s")
    lines.append(f"- **平均输出 Tokens**: {mean(r.output_tokens for r in results):.0f}")
    lines.append(f"- **总推理次数**: {len(results)}")
    lines.append("")

    # 输出示例
    lines.append("## 输出示例（每任务每游戏首次运行）")
    lines.append("")
    for task in TASKS:
        t_results = [r for r in results if r.task_id == task.id]
        if not t_results:
            continue
        lines.append(f"### {task.name} (`{task.id}`)")
        lines.append("")
        seen_games: set[str] = set()
        for r in t_results:
            if r.game in seen_games:
                continue
            seen_games.add(r.game)
            lines.append(f"#### {r.game} ({r.image_name})")
            lines.append(f"**Tokens**: {r.output_tokens} | **耗时**: {r.total_time_s}s")
            lines.append(f"\n```\n{r.response_text[:600]}\n```\n")

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    lg.info("报告已保存至: {}", report_path)


# ── 主流程 ────────────────────────────────────────────────────────

def run_experiment(
    base_url: str = "http://localhost:8000/v1",
    num_runs: int = 3,
    warmup_runs: int = 1,
    task_ids: list[str] | None = None,
) -> tuple[list[RunResult], Path]:
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    model = detect_model(client)
    short_name = model_short_name(model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = _REPORTS_DIR / f"{short_name}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    log_path = report_dir / "experiment.log"
    lg.add(str(log_path), level="DEBUG", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}")

    lg.info("=" * 60)
    lg.info("游戏画面分析专项实验启动")
    lg.info("模型: {} | 重复次数: {} | 预热次数: {}", model, num_runs, warmup_runs)
    lg.info("报告目录: {}", report_dir)
    lg.info("=" * 60)

    images = load_images(ASSETS_IMAGES_DIR)

    tasks = TASKS
    if task_ids:
        tasks = [TASK_MAP[tid] for tid in task_ids if tid in TASK_MAP]
        if not tasks:
            lg.error("未找到匹配的任务 ID: {}", task_ids)
            sys.exit(1)
    lg.info("测试任务: {}", [t.id for t in tasks])

    csv_path = report_dir / "raw_data.csv"
    init_csv(csv_path)

    bbox_dir = _LOGS_DIR / f"bbox_{short_name}_{timestamp}"
    bbox_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[RunResult] = []
    total_tasks = len(tasks) * len(images) * num_runs
    completed = 0

    for task in tasks:
        lg.info("─" * 40)
        lg.info("任务: {} ({})", task.name, task.id)

        for img_name, img_b64, img_mime, img_path in images:
            game_prefix, game_cn, game_desc = _get_game_info(img_name)

            for w in range(warmup_runs):
                lg.debug("预热 {}/{} | {} | {} ({})", w + 1, warmup_runs, task.id, img_name, game_cn)
                try:
                    run_single_inference(
                        client, model, task, game_cn, game_desc,
                        img_name, img_b64, img_mime, -1,
                    )
                except Exception:
                    lg.exception("预热失败: {} | {}", task.id, img_name)

            for run_idx in range(num_runs):
                completed += 1
                lg.info("[{}/{}] {} | {} ({}) | run {}",
                        completed, total_tasks, task.id, img_name, game_cn, run_idx + 1)
                try:
                    result = run_single_inference(
                        client, model, task, game_cn, game_desc,
                        img_name, img_b64, img_mime, run_idx + 1,
                    )
                    all_results.append(result)
                    append_csv(csv_path, result)
                    lg.info("  TTFT={:.3f}s | 吞吐={:.1f} tok/s | 总耗时={:.3f}s | Tokens={}",
                            result.ttft_s, result.throughput_tps,
                            result.total_time_s, result.output_tokens)

                    # bbox 任务：解析 JSON 并绘制标注图
                    if task.id == "ui_bbox":
                        bbox_data = _parse_bbox_json(result.response_text)
                        if bbox_data:
                            out_name = f"{game_prefix}_{img_name.replace('.', '_')}_run{run_idx + 1}.png"
                            draw_bboxes(img_path, bbox_data, bbox_dir / out_name)
                            lg.info("  bbox 标注: {} 个元素 -> {}", len(bbox_data), out_name)
                        else:
                            lg.warning("  bbox JSON 解析失败，跳过绘制")

                except Exception:
                    lg.exception("推理失败: {} | {} | run {}", task.id, img_name, run_idx + 1)

    lg.info("=" * 60)
    if all_results:
        generate_report(all_results, model, report_dir)
        lg.info("实验完成 | 成功: {}/{} | 报告: {}", len(all_results), total_tasks, report_dir)
    else:
        lg.error("没有成功的推理结果，无法生成报告")

    lg.info("CSV 数据: {}", csv_path)
    lg.info("bbox 标注图: {}", bbox_dir)
    return all_results, report_dir


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="游戏画面分析专项实验")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM 服务地址")
    parser.add_argument("--runs", type=int, default=3, help="每任务每图重复次数 (默认 3)")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数 (默认 1)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help=f"指定任务 ID (可选: {', '.join(t.id for t in TASKS)})")
    args = parser.parse_args()

    run_experiment(
        base_url=args.base_url,
        num_runs=args.runs,
        warmup_runs=args.warmup,
        task_ids=args.tasks,
    )


if __name__ == "__main__":
    main()
