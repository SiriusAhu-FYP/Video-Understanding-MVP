# 游戏画面分析专项实验 (Gameplay Analysis)

## 实验目的

针对视频理解 MVP 项目的实际需求——**游戏陪玩 AI**，测试 VLM 模型在以下 4 个核心任务上的表现：

| 任务 ID | 任务名称 | 说明 |
|---------|---------|------|
| `ui_bbox` | UI 元素检测 (Bounding Box) | 识别游戏截图中的 UI/HUD 元素，输出 JSON 格式的 bbox 坐标和标签。脚本自动将 bbox 绘制到图片上并保存。 |
| `scene_desc` | 画面描述 | 用中文详细描述当前画面内容（场景、角色、事件） |
| `player_intent` | 玩家意图推测 | 推测玩家当前的操作意图和接下来最可能的行为 |
| `situation_eval` | 情况评价 | 两部分输出：1) 情感支持（鼓励/共情/陪伴） 2) 客观评价（优劣势分析） |

## 与基准速度测试 (benchmark_speed) 的区别

- **提示词带游戏上下文**：每个 prompt 的 system message 包含游戏名称和简介
- **角色定位**：模型以"游戏陪玩 AI 助手"的身份输出
- **bbox 可视化**：`ui_bbox` 任务会解析模型输出的 JSON，将检测到的 UI 元素绘制到原图上并保存
- **评估维度**：不只关注速度指标 (TTFT/Throughput)，还关注输出内容的质量

## 使用方法

### 独立运行

```powershell
# 运行全部任务（需要先在 WSL 中启动 vLLM 服务）
uv run experiments/gameplay_analysis/run_experiment.py

# 指定重复次数
uv run experiments/gameplay_analysis/run_experiment.py --runs 5

# 只运行部分任务
uv run experiments/gameplay_analysis/run_experiment.py --tasks ui_bbox scene_desc

# 指定 vLLM 地址
uv run experiments/gameplay_analysis/run_experiment.py --base-url http://localhost:8000/v1
```

## 测试参数

- 每张图 x 每个任务 x 3 次重复 + 1 次预热
- 6 张游戏截图（明日方舟、赛博朋克2077、原神、我的世界、女神异闻录5、超级马里奥兄弟）
- 每个模型共 72 次推理

## 输出

独立运行时输出到 `experiments/gameplay_analysis/reports/{model}_{timestamp}/`：

```
{model}_{timestamp}/
├── raw_data.csv          # 每次推理的完整数据（含完整响应文本）
├── report.md             # Markdown 格式的实验报告
└── experiment.log        # 运行日志
```

bbox 标注图输出到 `experiments/gameplay_analysis/logs/bbox_{model}_{timestamp}/`。

## 关键指标

| 指标 | 说明 |
|------|------|
| TTFT (Time to First Token) | 首个 token 的延迟，反映模型的响应速度 |
| Throughput (tokens/s) | 生成吞吐量，反映模型的 decode 速度 |
| Total Time | 单次推理的总耗时 |
| Output Tokens | 输出的 token 数量 |
| VRAM (MB) | GPU 显存占用 |

## Bounding Box 坐标说明

模型输出的 `bbox_2d` 坐标归一化到 `[0, 1000]` 范围。脚本会将其映射到原图的实际像素坐标后绘制。
