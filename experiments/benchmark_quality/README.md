# 描述质量评估 (Benchmark Quality)

## 实验目的

使用 **LLM-as-Judge** 方法评估 VLM 在游戏场景描述任务上的输出质量。DeepSeek 作为评审模型，依据预定义的评分标准对 VLM 的描述输出进行 0-10 分评分。

## 评估维度

每个评估任务包含：
- **task_definition** — 任务定义（VLM 应当做什么）
- **reference_answer** — 参考答案
- **scoring_rubric** — 5 个评分维度（每维度 0-2 分，满分 10 分）
- **grading_prompt** — 给 Judge 模型的评分提示

### Prompt 模式

每个资产用两种 prompt 风格测试：
- **A_description** — 纯描述模式（客观描述画面内容）
- **B_assistant** — 助手模式（以游戏陪玩 AI 身份回应）

## 前置条件

- `.env` 中配置 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_API_BASE_URL`
- vLLM 在 WSL 中运行
- `assets/images/` 下有测试图片及对应的 JSON 评估定义文件

## 使用方法

### 独立运行

```powershell
uv run experiments/benchmark_quality/benchmark.py
uv run experiments/benchmark_quality/benchmark.py --runs 3
uv run experiments/benchmark_quality/benchmark.py --base-url http://localhost:8000/v1
```

### 通过编排器运行（推荐）

```powershell
uv run run_benchmark.py
```

编排器会自动管理 vLLM 启停，输出到 `results/v3/{timestamp}/benchmark_quality/` 目录。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url` | `http://localhost:8000/v1` | vLLM 服务地址 |
| `--runs` | `5` | 每个资产每个 prompt 模式重复次数 |
| `--judge-delay` | `1.0` | 每次 Judge 评估之间的延迟（秒） |

## 输出

通过编排器运行时，输出到 `results/v3/{timestamp}/benchmark_quality/{model}/`：

```
{model}/
├── scores.csv     # 所有评分数据
└── report.md      # 模型质量评估报告
```

多模型对比报告：`results/v3/{timestamp}/benchmark_quality/comparison_report.md`

### scores.csv 字段

```
asset_id, model_id, prompt_mode, total_score, max_score, [dimension_1], [dimension_2], ...
```

## 评分系统

- **ScoreAggregator** 计算均值、标准差、95% 置信区间
- 支持按 prompt mode 过滤统计
- 跨模型对比报告按均分排名
