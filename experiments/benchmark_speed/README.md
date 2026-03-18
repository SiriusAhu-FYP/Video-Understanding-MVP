# VLM 多场景基准测试 (Benchmark Speed)

对本地部署的视觉语言模型 (VLM) 进行多场景性能基准测试，使用游戏截图作为输入，测量不同场景下的推理速度和输出质量。

## 测试指标

| 指标 | 说明 |
|------|------|
| TTFT (Time to First Token) | 首个 token 的响应延迟 (秒) |
| Throughput | 生成吞吐量 (tokens/s) |
| Total Time | 单次推理总耗时 (秒) |
| Output Tokens | 输出 token 数 |
| VRAM | GPU 显存占用 (MB) |

## 测试场景 (5 个)

| 场景 ID | 名称 | 描述 | max_tokens |
|---------|------|------|------------|
| `short_desc` | 短文本描述 | 一句话概括画面 | 64 |
| `detailed_desc` | 详细描述 | 详细描述所有元素 | 512 |
| `object_detection` | JSON 物体检测 | 输出 bounding box + label 的 JSON | 512 |
| `action_analysis` | 角色动作分析 | 分析角色行为和状态 | 256 |
| `ui_recognition` | 游戏 UI 识别 | 识别 HUD/血条/小地图等 UI 元素 | 256 |

## 使用方法

### 前置条件

1. 在 WSL 中启动 vLLM 服务
2. 确保 `assets/images/` 目录下有测试图片 (PNG/JPG)

### 独立运行

```powershell
# 运行全部场景（自动检测模型）
uv run experiments/benchmark_speed/benchmark.py

# 指定重复次数
uv run experiments/benchmark_speed/benchmark.py --runs 5

# 只运行特定场景
uv run experiments/benchmark_speed/benchmark.py --scenarios short_desc detailed_desc

# 指定 vLLM 地址
uv run experiments/benchmark_speed/benchmark.py --base-url http://localhost:8000/v1
```

### 通过编排器运行（推荐）

```powershell
uv run run_benchmark.py
```

编排器会自动管理 vLLM 启停，输出到 `results/v3/{timestamp}/benchmark_speed/` 目录。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url` | `http://localhost:8000/v1` | vLLM 服务地址 |
| `--runs` | `3` | 每场景每图重复次数 |
| `--warmup` | `1` | 预热次数 |
| `--scenarios` | 全部 | 指定要测试的场景 ID |

## 输出

通过编排器运行时，输出到 `results/v3/{timestamp}/benchmark_speed/{model}/`：

```
{model}/
├── raw_data.csv      # 每次推理的原始数据
├── report.md         # 包含统计汇总的 Markdown 报告
└── benchmark.log     # 运行日志
```

### CSV 字段

```
model, scenario_id, scenario_name, image_name, run_idx,
ttft_s, throughput_tps, total_time_s, output_tokens, vram_mb,
response_preview
```
