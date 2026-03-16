# 多模型 VLM 基准测试实验

**实验日期**: 2026-03-16
**分支**: `feat/multi-model-benchmark-8models-02`

## 实验概述

对 8 款小型视觉语言模型（VLM）进行推理速度基准测试和视频理解实验，
评估其在游戏画面实时理解场景下的表现。

## 模型列表

| 模型 | 参数量 | 架构 |
|------|:---:|------|
| Qwen/Qwen3.5-2B | 2B | Mamba-Transformer 混合 |
| Qwen/Qwen3.5-0.8B | 0.8B | Mamba-Transformer 混合 |
| Qwen/Qwen3-VL-2B-Instruct | 2B | Transformer + 动态 ViT |
| OpenGVLab/InternVL2_5-2B | 2B | InternViT + InternLM2 |
| microsoft/Phi-3.5-vision-instruct | 4.2B | CLIP ViT + Phi-3.5 |
| mistralai/Ministral-3-3B-Instruct | 3B | Mistral + 视觉适配器 |
| deepseek-ai/deepseek-vl2-tiny | 3B | MoE (激活 ~1B) |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | 2.2B | SigLIP + 精简 LM |

## 实验内容

### 1. benchmark_speed
- 5 个场景（短描述、详细描述、目标检测、动作分析、UI 识别）
- 6 张游戏截图（明日方舟、赛博朋克2077、原神、Minecraft、P5R、超级马里奥）
- 3 次重复 + 1 次预热 = 每模型 90 次有效推理

### 2. video_understanding
- 2 个视频（原神 24.5s、MC 20s）
- 3 次重复 = 每模型 6 次完整流水线运行
- 包含截图采集、帧差检测、VLM 推理、DeepSeek 汇总

## 复现步骤

### 环境准备

```bash
# 1. 安装 Python 依赖
uv sync

# 2. WSL 中安装 vLLM
# 确保 WSL 中已配置 CUDA 和 vLLM 虚拟环境 (~~/vLLM_server/.venv)
pip install vllm num2words

# 3. 下载模型（在 WSL 中）
source ~/vLLM_server/.venv/bin/activate
hf download Qwen/Qwen3.5-2B
hf download Qwen/Qwen3.5-0.8B
hf download Qwen/Qwen3-VL-2B-Instruct
hf download OpenGVLab/InternVL2_5-2B
hf download microsoft/Phi-3.5-vision-instruct
hf download mistralai/Ministral-3-3B-Instruct-2512
hf download deepseek-ai/deepseek-vl2-tiny
hf download HuggingFaceTB/SmolVLM2-2.2B-Instruct

# 4. 配置 .env
# PLAYER_EXE_PATH=<PotPlayer 路径>
# DEEPSEEK_API_KEY=<DeepSeek API Key>
```

### 运行实验

```bash
# 完整实验（含预飞行验证）
uv run run_benchmark.py

# 跳过预飞行（已知模型可用时）
uv run run_benchmark.py --skip-preflight --skip-memory-probe
```

## 文件结构

```
2026-03-16_18-09-38/
├── meta.json                    # 实验元数据
├── config.toml                  # 实验配置快照
├── orchestrator.log             # 完整编排日志
├── final_report.md              # 综合研究报告
├── incident_report.md           # 事故报告
├── README.md                    # 本文件
├── charts/                      # 可视化图表
│   ├── throughput_comparison.png
│   ├── ttft_comparison.png
│   └── summary_table.png
├── benchmark_speed/             # 推理速度数据
│   ├── comparison_report.md
│   └── <模型名>/
│       ├── raw_data.csv
│       ├── report.md
│       └── benchmark.log
└── video_understanding/         # 视频理解数据
    ├── comparison_report.md
    └── <模型名>/
        ├── report.md
        ├── experiment.log
        └── <视频>_run<N>/
```

## 主要结论

- **速度冠军**: SmolVLM2-2.2B (114.9 tok/s)
- **综合最优**: InternVL2.5-2B (95.6 tok/s + 高质量描述)
- **描述最佳**: Qwen3-VL-2B (最准确的游戏元素识别)

详见 `final_report.md`。