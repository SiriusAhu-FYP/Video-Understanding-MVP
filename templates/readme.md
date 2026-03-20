# 多模型综合基准测试

本目录包含一次完整的多模型基准测试会话。

## 目的

系统评估多个 VLM 模型在游戏场景 AI 伴侣应用中的推理速度、描述质量和视频理解能力。

## 测试模型

{{model_list}}

## 实验内容

### 1. 推理速度基准测试 (benchmark_speed)

对游戏截图进行多场景推理，测量 TTFT、吞吐量和总推理时间。

**场景列表**:

{{scenario_list}}

### 2. 描述质量评估 (benchmark_quality)

使用 LLM-as-Judge（DeepSeek）对 VLM 生成的描述进行 0-10 分评分，评估两种 prompt 风格。

### 3. 视频理解 (video_understanding)

完整流水线: PotPlayer 播放 -> 窗口截图 -> 帧差检测 -> vLLM 描述 -> DeepSeek 汇总

## 复现步骤

### 前置条件

- Windows 10/11 + WSL2
- NVIDIA GPU (>= 12GB VRAM 推荐)
- Python >= 3.12, uv 包管理器
- WSL 中安装 vLLM
- PotPlayer（用于 video_understanding）

### 步骤

```bash
# 1. 安装依赖
uv sync

# 2. 在项目根目录创建 .env
# DEEPSEEK_API_KEY=your_key
# DEEPSEEK_API_BASE_URL=https://api.deepseek.com
# PLAYER_EXE_PATH=D:\Programs\PotPlayer\PotPlayerMini64.exe

# 3. 确保模型已缓存在 WSL 中 (~/.cache/huggingface/hub/)

# 4. 运行基准测试
uv run run_benchmark.py

# 可选参数:
# --config my_config.toml
# --models "Qwen/Qwen3-VL-2B-Instruct" "OpenGVLab/InternVL2_5-2B"
# --skip-speed   (跳过 benchmark_speed)
# --skip-video   (跳过 video_understanding)
```

## 目录结构

```
.
├── config.toml               # 配置快照
├── meta.json                 # 环境元数据
├── orchestrator.log          # 完整日志
├── README.md                 # 本文件
├── final_report.md           # 综合报告
├── incident_report.md        # 事件记录
├── charts/                   # 可视化图表
├── benchmark_speed/
│   ├── comparison_report.md
│   └── {model}/
│       ├── raw_data.csv
│       ├── report.md
│       └── benchmark.log
├── benchmark_quality/
│   ├── comparison_report.md
│   └── {model}/
│       ├── scores.csv
│       └── report.md
└── video_understanding/
    ├── comparison_report.md
    └── {model}/
        ├── report.md
        ├── experiment.log
        └── {video}_run{N}/
            ├── frames/
            ├── run_log.json
            └── summary.txt
```
