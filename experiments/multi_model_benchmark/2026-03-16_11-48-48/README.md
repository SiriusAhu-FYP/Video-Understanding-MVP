# 多模型综合基准测试

本目录包含一次完整的多模型基准测试会话。

## 目的

系统评估多个 VLM 模型在游戏场景 AI 伴侣应用中的推理速度和视频理解能力。

## 测试模型

1. `mistralai/Ministral-3-3B-Instruct-2512`

## 实验内容

### 1. 推理速度基准测试 (benchmark_speed)

对游戏截图进行多场景推理，测量 TTFT、吞吐量和总推理时间。

**场景列表**:

- `short_desc` (短文本描述): max_tokens=64
- `detailed_desc` (详细描述): max_tokens=512
- `object_detection` (JSON 物体检测): max_tokens=512
- `action_analysis` (角色动作分析): max_tokens=256
- `ui_recognition` (游戏 UI 识别): max_tokens=256

### 2. 视频理解 (video_understanding)

完整流水线: PotPlayer 播放 → 窗口截图 → 帧差检测 → vLLM 描述 → DeepSeek 汇总

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
# --models Qwen/Qwen3.5-2B Qwen/Qwen3.5-0.8B
# --skip-speed   (仅运行 video_understanding)
# --skip-video   (仅运行 benchmark_speed)
```

## 目录结构

```
.
├── config.toml               # 配置快照
├── meta.json                  # 环境元数据
├── orchestrator.log           # 完整日志
├── README.md                  # 本文件
├── final_report.md            # 综合报告
├── incident_report.md         # 事件记录
├── charts/                    # 可视化图表
├── benchmark_speed/
│   ├── comparison_report.md
│   └── {model}/
│       ├── raw_data.csv
│       ├── report.md
│       └── benchmark.log
└── video_understanding/
    ├── comparison_report.md
    └── {model}/
        ├── report.md
        └── {video}_run{N}/
            ├── frames/
            ├── run_log.json
            └── summary.txt
```
