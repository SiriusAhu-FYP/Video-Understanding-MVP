# Multi-Model Comprehensive Benchmark

This directory contains a complete multi-model benchmark session.

## Purpose

Systematically evaluate multiple VLM/LLM models for inference speed
and video understanding capability in a game-scene AI companion context.

## Models Tested

1. `Qwen/Qwen3.5-2B`
2. `Qwen/Qwen3.5-0.8B`
3. `Qwen/Qwen3-VL-2B-Instruct`
4. `OpenGVLab/InternVL2_5-2B`
5. `mistralai/Ministral-3-3B-Instruct-2512`
6. `deepseek-ai/deepseek-vl2-tiny`

## Experiments

### 1. Benchmark Speed

Measures TTFT, throughput, and total inference time across game screenshots.

**Scenarios**:

- `short_desc` (短文本描述): max_tokens=64
- `detailed_desc` (详细描述): max_tokens=512
- `object_detection` (JSON 物体检测): max_tokens=512
- `action_analysis` (角色动作分析): max_tokens=256
- `ui_recognition` (游戏 UI 识别): max_tokens=256

### 2. Video Understanding

Full pipeline: PotPlayer -> window capture -> frame-diff -> vLLM -> DeepSeek summary.

## Reproduction

### Prerequisites

- Windows 10/11 with WSL2
- NVIDIA GPU (>= 12GB VRAM recommended)
- Python >= 3.12, uv package manager
- vLLM installed in WSL
- PotPlayer (for video_understanding)

### Steps

```bash
# 1. Install dependencies
uv sync

# 2. Create .env in project root
# DEEPSEEK_API_KEY=your_key
# DEEPSEEK_API_BASE_URL=https://api.deepseek.com
# PLAYER_EXE_PATH=D:\Programs\PotPlayer\PotPlayerMini64.exe

# 3. Ensure models are cached in WSL (~/.cache/huggingface/hub/)

# 4. Run the benchmark
uv run run_benchmark.py

# Optional flags:
# --config my_config.toml
# --models Qwen/Qwen3.5-2B Qwen/Qwen3.5-0.8B
# --skip-speed   (only video_understanding)
# --skip-video   (only benchmark_speed)
```

## Directory Structure

```
.
├── config.toml               # Config snapshot
├── meta.json                  # Environment metadata
├── orchestrator.log           # Full log
├── README.md                  # This file
├── final_report.md            # Comprehensive report
├── incident_report.md         # Startup issues & solutions
├── charts/                    # Visualization PNGs
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
