# Video Understanding MVP

面向游戏场景的视觉语言模型（VLM）实时理解系统验证项目。本项目提供了**从图像/视频采集到模型推理、质量评估的全链路验证程序和工具库**，并通过系统化的实验测定了**最适合本地部署的开源小模型**。

## 核心成果

### 1. 全链路视频理解验证

本项目验证了一条完整的游戏场景实时理解 pipeline：

```
屏幕截图 (mss/win32gui) → 帧差过滤 (OpenCV) → 异步队列 (asyncio)
    → vLLM 逐帧描述 (本地 VLM) → DeepSeek 汇总 → 视频总结
```

该 pipeline 由 [`ahu_paimon_toolkit`](#ahu_paimon_toolkit-通用工具库) 提供核心能力，已在 Minecraft 和原神两款游戏上完成端到端验证。

### 2. 最优开源小模型选型

通过两阶段实验确定了最适合本项目的模型：

**阶段一：SOTA 校准（评估框架验证）**
- 使用 10 个 SOTA 大模型（GPT-5.4、GPT-4.1、Claude Sonnet 4.6 等）验证评估框架的合理性
- 顶级模型在我们的评分体系下获得 8-9/10 分，证明 rubric 设计有效
- 统一使用 DeepSeek 作为裁判模型，每个任务重复 5 次取均值

**阶段二：本地 2B 模型选型**
- 对 4 个候选模型进行质量 + 推理速度综合评测

| 模型 | 质量均分 | 吞吐量 | VRAM | 推荐 |
|------|:-------:|:------:|:----:|:----:|
| **Qwen3-VL-2B-Instruct** | **4.28/10** | ~67 tok/s | **7.5 GB** | **首选** |
| DeepSeek-VL2-tiny | 3.23/10 | ~59 tok/s | 8.6 GB | 备选 |
| InternVL2.5-2B | 2.83/10 | ~77 tok/s | 8.6 GB | — |
| SmolVLM2-2.2B | 1.88/10 | ~71 tok/s | 8.7 GB | — |

**结论：Qwen3-VL-2B-Instruct** 在质量与资源之间取得最佳平衡（质量最高、显存最低），推荐作为系统部署模型。

> 详细报告见 [`QA/dissertation_model_selection_and_validation/`](QA/dissertation_model_selection_and_validation/实验报告_模型选型与评估框架验证.md)

---

## `ahu_paimon_toolkit` 通用工具库

本项目的核心能力封装在 [`ahu_paimon_toolkit`](ahu_paimon_toolkit/) 子模块（submodule）中。该库设计为**可复用的通用工具库**，其他项目只需引入即可获得实时图像采集 + VLM 推理的完整能力，无需重复造轮子。

### 库结构与能力

| 模块 | 能力 | 对外接口 |
|------|------|---------|
| `capture.window_capture` | Windows 窗口发现 + 极速截图 | `find_window()`, `capture_window()`, `frame_to_base64()` |
| `capture.frame_diff` | MSE/SSIM 帧差关键帧提取 | `compute_mse()`, `compute_ssim()` |
| `capture.video_reader` | 视频文件帧提取（同样支持帧差过滤） | `VideoFrameExtractor` |
| `vlm.client` | 异步 VLM 推理（OpenAI 兼容 API） | `AsyncVLMClient.describe_frame()`, `.chat()` |
| `vlm.model_utils` | 模型发现、就绪轮询 | `detect_model()`, `wait_for_vllm_ready()` |
| `pipeline` | 完整 pipeline 编排（采集→推理→汇总） | `VideoUnderstandingPipeline.from_settings()` |
| `evaluation.judge` | LLM-as-Judge 质量评估 | `LLMJudge.evaluate()` |
| `evaluation.scoring` | 统计聚合（均值/标准差/CI/维度分析） | `ScoreAggregator` |
| `utils.visualization` | 实验结果可视化（排名/热力图/表格） | `plot_metric_ranking()`, `plot_quality_heatmap()` 等 |
| `config` / `models` | Pydantic 配置与数据模型 | `ToolkitSettings`, `KeyFrame`, `FrameDescription` 等 |

### 在其他项目中使用

其他项目只需 3 步即可接入实时图像理解能力：

```python
from ahu_paimon_toolkit.capture import find_window, capture_window, frame_to_base64
from ahu_paimon_toolkit.vlm import AsyncVLMClient

# 1. 捕获窗口画面
hwnd, title = find_window("你的应用标题")
frame = capture_window(hwnd)
b64 = frame_to_base64(frame)

# 2. 发送给 WSL 中的 vLLM
client = AsyncVLMClient(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="描述这张图片中的内容"
)
result = await client.describe_frame(KeyFrame(image_b64=b64, timestamp_ms=0))
print(result.description)

# 3. 或直接使用完整 pipeline
from ahu_paimon_toolkit.pipeline import VideoUnderstandingPipeline
pipeline = VideoUnderstandingPipeline.from_settings(settings)
result = await pipeline.run()
```

### 泛用性评估

| 能力 | 状态 | 说明 |
|------|:---:|------|
| 窗口画面采集 | ✅ | 基于 `win32gui` + `mss`，支持任意 Windows 窗口 |
| 帧差关键帧提取 | ✅ | MSE/SSIM 双算法，阈值可配 |
| vLLM 推理 | ✅ | URL 无关，支持任何 OpenAI 兼容接口（本地/WSL/远端） |
| LLM-as-Judge 评估 | ✅ | 可用于任何 VLM 输出的自动化质量评估 |
| 结果可视化 | ✅ | 开箱即用的排名/热力图/表格图生成 |
| Pydantic 配置管理 | ✅ | TOML + `.env` 双配置源 |
| 跨平台支持 | ⚠️ | 窗口捕获仅限 Windows；vLLM 客户端跨平台 |
| 网络发现 | ⚠️ | WSL 端口转发由用户/宿主机负责 |

**结论**：`ahu_paimon_toolkit` 已具备足够的泛用性——在 Windows + WSL vLLM 环境下，其他项目引入后仅需配置窗口标题和 vLLM 地址即可使用，无需编写额外的采集/推理/评估代码。

---

## 项目结构

```
├── ahu_paimon_toolkit/          # 通用工具库 (git submodule)
│   └── src/ahu_paimon_toolkit/  #   capture / vlm / pipeline / evaluation / utils
├── experiments/                 # 实验脚本
│   ├── benchmark_speed/         #   推理速度基准测试
│   ├── benchmark_quality/       #   LLM-as-Judge 质量评估
│   ├── sota_validation/         #   SOTA 模型校准实验
│   ├── video_understanding/     #   视频理解 pipeline 实验
│   └── gameplay_analysis/       #   实际游戏分析实验
├── assets/                      # 测试素材 (截图/视频 + JSON rubric)
│   ├── images/                  #   图片素材 (Minecraft, 原神)
│   └── videos/                  #   视频素材
├── results/                     # 实验结果 (按版本和时间戳组织)
│   ├── v3/                      #   本地模型基准结果
│   └── v3-sota/                 #   SOTA 校准实验结果
├── QA/                          # 质量分析报告
│   ├── dissertation_model_selection_and_validation/  # 毕设用完整报告
│   └── v3/                      #   各轮实验 QA 报告
├── blueprint/                   # 实验设计文档
├── templates/                   # 报告模板
├── config_benchmark.toml        # 本地模型基准测试配置
├── config_sota_validation.toml  # SOTA 校准实验配置
├── config.toml                  # Pipeline 配置
├── main.py                      # 视频理解 pipeline 入口
└── run_benchmark.py             # 基准测试入口
```

## 环境要求

- **操作系统**: Windows 10/11
- **Python**: >= 3.12
- **包管理器**: [uv](https://docs.astral.sh/uv/)
- **GPU**: NVIDIA GPU (>= 8 GB VRAM, 推荐 RTX 4080 SUPER)
- **本地模型服务**: vLLM 运行在 WSL 中，暴露 `http://localhost:8000/v1`
- **云端 API**: DeepSeek API Key (用于汇总和质量评估)

## 快速开始

```powershell
# 1. 安装依赖
uv sync

# 2. 配置（复制并编辑 .env）
cp .env_example .env

# 3. 在 WSL 中启动 vLLM
#    vllm serve Qwen/Qwen3-VL-2B-Instruct --host 0.0.0.0

# 4. 运行视频理解 pipeline
uv run main.py

# 5. 运行基准测试
uv run run_benchmark.py -c config_benchmark.toml
```

## 实验复现

### SOTA 校准实验

```powershell
# 需要 DMXAPI_API_KEY 配置在 .env 中
uv run -m experiments.sota_validation.run_sota_validation
```

### 模型选型基准测试

```powershell
uv run run_benchmark.py -c config_benchmark.toml
```

## 测试

```powershell
uv run pytest -v
```
