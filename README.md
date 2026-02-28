# Video Understanding MVP

一个非交互式的纯后台视频理解流水线 (Pipeline)。在 Windows 端自动高频截图，通过帧差算法 (Frame Differencing) 提取关键帧 (Keyframe)，发送给本地 vLLM 进行逐帧描述，最终汇总至云端 DeepSeek 生成视频总结。

## 架构概览

```
屏幕截图 (mss) → 帧差过滤 (OpenCV) → 异步队列 (asyncio.Queue)
    → vLLM 逐帧描述 (Qwen3-VL) → DeepSeek 汇总 → 视频总结
```

### 核心流程

1. **画面捕获 (Capture)**: 通过 `win32gui` 模糊匹配目标窗口，使用 `mss` 极速截图
2. **帧差过滤 (Frame Diff)**: 用 OpenCV 计算当前帧与上一关键帧的差异 (MSE / SSIM)，超过阈值才保留
3. **异步队列 (Queue)**: 关键帧带时间戳入队，消费者自动丢弃过期帧，保证数据时效性
4. **vLLM 推理 (VLM)**: 异步调用本地 vLLM 的 OpenAI 兼容接口，对每帧生成文本描述
5. **DeepSeek 汇总 (Summarize)**: 收集所有帧描述，调用云端 DeepSeek API 生成最终视频总结

## 环境要求

- **操作系统**: Windows 11
- **Python**: >= 3.12
- **包管理器**: [uv](https://docs.astral.sh/uv/)
- **本地模型服务**: vLLM 运行在 WSL 中，暴露 `http://localhost:8000/v1`
- **云端 API**: DeepSeek API (需配置 API Key)

## 快速开始

### 1. 安装依赖

```powershell
uv sync
```

### 2. 配置

编辑 `config.toml` 设置运行参数：

```toml
[capture]
window_title_keyword = "PotPlayer"   # 目标窗口标题关键词 (模糊匹配)
screenshot_interval_ms = 500         # 截图间隔 (毫秒)
max_size = 512                       # 图片长边压缩上限 (像素)
recording_duration_s = 20            # 录制时长 (秒)

[algorithm]
method = "mse"                       # 帧差算法: "mse" 或 "ssim"
diff_threshold = 500.0               # 关键帧判定阈值

[queue]
max_size = 50                        # 队列最大容量
expiry_time_ms = 10000               # 帧过期时间 (毫秒)

[llm]
vllm_base_url = "http://localhost:8000/v1"
vllm_model = "Qwen/Qwen3-VL-2B-Instruct"

[log]
log_dir = "logs"
console_level = "INFO"
file_level = "DEBUG"
```

在 `.env` 文件中配置 DeepSeek 密钥：

```
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_API_BASE_URL=https://api.deepseek.com
```

### 3. 运行

1. 在 WSL 中启动 vLLM 服务
2. 手动打开视频播放器并开始播放视频
3. 运行流水线：

```powershell
uv run main.py
```

程序会在 `recording_duration_s` 秒后自动停止截图，等待所有帧处理完毕后调用 DeepSeek 生成总结。

## 项目结构

```
├── main.py                  # 入口，编排整条流水线
├── config.toml              # 全局配置
├── .env                     # DeepSeek API 密钥 (不纳入版本控制)
├── prompts/
│   ├── vlm_prompt.md        # vLLM 逐帧描述提示词 (Prompt)
│   └── deepseek_prompt.md   # DeepSeek 汇总提示词 (Prompt)
├── pipeline/
│   ├── config.py            # 配置加载 (Config)
│   ├── models.py            # 数据模型 (Pydantic Models)
│   ├── capture.py           # 窗口截图与帧差过滤 (Capture)
│   ├── queue_manager.py     # 带过期淘汰的异步队列 (Queue)
│   ├── vlm.py               # vLLM 推理客户端 (VLM Client)
│   └── summarizer.py        # DeepSeek 汇总客户端 (Summarizer)
├── tests/                   # 功能测试
└── blueprint/               # 项目设计文档
```

## 提示词管理

vLLM 和 DeepSeek 的提示词 (Prompt) 分别存放在 `prompts/` 目录下的 Markdown 文件中，可以随时修改而无需改动代码：

- `prompts/vlm_prompt.md` — 控制 vLLM 如何描述单帧画面
- `prompts/deepseek_prompt.md` — 控制 DeepSeek 如何汇总所有帧描述

## 测试

```powershell
uv run pytest -v
```

## 依赖清单

| 包名 | 用途 |
|------|------|
| `opencv-python-headless` | 帧差计算 (Frame Diff) |
| `mss` | 极速截图 (Screen Capture) |
| `pywin32` | Windows 窗口发现 (Window Discovery) |
| `httpx` | 异步 HTTP 客户端 (Async HTTP) |
| `pydantic` | 数据模型验证 (Data Validation) |
| `pydantic-settings` | 环境变量加载 (.env Loading) |
| `loguru` | 日志系统 (Logging) |
| `numpy` | 图像矩阵运算 (Image Processing) |
