# Video-Understanding-MVP 项目上下文与架构说明

## 1. 我们在做什么？(Project Overview)
本项目是一个**实时视频理解 AI 伴侣系统**，同时包含一套完整的**多模型基准测试框架**。

**核心系统（ahu_paimon_toolkit）：** 在 Windows 端播放游戏视频，程序在后台自动高频截图，通过计算前后帧的差异（状态增量）提取出"关键帧"。这些关键帧被放入异步队列，发送给本地部署的 vLLM 获取局部动态描述，最后将所有描述打包发送给云端 DeepSeek API，生成对整个视频的总结。

**基准测试框架：** 系统评估多个小型 VLM（0.8B-4B 参数）在游戏场景下的推理速度、描述质量和视频理解能力，帮助选择最适合实时伴侣应用的模型。

## 2. 运行环境与依赖 (Environment)
* **代码执行端:** Windows 11 (原生 PowerShell)。所有 Python 业务逻辑都在这里运行。
* **依赖管理:** `uv`。
* **本地 AI 节点:** vLLM 运行在 WSL 中，暴露的 OpenAI 兼容接口地址为 `http://localhost:8000/v1`。
    * 通过 `experiments/utils/vllm_manager.py` 管理 vLLM 的启停。
    * 引擎端已配置了 `--enable-prefix-caching` 和显存分辨率限制。
* **云端 AI 节点:** DeepSeek API。(通过 `.env` 中的 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_API_BASE_URL` 配置)

## 3. 项目结构 (Project Structure)
```
Video-Understanding-MVP/
├── ahu_paimon_toolkit/         # 核心 Python 包（editable install）
│   └── src/ahu_paimon_toolkit/
│       ├── capture/            # 窗口截图 + 帧差检测
│       ├── evaluation/         # LLM-as-Judge 评分
│       ├── pipeline/           # 队列管理 + 汇总器
│       ├── vlm/                # VLM 客户端 + 模型工具
│       └── utils/              # 图片编码、GPU 信息、可视化
│
├── experiments/                # 实验脚本（仅代码，不含输出数据）
│   ├── benchmark_speed/        #   推理速度基准测试
│   │   └── benchmark.py
│   ├── benchmark_quality/      #   描述质量评估（LLM-as-Judge）
│   │   └── benchmark.py
│   ├── gameplay_analysis/      #   游戏画面分析专项实验
│   │   └── run_experiment.py
│   ├── video_understanding/    #   完整视频理解流水线测试
│   │   └── run_experiment.py
│   └── utils/                  #   实验间共享工具
│       ├── csv_io.py           #     CSV 读写 (init_csv, append_csv, read_csv_dicts)
│       ├── inference.py        #     通用推理 + RunResult 数据类
│       ├── logging.py          #     loguru 文件日志设置 (setup_experiment_log)
│       ├── reporting.py        #     模板驱动报告生成 + DeepSeek 分析
│       └── vllm_manager.py     #     WSL vLLM 服务管理 (start/stop/probe)
│
├── templates/                  # 报告 Markdown 模板
│   ├── speed_comparison.md     #   推理速度对比报告模板
│   ├── video_comparison.md     #   视频理解对比报告模板
│   ├── quality_comparison.md   #   描述质量对比报告模板
│   ├── final_report.md         #   综合报告模板
│   ├── incident_report.md      #   事件记录模板
│   └── readme.md               #   会话 README 模板
│
├── results/                    # 所有实验结果（.gitignore 中排除）
│   ├── v1/                     #   第一批实验结果
│   ├── v2/                     #   第二批实验结果
│   └── v3/                     #   第三批实验结果（当前）
│       └── {timestamp}/        #     每次运行的会话目录
│           ├── config.toml
│           ├── meta.json
│           ├── orchestrator.log
│           ├── final_report.md
│           ├── charts/
│           ├── benchmark_speed/{model}/
│           ├── benchmark_quality/{model}/
│           └── video_understanding/{model}/
│
├── assets/                     # 测试资源
│   ├── images/                 #   游戏截图 + JSON 评估定义
│   └── videos/                 #   测试视频
│
├── tests/                      # pytest 测试套件
├── blueprint/                  # 项目设计文档
├── config.toml                 # 流水线运行时配置
├── config_benchmark.toml       # 基准测试配置（模型列表、输出路径等）
└── run_benchmark.py            # 多模型基准测试编排器
```

## 4. 配置体系 (Configuration)

### config.toml（流水线运行时）
* `[capture]`：窗口关键字、截图频率、图片压缩上限、录制时长。
* `[algorithm]`：帧差算法（MSE/SSIM）和阈值。
* `[queue]`：队列最大长度和帧过期时间。
* `[llm]`：vLLM base_url 和模型名称。

### config_benchmark.toml（基准测试）
* `[general]`：实验名称、`output_dir`（如 `results/v3`）、时间戳格式。
* `[models]`：待测试的模型 ID 列表。
* `[vllm]`：GPU 显存利用率、启动超时、轮询间隔。
* `[benchmark_speed]`、`[benchmark_quality]`、`[video_understanding]`：各实验的开关和参数。

## 5. 核心工作流 (Core Pipeline)

### 实时视频理解流水线
1. **画面捕获与差异对比**: `win32gui` + `mss` 截图 → OpenCV 帧差对比 → 超过阈值保留为关键帧。
2. **异步队列与淘汰机制**: `KeyFrameQueue` 带过期淘汰，超时帧静默丢弃。
3. **vLLM 本地推理**: 异步发送关键帧到本地 vLLM，获取画面描述。
4. **DeepSeek 汇总**: 收集时序描述，调用 DeepSeek 生成视频总结。

### 多模型基准测试流程
1. **预飞行**: GPU 显存探测 → 健康检查 → 视觉检查。
2. **实验**: 对每个模型依次启动 vLLM → 运行各实验 → 停止 vLLM。
3. **报告**: 模板驱动生成 → DeepSeek 写分析 → 可视化图表。

## 6. 报告生成 (Report Generation)
- 报告模板位于 `templates/` 目录，使用 `{{placeholder}}` 注入数据。
- `<!-- ANALYSIS -->` 标记的部分由 DeepSeek 自动生成专业分析。
- 通过 `experiments/utils/reporting.generate_report_from_template()` 完成完整流程：
  加载模板 → 填充数据 → (可选) DeepSeek 分析 → 写入文件。

## 7. 关于提交
- 总是以 `Conventional Commits` 的规范提交代码。
- 总是在实现特定功能后进行一次提交，而不是囤积多个功能后再一次性提交。
- 总是为每个功能添加相应的测试用例。

## 其他
1. 总是使用 `from loguru import logger as lg` 作为日志记录器。
2. 实验脚本中使用 `experiments.utils.logging.setup_experiment_log()` 设置文件日志。
3. CSV 操作使用 `experiments.utils.csv_io` 中的工具函数。
4. 总是及时更新 README.md 以反映项目的最新状态。
