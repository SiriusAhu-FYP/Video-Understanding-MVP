# 视频理解专项实验 (Video Understanding)

## 实验目的

测试完整的视频理解流水线：PotPlayer 播放视频 → 窗口截图 → 帧差检测 → vLLM 描述 → DeepSeek 汇总。

## 流水线

```
PotPlayer 播放视频
       ↓
窗口截图 (mss, 每 500ms)
       ↓
帧差对比 (MSE, 阈值 500.0)
       ↓ 超过阈值 → 关键帧
KeyFrameQueue (带过期淘汰)
       ↓
vLLM 局部描述 (异步消费)
       ↓
DeepSeek 总结 (收集所有描述后调用)
```

每个视频默认重复运行 3 次，录制时长 = 视频时长 + 3 秒缓冲。

## 前置条件

- `.env` 中配置 `PLAYER_EXE_PATH`（PotPlayer 路径）
- `.env` 中配置 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_API_BASE_URL`
- vLLM 在 WSL 中运行
- 视频文件放在 `assets/videos/`（`.mp4` 格式）

## 使用方法

### 独立运行

```powershell
uv run experiments/video_understanding/run_experiment.py
uv run experiments/video_understanding/run_experiment.py --runs 5
```

### 通过编排器运行（推荐）

```powershell
uv run run_benchmark.py
```

编排器会自动管理 vLLM 启停，输出到 `results/v3/{timestamp}/video_understanding/` 目录。

### 生成单次运行的详细 review

```powershell
uv run experiments/video_understanding/generate_review.py <run_dir>
```

## 输出

通过编排器运行时，输出到 `results/v3/{timestamp}/video_understanding/{model}/`：

```
{model}/
├── report.md                    # 模型实验报告
├── experiment.log               # 运行日志
└── {video}_run{N}/
    ├── frames/                  # 截图（key/skip 标记）
    ├── run_log.json             # 结构化运行日志
    ├── summary.txt              # DeepSeek 总结文本
    └── review.md                # 详细 review（可选）
```

## 关键参数

| 参数 | 来源 | 默认值 |
|------|------|--------|
| 截图频率 | `config.toml` capture.screenshot_interval_ms | 500ms |
| 帧差算法 | `config.toml` algorithm.method | MSE |
| 帧差阈值 | `config.toml` algorithm.diff_threshold | 500.0 |
| 队列过期时间 | `config.toml` queue.expiry_time_ms | 10000ms |
| 重复次数 | `config_benchmark.toml` video_understanding.runs | 5 |

## 相关脚本

- `run_experiment.py` — 主实验脚本
- `generate_review.py` — 从 `run_log.json` 生成详细 review 文档
