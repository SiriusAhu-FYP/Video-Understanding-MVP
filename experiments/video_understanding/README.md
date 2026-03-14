# 视频理解专项实验 (Video Understanding Experiment)

## 实验目的

使用完整的视频理解流水线（截图 -> 帧差过滤 -> vLLM 描述 -> DeepSeek 汇总）处理真实游戏视频，对比不同模型的理解效果。

## 实验流程

1. 通过 PotPlayer 自动播放视频
2. 从播放器窗口实时截图
3. 帧差过滤提取关键帧
4. 发送关键帧给 vLLM 获取描述
5. 将所有描述发送给 DeepSeek 生成视频总结
6. 每个视频重复 3 次

## 测试参数

- 视频文件: `assets/videos/` 下的 mp4 文件
- 截图间隔: 500ms（来自 config.toml）
- 帧差算法: MSE，阈值 500.0
- 每个视频每个模型重复 3 次

## 使用方法

### 前置条件

1. 在 `.env` 中配置 `PLAYER_EXE_PATH`（PotPlayer 可执行文件路径）
2. 在 WSL 中启动 vLLM 服务
3. 确保 `assets/videos/` 下有视频文件

### 运行实验

```powershell
# 运行全部视频（每视频 3 次）
uv run experiments/video_understanding/run_experiment.py

# 自定义重复次数
uv run experiments/video_understanding/run_experiment.py --runs 5
```

### 生成审阅文档

实验结束后，可以对任意一次运行生成帧级审阅文档：

```powershell
# 指定运行目录
uv run experiments/video_understanding/generate_review.py reports/Qwen_xxx/video_MC_01_run1
```

## 输出结构

```
experiments/video_understanding/
├── reports/
│   ├── {model}_{timestamp}/
│   │   ├── {video}_run{N}/
│   │   │   ├── frames/           # 每次采样的帧图片（含关键帧和跳过帧）
│   │   │   ├── run_log.json      # 结构化帧级日志
│   │   │   ├── summary.txt       # DeepSeek 总结
│   │   │   └── review.md         # 审阅文档（由 generate_review.py 生成）
│   │   ├── experiment.log
│   │   └── report.md             # 单模型报告
│   └── comparison_report.md      # 对比报告
```

## 审阅文档内容

`generate_review.py` 生成的 Markdown 文档包含：

- **运行元信息**: 模型、视频、配置参数
- **帧时间线表格**: 帧序号 | 时间戳 | 差异值 | 是否关键帧 | 判定原因 | 帧图片链接 | VLM 描述
- **DeepSeek 最终总结**
- **关键帧详细描述**: 每个关键帧的图片和完整 VLM 描述
