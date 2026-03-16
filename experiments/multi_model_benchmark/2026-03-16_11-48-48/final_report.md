# 多模型综合基准测试报告

**生成时间**: 2026-03-16 13:17:14

## 实验目标

本实验旨在系统评估 8 个小型视觉语言模型（VLM, 0.8B-4B 参数）在游戏场景 AI 伴侣应用中的表现。
评估维度包括：

1. **推理速度**：首Token延迟（TTFT）、吞吐量（tok/s）、理论最大帧率
2. **视频理解**：实时截图→帧差检测→VLM描述→DeepSeek汇总的完整流水线效果
3. **资源效率**：各模型最小可用 GPU 显存配置

## 实验环境

- **GPU**: NVIDIA GeForce RTX 4080 SUPER (16376 MB)
- **Python**: 3.12.11
- **Git**: da9fa46fcccf
- **开始时间**: 2026-03-16T11:48:48.486266
- **结束时间**: 2026-03-16T13:17:13.906299

## 模型清单

| # | 模型ID | 简称 | 参数量级 | 状态 |
|---|--------|------|---------|------|
| 1 | mistralai/Ministral-3-3B-Instruct-2512 | mistralai_Ministral-3-3B-Instruct-2512 | 3B | 实验中失败 |

## GPU 显存优化

通过逐步探测，为每个模型找到了在 RTX 4080 SUPER (16GB) 上可用的最小 `gpu_memory_utilization` 值：

| 模型 | 最小 gpu_memory_utilization | 等效分配显存 |
|------|:---:|:---:|
| Qwen/Qwen3.5-2B | 0.55 | ~9006 MB |
| Qwen/Qwen3.5-0.8B | 0.30 | ~4912 MB |
| Qwen/Qwen3-VL-2B-Instruct | 0.40 | ~6550 MB |
| OpenGVLab/InternVL2_5-2B | 0.50 | ~8188 MB |
| microsoft/Phi-3.5-vision-instruct | 0.30 | ~4912 MB |
| mistralai/Ministral-3-3B-Instruct-2512 | 0.30 | ~4912 MB |
| deepseek-ai/deepseek-vl2-tiny | 0.30 | ~4912 MB |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | 0.45 | ~7369 MB |

> 更低的 gpu_memory_utilization 意味着模型可与其他任务共享 GPU 资源，这对实际部署非常重要。

## 视频理解测试结果

详细数据见 [comparison_report.md](video_understanding/comparison_report.md)。

## 研究问题回答

### 1. 理论最大帧率是多少？


### 2. 安全的截图采集帧率是多少？

建议取理论最大帧率的 70% 作为安全值，以预留网络传输、队列调度和帧差计算的开销。
在实际使用中，帧差过滤会大幅降低 VLM 调用频率——仅当画面发生显著变化时才触发推理。

### 3. 未处理的帧如何处理？

KeyFrameQueue 使用过期机制（默认 10 秒），超时的帧被静默丢弃，不阻塞流水线。
丢帧率高说明模型推理跟不上采集速度，应降低采集帧率或换用更快的模型。

## 应用建议

## 事件记录

- **Qwen/Qwen3.5-2B**（health_check阶段）：文本和视觉检查失败 → 从实验中排除
- **Qwen/Qwen3.5-0.8B**（startup阶段）：vLLM 在 300s 内未就绪: Connection error. → 从实验中排除
- **Qwen/Qwen3-VL-2B-Instruct**（health_check阶段）：视觉检查失败 → 从实验中排除
- **OpenGVLab/InternVL2_5-2B**（health_check阶段）：视觉检查失败 → 从实验中排除
- **microsoft/Phi-3.5-vision-instruct**（health_check阶段）：文本和视觉检查失败 → 从实验中排除
- **deepseek-ai/deepseek-vl2-tiny**（health_check阶段）：视觉检查失败 → 从实验中排除
- **HuggingFaceTB/SmolVLM2-2.2B-Instruct**（startup阶段）：vLLM 在 300s 内未就绪: Error code: 502 → 从实验中排除
- **mistralai/Ministral-3-3B-Instruct-2512**（experiment阶段）：详见 orchestrator.log → 已跳过
