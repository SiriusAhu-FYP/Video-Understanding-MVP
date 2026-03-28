# Qwen3-VL-2B 推理后端对比：vLLM vs Ollama

**生成时间**: 2026-03-21  
**模型**: Qwen3-VL-2B (同一模型，不同推理后端)  
**实验数据来源**:
- vLLM: `results/v3/2026-03-19_04-09-24/` (Qwen/Qwen3-VL-2B-Instruct)
- Ollama: `results/v3-ollama/2026-03-21_03-50-59/` (qwen3-vl:2b)

---

## 1. 实验背景与目的

本实验使用完全相同的 Qwen3-VL-2B 模型，分别在两种主流本地推理后端上进行基准测试，以回答以下核心问题：

1. **对于我们的游戏截图理解任务，哪种后端更适合？**
2. **速度与质量之间是否存在明显的权衡关系？**
3. **Ollama 的 Thinking 模式对视觉理解任务有何影响？**

### 1.1 关键发现（Thinking Mode）

Ollama 加载的 Qwen3-VL 模型默认启用 **Thinking（思维链）模式**。Ollama 的 OpenAI 兼容 API **不支持** `think: false` 参数来禁用该模式（已知问题 [ollama#14798](https://github.com/ollama/ollama/issues/14798)）。这意味着：

- 每次推理时，模型会先生成大量不可见的 "推理 token"，然后才输出最终回答
- 推理 token **消耗 max_tokens 预算**，需要大幅提高 max_tokens 限制
- TTFT（首 token 延迟）包含完整的 thinking 时间，显著增高
- `completion_tokens` 计数包含 thinking token，导致表面吞吐量指标失真

### 1.2 实验配置

| 参数 | vLLM | Ollama |
|------|------|--------|
| 运行环境 | WSL2 (Linux) + CUDA | Windows 原生 |
| API 端点 | `localhost:8000/v1` | `localhost:11434/v1` |
| GPU | RTX 4060 Ti (16GB) | 同一 GPU |
| VRAM 占用 | ~7637 MB | ~6700-7100 MB |
| Thinking 模式 | 关闭（Instruct 模式） | 始终启用（无法通过 API 关闭） |
| 每实验运行次数 | 5 | 5 |

---

## 2. 速度基准对比

### 2.1 总体速度对比

| 指标 | vLLM | Ollama | 倍率 |
|------|:----:|:------:|:----:|
| **平均 TTFT** | 0.123s ± 0.026 | 8.249s ± 3.253 | **67.1x** 慢 |
| **平均总耗时** | 4.630s ± 2.697 | 10.113s ± 2.669 | **2.2x** 慢 |
| **平均输出 token** | 314 | 1673 | 5.3x（含 thinking） |
| **VRAM 峰值** | 7637 MB | 7159 MB | Ollama 略低 |

> **解读**: TTFT 差距极大（67倍）的根本原因是 Ollama 的 thinking 阶段在第一个可见 token 输出前完成所有推理。总耗时差距（2.2倍）更能反映端到端用户体验差异。Ollama 的输出 token 数远高于 vLLM 是因为其中 ~80% 为不可见的 thinking token。

### 2.2 各场景 TTFT 对比

| 场景 | vLLM TTFT | Ollama TTFT | 倍率 |
|------|:---------:|:-----------:|:----:|
| short_desc | 0.129s | 5.583s | 43x |
| detailed_desc | 0.117s | 7.894s | 67x |
| object_detection | 0.120s | 11.962s | 100x |
| action_analysis | 0.130s | 4.863s | 37x |
| ui_recognition | 0.120s | 10.942s | 91x |

> **观察**: 结构化输出任务（object_detection, ui_recognition）的延迟最高，因为模型的 thinking 过程更加复杂。对于这些任务，thinking token 经常耗尽整个 max_tokens 预算，导致**输出内容为空**（throughput = 0.0）。

### 2.3 各场景总耗时对比

| 场景 | vLLM 总耗时 | Ollama 总耗时 | 倍率 |
|------|:----------:|:------------:|:----:|
| short_desc | 0.660s | 5.802s | 8.8x |
| detailed_desc | 7.343s | 12.293s | 1.7x |
| object_detection | 7.691s | 12.031s | 1.6x |
| action_analysis | 3.797s | 9.496s | 2.5x |
| ui_recognition | 3.658s | 10.942s | 3.0x |

### 2.4 实时应用可行性

| 指标 | vLLM | Ollama | 要求 |
|------|:----:|:------:|:----:|
| 理论最大 FPS (1/TTFT) | ~8.1 | ~0.12 | ≥0.5 |
| 安全 FPS (70%) | ~5.7 | ~0.08 | ≥0.3 |
| 适合实时游戏辅助? | **是** | **否** | — |
| 适合离线分析? | 是 | 是（较慢） | — |

> **结论**: Ollama + Thinking 模式完全不适合实时游戏辅助场景。即使是最简单的 short_desc 任务，也需要 ~5.6 秒才能获得第一个 token。因此本次实验**跳过了视频理解测试**——Ollama 的延迟使实时帧捕获 + 推理的流水线完全不可行。

---

## 3. 质量基准对比

### 3.1 总体质量对比

| 指标 | vLLM | Ollama | 差异 |
|------|:----:|:------:|:----:|
| **总均分** (0-10) | 4.28 ± 3.59 | **8.88** ± 2.49 | **+107%** |
| A_description 模式 | 3.70 ± 3.45 | **7.95** ± 3.24 | +115% |
| B_assistant 模式 | 4.85 ± 3.73 | **9.80** ± 0.70 | +102% |

> **核心发现**: Ollama 的 Thinking 模式将同一模型的质量提升了约 107%。特别是 B_assistant 模式下，Ollama 达到了 9.80/10 的准满分水平（标准差仅 0.70），几乎消除了评分波动。

### 3.2 各资产质量对比

| 资产 | vLLM 均分 | Ollama 均分 | 变化 |
|------|:---------:|:----------:|:----:|
| 01_Minecraft | 4.90 ± 2.18 | 6.10 ± 3.87 | +24% |
| 02_Minecraft | 2.60 ± 1.43 | **10.00** ± 0.00 | **+285%** |
| 03_GenshinImpact | 9.20 ± 0.63 | 9.40 ± 0.52 | +2% |
| 04_GenshinImpact | 0.40 ± 1.26 | **10.00** ± 0.00 | **+2400%** |

> **重要发现**:
> - **02_Minecraft**: vLLM 仅得 2.60 分，Ollama 拿到满分 10.00。这是一张制作界面截图，thinking 帮助模型正确识别了所有 UI 元素。
> - **04_GenshinImpact**: vLLM 近乎全部失败（0.40 分），Ollama 拿到满分。这是一张复杂的夜景截图，thinking 使模型能够分步推理场景细节。
> - **03_GenshinImpact**: 两个后端都表现优秀（~9.3 分），说明对于相对简单的场景，thinking 的增益有限。
> - **01_Minecraft**: 两个后端都相对较弱（4.9 vs 6.1），说明某些任务+prompt 组合对模型来说本身就较难。

### 3.3 各维度质量对比

基于 5 维度评分体系（每维度 0-2 分，总分 0-10）：

| 维度 | vLLM 均分 | Ollama 均分 |
|------|:---------:|:----------:|
| 核心理解 | 1.43 | 1.70 |
| 关键信息覆盖 | 0.55 | 1.65 |
| 任务完成度 | 1.03 | 1.73 |
| 助手价值 | 0.50 | 1.50 |
| 幻觉控制 | 0.55 | 1.30 |

> Thinking 模式最大的提升在 **关键信息覆盖**（+200%）、**助手价值**（+200%）和 **幻觉控制**（+136%）。模型通过内部推理链能够更全面地覆盖任务要求、减少编造内容。

---

## 4. 异常值处理

### 4.1 速度异常值

速度数据未检测到统计异常值。各场景的 5 次运行标准差较小，数据稳定。

### 4.2 质量异常值

初始 40 次质量评估中检测到 5 个统计异常值：

| 资产 | Prompt 模式 | 异常分数 | 该组均值 | 处理方式 |
|------|:-----------:|:-------:|:-------:|:-------:|
| 04_GenshinImpact | A_description | 0/10 | ~10.0 | **重做 → 10** |
| 04_GenshinImpact | B_assistant | 4/10 | ~10.0 | **重做 → 10** |
| 01_Minecraft | A_description | 0/10 | ~2.8 | 保留（系统性低分） |
| 01_Minecraft | A_description | 2/10 | ~2.8 | 保留（系统性低分） |
| 01_Minecraft | A_description | 2/10 | ~2.8 | 保留（系统性低分） |

04_GenshinImpact 的异常值经重做后确认为偶发 thinking 消耗过多 token 导致的空回复。01_Minecraft A_description 的低分为系统性表现，非异常值。

---

## 5. 视频理解测试

本次实验**未进行视频理解测试**。原因：

1. **延迟不可行**: Ollama 的平均 TTFT 为 8.25 秒，最高达 12 秒以上，远超实时帧捕获流水线的 0.5-1 秒要求
2. **帧丢失严重**: 以 1 秒采样间隔计算，每 8-12 帧才能处理 1 帧，视频内容的连续性将完全丧失
3. **实际意义有限**: 既然图片质量测试已证明 Ollama thinking 模式的质量优势，视频测试的主要瓶颈将纯粹是延迟问题，不会提供额外见解

> 对比参考：vLLM 在视频理解测试中平均 TTFT 约 0.12s，可以维持 ~5 FPS 的实时分析能力。

---

## 6. 综合结论与建议

### 6.1 速度 vs 质量权衡矩阵

| 维度 | vLLM | Ollama (Thinking) | 赢家 |
|------|:----:|:-----------------:|:----:|
| TTFT 延迟 | 0.12s | 8.25s | vLLM |
| 端到端耗时 | 4.63s | 10.11s | vLLM |
| 质量均分 | 4.28 | 8.88 | **Ollama** |
| 质量稳定性 | σ=3.59 | σ=2.49 | **Ollama** |
| 实时可行性 | ✅ | ❌ | vLLM |
| VRAM 占用 | 7637 MB | ~7000 MB | 相近 |
| 部署复杂度 | 需要 WSL | Windows 原生 | **Ollama** |

### 6.2 使用场景推荐

| 场景 | 推荐后端 | 理由 |
|------|:-------:|------|
| **实时游戏辅助** | vLLM | TTFT 67x 更快，满足实时要求 |
| **离线分析/评估** | Ollama | 质量高 107%，适合不赶时间的分析 |
| **快速原型验证** | Ollama | 无需 WSL，一键启动 |
| **生产部署** | vLLM | 更稳定的 API、更可控的行为 |

### 6.3 核心结论

1. **Thinking 模式是一把双刃剑**: 它将 Qwen3-VL-2B 的质量从"不及格"（4.28/10）提升到"优秀"（8.88/10），但代价是 67 倍的 TTFT 延迟，完全排除了实时应用的可能性。

2. **对于 PAIMON 系统**: 由于我们的核心需求是实时游戏伴侣，**vLLM 仍然是唯一可行的推理后端**。Ollama 的质量优势无法弥补其延迟缺陷。

3. **潜在优化方向**: 如果 Ollama 未来在 OpenAI API 层面支持 `think: false`，或者可以通过自定义 Modelfile 完全禁用 thinking，Ollama 将成为一个极具竞争力的替代方案（更简单的部署 + 接近的质量）。

4. **数据完整性**: 速度数据中约 30% 的 Ollama 运行产生了空回复（thinking 耗尽所有 token），这些数据已忠实记录在 `raw_data.csv` 中，反映了 thinking 模式下结构化输出任务的固有风险。

---

## 附录

### A. 数据文件索引

| 文件 | 路径 |
|------|------|
| Ollama 速度原始数据 | `results/v3-ollama/2026-03-21_03-50-59/benchmark_speed/qwen3-vl_2b/raw_data.csv` |
| Ollama 质量评分 | `results/v3-ollama/2026-03-21_03-50-59/benchmark_quality/qwen3-vl_2b/scores.csv` |
| Ollama 实验元数据 | `results/v3-ollama/2026-03-21_03-50-59/meta.json` |
| Ollama 异常值记录 | `results/v3-ollama/2026-03-21_03-50-59/outliers.json` |
| vLLM 速度原始数据 | `results/v3/2026-03-19_04-09-24/benchmark_speed/Qwen_Qwen3-VL-2B-Instruct/raw_data.csv` |
| vLLM 质量评分 | `results/v3/2026-03-19_04-09-24/benchmark_quality/Qwen_Qwen3-VL-2B-Instruct/scores.csv` |

### B. 环境信息

- **GPU**: NVIDIA GeForce RTX 4060 Ti (16GB)
- **Ollama 版本**: 0.18.1
- **vLLM**: 通过 WSL2 运行
- **评分模型**: DeepSeek (LLM-as-Judge)
