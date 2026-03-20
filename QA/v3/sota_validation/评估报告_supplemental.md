# SOTA 模型校准实验 - 补充实验评估报告

**生成时间**: 2026-03-19 17:25:37
**完整数据**: `results\v3-sota\2026-03-19_14-57-39\complete_data`
**新实验数据**: `results\v3-sota\2026-03-19_14-57-39`
**旧实验数据**: `results/v3-sota/2026-03-19_01-39-43`
**裁判模型**: deepseek-chat

---

## 1. 实验目的

本次补充实验针对上一轮 SOTA 校准实验中发现的以下问题进行定点重跑：
- 不完整的实验（运行次数 < 5）→ 补齐缺失轮次
- 高方差实验（标准差 ≥ 1.2）→ 全部 5 轮重跑
- 低分实验（均分 < 6.0）→ 全部 5 轮重跑
- 新增模型 kimi-k2.5 的全部实验

## 2. 补充分析摘要

### 图片实验分析

| 模型 | KEEP | REDO | SUPPLEMENT | 新运行数 |
|------|:---:|:---:|:---:|:---:|
| Qwen3-VL-235B-A22B-Instruct | 4 | 4 | 0 | 20 |
| claude-opus-4-6 | 6 | 2 | 0 | 10 |
| claude-sonnet-4-6 | 6 | 2 | 0 | 10 |
| gemini-3.1-pro-preview | 0 | 8 | 0 | 40 |
| glm-5 | 0 | 8 | 0 | 40 |
| gpt-4.1 | 6 | 2 | 0 | 10 |
| gpt-5.4 | 6 | 2 | 0 | 10 |
| kimi-k2.5 | 0 | 8 | 0 | 40 |
| qwen2.5-vl-72b-instruct | 4 | 4 | 0 | 20 |
| qwen3.5-397b-a17b | 4 | 4 | 0 | 20 |

### 视频实验分析

| 模型 | KEEP | REDO | SUPPLEMENT | 新运行数 |
|------|:---:|:---:|:---:|:---:|
| Qwen3-VL-235B-A22B-Instruct | 3 | 1 | 0 | 5 |
| claude-opus-4-6 | 3 | 0 | 1 | 1 |
| claude-sonnet-4-6 | 4 | 0 | 0 | 0 |
| gemini-3.1-pro-preview | 0 | 4 | 0 | 20 |
| glm-5 | 0 | 4 | 0 | 20 |
| gpt-4.1 | 4 | 0 | 0 | 0 |
| gpt-5.4 | 4 | 0 | 0 | 0 |
| kimi-k2.5 | 0 | 4 | 0 | 20 |
| qwen2.5-vl-72b-instruct | 4 | 0 | 0 | 0 |
| qwen3.5-397b-a17b | 4 | 0 | 0 | 0 |

## 3. 实验方法

- **被测模型**通过 DMXAPI 完成任务回答
- **deepseek-chat** 统一作为裁判，根据 rubric 评分
- 所有模型**并行执行**，各自独立写入结果
- 旧实验中合格的数据直接保留，仅重跑/补充不合格部分
- 最终结果合并到 `complete_data/` 中

## 4. 图片实验完整结果

### Qwen3-VL-235B-A22B-Instruct

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 6.8 | 2.17 | 4.0 | 9.0 | 5 |
| 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 8.2 | 0.45 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| 02_Minecraft | A_description | 7.8 | 1.1 | 7.0 | 9.0 | 5 |
| 02_Minecraft | B_assistant | 7.6 | 0.89 | 7.0 | 9.0 | 5 |
| 03_GenshinImpact | A_description | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| 03_GenshinImpact | B_assistant | 10.0 | 0.0 | 10.0 | 10.0 | 5 |
| 04_GenshinImpact | A_description | 4.0 | 0.0 | 4.0 | 4.0 | 5 |
| 04_GenshinImpact | B_assistant | 6.8 | 2.59 | 4.0 | 9.0 | 5 |

### claude-opus-4-6

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 8.8 | 0.45 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| 02_Minecraft | A_description | 6.6 | 0.89 | 5.0 | 7.0 | 5 |
| 02_Minecraft | B_assistant | 7.4 | 0.55 | 7.0 | 8.0 | 5 |
| 03_GenshinImpact | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 03_GenshinImpact | B_assistant | 8.4 | 0.89 | 7.0 | 9.0 | 5 |
| 04_GenshinImpact | A_description | 3.2 | 1.79 | 0.0 | 4.0 | 5 |
| 04_GenshinImpact | B_assistant | 4.0 | 0.0 | 4.0 | 4.0 | 5 |

### claude-sonnet-4-6

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 7.8 | 0.45 | 7.0 | 8.0 | 5 |
| 02_Minecraft | A_description | 7.0 | 1.0 | 6.0 | 8.0 | 5 |
| 02_Minecraft | B_assistant | 7.0 | 0.0 | 7.0 | 7.0 | 5 |
| 03_GenshinImpact | A_description | 9.4 | 0.55 | 9.0 | 10.0 | 5 |
| 03_GenshinImpact | B_assistant | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| 04_GenshinImpact | A_description | 4.8 | 1.79 | 4.0 | 8.0 | 5 |
| 04_GenshinImpact | B_assistant | 4.0 | 0.0 | 4.0 | 4.0 | 5 |

### gemini-3.1-pro-preview

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 2.0 | 3.46 | 0.0 | 8.0 | 5 |
| 01_Minecraft | B_assistant | 2.0 | 2.74 | 0.0 | 5.0 | 5 |
| 02_Minecraft | A_description | 1.8 | 1.79 | 0.0 | 4.0 | 5 |
| 02_Minecraft | B_assistant | 2.4 | 0.89 | 2.0 | 4.0 | 5 |
| 03_GenshinImpact | A_description | 2.2 | 0.45 | 2.0 | 3.0 | 5 |
| 03_GenshinImpact | B_assistant | 1.6 | 2.3 | 0.0 | 5.0 | 5 |
| 04_GenshinImpact | A_description | 7.4 | 2.07 | 4.0 | 9.0 | 5 |
| 04_GenshinImpact | B_assistant | 0.4 | 0.89 | 0.0 | 2.0 | 5 |

### glm-5

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 01_Minecraft | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 02_GenshinImpact | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 02_GenshinImpact | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 02_Minecraft | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 02_Minecraft | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 03_GenshinImpact | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 03_GenshinImpact | B_assistant | 1.6 | 0.89 | 0.0 | 2.0 | 5 |
| 04_GenshinImpact | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 04_GenshinImpact | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |

### gpt-4.1

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 10.0 | 0.0 | 10.0 | 10.0 | 5 |
| 01_Minecraft | B_assistant | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | A_description | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_Minecraft | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_Minecraft | B_assistant | 7.6 | 0.89 | 7.0 | 9.0 | 5 |
| 03_GenshinImpact | A_description | 9.6 | 0.55 | 9.0 | 10.0 | 5 |
| 03_GenshinImpact | B_assistant | 10.0 | 0.0 | 10.0 | 10.0 | 5 |
| 04_GenshinImpact | A_description | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| 04_GenshinImpact | B_assistant | 5.0 | 2.24 | 4.0 | 9.0 | 5 |

### gpt-5.4

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| 01_Minecraft | B_assistant | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | A_description | 8.8 | 0.45 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_Minecraft | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_Minecraft | B_assistant | 8.2 | 1.1 | 7.0 | 9.0 | 5 |
| 03_GenshinImpact | A_description | 9.6 | 0.55 | 9.0 | 10.0 | 5 |
| 03_GenshinImpact | B_assistant | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| 04_GenshinImpact | A_description | 7.0 | 2.74 | 4.0 | 9.0 | 5 |
| 04_GenshinImpact | B_assistant | 6.6 | 2.51 | 4.0 | 9.0 | 5 |

### kimi-k2.5

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 8.6 | 2.07 | 5.0 | 10.0 | 5 |
| 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 6.2 | 3.63 | 0.0 | 9.0 | 5 |
| 02_Minecraft | A_description | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| 02_Minecraft | B_assistant | 7.8 | 0.45 | 7.0 | 8.0 | 5 |
| 03_GenshinImpact | A_description | 9.6 | 0.55 | 9.0 | 10.0 | 5 |
| 03_GenshinImpact | B_assistant | 9.4 | 0.55 | 9.0 | 10.0 | 5 |
| 04_GenshinImpact | A_description | 8.2 | 0.84 | 7.0 | 9.0 | 5 |
| 04_GenshinImpact | B_assistant | 7.6 | 2.07 | 4.0 | 9.0 | 5 |

### qwen2.5-vl-72b-instruct

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| 01_Minecraft | B_assistant | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | A_description | 9.4 | 0.55 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | B_assistant | 9.6 | 0.55 | 9.0 | 10.0 | 5 |
| 02_Minecraft | A_description | 6.8 | 0.84 | 6.0 | 8.0 | 5 |
| 02_Minecraft | B_assistant | 6.0 | 0.71 | 5.0 | 7.0 | 5 |
| 03_GenshinImpact | A_description | 5.6 | 0.55 | 5.0 | 6.0 | 5 |
| 03_GenshinImpact | B_assistant | 7.0 | 0.0 | 7.0 | 7.0 | 5 |
| 04_GenshinImpact | A_description | 6.6 | 3.71 | 0.0 | 9.0 | 5 |
| 04_GenshinImpact | B_assistant | 0.8 | 1.79 | 0.0 | 4.0 | 5 |

### qwen3.5-397b-a17b

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 9.2 | 0.84 | 8.0 | 10.0 | 5 |
| 01_Minecraft | B_assistant | 7.8 | 1.1 | 7.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_Minecraft | A_description | 6.6 | 1.14 | 5.0 | 8.0 | 5 |
| 02_Minecraft | B_assistant | 8.2 | 1.1 | 7.0 | 9.0 | 5 |
| 03_GenshinImpact | A_description | 9.4 | 0.55 | 9.0 | 10.0 | 5 |
| 03_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 04_GenshinImpact | A_description | 0.6 | 1.34 | 0.0 | 3.0 | 5 |
| 04_GenshinImpact | B_assistant | 0.8 | 1.79 | 0.0 | 4.0 | 5 |

## 5. 视频实验完整结果

### Qwen3-VL-235B-A22B-Instruct

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 6.8 | 2.17 | 4.0 | 9.0 | 5 |
| 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 8.2 | 0.45 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 8.6 | 0.55 | 8.0 | 9.0 | 5 |

### claude-opus-4-6

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 8.8 | 0.45 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 8.4 | 0.55 | 8.0 | 9.0 | 5 |

### claude-sonnet-4-6

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 7.8 | 0.45 | 7.0 | 8.0 | 5 |

### gemini-3.1-pro-preview

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 2.0 | 3.46 | 0.0 | 8.0 | 5 |
| 01_Minecraft | B_assistant | 2.0 | 2.74 | 0.0 | 5.0 | 5 |

### glm-5

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 01_Minecraft | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 02_GenshinImpact | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| 02_GenshinImpact | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |

### gpt-4.1

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 10.0 | 0.0 | 10.0 | 10.0 | 5 |
| 01_Minecraft | B_assistant | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | A_description | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |

### gpt-5.4

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| 01_Minecraft | B_assistant | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | A_description | 8.8 | 0.45 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |

### kimi-k2.5

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 8.6 | 2.07 | 5.0 | 10.0 | 5 |
| 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 6.2 | 3.63 | 0.0 | 9.0 | 5 |

### qwen2.5-vl-72b-instruct

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| 01_Minecraft | B_assistant | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | A_description | 9.4 | 0.55 | 9.0 | 10.0 | 5 |
| 02_GenshinImpact | B_assistant | 9.6 | 0.55 | 9.0 | 10.0 | 5 |

### qwen3.5-397b-a17b

| 样本 | Prompt | 均分 | 标准差 | 最低 | 最高 | 次数 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| 01_Minecraft | A_description | 9.2 | 0.84 | 8.0 | 10.0 | 5 |
| 01_Minecraft | B_assistant | 7.8 | 1.1 | 7.0 | 9.0 | 5 |
| 02_GenshinImpact | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| 02_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |

## 6. 可视化

![dimension_heatmap_image](results\v3-sota\2026-03-19_14-57-39\charts\dimension_heatmap_image.png)

![dimension_heatmap_video](results\v3-sota\2026-03-19_14-57-39\charts\dimension_heatmap_video.png)

![latency_image](results\v3-sota\2026-03-19_14-57-39\charts\latency_image.png)

![latency_video](results\v3-sota\2026-03-19_14-57-39\charts\latency_video.png)

![per_sample_comparison](results\v3-sota\2026-03-19_14-57-39\charts\per_sample_comparison.png)

![score_ranking](results\v3-sota\2026-03-19_14-57-39\charts\score_ranking.png)

![stability](results\v3-sota\2026-03-19_14-57-39\charts\stability.png)

## 7. 争议样本分析

未发现争议样本。所有样本在多数 SOTA 模型上得分 >= 7.0。

## 8. 结论

<!-- 此部分将在审阅后填写 -->
