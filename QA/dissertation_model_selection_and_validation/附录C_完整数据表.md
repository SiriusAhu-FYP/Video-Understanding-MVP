# 附录 C：完整数据表

本附录包含所有实验的汇总数据。原始逐次运行数据（含 VLM 完整回复、裁判评分 JSON 等）以 CSV 格式存储在 `data/` 目录中。

---

## C.1 SOTA 模型图片质量 — 聚合结果

**数据来源**: `data/sota_image_aggregated.csv`

| 模型 | 素材 | Prompt | 均分 | σ | 最低 | 最高 | 次数 |
|------|------|--------|:---:|:---:|:---:|:---:|:---:|
| claude-opus-4-6 | 01_Minecraft | A_description | 8.0 | 0.71 | 7.0 | 9.0 | 5 |
| claude-opus-4-6 | 01_Minecraft | B_assistant | 6.6 | 0.55 | 6.0 | 7.0 | 5 |
| claude-opus-4-6 | 02_Minecraft | A_description | 6.6 | 0.89 | 5.0 | 7.0 | 5 |
| claude-opus-4-6 | 02_Minecraft | B_assistant | 7.4 | 0.55 | 7.0 | 8.0 | 5 |
| claude-opus-4-6 | 03_GenshinImpact | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| claude-opus-4-6 | 03_GenshinImpact | B_assistant | 8.4 | 0.89 | 7.0 | 9.0 | 5 |
| claude-opus-4-6 | 04_GenshinImpact | A_description | 3.2 | 1.79 | 0.0 | 4.0 | 5 |
| claude-opus-4-6 | 04_GenshinImpact | B_assistant | 4.0 | 0.0 | 4.0 | 4.0 | 5 |
| claude-sonnet-4-6 | 01_Minecraft | A_description | 8.0 | 0.0 | 8.0 | 8.0 | 5 |
| claude-sonnet-4-6 | 01_Minecraft | B_assistant | 7.0 | 0.0 | 7.0 | 7.0 | 5 |
| claude-sonnet-4-6 | 02_Minecraft | A_description | 7.0 | 1.0 | 6.0 | 8.0 | 5 |
| claude-sonnet-4-6 | 02_Minecraft | B_assistant | 7.0 | 0.0 | 7.0 | 7.0 | 5 |
| claude-sonnet-4-6 | 03_GenshinImpact | A_description | 9.4 | 0.55 | 9.0 | 10.0 | 5 |
| claude-sonnet-4-6 | 03_GenshinImpact | B_assistant | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| claude-sonnet-4-6 | 04_GenshinImpact | A_description | 4.8 | 1.79 | 4.0 | 8.0 | 5 |
| claude-sonnet-4-6 | 04_GenshinImpact | B_assistant | 4.0 | 0.0 | 4.0 | 4.0 | 5 |
| gemini-3.1-pro-preview | 01_Minecraft | A_description | 0.2 | 0.45 | 0.0 | 1.0 | 5 |
| gemini-3.1-pro-preview | 01_Minecraft | B_assistant | 0.2 | 0.45 | 0.0 | 1.0 | 5 |
| gemini-3.1-pro-preview | 02_Minecraft | A_description | 1.8 | 1.79 | 0.0 | 4.0 | 5 |
| gemini-3.1-pro-preview | 02_Minecraft | B_assistant | 2.4 | 0.89 | 2.0 | 4.0 | 5 |
| gemini-3.1-pro-preview | 03_GenshinImpact | A_description | 2.2 | 0.45 | 2.0 | 3.0 | 5 |
| gemini-3.1-pro-preview | 03_GenshinImpact | B_assistant | 1.6 | 2.3 | 0.0 | 5.0 | 5 |
| gemini-3.1-pro-preview | 04_GenshinImpact | A_description | 7.4 | 2.07 | 4.0 | 9.0 | 5 |
| gemini-3.1-pro-preview | 04_GenshinImpact | B_assistant | 0.4 | 0.89 | 0.0 | 2.0 | 5 |
| glm-5 | 01_Minecraft | A_description | 0.8 | 0.45 | 0.0 | 1.0 | 5 |
| glm-5 | 01_Minecraft | B_assistant | 0.6 | 0.55 | 0.0 | 1.0 | 5 |
| glm-5 | 02_Minecraft | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| glm-5 | 02_Minecraft | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| glm-5 | 03_GenshinImpact | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| glm-5 | 03_GenshinImpact | B_assistant | 1.6 | 0.89 | 0.0 | 2.0 | 5 |
| glm-5 | 04_GenshinImpact | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| glm-5 | 04_GenshinImpact | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| gpt-4.1 | 01_Minecraft | A_description | 8.0 | 0.71 | 7.0 | 9.0 | 5 |
| gpt-4.1 | 01_Minecraft | B_assistant | 10.0 | 0.0 | 10.0 | 10.0 | 5 |
| gpt-4.1 | 02_Minecraft | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| gpt-4.1 | 02_Minecraft | B_assistant | 7.6 | 0.89 | 7.0 | 9.0 | 5 |
| gpt-4.1 | 03_GenshinImpact | A_description | 9.6 | 0.55 | 9.0 | 10.0 | 5 |
| gpt-4.1 | 03_GenshinImpact | B_assistant | 10.0 | 0.0 | 10.0 | 10.0 | 5 |
| gpt-4.1 | 04_GenshinImpact | A_description | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| gpt-4.1 | 04_GenshinImpact | B_assistant | 5.0 | 2.24 | 4.0 | 9.0 | 5 |
| gpt-5.4 | 01_Minecraft | A_description | 8.8 | 0.84 | 8.0 | 10.0 | 5 |
| gpt-5.4 | 01_Minecraft | B_assistant | 9.0 | 1.0 | 8.0 | 10.0 | 5 |
| gpt-5.4 | 02_Minecraft | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| gpt-5.4 | 02_Minecraft | B_assistant | 8.2 | 1.1 | 7.0 | 9.0 | 5 |
| gpt-5.4 | 03_GenshinImpact | A_description | 9.6 | 0.55 | 9.0 | 10.0 | 5 |
| gpt-5.4 | 03_GenshinImpact | B_assistant | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| gpt-5.4 | 04_GenshinImpact | A_description | 7.0 | 2.74 | 4.0 | 9.0 | 5 |
| gpt-5.4 | 04_GenshinImpact | B_assistant | 6.6 | 2.51 | 4.0 | 9.0 | 5 |
| kimi-k2.5 | 01_Minecraft | A_description | 8.4 | 1.34 | 7.0 | 10.0 | 5 |
| kimi-k2.5 | 01_Minecraft | B_assistant | 7.8 | 1.64 | 6.0 | 9.0 | 5 |
| kimi-k2.5 | 02_Minecraft | A_description | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| kimi-k2.5 | 02_Minecraft | B_assistant | 7.8 | 0.45 | 7.0 | 8.0 | 5 |
| kimi-k2.5 | 03_GenshinImpact | A_description | 9.6 | 0.55 | 9.0 | 10.0 | 5 |
| kimi-k2.5 | 03_GenshinImpact | B_assistant | 9.4 | 0.55 | 9.0 | 10.0 | 5 |
| kimi-k2.5 | 04_GenshinImpact | A_description | 8.2 | 0.84 | 7.0 | 9.0 | 5 |
| kimi-k2.5 | 04_GenshinImpact | B_assistant | 7.6 | 2.07 | 4.0 | 9.0 | 5 |
| qwen2.5-vl-72b-instruct | 01_Minecraft | A_description | 7.0 | 0.0 | 7.0 | 7.0 | 5 |
| qwen2.5-vl-72b-instruct | 01_Minecraft | B_assistant | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| qwen2.5-vl-72b-instruct | 02_Minecraft | A_description | 6.8 | 0.84 | 6.0 | 8.0 | 5 |
| qwen2.5-vl-72b-instruct | 02_Minecraft | B_assistant | 6.0 | 0.71 | 5.0 | 7.0 | 5 |
| qwen2.5-vl-72b-instruct | 03_GenshinImpact | A_description | 5.6 | 0.55 | 5.0 | 6.0 | 5 |
| qwen2.5-vl-72b-instruct | 03_GenshinImpact | B_assistant | 7.0 | 0.0 | 7.0 | 7.0 | 5 |
| qwen2.5-vl-72b-instruct | 04_GenshinImpact | A_description | 6.6 | 3.71 | 0.0 | 9.0 | 5 |
| qwen2.5-vl-72b-instruct | 04_GenshinImpact | B_assistant | 0.8 | 1.79 | 0.0 | 4.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 01_Minecraft | A_description | 5.6 | 1.52 | 4.0 | 8.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 01_Minecraft | B_assistant | 10.0 | 0.0 | 10.0 | 10.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 02_Minecraft | A_description | 7.8 | 1.1 | 7.0 | 9.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 02_Minecraft | B_assistant | 7.6 | 0.89 | 7.0 | 9.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 03_GenshinImpact | A_description | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 03_GenshinImpact | B_assistant | 10.0 | 0.0 | 10.0 | 10.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 04_GenshinImpact | A_description | 4.0 | 0.0 | 4.0 | 4.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 04_GenshinImpact | B_assistant | 6.8 | 2.59 | 4.0 | 9.0 | 5 |
| qwen3.5-397b-a17b | 01_Minecraft | A_description | 7.0 | 0.0 | 7.0 | 7.0 | 5 |
| qwen3.5-397b-a17b | 01_Minecraft | B_assistant | 6.8 | 0.45 | 6.0 | 7.0 | 5 |
| qwen3.5-397b-a17b | 02_Minecraft | A_description | 6.6 | 1.14 | 5.0 | 8.0 | 5 |
| qwen3.5-397b-a17b | 02_Minecraft | B_assistant | 8.2 | 1.1 | 7.0 | 9.0 | 5 |
| qwen3.5-397b-a17b | 03_GenshinImpact | A_description | 9.4 | 0.55 | 9.0 | 10.0 | 5 |
| qwen3.5-397b-a17b | 03_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| qwen3.5-397b-a17b | 04_GenshinImpact | A_description | 0.6 | 1.34 | 0.0 | 3.0 | 5 |
| qwen3.5-397b-a17b | 04_GenshinImpact | B_assistant | 0.8 | 1.79 | 0.0 | 4.0 | 5 |

---

## C.2 SOTA 模型视频质量 — 聚合结果

**数据来源**: `data/sota_video_aggregated.csv`

| 模型 | 素材 | Prompt | 均分 | σ | 最低 | 最高 | 次数 |
|------|------|--------|:---:|:---:|:---:|:---:|:---:|
| claude-opus-4-6 | 01_Minecraft | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| claude-opus-4-6 | 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| claude-opus-4-6 | 02_GenshinImpact | A_description | 8.8 | 0.45 | 8.0 | 9.0 | 5 |
| claude-opus-4-6 | 02_GenshinImpact | B_assistant | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| claude-sonnet-4-6 | 01_Minecraft | A_description | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| claude-sonnet-4-6 | 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| claude-sonnet-4-6 | 02_GenshinImpact | A_description | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| claude-sonnet-4-6 | 02_GenshinImpact | B_assistant | 7.8 | 0.45 | 7.0 | 8.0 | 5 |
| gemini-3.1-pro-preview | 01_Minecraft | A_description | 2.0 | 3.46 | 0.0 | 8.0 | 5 |
| gemini-3.1-pro-preview | 01_Minecraft | B_assistant | 2.0 | 2.74 | 0.0 | 5.0 | 5 |
| glm-5 | 01_Minecraft | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| glm-5 | 01_Minecraft | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| glm-5 | 02_GenshinImpact | A_description | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| glm-5 | 02_GenshinImpact | B_assistant | 2.0 | 0.0 | 2.0 | 2.0 | 5 |
| gpt-4.1 | 01_Minecraft | A_description | 10.0 | 0.0 | 10.0 | 10.0 | 5 |
| gpt-4.1 | 01_Minecraft | B_assistant | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| gpt-4.1 | 02_GenshinImpact | A_description | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| gpt-4.1 | 02_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| gpt-5.4 | 01_Minecraft | A_description | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| gpt-5.4 | 01_Minecraft | B_assistant | 9.8 | 0.45 | 9.0 | 10.0 | 5 |
| gpt-5.4 | 02_GenshinImpact | A_description | 8.8 | 0.45 | 8.0 | 9.0 | 5 |
| gpt-5.4 | 02_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| kimi-k2.5 | 01_Minecraft | A_description | 8.6 | 2.07 | 5.0 | 10.0 | 5 |
| kimi-k2.5 | 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| kimi-k2.5 | 02_GenshinImpact | A_description | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| kimi-k2.5 | 02_GenshinImpact | B_assistant | 6.2 | 3.63 | 0.0 | 9.0 | 5 |
| qwen2.5-vl-72b-instruct | 01_Minecraft | A_description | 8.4 | 0.55 | 8.0 | 9.0 | 5 |
| qwen2.5-vl-72b-instruct | 01_Minecraft | B_assistant | 9.2 | 0.45 | 9.0 | 10.0 | 5 |
| qwen2.5-vl-72b-instruct | 02_GenshinImpact | A_description | 9.4 | 0.55 | 9.0 | 10.0 | 5 |
| qwen2.5-vl-72b-instruct | 02_GenshinImpact | B_assistant | 9.6 | 0.55 | 9.0 | 10.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 01_Minecraft | A_description | 6.8 | 2.17 | 4.0 | 9.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 01_Minecraft | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 02_GenshinImpact | A_description | 8.2 | 0.45 | 8.0 | 9.0 | 5 |
| Qwen3-VL-235B-A22B-Instruct | 02_GenshinImpact | B_assistant | 8.6 | 0.55 | 8.0 | 9.0 | 5 |
| qwen3.5-397b-a17b | 01_Minecraft | A_description | 9.2 | 0.84 | 8.0 | 10.0 | 5 |
| qwen3.5-397b-a17b | 01_Minecraft | B_assistant | 7.8 | 1.1 | 7.0 | 9.0 | 5 |
| qwen3.5-397b-a17b | 02_GenshinImpact | A_description | 9.0 | 0.0 | 9.0 | 9.0 | 5 |
| qwen3.5-397b-a17b | 02_GenshinImpact | B_assistant | 9.0 | 0.0 | 9.0 | 9.0 | 5 |

---

## C.3 本地模型图片质量 — 逐次评分

**数据来源**: `data/local_Qwen3-VL-2B_scores.csv` 等

### Qwen3-VL-2B-Instruct

| 素材 | Prompt | 逐次得分 | 均分 |
|------|--------|---------|:---:|
| 01_Minecraft | A_description | 4, 2, 2, 4, 4 | 3.2 |
| 01_Minecraft | B_assistant | 7, 5, 8, 5, 8 | 6.6 |
| 02_Minecraft | A_description | 2, 2, 3, 0, 2 | 1.8 |
| 02_Minecraft | B_assistant | 4, 4, 2, 5, 2 | 3.4 |
| 03_GenshinImpact | A_description | 9, 8, 9, 9, 10 | **9.0** |
| 03_GenshinImpact | B_assistant | 10, 9, 9, 10, 9 | **9.4** |
| 04_GenshinImpact | A_description | 0, 4, 0, 0, 0 | 0.8 |
| 04_GenshinImpact | B_assistant | 0, 0, 0, 0, 0 | 0.0 |

### InternVL2.5-2B

| 素材 | Prompt | 逐次得分 | 均分 |
|------|--------|---------|:---:|
| 01_Minecraft | A_description | 0, 2, 0, 0, 2 | 0.8 |
| 01_Minecraft | B_assistant | 3, 0, 7, 4, 3 | 3.4 |
| 02_Minecraft | A_description | 4, 4, 5, 5, 4 | 4.4 |
| 02_Minecraft | B_assistant | 5, 1, 3, 1, 3 | 2.6 |
| 03_GenshinImpact | A_description | 3, 5, 5, 5, 5 | 4.6 |
| 03_GenshinImpact | B_assistant | 7, 7, 7, 6, 7 | 6.8 |
| 04_GenshinImpact | A_description | 0, 0, 0, 0, 0 | 0.0 |
| 04_GenshinImpact | B_assistant | 0, 0, 0, 0, 0 | 0.0 |

### DeepSeek-VL2-tiny

| 素材 | Prompt | 逐次得分 | 均分 |
|------|--------|---------|:---:|
| 01_Minecraft | A_description | 5, 3, 3, 3, 2 | 3.2 |
| 01_Minecraft | B_assistant | 0, 0, 0, 0, 2 | 0.4 |
| 02_Minecraft | A_description | 6, 5, 5, 5, 6 | 5.4 |
| 02_Minecraft | B_assistant | 5, 4, 2, 5, 4 | 4.0 |
| 03_GenshinImpact | A_description | 6, 8, 5, 1, 3 | 4.6 |
| 03_GenshinImpact | B_assistant | 8, 9, 9, 2, 4 | 6.4 |
| 04_GenshinImpact | A_description | 0, 0, 3, 2, 0 | 1.0 |
| 04_GenshinImpact | B_assistant | 0, 0, 0, 0, 4 | 0.8 |

### SmolVLM2-2.2B-Instruct

| 素材 | Prompt | 逐次得分 | 均分 |
|------|--------|---------|:---:|
| 01_Minecraft | A_description | 4, 1, 2, 5, 4 | 3.2 |
| 01_Minecraft | B_assistant | 0, 1, 1, 0, 0 | 0.4 |
| 02_Minecraft | A_description | 1, 5, 1, 5, 5 | 3.4 |
| 02_Minecraft | B_assistant | 4, 4, 1, 4, 1 | 2.8 |
| 03_GenshinImpact | A_description | 0, 2, 0, 0, 0 | 0.4 |
| 03_GenshinImpact | B_assistant | 3, 3, 3, 3, 3 | 3.0 |
| 04_GenshinImpact | A_description | 3, 3, 3, 0, 0 | 1.8 |
| 04_GenshinImpact | B_assistant | 0, 0, 0, 0, 0 | 0.0 |

---

## C.4 本地模型推理速度汇总

**数据来源**: `data/local_speed_*.csv`（每模型 120 条原始测量数据）

| 模型 | 平均 TTFT (ms) | TTFT σ (ms) | 平均吞吐量 (tok/s) | 吞吐量 σ | VRAM (MB) |
|------|:-------------:|:-----------:|:------------------:|:--------:|:---------:|
| Qwen3-VL-2B | 125.3 | 24.1 | 67.4 | 12.8 | 7,637 |
| InternVL2.5-2B | 107.6 | 20.3 | 77.2 | 14.5 | 8,801 |
| DeepSeek-VL2-tiny | 99.8 | 11.2 | 59.3 | 9.7 | 8,809 |
| SmolVLM2-2.2B | 131.4 | 22.8 | 71.1 | 13.2 | 8,879 |

---

## C.5 原始 CSV 文件索引

`data/` 目录下的所有 CSV 文件列表：

### SOTA 模型数据
| 文件名 | 内容 | 记录数 |
|--------|------|:------:|
| sota_image_aggregated.csv | 图片质量聚合统计 | 80 |
| sota_video_aggregated.csv | 视频质量聚合统计 | 38 |
| sota_image_{model}_results.csv | 各模型图片质量逐次结果（含完整 VLM 回复） | 各 50 条 |
| sota_video_{model}_results.csv | 各模型视频质量逐次结果（含完整 VLM 回复） | 各 20 条 |

### 本地模型数据
| 文件名 | 内容 | 记录数 |
|--------|------|:------:|
| local_Qwen3-VL-2B_scores.csv | Qwen3-VL-2B 图片质量维度评分 | 40 |
| local_InternVL2_5-2B_scores.csv | InternVL2.5-2B 图片质量维度评分 | 40 |
| local_DeepSeek-VL2-tiny_scores.csv | DeepSeek-VL2-tiny 图片质量维度评分 | 40 |
| local_SmolVLM2-2.2B_scores.csv | SmolVLM2-2.2B 图片质量维度评分 | 40 |
| local_speed_{model}.csv | 各模型推理速度原始数据 | 各 120 条 |

*所有 CSV 文件均使用 UTF-8 编码，可直接用 Python、R 或 Excel 打开。SOTA 模型的 `results.csv` 文件包含完整的 VLM 原始回复文本（`vlm_response` 列），适合做详细的定性分析。*
