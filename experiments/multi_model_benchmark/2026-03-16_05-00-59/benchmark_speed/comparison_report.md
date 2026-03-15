# Benchmark Speed - Multi-Model Comparison

**Generated**: 2026-03-16 06:08:54

## Performance Overview

| Model | Avg TTFT (s) | Avg Throughput (tok/s) | Avg Total (s) | Avg Tokens | Runs |
|-------|:---:|:---:|:---:|:---:|:---:|
| Qwen/Qwen3.5-2B | 0.118 | 148.2 | 2.163 | 302 | 90 |
| Qwen/Qwen3.5-0.8B | 0.122 | 305.3 | 1.134 | 305 | 90 |
| Qwen/Qwen3-VL-2B-Instruct | 0.109 | 150.1 | 2.175 | 307 | 90 |
| OpenGVLab/InternVL2_5-2B | 0.112 | 163.4 | 1.598 | 240 | 64 |
| mistralai/Ministral-3-3B-Instruct-2512 | 0.145 | 133.5 | 2.495 | 312 | 90 |
| deepseek-ai/deepseek-vl2-tiny | 0.100 | 347.3 | 0.690 | 202 | 90 |

## Frame Rate Analysis

| Model | Avg TTFT (s) | Max FPS | Safe FPS (70%) |
|-------|:---:|:---:|:---:|
| Qwen/Qwen3.5-2B | 0.118 | 8.5 | 5.9 |
| Qwen/Qwen3.5-0.8B | 0.122 | 8.2 | 5.7 |
| Qwen/Qwen3-VL-2B-Instruct | 0.109 | 9.2 | 6.4 |
| OpenGVLab/InternVL2_5-2B | 0.112 | 8.9 | 6.2 |
| mistralai/Ministral-3-3B-Instruct-2512 | 0.145 | 6.9 | 4.8 |
| deepseek-ai/deepseek-vl2-tiny | 0.100 | 10.0 | 7.0 |

> Max FPS = 1 / TTFT. Safe range = 70% of max to allow for overhead.