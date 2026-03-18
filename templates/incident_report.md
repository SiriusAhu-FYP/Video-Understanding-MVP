# 事件报告

**生成时间**: {{generated_at}}

{{incident_summary}}

{{incident_details}}

## 根因分析与预防

<!-- ANALYSIS -->
根据上述事件记录，进行根因分析：
1. 将事件按阶段分类（gpu_probe / startup / health_check / experiment），哪个阶段失败最多？
2. 常见失败模式及根因推测（显存不足、模型架构不兼容 vLLM、网络超时等）。
3. 已实施的修复措施（调高 gpu_memory_utilization、增加超时时间、排除不兼容模型）。
4. 对后续实验运维的预防建议（如何在预飞行阶段更早发现问题）。
