# 实验事故报告 (Incident Report)

**实验日期**: 2026-03-16
**生成时间**: 2026-03-16 20:42:07

---

## 事故列表

### 1. SmolVLM2-2.2B-Instruct 首次启动失败

**现象**: vLLM 服务在 300s 超时内未能启动，导致该模型在主批次实验中被标记为失败。

**根本原因**: SmolVLM2 的 `transformers` 处理器 (`SmolVLMProcessor`) 依赖 `num2words` Python 包，但 WSL vLLM 虚拟环境中未安装该包。vLLM 启动时加载处理器即报错 `ImportError: Package 'num2words' is required`。

**解决方案**: 在 WSL 虚拟环境中执行 `pip install num2words`，随后重新启动 vLLM 即可正常运行。

**后续操作**: 手动补跑 SmolVLM2 的 benchmark_speed（90/90 成功）和 video_understanding（6/6 成功），结果已整合至同一实验会话中。

### 2. gpu-memory-utilization=0.5 导致 Mamba 模型启动失败（前置实验）

**现象**: 在更早的实验尝试中，Qwen3.5-2B 和 Qwen3.5-0.8B 因 `AssertionError: num_cache_lines >= batch` （Mamba causal_conv1d 层）而启动失败。

**根本原因**: `gpu-memory-utilization=0.5` 分配给 KV Cache 的显存不足以支撑 Mamba 混合架构的额外状态缓存需求。

**解决方案**: 将 `gpu-memory-utilization` 从 0.5 提升至 0.7。同时发现 `run_benchmark.py` 中存在硬编码 fallback `gpu_util_map.get(model_id, 0.5)`，在 `--skip-preflight` 模式下忽略配置文件中的 0.7。已修复为从 `config_benchmark.toml` 读取默认值。

### 3. PotPlayer 播放器清理不彻底

**现象**: video_understanding 实验中，MC 视频播放完毕后 PotPlayer 不自动关闭，导致后续运行可能同时存在多个 PotPlayer 实例。

**根本原因**: `close_window()` 函数使用 `WM_CLOSE` 消息关闭窗口，但 PotPlayer 在某些情况下不响应 （如播放结束停留在最后一帧）。

**解决方案**: 
1. 重构 `launch_player()` 为返回 `subprocess.Popen` 句柄；
2. 新增 `_kill_all_players()` 使用 `psutil` 强制终止所有 PotPlayer 进程；
3. 新增 `close_player()` 实现「先礼后兵」策略：WM_CLOSE → wait(3s) → kill → kill_all；
4. 用 `try/finally` 包裹整个流水线，确保异常时也能清理播放器。

### 4. 帧差计算因分辨率不一致报错（前置实验）

**现象**: `ValueError: operands could not be broadcast together with shapes (309,416,3) (382,512,3)`

**根本原因**: PotPlayer 窗口大小在录制过程中可能变化（如自动调整到视频原始分辨率），导致前后两帧截图的分辨率不同，MSE 计算时 numpy 广播失败。

**解决方案**: 在 `compute_diff()` 中添加形状检查，若不一致则用 `cv2.resize()` 将 frame_b 缩放至 frame_a 尺寸。

### 5. Phi-3.5-vision-instruct 前批次启动失败

**现象**: 在 gpu-memory-utilization=0.5 的前置实验中，Phi-3.5 无法启动（300s 超时，持续 Connection error/502）。

**根本原因**: 与事故 #2 相同，0.5 的显存利用率不足。

**解决方案**: 提升至 0.7 后在正式实验中成功启动并完成全部测试。

---

## 总结

| 事故 | 影响范围 | 是否解决 |
|------|------|:---:|
| SmolVLM2 缺少 num2words | 单模型 | ✅ 已解决 + 补跑 |
| GPU 显存不足 | 多模型 | ✅ 已修复 (0.7) |
| PotPlayer 未清理 | video_understanding | ✅ 已重构 |
| 帧差分辨率不一致 | video_understanding | ✅ 已修复 |
| Phi-3.5 启动失败 | 单模型 | ✅ 随 GPU 修复一并解决 |