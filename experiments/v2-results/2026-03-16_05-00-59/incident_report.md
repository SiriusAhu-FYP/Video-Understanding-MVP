# Incident Report

**Generated**: 2026-03-16
**Session**: 2026-03-16_05-00-59

## Summary

| Model | Phase | Result | Root Cause |
|-------|-------|--------|------------|
| Zero-Point-AI/MARTHA-2B | pre-flight | Excluded | GGUF-only, PixtralForConditionalGeneration not inspectable |
| vikhyatk/moondream2 | pre-flight | Excluded | HfMoondream architecture not supported by vLLM 0.17.0 |
| OpenGVLab/InternVL2_5-2B | benchmark_speed | Partial (64/90) | vLLM process crashed mid-run (suspected GPU memory pressure from CUDA graph compilation) |
| All 6 models | video_understanding | 0 descriptions | VLMClient used localhost instead of WSL2 IP (fix committed but not active during this run) |

Total pre-flight pass: **6/8** models

---

## Incident 1: Zero-Point-AI/MARTHA-2B — Pre-flight Failure

- **Phase**: pre-flight startup
- **Error**: `Model architectures ['PixtralForConditionalGeneration'] failed to be inspected`
- **Root Cause**: MARTHA-2B is distributed only in GGUF format (Q4_K_M, Q5_K_M, Q6_K, Q8_0). No `config.json` or SafeTensors weights exist in the HuggingFace repository. vLLM cannot load GGUF-only models that use `PixtralForConditionalGeneration` architecture. Additionally, the xFormers version incompatibility (`xFormers built for PyTorch 2.5.1+cu124, have 2.10.0+cu128`) caused the architecture inspection subprocess to fail.
- **Resolution**: Excluded from experiment. The model would need to be served via llama.cpp or Ollama instead of vLLM. xFormers was upgraded from 0.0.28.post3 to 0.0.35 (fixed for other Pixtral models).

## Incident 2: vikhyatk/moondream2 — Pre-flight Failure

- **Phase**: pre-flight startup
- **Error**: vLLM timed out after 300s (Connection error)
- **Root Cause**: moondream2 uses the `HfMoondream` architecture (registered as `model_type: moondream1` in config.json). While vLLM added moondream support via PR #4228 for the older LLaVA-like architecture, the current `HfMoondream` architecture is not supported by vLLM 0.17.0. The vLLM process likely crashed immediately during model loading.
- **Resolution**: Excluded from experiment. Would need a vLLM version with HfMoondream support or a separate serving framework.

## Incident 3: OpenGVLab/InternVL2_5-2B — Partial Results

- **Phase**: benchmark_speed (experiment)
- **Error**: 26 out of 90 inference runs failed with `APIConnectionError: Connection error`
- **Root Cause**: The vLLM process crashed during the benchmark run, likely due to GPU memory pressure. With `gpu-memory-utilization: 0.5` and `max-model-len: 4096`, InternVL2.5-2B uses dynamic image patching which can cause variable memory consumption. CUDA graph compilation may have exceeded available VRAM on specific prompt/image combinations.
- **Resolution**: 64/90 results were successfully collected (71% success rate). Data is usable for comparison. Adding `--enforce-eager` or reducing `max-model-len` could improve stability at the cost of performance.

## Incident 4: Video Understanding — Zero Descriptions (All Models)

- **Phase**: video_understanding experiment
- **Error**: All frame descriptions returned 0 (e.g., `采样=53 关键帧=45 描述=0`)
- **Root Cause**: The `VLMClient` in `pipeline/vlm.py` was initialized with `cfg.llm.vllm_base_url` which defaults to `http://localhost:8000/v1`. In WSL2 environments, `localhost` port forwarding is broken, requiring the WSL2 VM's IP address (`172.17.19.50`) instead. While the orchestrator correctly detected and used the WSL2 IP for `benchmark_speed`, the `video_understanding` module's `VLMClient` was not receiving the corrected URL.
- **Resolution**: Fix committed (commit `34c816c`): `run_experiment()` now accepts a `base_url` parameter, and `VLMClient` is initialized with `base_url=cfg.llm.vllm_base_url` (which now receives the WSL2 IP via the modified config dict). The video_understanding experiment would need to be re-run with the fix active.

## Environment Issues Resolved Before Experiment

1. **xFormers version mismatch**: Upgraded from 0.0.28.post3 to 0.0.35 to support PyTorch 2.10 + CUDA 12.8. This fixed PixtralForConditionalGeneration inspection failures for Ministral-3-3B.
2. **Missing `timm` module**: Installed `timm` 1.0.25 in the WSL vLLM environment. Required by deepseek-vl2-tiny's vision module (`DeepseekVLV2ForCausalLM`).
3. **WSL2 localhost forwarding broken**: Detected that Windows cannot reach WSL2's `localhost:8000`. Implemented dynamic WSL2 IP detection via `ip -4 -o addr show eth0` to construct the correct vLLM base URL.
4. **Zombie vLLM processes**: Added `_kill_all_vllm()` cleanup function that runs before each model start to prevent stale processes from occupying port 8000 and causing false-positive health checks.
