# 事件报告

**生成时间**: 2026-03-16 13:17:14

共记录 8 个事件。

## 事件 1: Qwen/Qwen3.5-2B

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 2: Qwen/Qwen3.5-0.8B

- **阶段**: startup
- **错误**: vLLM 在 300s 内未就绪: Connection error.
- **处置**: 从实验中排除

**vLLM 输出（末尾片段）**:
```
(APIServer pid=19725)                ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=19725)     return await anext(self.gen)
(APIServer pid=19725)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 96, in build_async_engine_client
(APIServer pid=19725)     async with build_async_engine_client_from_engine_args(
(APIServer pid=19725)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=19725)     return await anext(self.gen)
(APIServer pid=19725)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 137, in build_async_engine_client_from_engine_args
(APIServer pid=19725)     async_llm = AsyncLLM.from_vllm_config(
(APIServer pid=19725)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 225, in from_vllm_config
(APIServer pid=19725)     return cls(
(APIServer pid=19725)            ^^^^
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 154, in __init__
(APIServer pid=19725)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=19725)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=19725)     return func(*args, **kwargs)
(APIServer pid=19725)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 127, in make_async_mp_client
(APIServer pid=19725)     return AsyncMPClient(*client_args)
(APIServer pid=19725)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=19725)     return func(*args, **kwargs)
(APIServer pid=19725)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 911, in __init__
(APIServer pid=19725)     super().__init__(
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 569, in __init__
(APIServer pid=19725)     with launch_core_engines(
(APIServer pid=19725)          ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=19725)   File "/home/playerAhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=19725)     next(self.gen)
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 951, in launch_core_engines
(APIServer pid=19725)     wait_for_engine_startup(
(APIServer pid=19725)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1010, in wait_for_engine_startup
(APIServer pid=19725)     raise RuntimeError(
(APIServer pid=19725) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 事件 3: Qwen/Qwen3-VL-2B-Instruct

- **阶段**: health_check
- **错误**: 视觉检查失败
- **处置**: 从实验中排除

## 事件 4: OpenGVLab/InternVL2_5-2B

- **阶段**: health_check
- **错误**: 视觉检查失败
- **处置**: 从实验中排除

## 事件 5: microsoft/Phi-3.5-vision-instruct

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 6: deepseek-ai/deepseek-vl2-tiny

- **阶段**: health_check
- **错误**: 视觉检查失败
- **处置**: 从实验中排除

## 事件 7: HuggingFaceTB/SmolVLM2-2.2B-Instruct

- **阶段**: startup
- **错误**: vLLM 在 300s 内未就绪: Error code: 502
- **处置**: 从实验中排除

**vLLM 输出（末尾片段）**:
```
(EngineCore_DP0 pid=303) INFO 03-16 13:09:10 [kv_cache_utils.py:1314] GPU KV cache size: 12,880 tokens
(EngineCore_DP0 pid=303) INFO 03-16 13:09:10 [kv_cache_utils.py:1319] Maximum concurrency for 8,192 tokens per request: 1.57x
(EngineCore_DP0 pid=303) INFO 03-16 13:09:10 [core.py:282] init engine (profile, create kv cache, warmup model) took 2.67 seconds
(EngineCore_DP0 pid=303) WARNING 03-16 13:09:10 [vllm.py:781] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(EngineCore_DP0 pid=303) WARNING 03-16 13:09:10 [vllm.py:792] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore_DP0 pid=303) INFO 03-16 13:09:10 [vllm.py:957] Cudagraph is disabled under eager mode
(APIServer pid=247) INFO 03-16 13:09:10 [api_server.py:495] Supported tasks: ['generate']
(APIServer pid=247) INFO 03-16 13:09:10 [serving.py:185] Warming up chat template processing...
(APIServer pid=247) INFO 03-16 13:09:10 [serving.py:210] Chat template warmup completed in 1.2ms
(APIServer pid=247) INFO 03-16 13:09:10 [api_server.py:500] Starting vLLM API server 0 on http://0.0.0.0:8000
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:38] Available routes are:
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /docs, Methods: HEAD, GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /redoc, Methods: HEAD, GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /tokenize, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /detokenize, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /load, Methods: GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /version, Methods: GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /health, Methods: GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /metrics, Methods: GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/models, Methods: GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /ping, Methods: GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /ping, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /invocations, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/chat/completions, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/responses, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/completions, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/completions/render, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/messages, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /inference/v1/generate, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=247) INFO 03-16 13:09:10 [launcher.py:47] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=247) INFO:     Started server process [247]
(APIServer pid=247) INFO:     Waiting for application startup.
(APIServer pid=247) INFO:     Application startup complete.

```

## 事件 8: mistralai/Ministral-3-3B-Instruct-2512

- **阶段**: experiment
- **错误**: 详见 orchestrator.log
- **处置**: 已跳过
