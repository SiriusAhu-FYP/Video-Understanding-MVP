# 事件报告

**生成时间**: 2026-03-16 12:49:15

共记录 8 个事件。

## 事件 1: Qwen/Qwen3.5-2B

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 2: Qwen/Qwen3.5-0.8B

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 3: Qwen/Qwen3-VL-2B-Instruct

- **阶段**: startup
- **错误**: vLLM 在 300s 内未就绪: Connection error.
- **处置**: 从实验中排除

**vLLM 输出（末尾片段）**:
```
(APIServer pid=24273)                ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=24273)     return await anext(self.gen)
(APIServer pid=24273)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 96, in build_async_engine_client
(APIServer pid=24273)     async with build_async_engine_client_from_engine_args(
(APIServer pid=24273)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=24273)     return await anext(self.gen)
(APIServer pid=24273)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 137, in build_async_engine_client_from_engine_args
(APIServer pid=24273)     async_llm = AsyncLLM.from_vllm_config(
(APIServer pid=24273)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 225, in from_vllm_config
(APIServer pid=24273)     return cls(
(APIServer pid=24273)            ^^^^
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 154, in __init__
(APIServer pid=24273)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=24273)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=24273)     return func(*args, **kwargs)
(APIServer pid=24273)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 127, in make_async_mp_client
(APIServer pid=24273)     return AsyncMPClient(*client_args)
(APIServer pid=24273)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=24273)     return func(*args, **kwargs)
(APIServer pid=24273)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 911, in __init__
(APIServer pid=24273)     super().__init__(
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 569, in __init__
(APIServer pid=24273)     with launch_core_engines(
(APIServer pid=24273)          ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=24273)   File "/home/playerAhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=24273)     next(self.gen)
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 951, in launch_core_engines
(APIServer pid=24273)     wait_for_engine_startup(
(APIServer pid=24273)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1010, in wait_for_engine_startup
(APIServer pid=24273)     raise RuntimeError(
(APIServer pid=24273) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 事件 4: OpenGVLab/InternVL2_5-2B

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 5: microsoft/Phi-3.5-vision-instruct

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 6: mistralai/Ministral-3-3B-Instruct-2512

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 7: deepseek-ai/deepseek-vl2-tiny

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 8: HuggingFaceTB/SmolVLM2-2.2B-Instruct

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除
