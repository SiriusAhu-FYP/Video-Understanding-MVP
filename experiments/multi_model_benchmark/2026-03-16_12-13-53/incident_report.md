# 事件报告

**生成时间**: 2026-03-16 13:54:06

共记录 8 个事件。

## 事件 1: Qwen/Qwen3.5-0.8B

- **阶段**: gpu_probe
- **错误**: 所有 gpu_memory_utilization (0.3-0.6) 均无法启动
- **处置**: 从实验中排除

## 事件 2: Qwen/Qwen3-VL-2B-Instruct

- **阶段**: startup
- **错误**: vLLM 在 300s 内未就绪: Connection error.
- **处置**: 从实验中排除

**vLLM 输出（末尾片段）**:
```
(APIServer pid=27160)                ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=27160)     return await anext(self.gen)
(APIServer pid=27160)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 96, in build_async_engine_client
(APIServer pid=27160)     async with build_async_engine_client_from_engine_args(
(APIServer pid=27160)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=27160)     return await anext(self.gen)
(APIServer pid=27160)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 137, in build_async_engine_client_from_engine_args
(APIServer pid=27160)     async_llm = AsyncLLM.from_vllm_config(
(APIServer pid=27160)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 225, in from_vllm_config
(APIServer pid=27160)     return cls(
(APIServer pid=27160)            ^^^^
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 154, in __init__
(APIServer pid=27160)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=27160)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=27160)     return func(*args, **kwargs)
(APIServer pid=27160)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 127, in make_async_mp_client
(APIServer pid=27160)     return AsyncMPClient(*client_args)
(APIServer pid=27160)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=27160)     return func(*args, **kwargs)
(APIServer pid=27160)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 911, in __init__
(APIServer pid=27160)     super().__init__(
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 569, in __init__
(APIServer pid=27160)     with launch_core_engines(
(APIServer pid=27160)          ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=27160)   File "/home/playerAhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=27160)     next(self.gen)
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 951, in launch_core_engines
(APIServer pid=27160)     wait_for_engine_startup(
(APIServer pid=27160)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1010, in wait_for_engine_startup
(APIServer pid=27160)     raise RuntimeError(
(APIServer pid=27160) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 事件 3: OpenGVLab/InternVL2_5-2B

- **阶段**: startup
- **错误**: vLLM 在 300s 内未就绪: Connection error.
- **处置**: 从实验中排除

**vLLM 输出（末尾片段）**:
```
(APIServer pid=30792)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/multimodal/encoder_budget.py", line 87, in __init__
(APIServer pid=30792)     all_mm_max_toks_per_item = get_mm_max_toks_per_item(
(APIServer pid=30792)                                ^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/multimodal/encoder_budget.py", line 32, in get_mm_max_toks_per_item
(APIServer pid=30792)     mm_inputs = mm_registry.get_dummy_mm_inputs(
(APIServer pid=30792)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/multimodal/registry.py", line 235, in get_dummy_mm_inputs
(APIServer pid=30792)     processor_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
(APIServer pid=30792)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/multimodal/processing/dummy_inputs.py", line 82, in get_dummy_processor_inputs
(APIServer pid=30792)     dummy_text = self.get_dummy_text(mm_counts)
(APIServer pid=30792)                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/model_executor/models/idefics3.py", line 271, in get_dummy_text
(APIServer pid=30792)     processor = self.info.get_hf_processor()
(APIServer pid=30792)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/model_executor/models/smolvlm.py", line 17, in get_hf_processor
(APIServer pid=30792)     return self.ctx.get_hf_processor(SmolVLMProcessor, **kwargs)
(APIServer pid=30792)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/multimodal/processing/context.py", line 203, in get_hf_processor
(APIServer pid=30792)     return cached_processor_from_config(
(APIServer pid=30792)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/transformers_utils/processor.py", line 315, in cached_processor_from_config
(APIServer pid=30792)     return cached_get_processor_without_dynamic_kwargs(
(APIServer pid=30792)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/transformers_utils/processor.py", line 272, in cached_get_processor_without_dynamic_kwargs
(APIServer pid=30792)     processor = cached_get_processor(
(APIServer pid=30792)                 ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/transformers_utils/processor.py", line 164, in get_processor
(APIServer pid=30792)     processor = processor_cls.from_pretrained(
(APIServer pid=30792)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/transformers/processing_utils.py", line 1404, in from_pretrained
(APIServer pid=30792)     return cls.from_args_and_dict(args, processor_dict, **instantiation_kwargs)
(APIServer pid=30792)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/transformers/processing_utils.py", line 1171, in from_args_and_dict
(APIServer pid=30792)     processor = cls(*args, **valid_kwargs)
(APIServer pid=30792)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=30792)   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/transformers/models/smolvlm/processing_smolvlm.py", line 140, in __init__
(APIServer pid=30792)     raise ImportError(
(APIServer pid=30792) ImportError: Package `num2words` is required to run SmolVLM processor. Install it with `pip install num2words`.

```

## 事件 4: microsoft/Phi-3.5-vision-instruct

- **阶段**: startup
- **错误**: vLLM 在 300s 内未就绪: Connection error.
- **处置**: 从实验中排除

## 事件 5: mistralai/Ministral-3-3B-Instruct-2512

- **阶段**: gpu_probe
- **错误**: 所有 gpu_memory_utilization (0.3-0.6) 均无法启动
- **处置**: 从实验中排除

## 事件 6: deepseek-ai/deepseek-vl2-tiny

- **阶段**: gpu_probe
- **错误**: 所有 gpu_memory_utilization (0.3-0.6) 均无法启动
- **处置**: 从实验中排除

## 事件 7: HuggingFaceTB/SmolVLM2-2.2B-Instruct

- **阶段**: gpu_probe
- **错误**: 所有 gpu_memory_utilization (0.3-0.6) 均无法启动
- **处置**: 从实验中排除

## 事件 8: Qwen/Qwen3.5-2B

- **阶段**: experiment
- **错误**: 详见 orchestrator.log
- **处置**: 已跳过
