# 事件报告

**生成时间**: 2026-03-16 12:59:31

共记录 7 个事件。

## 事件 1: Qwen/Qwen3.5-2B

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 2: Qwen/Qwen3.5-0.8B

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 3: Qwen/Qwen3-VL-2B-Instruct

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 4: OpenGVLab/InternVL2_5-2B

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 5: microsoft/Phi-3.5-vision-instruct

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 6: deepseek-ai/deepseek-vl2-tiny

- **阶段**: health_check
- **错误**: 文本和视觉检查失败
- **处置**: 从实验中排除

## 事件 7: HuggingFaceTB/SmolVLM2-2.2B-Instruct

- **阶段**: startup
- **错误**: vLLM 在 300s 内未就绪: Connection error.
- **处置**: 从实验中排除

**vLLM 输出（末尾片段）**:
```
(APIServer pid=29592) The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
(APIServer pid=29592) You are using a model of type `deepseek_vl_v2` to instantiate a model of type ``. This may be expected if you are loading a checkpoint that shares a subset of the architecture (e.g., loading a `sam2_video` checkpoint into `Sam2Model`), but is otherwise not supported and can yield errors. Please verify that the checkpoint is compatible with the model you are instantiating.
(APIServer pid=29592) ERROR 03-16 12:48:29 [processor.py:229] Failed to collect processor kwargs
(APIServer pid=29592) ERROR 03-16 12:48:29 [processor.py:229] Traceback (most recent call last):
(APIServer pid=29592) ERROR 03-16 12:48:29 [processor.py:229]   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/transformers_utils/processor.py", line 219, in get_processor_kwargs_type
(APIServer pid=29592) ERROR 03-16 12:48:29 [processor.py:229]     return get_args(call_kwargs_annotations)[0]
(APIServer pid=29592) ERROR 03-16 12:48:29 [processor.py:229]            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
(APIServer pid=29592) ERROR 03-16 12:48:29 [processor.py:229] IndexError: tuple index out of range
(APIServer pid=29592) You are using a model of type `deepseek_vl_v2` to instantiate a model of type ``. This may be expected if you are loading a checkpoint that shares a subset of the architecture (e.g., loading a `sam2_video` checkpoint into `Sam2Model`), but is otherwise not supported and can yield errors. Please verify that the checkpoint is compatible with the model you are instantiating.
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:36 [core.py:101] Initializing a V1 LLM engine (v0.17.0) with config: model='/home/playerAhu/.cache/huggingface/hub/models--deepseek-ai--deepseek-vl2-tiny/snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa', speculative_config=None, tokenizer='/home/playerAhu/.cache/huggingface/hub/models--deepseek-ai--deepseek-vl2-tiny/snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=/home/playerAhu/.cache/huggingface/hub/models--deepseek-ai--deepseek-vl2-tiny/snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore_DP0 pid=29670) WARNING 03-16 12:48:36 [interface.py:472] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
(EngineCore_DP0 pid=29670) You are using a model of type `deepseek_vl_v2` to instantiate a model of type ``. This may be expected if you are loading a checkpoint that shares a subset of the architecture (e.g., loading a `sam2_video` checkpoint into `Sam2Model`), but is otherwise not supported and can yield errors. Please verify that the checkpoint is compatible with the model you are instantiating.
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:37 [parallel_state.py:1393] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.17.19.50:57981 backend=nccl
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:37 [parallel_state.py:1715] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0, EPLB rank N/A
(EngineCore_DP0 pid=29670) You are using a model of type `deepseek_vl_v2` to instantiate a model of type ``. This may be expected if you are loading a checkpoint that shares a subset of the architecture (e.g., loading a `sam2_video` checkpoint into `Sam2Model`), but is otherwise not supported and can yield errors. Please verify that the checkpoint is compatible with the model you are instantiating.
(EngineCore_DP0 pid=29670) ERROR 03-16 12:48:38 [processor.py:229] Failed to collect processor kwargs
(EngineCore_DP0 pid=29670) ERROR 03-16 12:48:38 [processor.py:229] Traceback (most recent call last):
(EngineCore_DP0 pid=29670) ERROR 03-16 12:48:38 [processor.py:229]   File "/home/playerAhu/vLLM_server/.venv/lib/python3.12/site-packages/vllm/transformers_utils/processor.py", line 219, in get_processor_kwargs_type
(EngineCore_DP0 pid=29670) ERROR 03-16 12:48:38 [processor.py:229]     return get_args(call_kwargs_annotations)[0]
(EngineCore_DP0 pid=29670) ERROR 03-16 12:48:38 [processor.py:229]            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
(EngineCore_DP0 pid=29670) ERROR 03-16 12:48:38 [processor.py:229] IndexError: tuple index out of range
(EngineCore_DP0 pid=29670) You are using a model of type `deepseek_vl_v2` to instantiate a model of type ``. This may be expected if you are loading a checkpoint that shares a subset of the architecture (e.g., loading a `sam2_video` checkpoint into `Sam2Model`), but is otherwise not supported and can yield errors. Please verify that the checkpoint is compatible with the model you are instantiating.
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:40 [base.py:106] Offloader set to NoopOffloader
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:40 [gpu_model_runner.py:4255] Starting to load model /home/playerAhu/.cache/huggingface/hub/models--deepseek-ai--deepseek-vl2-tiny/snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa...
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:41 [vllm.py:747] Asynchronous scheduling is enabled.
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:41 [cuda.py:405] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:41 [flash_attn.py:587] Using FlashAttention version 2
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:41 [unquantized.py:186] Using TRITON backend for Unquantized MoE
(EngineCore_DP0 pid=29670) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
(EngineCore_DP0 pid=29670) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
(EngineCore_DP0 pid=29670) 
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
(EngineCore_DP0 pid=29670) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:02<00:00,  2.70s/it]
(EngineCore_DP0 pid=29670) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:02<00:00,  2.70s/it]
(EngineCore_DP0 pid=29670) 
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:44 [default_loader.py:293] Loading weights took 2.88 seconds
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:45 [gpu_model_runner.py:4338] Model loading took 6.31 GiB memory and 3.655868 seconds
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:45 [gpu_model_runner.py:5254] Encoder cache will be initialized with a budget of 2101 tokens, and profiled with 1 image items of the maximum feature size.
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:46 [decorators.py:465] Directly load AOT compilation from path /home/playerAhu/.cache/vllm/torch_compile_cache/torch_aot_compile/9c839203a75d7c12dbcaad7768d455fe31995cdf082a0fbbeda4c24fe1b30857/rank_0_0/model
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:46 [backends.py:916] Using cache directory: /home/playerAhu/.cache/vllm/torch_compile_cache/e7d08cfcaa/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=29670) INFO 03-16 12:48:46 [backends.py:976] Dynamo bytecode transform time: 0.61 s

```
