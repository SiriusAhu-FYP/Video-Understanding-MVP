"""Common inference dataclass and single-shot VLM call used by benchmark experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger as lg
from openai import OpenAI

from ahu_paimon_toolkit.utils.gpu import get_gpu_memory_mb


@dataclass
class RunResult:
    """Unified result container for a single VLM inference run.

    Fields that are experiment-specific (e.g. ``game``) may be left as
    empty strings when unused.
    """
    model: str
    test_id: str
    test_name: str
    image_name: str
    run_idx: int
    ttft_s: float
    throughput_tps: float
    total_time_s: float
    output_tokens: int
    vram_mb: int
    response_text: str
    extra_label: str = ""


def run_single_inference(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    image_name: str,
    image_b64: str,
    image_mime: str,
    run_idx: int,
    *,
    test_id: str = "",
    test_name: str = "",
    extra_label: str = "",
    system_prompt: str | None = None,
) -> RunResult:
    """Execute one VLM inference call and return timing metrics.

    This is the shared core used by ``benchmark_speed`` and
    ``gameplay_analysis``.  Callers should pass experiment-specific
    metadata via *test_id*, *test_name*, and *extra_label*.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ],
    })

    vram_before = get_gpu_memory_mb()
    chunks: list[str] = []
    ttft: float = 0.0
    t0 = time.perf_counter()

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            if not chunks:
                ttft = time.perf_counter() - t0
            chunks.append(delta.content)
    total_time = time.perf_counter() - t0
    vram_after = get_gpu_memory_mb()

    response_text = "".join(chunks)
    output_tokens = max(len(chunks), 1)
    throughput = output_tokens / total_time if total_time > 0 else 0.0

    return RunResult(
        model=model,
        test_id=test_id,
        test_name=test_name,
        image_name=image_name,
        run_idx=run_idx,
        ttft_s=round(ttft, 4),
        throughput_tps=round(throughput, 2),
        total_time_s=round(total_time, 4),
        output_tokens=output_tokens,
        vram_mb=vram_after,
        response_text=response_text,
        extra_label=extra_label,
    )
