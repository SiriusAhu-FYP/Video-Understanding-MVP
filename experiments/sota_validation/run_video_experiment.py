"""Video quality experiment for SOTA validation.

For each model (in parallel) x each video x 2 prompt modes x N runs:
1. Load pre-extracted cached frames (no real-time capture)
2. Send frame sequence + prompt to DMXAPI as multi-image input
3. DeepSeek judges the temporal summary
4. Save per-run structured result

All models run concurrently via asyncio.gather; each writes to its own directory.
"""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger as lg

from experiments.sota_validation.dmxapi_client import DMXAPIClient
from experiments.sota_validation.frame_extractor import load_frames_as_b64
from experiments.sota_validation.run_image_experiment import (
    build_judge_prompt,
    parse_judge_response,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_VIDEOS_DIR = _PROJECT_ROOT / "assets" / "videos"


def _load_video_jsons() -> list[dict]:
    """Load all video asset JSONs (supports per-folder layout)."""
    jsons: list[dict] = []
    candidates: list[Path] = []
    for entry in sorted(ASSETS_VIDEOS_DIR.iterdir()):
        if entry.is_dir():
            candidates.extend(sorted(entry.glob("*.json")))
        elif entry.suffix == ".json":
            candidates.append(entry)
    for p in candidates:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if data:
            data["_json_path"] = str(p)
            jsons.append(data)
    return jsons


async def _run_single_model(
    client: DMXAPIClient,
    judge_client: DMXAPIClient,
    judge_model: str,
    model_id: str,
    video_assets: list[dict],
    cached_frames: dict[str, list[Path]],
    output_dir: Path,
    runs: int,
) -> tuple[str, list[dict]]:
    """Process all video assets for a single model. Returns (model_id, results)."""
    lg.info("Video experiment started: {}", model_id)
    model_results: list[dict] = []
    model_dir = output_dir / model_id.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    for asset in video_assets:
        video_id = asset["id"]
        frames = cached_frames.get(video_id)
        if not frames:
            lg.warning("[{}] No cached frames for {}, skipping", model_id, video_id)
            continue

        frame_images = load_frames_as_b64(frames)
        lg.info("  [{}] {} | {} frames loaded", model_id, video_id, len(frame_images))

        for prompt_mode in ["A_description", "B_assistant"]:
            prompt_text = asset["prompts"][prompt_mode]
            frame_context = (
                f"The following {len(frame_images)} images are sequential frames "
                f"extracted from a game video at regular intervals. "
                f"Analyze them as a video sequence.\n\n{prompt_text}"
            )

            for run_idx in range(1, runs + 1):
                lg.info(
                    "  {} | {} | {} | run {}/{}",
                    video_id, model_id.split("/")[-1], prompt_mode, run_idx, runs,
                )

                vlm_result = await client.chat(
                    model=model_id,
                    prompt=frame_context,
                    images_b64=frame_images,
                    max_tokens=1024,
                    temperature=0.1,
                )

                record: dict[str, Any] = {
                    "model": model_id,
                    "asset_id": video_id,
                    "asset_type": "video",
                    "prompt_mode": prompt_mode,
                    "run": run_idx,
                    "num_frames": len(frame_images),
                    "timestamp": datetime.now().isoformat(),
                    "vlm_success": vlm_result.success,
                    "vlm_response": vlm_result.content[:2000] if vlm_result.success else "",
                    "vlm_latency_s": round(vlm_result.latency_s, 3),
                    "vlm_retries": vlm_result.retries,
                    "vlm_error_type": vlm_result.error_type,
                    "vlm_error": vlm_result.error_message[:300] if not vlm_result.success else "",
                }

                if not vlm_result.success:
                    lg.warning("    [{}] VLM failed: {}", model_id, vlm_result.error_type)
                    record.update({"judge_success": False, "total_score": -1, "max_score": 10})
                    model_results.append(record)
                    continue

                judge_prompt = build_judge_prompt(asset, prompt_mode, vlm_result.content)
                judge_result = await judge_client.chat(
                    model=judge_model,
                    prompt=judge_prompt,
                    max_tokens=1024,
                    temperature=0.0,
                )

                if judge_result.success:
                    parsed = parse_judge_response(judge_result.content)
                    if parsed:
                        record.update({
                            "judge_success": True,
                            "total_score": parsed.get("total_score", -1),
                            "max_score": parsed.get("max_score", 10),
                            "dimension_scores": json.dumps(parsed.get("dimension_scores", {})),
                            "strengths": json.dumps(parsed.get("strengths", []), ensure_ascii=False),
                            "weaknesses": json.dumps(parsed.get("weaknesses", []), ensure_ascii=False),
                            "missing_points": json.dumps(parsed.get("missing_points", []), ensure_ascii=False),
                            "hallucinations": json.dumps(parsed.get("hallucinations", []), ensure_ascii=False),
                            "judge_latency_s": round(judge_result.latency_s, 3),
                            "judge_retries": judge_result.retries,
                        })
                    else:
                        lg.warning("    [{}] Judge response not valid JSON", model_id)
                        record.update({
                            "judge_success": False, "total_score": -1, "max_score": 10,
                            "judge_raw": judge_result.content[:500],
                        })
                else:
                    lg.warning("    [{}] Judge call failed: {}", model_id, judge_result.error_type)
                    record.update({"judge_success": False, "total_score": -1, "max_score": 10})

                model_results.append(record)

    _save_results_csv(model_results, model_dir / "results.csv")
    lg.info("Video experiment done: {} ({} results)", model_id, len(model_results))
    return model_id, model_results


async def run_video_experiment(
    client: DMXAPIClient,
    judge_client: DMXAPIClient,
    judge_model: str,
    models: list[str],
    cached_frames: dict[str, list[Path]],
    output_dir: Path,
    runs: int = 5,
) -> dict[str, list[dict]]:
    """Run video quality experiment for all models in parallel.

    Returns: {model_id: [result_dicts]}
    """
    video_assets = _load_video_jsons()
    if not video_assets:
        lg.error("No video assets found in {}", ASSETS_VIDEOS_DIR)
        return {}

    lg.info("Loaded {} video assets, launching {} models in parallel", len(video_assets), len(models))

    tasks = [
        _run_single_model(client, judge_client, judge_model, m, video_assets, cached_frames, output_dir, runs)
        for m in models
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: dict[str, list[dict]] = {}
    for r in results:
        if isinstance(r, Exception):
            lg.error("Model task failed with exception: {}", r)
            continue
        model_id, model_results = r
        all_results[model_id] = model_results

    return all_results


def _save_results_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    lg.info("Saved CSV: {}", path)
