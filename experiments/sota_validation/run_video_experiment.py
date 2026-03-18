"""Video quality experiment for SOTA validation.

For each model x each video x 2 prompt modes x N runs:
1. Load pre-extracted cached frames (no real-time capture)
2. Send frame sequence + prompt to DMXAPI as multi-image input
3. The SAME model self-judges its own temporal summary
4. Save per-run structured result
"""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path

from loguru import logger as lg

from experiments.sota_validation.dmxapi_client import DMXAPIClient
from experiments.sota_validation.frame_extractor import load_frames_as_b64
from experiments.sota_validation.run_image_experiment import _build_judge_prompt, _parse_judge_response

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_VIDEOS_DIR = _PROJECT_ROOT / "assets" / "videos"


def _load_video_jsons() -> list[dict]:
    """Load all video asset JSONs (supports per-folder layout)."""
    jsons = []
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


async def run_video_experiment(
    client: DMXAPIClient,
    models: list[str],
    cached_frames: dict[str, list[Path]],
    output_dir: Path,
    runs: int = 5,
) -> dict[str, list[dict]]:
    """Run the full video quality experiment.

    Args:
        client: DMXAPI client instance.
        models: List of model IDs to test.
        cached_frames: {video_id: [frame_paths]} from frame_extractor.
        output_dir: Directory for per-model results.
        runs: Number of repetitions per model x video x prompt_mode.

    Returns:
        {model_id: [result_dicts]}
    """
    video_assets = _load_video_jsons()
    if not video_assets:
        lg.error("No video assets found in {}", ASSETS_VIDEOS_DIR)
        return {}

    lg.info("Loaded {} video assets, {} cached video frame sets", len(video_assets), len(cached_frames))
    all_results: dict[str, list[dict]] = {}

    for model_id in models:
        lg.info("=" * 60)
        lg.info("Video experiment: {}", model_id)
        model_results: list[dict] = []
        model_dir = output_dir / model_id.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        for asset in video_assets:
            video_id = asset["id"]
            frames = cached_frames.get(video_id)
            if not frames:
                lg.warning("No cached frames for {}, skipping", video_id)
                continue

            frame_images = load_frames_as_b64(frames)
            lg.info("  {} | {} frames loaded", video_id, len(frame_images))

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

                    record: dict = {
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
                        lg.warning("    VLM failed: {}", vlm_result.error_type)
                        record.update({"judge_success": False, "total_score": -1, "max_score": 10})
                        model_results.append(record)
                        continue

                    # Self-judge
                    judge_prompt = _build_judge_prompt(asset, prompt_mode, vlm_result.content)
                    judge_result = await client.chat(
                        model=model_id,
                        prompt=judge_prompt,
                        max_tokens=1024,
                        temperature=0.0,
                    )

                    if judge_result.success:
                        parsed = _parse_judge_response(judge_result.content)
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
                            lg.warning("    Judge response not valid JSON")
                            record.update({
                                "judge_success": False, "total_score": -1, "max_score": 10,
                                "judge_raw": judge_result.content[:500],
                            })
                    else:
                        lg.warning("    Judge call failed: {}", judge_result.error_type)
                        record.update({"judge_success": False, "total_score": -1, "max_score": 10})

                    model_results.append(record)
                    await asyncio.sleep(2.0)

        _save_results_csv(model_results, model_dir / "results.csv")
        all_results[model_id] = model_results
        lg.info("Saved {} results for {}", len(model_results), model_id)

    return all_results


def _save_results_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    lg.info("Saved CSV: {}", path)
