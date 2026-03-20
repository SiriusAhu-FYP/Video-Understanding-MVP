"""Image quality experiment for SOTA validation.

For each model (in parallel) x each image asset x 2 prompt modes x N runs:
1. Send image + prompt to DMXAPI (the SOTA model answers)
2. DeepSeek judges the response using the asset rubric
3. Save per-run structured result

All models run concurrently via asyncio.gather; each writes to its own directory.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger as lg

from experiments.sota_validation.dmxapi_client import DMXAPIClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_IMAGES_DIR = _PROJECT_ROOT / "assets" / "images"


def load_asset_jsons(assets_dir: Path) -> list[dict]:
    """Load all JSON task-description files (supports per-folder layout)."""
    jsons: list[dict] = []
    candidates: list[Path] = []
    for entry in sorted(assets_dir.iterdir()):
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


def load_image_b64(asset: dict) -> tuple[str, str] | None:
    """Load image as (base64, mime) from asset's co-located directory."""
    json_dir = Path(asset["_json_path"]).parent
    image_file = asset.get("image_file", "")
    image_path = json_dir / image_file
    if not image_path.exists():
        image_path = ASSETS_IMAGES_DIR / image_file
    if not image_path.exists():
        return None

    suffix = image_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".bmp": "image/bmp"}
    mime = mime_map.get(suffix, "image/png")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return b64, mime


def build_judge_prompt(
    asset: dict,
    prompt_mode: str,
    vlm_response: str,
) -> str:
    """Build the judge prompt from asset rubric fields."""
    return (
        f"{asset['grading_prompt_for_judge_model']}\n\n"
        f"## Task Definition\n{json.dumps(asset['task_definition'], ensure_ascii=False)}\n\n"
        f"## Reference Answer\n{json.dumps(asset['reference_answer'], ensure_ascii=False)}\n\n"
        f"## Scoring Rubric\n{json.dumps(asset['scoring'], ensure_ascii=False)}\n\n"
        f"## Prompt Mode\n{prompt_mode}\n\n"
        f"## Model Response to Evaluate\n{vlm_response}\n\n"
        "Return ONLY valid JSON with keys: dimension_scores, total_score, max_score, "
        "strengths, weaknesses, missing_points, hallucinations."
    )


def parse_judge_response(text: str) -> dict | None:
    """Extract JSON from judge response (handles markdown fences)."""
    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    raw = fence.group(1) if fence else text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def _run_single_model(
    client: DMXAPIClient,
    judge_client: DMXAPIClient,
    judge_model: str,
    model_id: str,
    assets: list[dict],
    output_dir: Path,
    runs: int,
) -> tuple[str, list[dict]]:
    """Process all assets for a single model. Returns (model_id, results)."""
    lg.info("Image experiment started: {}", model_id)
    model_results: list[dict] = []
    model_dir = output_dir / model_id.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        img_data = load_image_b64(asset)
        if img_data is None:
            lg.warning("[{}] Image not found for {}, skipping", model_id, asset["id"])
            continue

        for prompt_mode in ["A_description", "B_assistant"]:
            prompt_text = asset["prompts"][prompt_mode]

            for run_idx in range(1, runs + 1):
                lg.info(
                    "  {} | {} | {} | run {}/{}",
                    asset["id"], model_id.split("/")[-1], prompt_mode, run_idx, runs,
                )

                vlm_result = await client.chat(
                    model=model_id,
                    prompt=prompt_text,
                    images_b64=[img_data],
                    max_tokens=1024,
                    temperature=0.1,
                )

                record: dict[str, Any] = {
                    "model": model_id,
                    "asset_id": asset["id"],
                    "prompt_mode": prompt_mode,
                    "run": run_idx,
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
    lg.info("Image experiment done: {} ({} results)", model_id, len(model_results))
    return model_id, model_results


async def run_image_experiment(
    client: DMXAPIClient,
    judge_client: DMXAPIClient,
    judge_model: str,
    models: list[str],
    output_dir: Path,
    runs: int = 5,
) -> dict[str, list[dict]]:
    """Run image quality experiment for all models in parallel.

    Returns: {model_id: [result_dicts]}
    """
    assets = load_asset_jsons(ASSETS_IMAGES_DIR)
    if not assets:
        lg.error("No image assets found in {}", ASSETS_IMAGES_DIR)
        return {}

    lg.info("Loaded {} image assets, launching {} models in parallel", len(assets), len(models))

    tasks = [
        _run_single_model(client, judge_client, judge_model, m, assets, output_dir, runs)
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
