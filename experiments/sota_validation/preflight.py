"""Low-cost preflight check for each SOTA model via DMXAPI.

Sends a single real game screenshot (resized to 512px shortest side) to verify:
- Authentication works
- Model accepts multimodal input
- A non-empty, meaningful response is returned

All model checks run in parallel via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

from loguru import logger as lg
from PIL import Image

from experiments.sota_validation.dmxapi_client import DMXAPIClient, RequestResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_IMAGES_DIR = _PROJECT_ROOT / "assets" / "images"


def _make_preflight_image() -> tuple[str, str]:
    """Load the first real game screenshot, resize to 512px shortest side."""
    candidates: list[Path] = []
    for entry in sorted(ASSETS_IMAGES_DIR.iterdir()):
        if entry.is_dir():
            candidates.extend(sorted(entry.glob("*.png")))
            candidates.extend(sorted(entry.glob("*.jpg")))
        elif entry.suffix.lower() in (".png", ".jpg", ".jpeg"):
            candidates.append(entry)

    if not candidates:
        lg.warning("No images in assets, falling back to synthetic 512x512")
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    else:
        img = Image.open(candidates[0]).convert("RGB")
        w, h = img.size
        short_side = min(w, h)
        if short_side > 512:
            scale = 512 / short_side
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    lg.info("Preflight image: {}x{}, {:.1f} KB", img.width, img.height, len(buf.getvalue()) / 1024)
    return b64, "image/jpeg"


async def _check_single_model(
    client: DMXAPIClient,
    model_id: str,
    test_b64: str,
    test_mime: str,
    idx: int,
    total: int,
) -> tuple[str | None, dict | None]:
    """Check a single model. Returns (model_id, None) on pass, (None, incident) on fail."""
    lg.info("[Preflight {}/{}] {}", idx, total, model_id)

    result: RequestResult = await client.chat(
        model=model_id,
        prompt="Describe this image in one sentence.",
        images_b64=[(test_b64, test_mime)],
        max_tokens=64,
        temperature=0.0,
    )

    if result.success and len(result.content.strip()) > 3:
        lg.info(
            "[PASS] {} | response: '{}' | latency: {:.1f}s",
            model_id, result.content[:60], result.latency_s,
        )
        return model_id, None

    reason = result.error_message or "empty/short response"
    if result.success:
        reason = f"response too short: '{result.content[:30]}'"
    lg.warning("[FAIL] {} | {} | {}", model_id, result.error_type, reason)
    return None, {
        "model": model_id,
        "phase": "preflight",
        "error_type": result.error_type or "short_response",
        "error": reason,
        "retries": result.retries,
    }


async def run_preflight(
    client: DMXAPIClient,
    models: list[str],
) -> tuple[list[str], list[dict]]:
    """Check all models in parallel with a real game screenshot.

    Returns:
        (passed_models, incidents) where incidents is a list of dicts
        describing failures.
    """
    test_b64, test_mime = _make_preflight_image()

    lg.info("━" * 50)
    lg.info("Running preflight for {} models (parallel)", len(models))

    tasks = [
        _check_single_model(client, model_id, test_b64, test_mime, idx, len(models))
        for idx, model_id in enumerate(models, 1)
    ]
    results = await asyncio.gather(*tasks)

    passed: list[str] = []
    incidents: list[dict] = []
    for model_id, incident in results:
        if model_id is not None:
            passed.append(model_id)
        if incident is not None:
            incidents.append(incident)

    lg.info("━" * 50)
    lg.info("Preflight done: {}/{} passed", len(passed), len(models))
    if incidents:
        lg.warning("Failed: {}", [i["model"] for i in incidents])

    return passed, incidents
