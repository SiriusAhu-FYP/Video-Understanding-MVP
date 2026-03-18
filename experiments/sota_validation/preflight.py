"""Low-cost preflight check for each SOTA model via DMXAPI.

Sends a single tiny compressed image to verify:
- Authentication works
- Model accepts multimodal input
- A non-empty response is returned
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from loguru import logger as lg
from PIL import Image

from experiments.sota_validation.dmxapi_client import DMXAPIClient, RequestResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_tiny_test_image() -> tuple[str, str]:
    """Create a small 64x64 test image, return (base64, mime)."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, "image/jpeg"


async def run_preflight(
    client: DMXAPIClient,
    models: list[str],
) -> tuple[list[str], list[dict]]:
    """Check each model with a tiny image request.

    Returns:
        (passed_models, incidents) where incidents is a list of dicts
        describing failures.
    """
    test_b64, test_mime = _make_tiny_test_image()
    passed: list[str] = []
    incidents: list[dict] = []

    for idx, model_id in enumerate(models, 1):
        lg.info("━" * 50)
        lg.info("[Preflight {}/{}] {}", idx, len(models), model_id)

        result: RequestResult = await client.chat(
            model=model_id,
            prompt="Describe this image in one sentence.",
            images_b64=[(test_b64, test_mime)],
            max_tokens=32,
            temperature=0.0,
        )

        if result.success and len(result.content.strip()) > 3:
            lg.info(
                "[PASS] {} | response: '{}' | latency: {:.1f}s",
                model_id, result.content[:60], result.latency_s,
            )
            passed.append(model_id)
        else:
            reason = result.error_message or "empty/short response"
            if result.success:
                reason = f"response too short: '{result.content[:30]}'"
            lg.warning("[FAIL] {} | {} | {}", model_id, result.error_type, reason)
            incidents.append({
                "model": model_id,
                "phase": "preflight",
                "error_type": result.error_type or "short_response",
                "error": reason,
                "retries": result.retries,
            })

    lg.info("━" * 50)
    lg.info("Preflight done: {}/{} passed", len(passed), len(models))
    if incidents:
        lg.warning("Failed: {}", [i["model"] for i in incidents])

    return passed, incidents
