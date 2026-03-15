"""Shared utilities for VLM experiment scripts.

Provides model detection, image loading, GPU memory query,
and other helper functions used across experiments.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from loguru import logger as lg
from openai import OpenAI

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_IMAGES_DIR = PROJECT_ROOT / "assets" / "images"
ASSETS_VIDEOS_DIR = PROJECT_ROOT / "assets" / "videos"


def encode_image(path: Path) -> str:
    """Encode an image file to a Base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(path: Path) -> str:
    """Return MIME type based on file extension."""
    ext = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/png")


def get_gpu_memory_mb() -> int:
    """Query current GPU memory usage (MB) via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def get_gpu_info() -> dict:
    """Return a dict with GPU name, total/free/used memory."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "name": parts[0],
            "memory_total_mb": int(parts[1]),
            "memory_free_mb": int(parts[2]),
            "memory_used_mb": int(parts[3]),
        }
    except Exception:
        return {"name": "unknown", "memory_total_mb": 0, "memory_free_mb": 0, "memory_used_mb": 0}


def detect_model(client: OpenAI) -> str:
    """Auto-detect the currently running model via /v1/models."""
    models = client.models.list()
    if models.data:
        model_id = models.data[0].id
        lg.info("自动检测到模型: {}", model_id)
        return model_id
    raise RuntimeError("未检测到任何运行中的模型")


def detect_model_from_url(base_url: str = "http://localhost:8000/v1") -> str:
    """Create a throwaway client and detect the running model."""
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    return detect_model(client)


def model_short_name(model_id: str) -> str:
    """Convert 'Org/Model-Name' to 'Org_Model-Name' for directory naming."""
    return model_id.replace("/", "_").replace(" ", "_")


def load_images(
    assets_dir: Path | None = None,
) -> list[tuple[str, str, str, Path]]:
    """Load all supported images from a directory.

    Returns: [(filename, base64_data, mime_type, file_path), ...]
    """
    if assets_dir is None:
        assets_dir = ASSETS_IMAGES_DIR

    images: list[tuple[str, str, str, Path]] = []
    for p in sorted(assets_dir.iterdir()):
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            b64 = encode_image(p)
            mime = get_image_mime(p)
            images.append((p.name, b64, mime, p))
            lg.debug("已加载图片: {} ({:.1f} KB)", p.name, len(b64) * 3 / 4 / 1024)
    if not images:
        raise FileNotFoundError(f"目录中未找到支持的图片: {assets_dir}")
    lg.info("共加载 {} 张图片", len(images))
    return images


def wait_for_vllm_ready(
    base_url: str = "http://localhost:8000/v1",
    timeout_s: float = 300.0,
    poll_interval_s: float = 10.0,
) -> str:
    """Poll the vLLM /v1/models endpoint until it responds.

    Returns the detected model ID.
    """
    import time

    start = time.monotonic()
    last_err = None
    while time.monotonic() - start < timeout_s:
        try:
            return detect_model_from_url(base_url)
        except Exception as e:
            last_err = e
            elapsed = time.monotonic() - start
            lg.debug("vLLM 尚未就绪 ({:.0f}s / {:.0f}s): {}", elapsed, timeout_s, e)
            time.sleep(poll_interval_s)
    raise TimeoutError(
        f"vLLM 在 {timeout_s}s 内未就绪: {last_err}"
    )
