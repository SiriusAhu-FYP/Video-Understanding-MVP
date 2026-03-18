"""Video frame extractor for SOTA validation.

Extracts a fixed keyframe sequence from each video and caches them to disk.
All models and all runs reuse the SAME cached frames, ensuring:
- Identical visual input across models
- No real-time capture overhead
- Reproducible results
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
from loguru import logger as lg

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_VIDEOS_DIR = _PROJECT_ROOT / "assets" / "videos"


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_s: float = 1.0,
    max_frames: int = 30,
) -> list[Path]:
    """Extract frames from a video at fixed intervals.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        interval_s: Seconds between frame extractions.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of paths to saved frame images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    if fps <= 0:
        lg.error("Cannot read FPS from {}", video_path)
        cap.release()
        return []

    frame_interval = int(fps * interval_s)
    lg.info(
        "Extracting frames: {} | fps={:.1f} duration={:.1f}s interval={}frames max={}",
        video_path.name, fps, duration_s, frame_interval, max_frames,
    )

    saved: list[Path] = []
    frame_idx = 0
    extracted = 0

    while cap.isOpened() and extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"frame_{extracted:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved.append(out_path)
            extracted += 1
        frame_idx += 1

    cap.release()
    lg.info("Extracted {} frames -> {}", len(saved), output_dir)
    return saved


def extract_all_videos(
    cache_dir: Path,
    interval_s: float = 1.0,
    max_frames: int = 30,
) -> dict[str, list[Path]]:
    """Extract frames from all videos in assets/videos/.

    Returns: {video_id: [frame_paths]}
    """
    videos: list[Path] = sorted(ASSETS_VIDEOS_DIR.glob("*/*.mp4"))
    if not videos:
        videos = sorted(ASSETS_VIDEOS_DIR.glob("*.mp4"))

    if not videos:
        lg.error("No video files found in {}", ASSETS_VIDEOS_DIR)
        return {}

    results: dict[str, list[Path]] = {}
    for video_path in videos:
        video_id = video_path.stem
        video_cache = cache_dir / video_id
        existing = sorted(video_cache.glob("frame_*.jpg"))
        if existing:
            lg.info("Using cached frames for {}: {} frames", video_id, len(existing))
            results[video_id] = existing
        else:
            frames = extract_frames(video_path, video_cache, interval_s, max_frames)
            results[video_id] = frames

    return results


def load_frames_as_b64(frame_paths: list[Path]) -> list[tuple[str, str]]:
    """Load frame images as (base64, mime) tuples for API submission."""
    images: list[tuple[str, str]] = []
    for p in frame_paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        images.append((b64, "image/jpeg"))
    return images
