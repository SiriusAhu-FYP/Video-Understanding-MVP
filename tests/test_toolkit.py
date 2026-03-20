"""Minimal tests for ahu_paimon_toolkit to verify key paths work.

These tests use mocks where external services (vLLM, DeepSeek) are needed,
and real files where available (asset JSONs, images).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_IMAGES_DIR = PROJECT_ROOT / "assets" / "images"
ASSETS_VIDEOS_DIR = PROJECT_ROOT / "assets" / "videos"


# ── JSON Loading Tests ───────────────────────────────────────────


class TestAssetJSONLoading:
    """Verify all asset JSON files load correctly with consistent schema."""

    REQUIRED_FIELDS = {
        "id", "type", "game", "requirement", "focus",
        "task_definition", "prompts", "reference_answer",
        "scoring", "grading_prompt_for_judge_model", "metadata",
    }

    def _load_all_jsons(self, directory: Path) -> list[dict]:
        jsons = []
        candidates: list[Path] = []
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                candidates.extend(sorted(entry.glob("*.json")))
            elif entry.suffix == ".json":
                candidates.append(entry)
        for p in candidates:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if data:
                data["_path"] = str(p)
                jsons.append(data)
        return jsons

    def test_image_jsons_load(self):
        jsons = self._load_all_jsons(ASSETS_IMAGES_DIR)
        assert len(jsons) == 4, f"Expected 4 image JSONs, got {len(jsons)}"

    def test_video_jsons_load(self):
        jsons = self._load_all_jsons(ASSETS_VIDEOS_DIR)
        assert len(jsons) == 2, f"Expected 2 video JSONs, got {len(jsons)}"

    def test_image_json_schema(self):
        for data in self._load_all_jsons(ASSETS_IMAGES_DIR):
            missing = self.REQUIRED_FIELDS - set(data.keys())
            filtered_missing = {k for k in missing if not k.startswith("_")}
            assert not filtered_missing, f"{data['id']} missing fields: {filtered_missing}"
            assert "image_file" in data, f"{data['id']} missing image_file"
            assert data["type"] == "image"

    def test_video_json_schema(self):
        for data in self._load_all_jsons(ASSETS_VIDEOS_DIR):
            missing = self.REQUIRED_FIELDS - set(data.keys())
            filtered_missing = {k for k in missing if not k.startswith("_")}
            assert not filtered_missing, f"{data['id']} missing fields: {filtered_missing}"
            assert "video_file" in data, f"{data['id']} missing video_file"
            assert data["type"] == "video"

    def test_scoring_dimensions_count(self):
        """Each asset should have exactly 5 scoring dimensions."""
        all_jsons = self._load_all_jsons(ASSETS_IMAGES_DIR) + self._load_all_jsons(ASSETS_VIDEOS_DIR)
        for data in all_jsons:
            dims = data["scoring"]["dimensions"]
            assert len(dims) == 5, f"{data['id']} has {len(dims)} dimensions, expected 5"

    def test_prompts_have_both_modes(self):
        """Each asset should have A_description and B_assistant prompts."""
        all_jsons = self._load_all_jsons(ASSETS_IMAGES_DIR) + self._load_all_jsons(ASSETS_VIDEOS_DIR)
        for data in all_jsons:
            prompts = data["prompts"]
            assert "A_description" in prompts, f"{data['id']} missing A_description"
            assert "B_assistant" in prompts, f"{data['id']} missing B_assistant"

    def test_field_order_consistency(self):
        """Verify image_file/video_file comes after 'game' field."""
        all_jsons = self._load_all_jsons(ASSETS_IMAGES_DIR) + self._load_all_jsons(ASSETS_VIDEOS_DIR)
        for data in all_jsons:
            keys = list(data.keys())
            keys = [k for k in keys if not k.startswith("_")]
            game_idx = keys.index("game")
            file_key = "image_file" if data["type"] == "image" else "video_file"
            file_idx = keys.index(file_key)
            assert file_idx == game_idx + 1, (
                f"{data['id']}: {file_key} at index {file_idx}, "
                f"expected {game_idx + 1} (right after game)"
            )


# ── Image Encoding Tests ─────────────────────────────────────────


class TestImageEncoding:
    def test_encode_image(self):
        from ahu_paimon_toolkit.utils.image import encode_image

        images = list(ASSETS_IMAGES_DIR.glob("*/*.png"))
        assert len(images) > 0
        b64 = encode_image(images[0])
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_get_image_mime(self):
        from ahu_paimon_toolkit.utils.image import get_image_mime

        assert get_image_mime(Path("test.png")) == "image/png"
        assert get_image_mime(Path("test.jpg")) == "image/jpeg"
        assert get_image_mime(Path("test.jpeg")) == "image/jpeg"
        assert get_image_mime(Path("test.bmp")) == "image/bmp"

    def test_load_images(self):
        from ahu_paimon_toolkit.utils.image import load_images

        images = load_images(ASSETS_IMAGES_DIR)
        assert len(images) == 4
        for name, b64, mime, path in images:
            assert name.endswith(".png")
            assert len(b64) > 0
            assert mime == "image/png"
            assert path.exists()


# ── Model Detection Tests ─────────────────────────────────────────


class TestModelUtils:
    MODELS = [
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3-VL-2B-Instruct",
        "OpenGVLab/InternVL2_5-2B",
        "microsoft/Phi-3.5-vision-instruct",
        "mistralai/Ministral-3-3B-Instruct-2512",
        "deepseek-ai/deepseek-vl2-tiny",
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    ]

    def test_is_vision_model_all_true(self):
        from ahu_paimon_toolkit.vlm.model_utils import is_vision_model

        for model_id in self.MODELS:
            assert is_vision_model(model_id), f"{model_id} should be detected as VLM"

    def test_model_short_name(self):
        from ahu_paimon_toolkit.vlm.model_utils import model_short_name

        assert model_short_name("Qwen/Qwen3.5-2B") == "Qwen_Qwen3.5-2B"
        assert model_short_name("OpenGVLab/InternVL2_5-2B") == "OpenGVLab_InternVL2_5-2B"
        assert model_short_name("simple-model") == "simple-model"

    def test_model_short_name_no_spaces(self):
        from ahu_paimon_toolkit.vlm.model_utils import model_short_name

        for model_id in self.MODELS:
            short = model_short_name(model_id)
            assert "/" not in short
            assert " " not in short


# ── Frame Diff Tests ──────────────────────────────────────────────


class TestFrameDiff:
    def test_identical_frames_mse_zero(self):
        from ahu_paimon_toolkit.capture.frame_diff import compute_mse

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert compute_mse(frame, frame) == 0.0

    def test_different_frames_mse_positive(self):
        from ahu_paimon_toolkit.capture.frame_diff import compute_mse

        frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 128, dtype=np.uint8)
        mse = compute_mse(frame_a, frame_b)
        assert mse > 0
        assert abs(mse - 128 * 128) < 1

    def test_identical_frames_ssim_zero(self):
        from ahu_paimon_toolkit.capture.frame_diff import compute_ssim

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ssim = compute_ssim(frame, frame)
        assert ssim < 0.01  # 1 - SSIM should be ~0

    def test_compute_diff_mse(self):
        from ahu_paimon_toolkit.capture.frame_diff import compute_diff

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert compute_diff(frame, frame, "mse") == 0.0

    def test_compute_diff_ssim(self):
        from ahu_paimon_toolkit.capture.frame_diff import compute_diff

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert compute_diff(frame, frame, "ssim") < 0.01

    def test_compute_diff_auto_resize(self):
        from ahu_paimon_toolkit.capture.frame_diff import compute_diff

        frame_a = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame_b = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        diff = compute_diff(frame_a, frame_b, "mse")
        assert diff >= 0

    def test_compute_diff_invalid_method(self):
        from ahu_paimon_toolkit.capture.frame_diff import compute_diff

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown diff method"):
            compute_diff(frame, frame, "invalid")


# ── VLM Client Tests (Mocked) ────────────────────────────────────


class TestAsyncVLMClient:
    def test_describe_frame_payload(self):
        """Verify the VLM client sends correct payload format."""
        from ahu_paimon_toolkit.vlm.client import AsyncVLMClient
        from ahu_paimon_toolkit.models import KeyFrame

        client = AsyncVLMClient(
            base_url="http://localhost:8000/v1",
            model="test-model",
            prompt="Describe this.",
        )

        keyframe = KeyFrame(
            frame_id=0,
            timestamp_ms=500,
            base64_image="dGVzdA==",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "A test scene."}}]
        }
        mock_response.raise_for_status = MagicMock()

        async def run():
            with patch.object(client, "_ensure_client") as mock_ensure:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_ensure.return_value = mock_client

                desc = await client.describe_frame(keyframe)

                call_args = mock_client.post.call_args
                payload = call_args.kwargs["json"]
                assert payload["model"] == "test-model"
                assert payload["messages"][0]["content"][0]["text"] == "Describe this."
                assert "base64" in payload["messages"][0]["content"][1]["image_url"]["url"]

                assert desc.frame_id == 0
                assert desc.timestamp_ms == 500
                assert desc.description == "A test scene."

        asyncio.run(run())


# ── LLM Judge Tests (Mocked) ─────────────────────────────────────


class TestLLMJudge:
    def test_evaluate_returns_structured_score(self):
        from ahu_paimon_toolkit.evaluation.judge import LLMJudge

        judge = LLMJudge(api_key="test-key", api_base_url="https://api.test.com")

        mock_judge_response = json.dumps({
            "dimension_scores": {
                "core_understanding": 2,
                "key_information_coverage": 1,
                "task_completion": 2,
                "assistant_value": 1,
                "hallucination_control": 2,
            },
            "total_score": 8,
            "strengths": ["Good threat identification"],
            "weaknesses": ["Missed some HUD details"],
            "missing_points": ["Player level"],
            "hallucinations": [],
        })

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": mock_judge_response}}]
        }
        mock_response.raise_for_status = MagicMock()

        async def run():
            with patch.object(judge, "_ensure_client") as mock_ensure:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_ensure.return_value = mock_client

                score = await judge.evaluate(
                    asset_id="01_Minecraft",
                    model_id="test-vlm",
                    prompt_mode="A_description",
                    vlm_response="The player is in a cave with a zombie.",
                    task_definition={"scene_summary": "Cave combat", "evaluation_intent": "Test"},
                    reference_answer={"minimum_expected_points": ["zombie"], "good_additional_points": []},
                    scoring_rubric={
                        "dimensions": [{"name": "core_understanding", "description": "...", "criteria": {}}],
                        "hard_penalties": [],
                    },
                    grading_prompt="Grade this response.",
                )

                assert score.asset_id == "01_Minecraft"
                assert score.model_id == "test-vlm"
                assert score.total_score == 8
                assert score.dimension_scores["core_understanding"] == 2
                assert len(score.strengths) == 1
                assert len(score.hallucinations) == 0

        asyncio.run(run())


# ── Score Aggregator Tests ────────────────────────────────────────


class TestScoreAggregator:
    def test_basic_aggregation(self):
        from ahu_paimon_toolkit.evaluation.scoring import ScoreAggregator
        from ahu_paimon_toolkit.models import JudgeScore

        scores = [
            JudgeScore(
                asset_id="01", model_id="m1", prompt_mode="A_description",
                dimension_scores={"d1": 2, "d2": 1}, total_score=7,
            ),
            JudgeScore(
                asset_id="01", model_id="m1", prompt_mode="A_description",
                dimension_scores={"d1": 1, "d2": 2}, total_score=9,
            ),
        ]

        agg = ScoreAggregator(scores)
        assert agg.count == 2
        assert agg.mean_total() == 8.0
        assert agg.stdev_total() > 0

        dims = agg.mean_per_dimension()
        assert dims["d1"] == 1.5
        assert dims["d2"] == 1.5

    def test_ci_95(self):
        from ahu_paimon_toolkit.evaluation.scoring import ScoreAggregator
        from ahu_paimon_toolkit.models import JudgeScore

        scores = [
            JudgeScore(asset_id="01", model_id="m1", prompt_mode="A", total_score=s)
            for s in [7, 8, 7, 9, 8]
        ]
        agg = ScoreAggregator(scores)
        lo, hi = agg.ci_95_total()
        assert lo < agg.mean_total() < hi
        assert lo > 5
        assert hi < 10
