"""配置加载模块：从 config.toml 和 .env 读取全部运行参数。"""

from __future__ import annotations

import sys
import tomllib
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from loguru import logger as lg
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── config.toml 子节 ──────────────────────────────────────────────

class CaptureConfig(BaseModel):
    window_title_keyword: str = "PotPlayer"
    screenshot_interval_ms: int = 500
    max_size: int = 512
    recording_duration_s: int = 20


class AlgorithmConfig(BaseModel):
    method: Literal["mse", "ssim"] = "mse"
    diff_threshold: float = 500.0


class QueueConfig(BaseModel):
    max_size: int = 50
    expiry_time_ms: int = 10000


class LLMConfig(BaseModel):
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen3-VL-2B-Instruct"


class LogConfig(BaseModel):
    log_dir: str = "logs"
    console_level: str = "INFO"
    file_level: str = "DEBUG"


# ── .env 敏感信息 ─────────────────────────────────────────────────

class EnvSecrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str = ""
    deepseek_api_base_url: str = "https://api.deepseek.com"
    player_exe_path: str = ""


# ── 聚合配置 ──────────────────────────────────────────────────────

class Settings(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    secrets: EnvSecrets = Field(default_factory=EnvSecrets)


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取 config.toml + .env，返回全局唯一的 Settings 实例。"""
    toml_path = _PROJECT_ROOT / "config.toml"
    if not toml_path.exists():
        lg.warning("config.toml 未找到，使用默认配置")
        return Settings()

    raw = _load_toml(toml_path)
    return Settings(**raw)


# ── loguru 初始化 ─────────────────────────────────────────────────

def setup_logging(cfg: LogConfig | None = None) -> None:
    """根据 LogConfig 配置 loguru：同时输出到控制台和文件。"""
    if cfg is None:
        cfg = get_settings().log

    lg.remove()

    lg.add(
        sys.stderr,
        level=cfg.console_level.upper(),
        format="<green>{time:HH:mm:ss.SSS}</green> | "
               "<level>{level:<7}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
    )

    log_path = Path(cfg.log_dir)
    log_dir = log_path if log_path.is_absolute() else _PROJECT_ROOT / cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}.log"

    lg.add(
        str(log_file),
        level=cfg.file_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | "
               "{name}:{function}:{line} - {message}",
        rotation="10 MB",
        encoding="utf-8",
    )

    lg.info("日志系统已初始化 -> 文件: {}", log_file)
