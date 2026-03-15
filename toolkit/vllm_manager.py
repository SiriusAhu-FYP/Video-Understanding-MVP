"""WSL-based vLLM service lifecycle management.

Provides functions to create launch scripts, start/stop vLLM services
running inside WSL, and poll for readiness.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import PurePosixPath

from loguru import logger as lg

VLLM_SERVER_DIR = PurePosixPath("/home/playerAhu/vLLM_server")
VLLM_VENV_ACTIVATE = str(VLLM_SERVER_DIR / ".venv" / "bin" / "activate")

DEFAULT_VLLM_ARGS: dict[str, str | int | float] = {
    "trust-remote-code": True,
    "max-model-len": 8192,
    "gpu-memory-utilization": 0.5,
    "enable-prefix-caching": True,
}

MODEL_SPECIFIC_ARGS: dict[str, dict] = {
    "Qwen/Qwen3.5-2B": {},
    "Qwen/Qwen3.5-0.8B": {},
    "Qwen/Qwen3-VL-2B-Instruct": {},
    "OpenGVLab/InternVL2_5-2B": {
        "max-model-len": 4096,
    },
    "Zero-Point-AI/MARTHA-2B": {
        "max-model-len": 4096,
    },
    "mistralai/Ministral-3-3B-Instruct-2512": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "max-model-len": 8192,
    },
    "deepseek-ai/deepseek-vl2-tiny": {
        "hf-overrides": '\'{"architectures": ["DeepseekVLV2ForCausalLM"]}\'',
        "max-model-len": 4096,
    },
    "vikhyatk/moondream2": {
        "max-model-len": 2048,
    },
}


def _sanitize_script_name(model_id: str) -> str:
    """Convert model ID to a safe filename for the launch script."""
    return "run_" + model_id.replace("/", "_").replace(" ", "_") + ".sh"


def build_vllm_command(model_id: str, extra_args: dict | None = None) -> str:
    """Build the vllm serve command string for a given model."""
    args = {**DEFAULT_VLLM_ARGS}
    if model_id in MODEL_SPECIFIC_ARGS:
        args.update(MODEL_SPECIFIC_ARGS[model_id])
    if extra_args:
        args.update(extra_args)

    parts = [f'vllm serve "{model_id}"']
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                parts.append(f"    --{key}")
        else:
            parts.append(f"    --{key} {value}")

    return " \\\n".join(parts)


def create_launch_script(model_id: str, extra_args: dict | None = None) -> str:
    """Create a vLLM launch script in WSL and return its WSL path."""
    script_name = _sanitize_script_name(model_id)
    script_path = str(VLLM_SERVER_DIR / script_name)

    vllm_cmd = build_vllm_command(model_id, extra_args)
    script_content = f"source {VLLM_VENV_ACTIVATE}\n{vllm_cmd}\n"

    subprocess.run(
        ["wsl", "bash", "-c",
         f"tr -d '\\r' > {script_path} && chmod +x {script_path}"],
        input=script_content,
        text=True, check=True, timeout=10,
    )

    lg.info("已创建 vLLM 启动脚本: {}", script_path)
    return script_path


_VLLM_LOG_PATH = str(VLLM_SERVER_DIR / "vllm_output.log")


def start_vllm(model_id: str, extra_args: dict | None = None) -> subprocess.Popen:
    """Start vLLM service in WSL for the given model.

    Uses HF_HUB_OFFLINE=1 to avoid network issues in WSL.
    Returns the Popen handle for the background process.
    """
    script_path = create_launch_script(model_id, extra_args)

    lg.info("正在启动 vLLM 服务: {} ...", model_id)
    proc = subprocess.Popen(
        [
            "wsl",
            "bash",
            "-c",
            f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash {script_path} "
            f"> {_VLLM_LOG_PATH} 2>&1",
        ],
    )
    lg.info("vLLM 进程已启动 (PID: {})", proc.pid)
    return proc


def read_vllm_log(tail: int = 80) -> str:
    """Read the last N lines of the vLLM output log."""
    try:
        result = subprocess.run(
            ["wsl", "bash", "-c", f"tail -n {tail} {_VLLM_LOG_PATH}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout
    except Exception:
        return ""


def stop_vllm(proc: subprocess.Popen | None = None) -> None:
    """Stop vLLM service. Tries graceful then forceful termination."""
    if proc is not None and proc.poll() is None:
        lg.info("正在终止 vLLM 进程 (PID: {}) ...", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=15)
            lg.info("vLLM 进程已正常终止")
        except subprocess.TimeoutExpired:
            lg.warning("vLLM 进程未响应终止信号，强制杀死")
            proc.kill()
            proc.wait()
        return

    # Fallback: kill any vllm process in WSL
    lg.info("尝试通过 WSL 终止所有 vLLM 进程 ...")
    try:
        subprocess.run(
            ["wsl", "bash", "-c", "pkill -f 'vllm serve' || true"],
            timeout=10,
        )
        time.sleep(2)
        lg.info("已发送终止信号")
    except Exception as e:
        lg.warning("终止 vLLM 失败: {}", e)


def wait_for_ready(
    base_url: str = "http://localhost:8000/v1",
    timeout_s: float = 300.0,
    poll_interval_s: float = 15.0,
) -> str:
    """Poll vLLM until it's ready and return the detected model ID."""
    from toolkit.common import wait_for_vllm_ready

    return wait_for_vllm_ready(base_url, timeout_s, poll_interval_s)
