"""Template-based report generation with optional DeepSeek analysis.

Reports are built by:
1. Loading a markdown template from ``templates/``.
2. Substituting ``{{placeholder}}`` tokens with raw data tables/stats.
3. Sending the filled template to DeepSeek for analysis sections.
4. Writing the final combined report to disk.

If the DeepSeek API is unavailable, placeholders that require AI
analysis are filled with *"(analysis pending)"*.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger as lg

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEMPLATES_DIR = _PROJECT_ROOT / "templates"


def _load_template(template_name: str) -> str:
    """Read a template file from the templates/ directory."""
    path = _TEMPLATES_DIR / template_name
    if not path.exists():
        lg.warning("Template not found: {}", path)
        return ""
    return path.read_text(encoding="utf-8")


def fill_template(template_text: str, data: dict[str, str]) -> str:
    """Replace ``{{key}}`` placeholders with values from *data*.

    Keys present in the template but missing from *data* are left
    unchanged so the caller can see what was not filled.
    """
    def _replace(m: re.Match) -> str:
        key = m.group(1).strip()
        return data.get(key, m.group(0))

    return re.sub(r"\{\{(.+?)\}\}", _replace, template_text)


def generate_report_via_deepseek(
    filled_template: str,
    *,
    language: str = "中文",
    timeout: float = 120.0,
) -> str:
    """Send the data-filled template to DeepSeek and ask it to write analysis.

    The template contains section headers and raw data; DeepSeek is asked
    to produce a full report following the template structure.

    Returns the AI-generated report text, or the original template with
    a note if the API call fails.
    """
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_base = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        lg.warning("DEEPSEEK_API_KEY not set; skipping AI analysis")
        return filled_template

    system_prompt = (
        f"你是一位专业的 AI/ML 实验分析师。请根据以下模板和实验数据，用{language}"
        "撰写完整的实验报告。保留模板的 Markdown 结构和标题层级。"
        "对于标记为 <!-- ANALYSIS --> 的部分，根据数据写出深入的专业分析，"
        "而不是简单罗列数字。其他部分保持原样。"
    )

    try:
        with httpx.Client(
            base_url=api_base,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": filled_template},
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        lg.exception("DeepSeek report generation failed; returning raw template")
        return filled_template


def write_report(path: Path, content: str) -> None:
    """Write report text to *path*, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    lg.info("Report written: {}", path)


def generate_report_from_template(
    template_name: str,
    data: dict[str, str],
    output_path: Path,
    *,
    use_deepseek: bool = True,
    language: str = "中文",
) -> str:
    """End-to-end: load template -> fill data -> (optional) DeepSeek -> write.

    Returns the final report text.
    """
    template = _load_template(template_name)
    if not template:
        lg.error("Cannot generate report: template '{}' not found", template_name)
        return ""

    filled = fill_template(template, data)

    if use_deepseek:
        final = generate_report_via_deepseek(filled, language=language)
    else:
        final = filled

    write_report(output_path, final)
    return final
