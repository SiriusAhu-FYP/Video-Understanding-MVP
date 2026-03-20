"""Template-based report generation with section-by-section DeepSeek analysis.

Reports are built by:
1. Loading a markdown template from ``templates/``.
2. Substituting ``{{placeholder}}`` tokens with raw data tables/stats.
3. For each ``<!-- ANALYSIS -->`` section, sending the *surrounding context*
   (section title + data + instructions) to DeepSeek as an isolated request.
4. Assembling all sections and writing the final report to disk.

Each ``<!-- ANALYSIS -->`` block is followed by instructions that tell
DeepSeek what to write. The model receives:
- A *system prompt* establishing its role as an experiment analyst.
- A *user prompt* containing the section header, nearby data tables,
  and the specific instructions for that section.

If the DeepSeek API is unavailable, analysis placeholders are replaced
with ``(analysis pending)``.
"""

from __future__ import annotations

import os
import re
import tomllib
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger as lg

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _PROJECT_ROOT / "templates"

_SYSTEM_PROMPT = (
    "You are a senior AI/ML experiment analyst writing a benchmark report. "
    "You have deep expertise in Vision-Language Models (VLMs), inference "
    "optimization, and real-time gaming AI systems.\n\n"
    "Rules:\n"
    "- Write in {language}.\n"
    "- Be analytical: explain *why*, not just *what*. Correlate metrics.\n"
    "- Use concrete numbers from the provided data.\n"
    "- Keep each section focused and concise (3-6 paragraphs max).\n"
    "- Do NOT repeat raw data tables — they are already in the report.\n"
    "- Do NOT add markdown headers — the section header is already set.\n"
    "- Output ONLY the analysis text, no preamble or sign-off."
)


def _load_config() -> dict:
    """Load reporting config from config_benchmark.toml."""
    cfg_path = _PROJECT_ROOT / "config_benchmark.toml"
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            return tomllib.load(f).get("reporting", {})
    return {}


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


# ── Section-level analysis ───────────────────────────────────────


_ANALYSIS_PATTERN = re.compile(
    r"(<!-- ANALYSIS -->\n)(.*?)(?=\n##|\n<!-- ANALYSIS -->|\Z)",
    re.DOTALL,
)


def _extract_sections(filled_template: str) -> list[dict]:
    """Parse the filled template into sections for individual analysis.

    Returns a list of dicts:
        {
            "start": int,          # char offset of <!-- ANALYSIS --> marker
            "end": int,            # char offset of section end
            "instructions": str,   # the text after <!-- ANALYSIS -->
            "context_before": str, # ~2000 chars before the marker (data context)
            "header": str,         # the ## header this section belongs to
        }
    """
    sections = []
    for m in _ANALYSIS_PATTERN.finditer(filled_template):
        marker_start = m.start()
        section_end = m.end()
        instructions = m.group(2).strip()

        context_start = max(0, marker_start - 2000)
        context_before = filled_template[context_start:marker_start]

        header = ""
        header_match = re.findall(r"^(##+ .+)$", context_before, re.MULTILINE)
        if header_match:
            header = header_match[-1]

        sections.append({
            "start": marker_start,
            "end": section_end,
            "instructions": instructions,
            "context_before": context_before,
            "header": header,
        })
    return sections


def _call_deepseek(
    user_prompt: str,
    *,
    language: str = "中文",
    model: str = "deepseek-chat",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: float = 120.0,
) -> str | None:
    """Make a single DeepSeek API call and return the response text."""
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_base = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        return None

    system = _SYSTEM_PROMPT.format(language=language)

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
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        lg.exception("DeepSeek API call failed")
        return None


def _build_section_prompt(section: dict) -> str:
    """Build a focused prompt for one analysis section."""
    parts = []
    if section["header"]:
        parts.append(f"You are writing the analysis for the report section: {section['header']}")
    parts.append("")
    parts.append("Here is the relevant data context from the report:")
    parts.append("---")
    parts.append(section["context_before"][-1500:])
    parts.append("---")
    parts.append("")
    parts.append("Instructions for this section:")
    parts.append(section["instructions"])
    parts.append("")
    parts.append(
        "Write a thorough, data-driven analysis. Reference specific numbers. "
        "Explain implications for real-time gaming AI companion deployment."
    )
    return "\n".join(parts)


def generate_report_via_deepseek(
    filled_template: str,
    *,
    language: str = "中文",
    timeout: float = 120.0,
) -> str:
    """Analyze each <!-- ANALYSIS --> section individually via DeepSeek.

    Each section is sent as a separate API call with focused context,
    producing higher-quality analysis than a single monolithic request.
    """
    cfg = _load_config()
    model = cfg.get("deepseek_model", "deepseek-chat")
    max_tokens = cfg.get("max_tokens_per_section", 1024)
    temperature = cfg.get("temperature", 0.3)

    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")

    if not api_key:
        lg.warning("DEEPSEEK_API_KEY not set; skipping AI analysis")
        return _strip_analysis_markers(filled_template)

    sections = _extract_sections(filled_template)
    if not sections:
        return filled_template

    lg.info("Generating analysis for {} sections via DeepSeek ...", len(sections))

    result = filled_template
    offset = 0

    for i, section in enumerate(sections):
        header_short = section["header"] or f"section {i + 1}"
        lg.info("  [{}/{}] Analyzing: {}", i + 1, len(sections), header_short)

        prompt = _build_section_prompt(section)
        analysis = _call_deepseek(
            prompt,
            language=language,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

        if analysis:
            marker_and_instructions = result[
                section["start"] + offset : section["end"] + offset
            ]
            replacement = analysis.strip()
            result = (
                result[: section["start"] + offset]
                + replacement
                + result[section["end"] + offset :]
            )
            offset += len(replacement) - len(marker_and_instructions)
        else:
            lg.warning("  Analysis failed for: {}", header_short)
            old = result[section["start"] + offset : section["end"] + offset]
            fallback = "*(analysis pending — DeepSeek API unavailable)*"
            result = (
                result[: section["start"] + offset]
                + fallback
                + result[section["end"] + offset :]
            )
            offset += len(fallback) - len(old)

    return result


def _strip_analysis_markers(text: str) -> str:
    """Remove <!-- ANALYSIS --> markers and their instructions when not using DeepSeek."""
    return _ANALYSIS_PATTERN.sub("*(analysis pending)*", text)


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

    cfg = _load_config()
    if use_deepseek and cfg.get("use_deepseek", True):
        final = generate_report_via_deepseek(filled, language=language)
    else:
        final = _strip_analysis_markers(filled)

    write_report(output_path, final)
    return final
