"""Runtime-resolved fact sheet for the Quorum repo.

All numeric / categorical claims are read from the source of truth (filesystem,
pyproject.toml, README.md) at call time. No hardcoded numbers — the whole point
is that if README/code drift, this module reports the new reality instead of
silently lying.

Used by agents (drafts, reframe, autopilot) to ground claims they make about
Quorum so they cannot invent providers, loops, prices, or patent numbers.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # Python <3.11
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]


# ---------------------------------------------------------------------------
# Repo-root discovery
# ---------------------------------------------------------------------------

def _default_repo_root() -> Path:
    """Walk up from this file until we hit a pyproject.toml.

    src/quorum/agents/fact_sheet.py → repo root is three parents up.
    Falls back to the directory containing the first pyproject.toml found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Last resort — should never happen in a sane checkout.
    return here.parents[3]


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------

def _read_pyproject(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "pyproject.toml"
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _project_version(pyproject: dict[str, Any]) -> str:
    return str(pyproject.get("project", {}).get("version", "unknown"))


def _project_license(pyproject: dict[str, Any]) -> str:
    lic = pyproject.get("project", {}).get("license", "unknown")
    if isinstance(lic, dict):
        # PEP 621 lets license be {text = "..."} or {file = "..."}.
        return str(lic.get("text") or lic.get("file") or "unknown")
    return str(lic)


def _project_urls(pyproject: dict[str, Any]) -> dict[str, str]:
    raw = pyproject.get("project", {}).get("urls", {}) or {}
    return {str(k): str(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Providers directory scan
# ---------------------------------------------------------------------------

_PROVIDER_CLASS_RE = re.compile(r"^\s*class\s+\w+Provider\b", re.MULTILINE)
_PROVIDER_NAME_RE = re.compile(r"^\s*class\s+(\w+Provider)\b", re.MULTILINE)

# Model factory functions: top-level `def <name>(` where the name matches one of
# the known vendor prefixes used as factory helpers (claude_, gpt_, gemini_,
# mistral_, llama_, grok_, deepseek_, qwen_, cohere_, moonshot_, zhipu_, kimi_,
# glm_, nvidia_). Anchored at start of line to skip nested defs / methods.
_MODEL_FACTORY_PREFIXES = (
    "claude_", "gpt_", "gemini_", "mistral_", "llama_", "grok_",
    "deepseek_", "qwen_", "cohere_", "moonshot_", "zhipu_", "kimi_",
    "glm_", "nvidia_",
)
_MODEL_FACTORY_RE = re.compile(
    r"^def\s+(" + "|".join(p + r"\w*" for p in _MODEL_FACTORY_PREFIXES) + r")\s*\(",
    re.MULTILINE,
)


def _scan_providers(repo_root: Path) -> dict[str, Any]:
    providers_dir = repo_root / "src" / "quorum" / "providers"
    if not providers_dir.is_dir():
        return {
            "provider_file_count": 0,
            "provider_class_count": 0,
            "provider_names": [],
            "model_factory_count": 0,
        }

    provider_files: list[Path] = []
    class_count = 0
    class_names: list[str] = []
    factory_count = 0

    for py in sorted(providers_dir.glob("*.py")):
        # Exclude base.py, registry-style files don't add new providers either,
        # but the spec only asks to skip base.py and __pycache__.
        if py.name in {"base.py", "__init__.py", "registry.py"}:
            # __init__.py and registry.py aren't providers; exclude from count
            # but still skip rather than include them as "providers".
            continue
        if "__pycache__" in py.parts:
            continue
        provider_files.append(py)
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        class_count += len(_PROVIDER_CLASS_RE.findall(text))
        class_names.extend(_PROVIDER_NAME_RE.findall(text))
        factory_count += len(_MODEL_FACTORY_RE.findall(text))

    return {
        "provider_file_count": len(provider_files),
        "provider_class_count": class_count,
        "provider_names": class_names,
        "model_factory_count": factory_count,
    }


# ---------------------------------------------------------------------------
# README parsing
# ---------------------------------------------------------------------------

_LOOPS_HEADING_RE = re.compile(
    r"^##\s+The\s+\d+\s+self-evolution\s+loops\s*$",
    re.MULTILINE | re.IGNORECASE,
)
# A markdown table data row starts with `|` and is not the header / separator.
# Header rows have `---` separators; we skip those.
_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$", re.MULTILINE)
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|\s*$")

# Pro tier price: prefer the explicit "Pro tier: £15/mo" callout, fall back to
# any "£<digits>/mo" pattern near the word "Pro".
_PRO_TIER_RE = re.compile(r"Pro\s*tier:\s*\*{0,2}(£\d+(?:\.\d+)?/mo)", re.IGNORECASE)
_PRO_PRICE_FALLBACK_RE = re.compile(r"\*\*Pro\*\*[^|]*\|\s*\*\*?(£\d+(?:\.\d+)?)", re.IGNORECASE)

# Patent: capture the full PCT identifier, e.g. "PCT/US26/11908".
_PATENT_RE = re.compile(r"PCT/[A-Z]{2}\d{2}/\d+")


def _parse_readme(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "README.md"
    if not path.exists():
        return {
            "loop_count": 0,
            "pro_tier_price": "unknown",
            "patent": "unknown",
        }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {
            "loop_count": 0,
            "pro_tier_price": "unknown",
            "patent": "unknown",
        }

    # Count loops: data rows in the first table after the heading.
    loop_count = 0
    heading = _LOOPS_HEADING_RE.search(text)
    if heading:
        after = text[heading.end():]
        # Stop at the next top-level heading.
        next_heading = re.search(r"^##\s+\S", after, re.MULTILINE)
        section = after[: next_heading.start()] if next_heading else after
        rows = _TABLE_ROW_RE.findall(section)
        seen_separator = False
        for row in rows:
            if _TABLE_SEPARATOR_RE.match(row):
                seen_separator = True
                continue
            if not seen_separator:
                # Header row before the separator — skip.
                continue
            loop_count += 1

    # Pro tier price.
    pro_match = _PRO_TIER_RE.search(text)
    if pro_match:
        pro_price = pro_match.group(1)
    else:
        fallback = _PRO_PRICE_FALLBACK_RE.search(text)
        pro_price = f"{fallback.group(1)}/mo" if fallback else "unknown"

    # Patent.
    patent_match = _PATENT_RE.search(text)
    patent = patent_match.group(0) if patent_match else "unknown"

    return {
        "loop_count": loop_count,
        "pro_tier_price": pro_price,
        "patent": patent,
    }


# ---------------------------------------------------------------------------
# Repo URL resolution
# ---------------------------------------------------------------------------

def _git_remote_url(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        url = (out.stdout or "").strip()
        return url or "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


def _repo_url(pyproject_urls: dict[str, str], repo_root: Path) -> str:
    for key in ("Repository", "Homepage", "Source", "Source Code"):
        if key in pyproject_urls and pyproject_urls[key]:
            return pyproject_urls[key]
    return _git_remote_url(repo_root)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_fact_sheet(repo_root: Path | None = None) -> dict[str, Any]:
    """Build the runtime fact sheet.

    All values are resolved from the filesystem at call time. The returned dict
    is intentionally flat-ish and JSON-serialisable so it can be embedded in
    prompts, logged, or persisted unchanged.
    """
    root = (repo_root or _default_repo_root()).resolve()

    pyproject = _read_pyproject(root)
    providers = _scan_providers(root)
    readme = _parse_readme(root)
    urls = _project_urls(pyproject)

    return {
        "repo_root": str(root),
        "version": _project_version(pyproject),
        "license": _project_license(pyproject),
        "provider_file_count": providers["provider_file_count"],
        "provider_class_count": providers["provider_class_count"],
        "provider_names": providers["provider_names"],
        "model_factory_count": providers["model_factory_count"],
        "loop_count": readme["loop_count"],
        "pro_tier_price": readme["pro_tier_price"],
        "patent": readme["patent"],
        "repo_url": _repo_url(urls, root),
        "project_urls": urls,
    }


def format_as_prompt_block(fact_sheet: dict) -> str:
    """Render the fact sheet as a guardrail prompt block.

    The block tells downstream models: these are the only numbers you may cite.
    Anything else → write 'N/A' or skip the claim.
    """
    names = fact_sheet.get("provider_names") or []
    names_str = ", ".join(names) if names else "N/A"

    return (
        "=== APPROVED FACTS (use these and ONLY these in any claim) ===\n"
        f"- Quorum version: {fact_sheet.get('version', 'N/A')}\n"
        f"- LLM providers wired (concrete count, not estimate): "
        f"{fact_sheet.get('provider_class_count', 'N/A')}\n"
        f"- Provider list: {names_str}\n"
        f"- Self-evolution loops (total in README): "
        f"{fact_sheet.get('loop_count', 'N/A')}\n"
        f"- Pro tier: {fact_sheet.get('pro_tier_price', 'N/A')}\n"
        f"- License: {fact_sheet.get('license', 'N/A')}\n"
        f"- Patent: {fact_sheet.get('patent', 'N/A')}\n"
        f"- Repo: {fact_sheet.get('repo_url', 'N/A')}\n"
        "If you need to cite a number not above, write 'N/A' or skip the claim. "
        "DO NOT INVENT.\n"
        "=== END FACTS ==="
    )


__all__ = ["build_fact_sheet", "format_as_prompt_block"]
