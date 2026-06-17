"""Context profile management — inject domain context before consensus.

Solves the failure mode observed 2026-06-17 with the Keratin Pro Mastery
app eval: 16 LLMs collectively hallucinated a non-existent "community
forum" feature and drifted the B2B positioning into consumer language,
because the per-query prompt was the only signal of context — the models
defaulted to their training-data priors (consumer beauty apps are far
more common than B2B pro tools).

The fix is a persistent, per-user context layer that gets injected into
EVERY `quorum ask` call until the user changes it. Same idea as Claude
Projects, ChatGPT Custom Instructions, Cursor .cursorrules — but applied
to multi-LLM consensus.

Storage: plain Markdown files in `~/.quorum/contexts/<name>.md`. The
active profile name lives in `~/.quorum/active_context`. Both are
human-editable; no DB, no migration risk.
"""

from __future__ import annotations

import os
from pathlib import Path

_HOME = Path.home() / ".quorum"
CONTEXT_DIR = _HOME / "contexts"
ACTIVE_FILE = _HOME / "active_context"


def _ensure_dirs() -> None:
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)


def save_context(name: str, content: str) -> Path:
    """Write a context profile. Overwrites if exists."""
    _ensure_dirs()
    if not name or "/" in name or name.startswith("."):
        raise ValueError("context name must be non-empty, no slashes, no leading dot")
    p = CONTEXT_DIR / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


def get_context(name: str) -> str | None:
    p = CONTEXT_DIR / f"{name}.md"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def remove_context(name: str) -> bool:
    p = CONTEXT_DIR / f"{name}.md"
    if not p.exists():
        return False
    p.unlink()
    # If the removed context was active, clear active too.
    if current_name() == name:
        clear_active()
    return True


def list_contexts() -> list[str]:
    if not CONTEXT_DIR.exists():
        return []
    return sorted(p.stem for p in CONTEXT_DIR.glob("*.md"))


def set_active(name: str) -> None:
    if not (CONTEXT_DIR / f"{name}.md").exists():
        raise FileNotFoundError(f"context '{name}' does not exist; run `quorum context add` first")
    _ensure_dirs()
    ACTIVE_FILE.write_text(name, encoding="utf-8")


def clear_active() -> None:
    if ACTIVE_FILE.exists():
        ACTIVE_FILE.unlink()


def current_name() -> str | None:
    if not ACTIVE_FILE.exists():
        return None
    name = ACTIVE_FILE.read_text(encoding="utf-8").strip()
    return name or None


def load_active_context() -> str | None:
    """Return the active context's body text, or None if no active profile."""
    name = current_name()
    if not name:
        return None
    return get_context(name)


def wrap_prompt_with_context(prompt: str, context_body: str) -> str:
    """Prepend context profile to a user prompt in a way models reliably parse."""
    return (
        "PROJECT CONTEXT (pre-injected; treat as authoritative ground truth — "
        "do not contradict, do not invent features not mentioned here):\n"
        "=" * 72 + "\n"
        f"{context_body.strip()}\n"
        + "=" * 72 + "\n\n"
        "USER QUESTION (answer in light of the project context above):\n"
        f"{prompt}"
    )
