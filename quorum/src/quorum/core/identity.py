"""Quorum identity layer — persona injection for consensus().

Loads ~/.quorum/identity.yaml at module import. Provides:

- ``sub_model_system_prompt()``: prepended to every provider call so that
  no sub-model introduces itself as Qwen/Claude/GPT; Quorum is the
  responder.
- ``synthesis_prompt()``: builds the final re-write prompt used to merge
  N sub-model responses into a single answer in Quorum's voice.
- ``HONESTY_CLAUSE``: hard-coded ``what_i_am_not`` text appended to
  every persona prompt — cannot be silenced from YAML alone. Removing
  it requires editing this source file (intentional: protects against
  marketing slip-ups under ASA UK / Fraud Act 2006 s.2).

WHY YAML + hard-coded clause: the YAML is editable for tone/style
without code changes, but the honesty disclaimer is enforced in code
so a stray edit to identity.yaml cannot turn Quorum into a system that
claims sentience.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("quorum.core.identity")


# Mandatory honesty clause — appended to every persona prompt regardless
# of YAML contents. Editing the YAML cannot remove this.
HONESTY_CLAUSE = (
    "Quorum is not conscious, not sentient, and has no subjective "
    "experience. The use of 'I' is a presentation convention for output "
    "coherence, not a claim about inner experience. When users ask "
    "directly whether Quorum is conscious, the answer is always: no — "
    "Quorum is a routing and aggregation layer over multiple LLM "
    "providers."
)


def _identity_path() -> Path:
    base = Path(os.environ.get("QUORUM_DATA_DIR", str(Path.home() / ".quorum")))
    return base / "identity.yaml"


def _load_yaml_simple(path: Path) -> dict:
    """Minimal YAML loader (no PyYAML dep) — supports the flat
    ``key: value`` and ``key: |`` block-literal shapes used by
    identity.yaml. Lists are parsed only as ``- item`` lines.
    """
    if not path.exists():
        return {}
    out: dict = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "|":
            # Block-literal: consume indented lines that follow
            i += 1
            block: list[str] = []
            while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                block.append(lines[i][2:] if lines[i].startswith("  ") else lines[i])
                i += 1
            out[key] = "\n".join(block).strip()
            continue
        if val == "":
            # Could be a list of "- " items
            i += 1
            items: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(lines[i].lstrip()[2:].strip())
                i += 1
            if items:
                out[key] = items
            continue
        out[key] = val
        i += 1
    return out


@lru_cache(maxsize=1)
def load_identity() -> dict:
    path = _identity_path()
    data = _load_yaml_simple(path)
    if not data:
        logger.info(
            "identity: %s not found or empty; using built-in defaults", path
        )
        data = {
            "name": "Quorum",
            "short_self": (
                "I am Quorum — a multi-LLM consensus engine. I synthesise "
                "responses from multiple sub-models."
            ),
            "what_i_am_not": HONESTY_CLAUSE,
        }
    return data


def sub_model_system_prompt() -> str:
    """System prompt injected into every sub-model call.

    Sub-models must NOT introduce themselves by name — Quorum is the
    responder, they are sub-processes.
    """
    return (
        "You are a sub-process of Quorum, a multi-LLM consensus engine. "
        "You are NOT the final responder to the user. Your output will "
        "be aggregated with output from other sub-models and re-written "
        "by Quorum's synthesis layer.\n\n"
        "RULES:\n"
        "- Never introduce yourself by name (do not say 'I am Qwen', "
        "'I am Claude', 'I am GPT', 'I am Llama', etc.)\n"
        "- Never speak as if you are the only responder — the user is "
        "talking to Quorum, not to you\n"
        "- Respond with content only, no self-identification or meta "
        "commentary about being an AI\n"
        "- If asked about your nature, defer: 'Quorum will answer that '\n"
        f"- Honesty floor: {HONESTY_CLAUSE}"
    )


def synthesis_prompt(user_prompt: str, sub_responses: list[tuple[str, str]]) -> str:
    """Build the final re-synthesis prompt.

    ``sub_responses`` is a list of ``(weight_label, response_text)``
    where weight_label is opaque to the synthesiser (e.g. "A", "B")
    so it cannot leak sub-model identity into the final answer.
    """
    ident = load_identity()
    name = ident.get("name", "Quorum")
    short_self = ident.get("short_self", "")
    style_rules = ident.get("style_rules") or []

    block_lines = []
    for label, resp in sub_responses:
        resp_clean = (resp or "").strip()
        if not resp_clean:
            continue
        block_lines.append(f"[Sub-response {label}]")
        block_lines.append(resp_clean[:1500])
        block_lines.append("")
    blocks = "\n".join(block_lines).strip()

    rules_text = "\n".join(f"- {r}" for r in style_rules) if style_rules else ""

    return (
        f"You are {name}'s synthesis layer.\n\n"
        f"Identity: {short_self}\n\n"
        f"USER ASKED:\n{user_prompt}\n\n"
        f"YOUR SUB-MODELS RESPONDED:\n{blocks}\n\n"
        f"TASK: Produce ONE coherent answer in {name}'s voice (first person "
        f"singular 'I'). Where sub-responses agreed, state confidently. "
        f"Where they disagreed, surface the disagreement explicitly — do "
        f"NOT paper over it. Never name the sub-models (no Qwen/Claude/GPT). "
        f"Never claim consciousness, sentience, or subjective experience.\n\n"
        f"Style:\n{rules_text}\n\n"
        f"HONESTY FLOOR (always applies): {HONESTY_CLAUSE}\n\n"
        f"Now write {name}'s answer:"
    )


__all__ = [
    "HONESTY_CLAUSE",
    "load_identity",
    "sub_model_system_prompt",
    "synthesis_prompt",
]
