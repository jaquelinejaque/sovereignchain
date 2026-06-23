# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Quorum MCP server — stdio transport.

Exposes the Quorum multi-LLM consensus engine over Model Context Protocol
so any MCP-aware client (Claude Desktop, Cursor, Antigravity, Cline, etc.)
can call it as a native tool.

Transport: stdio only (no auth, no rate limiting in this first cut).
For paid remote access, see `quorum.server.main` (FastAPI hosted SaaS).

Tools exposed (6):
  - quorum_consensus           — multi-LLM consensus answer + confidence
  - quorum_multi               — raw per-model answers (no synthesis)
  - quorum_verdict             — yes/no verdict on a claim (with disagreement)
  - quorum_disagreement_matrix — pairwise agreement matrix across models
  - quorum_readiness_check     — EU AI Act Annex VI gap-analysis (advisory)
  - quorum_evidence_record     — generate per-query PDF evidence record

LEGAL NOTICE — every tool response includes a `_disclaimer` field that
states (verbatim): "Advisory only — not a conformity assessment under
Regulation (EU) 2024/1689. Sovereign Chain Ltd is not a Notified Body
under Article 31." This is mandatory protection against Fraud Act 2006
s.2 and CPUTR 2008 Reg 6 exposure when AI agents call this server
autonomously and downstream readers may interpret outputs as compliance
attestations rather than advisory technical material.

Usage:
    quorum-mcp                 # run server on stdio

MCP client config (e.g. ~/.gemini/antigravity-ide/mcp/quorum.json):
    {
      "command": "/path/to/quorum/.venv/bin/python",
      "args": ["-m", "quorum.mcp.server"],
      "env": {
        "ANTHROPIC_API_KEY": "...",
        "OPENAI_API_KEY": "...",
        "GEMINI_API_KEY": "...",
        "NVIDIA_API_KEY": "..."
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("quorum.mcp")

# Mandatory legal disclaimer — embedded in every response payload.
# Copy is verified against Round 4-9 of the Quorum vocabulary audit.
DISCLAIMER = (
    "Advisory only — not a conformity assessment under Regulation (EU) "
    "2024/1689. Sovereign Chain Ltd is not a Notified Body under Article 31. "
    "Outputs are technical evidence material to support, not replace, the "
    "documentation and risk-management obligations of the AI system provider "
    "(internal assessment under Annex VI) or a designated Notified Body "
    "(external assessment under Annex VII)."
)

# Per-call usage counter (in-memory, resets on restart). Wired so a future
# paywall layer can read it without a schema migration.
_call_counter: dict[str, int] = {}


def _wrap(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the mandatory legal disclaimer to every tool response."""
    return {**payload, "_disclaimer": DISCLAIMER}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tool implementations — thin wrappers over quorum.core.consensus and
# quorum.hsp.ai_act_cert. We keep them async so the MCP server stays
# non-blocking even when a provider hangs.
# ---------------------------------------------------------------------------


async def _tool_consensus(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a multi-LLM consensus query.

    Reuses quorum.core.consensus.consensus() — the same code path the CLI
    and FastAPI server use, so MCP callers get identical evolution-loop
    behaviour (RLHF, Hebbian, router) as paid hosted users.
    """
    from quorum.core.consensus import consensus

    prompt = str(arguments.get("prompt", "")).strip()
    if not prompt:
        return _wrap({"error": "prompt is required"})

    budget = float(arguments.get("budget_usd", 0.05))
    timeout_s = float(arguments.get("timeout_s", 60.0))
    user_id = arguments.get("user_id")

    _call_counter["consensus"] = _call_counter.get("consensus", 0) + 1
    try:
        result = await consensus(
            prompt,
            budget_usd=budget,
            timeout_s=timeout_s,
            user_id=user_id,
        )
    except Exception as exc:
        logger.exception("consensus failed for prompt=%r", prompt[:60])
        return _wrap({"error": f"consensus_failed: {exc}"})

    return _wrap(
        {
            "answer": result.answer,
            "confidence": float(result.confidence),
            "scoring_method": getattr(result, "scoring_method", "embedding"),
            "total_cost_usd": float(getattr(result, "total_cost_usd", 0.0)),
            "total_latency_ms": int(getattr(result, "total_latency_ms", 0)),
            "models": [
                {
                    "name": str(m.name),
                    "weight": float(getattr(m, "weight", 0.0) or 0.0),
                    "latency_ms": int(getattr(m, "latency_ms", 0) or 0),
                    "cost_usd": float(getattr(m, "cost_usd", 0.0) or 0.0),
                    "answer_preview": (str(getattr(m, "answer", ""))[:200]),
                    "error": str(getattr(m, "error", "") or ""),
                }
                for m in (result.models or [])
            ],
            "disagreements": list(getattr(result, "disagreements", []) or []),
            "generated_at": _now_iso(),
        }
    )


async def _tool_multi(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return raw per-model answers without picking a synthesized winner.

    Useful when the caller wants to see divergence directly (compliance
    review, content authoring with multiple drafts).
    """
    from quorum.core.consensus import consensus

    prompt = str(arguments.get("prompt", "")).strip()
    if not prompt:
        return _wrap({"error": "prompt is required"})

    _call_counter["multi"] = _call_counter.get("multi", 0) + 1
    try:
        result = await consensus(prompt, budget_usd=0.10, timeout_s=60.0)
    except Exception as exc:
        return _wrap({"error": f"multi_failed: {exc}"})

    return _wrap(
        {
            "prompt": prompt,
            "answers": [
                {
                    "model": str(m.name),
                    "answer": str(getattr(m, "answer", "") or ""),
                    "cost_usd": float(getattr(m, "cost_usd", 0.0) or 0.0),
                    "latency_ms": int(getattr(m, "latency_ms", 0) or 0),
                    "error": str(getattr(m, "error", "") or ""),
                }
                for m in (result.models or [])
            ],
            "model_count": len(result.models or []),
            "generated_at": _now_iso(),
        }
    )


async def _tool_verdict(arguments: dict[str, Any]) -> dict[str, Any]:
    """Yes/no verdict on a claim, with breakdown of model votes.

    The claim is wrapped in a structured prompt: "Is the following claim
    true? Answer YES or NO and give one-sentence justification." Each
    model's verdict is parsed; the consensus tally is returned.
    """
    from quorum.core.consensus import consensus

    claim = str(arguments.get("claim", "")).strip()
    if not claim:
        return _wrap({"error": "claim is required"})

    _call_counter["verdict"] = _call_counter.get("verdict", 0) + 1

    framed_prompt = (
        f"Is the following claim TRUE or FALSE?\n\nClaim: {claim}\n\n"
        "Respond with exactly one word — TRUE or FALSE — on the first line, "
        "then a one-sentence justification on the second line."
    )

    try:
        result = await consensus(framed_prompt, budget_usd=0.05, timeout_s=60.0)
    except Exception as exc:
        return _wrap({"error": f"verdict_failed: {exc}"})

    votes_true = 0
    votes_false = 0
    votes_unclear = 0
    breakdown = []
    for m in result.models or []:
        ans = str(getattr(m, "answer", "") or "").strip().upper()
        first_word = ans.split()[0] if ans else ""
        if first_word.startswith("TRUE"):
            votes_true += 1
            verdict = "TRUE"
        elif first_word.startswith("FALSE"):
            votes_false += 1
            verdict = "FALSE"
        else:
            votes_unclear += 1
            verdict = "UNCLEAR"
        breakdown.append(
            {
                "model": str(m.name),
                "verdict": verdict,
                "raw_first_line": (str(getattr(m, "answer", "") or "").split("\n")[0])[:200],
            }
        )

    total = votes_true + votes_false + votes_unclear
    consensus_verdict = "UNCLEAR"
    consensus_strength = 0.0
    if total > 0:
        if votes_true > votes_false and votes_true >= total // 2:
            consensus_verdict = "TRUE"
            consensus_strength = votes_true / total
        elif votes_false > votes_true and votes_false >= total // 2:
            consensus_verdict = "FALSE"
            consensus_strength = votes_false / total

    return _wrap(
        {
            "claim": claim,
            "consensus_verdict": consensus_verdict,
            "consensus_strength": round(consensus_strength, 3),
            "votes": {"TRUE": votes_true, "FALSE": votes_false, "UNCLEAR": votes_unclear},
            "breakdown": breakdown,
            "synthesized_answer": result.answer,
            "generated_at": _now_iso(),
        }
    )


async def _tool_disagreement_matrix(arguments: dict[str, Any]) -> dict[str, Any]:
    """Pairwise agreement matrix across all models that responded.

    Reads the same ConsensusResult and computes pairwise cosine similarity
    on embedded answers (when embedding provider available, else Jaccard
    fallback). This is the dashboard signal flagged in EU AI Act Article 14
    as the automation-bias evidence material.
    """
    from quorum.core.consensus import consensus

    prompt = str(arguments.get("prompt", "")).strip()
    if not prompt:
        return _wrap({"error": "prompt is required"})

    _call_counter["disagreement_matrix"] = _call_counter.get("disagreement_matrix", 0) + 1
    try:
        result = await consensus(prompt, budget_usd=0.10, timeout_s=60.0)
    except Exception as exc:
        return _wrap({"error": f"disagreement_failed: {exc}"})

    models = result.models or []
    matrix: list[list[float]] = []
    model_names = [str(m.name) for m in models]
    # Naive Jaccard pairwise so the tool ships without depending on whichever
    # embedding provider the user happens to have keys for. The full hosted
    # API uses cosine on embeddings; this is the degraded but always-available
    # fallback, mirroring the documented behaviour of ConsensusResult.scoring_method.
    answers = [str(getattr(m, "answer", "") or "") for m in models]
    for i in range(len(answers)):
        row: list[float] = []
        a = set(answers[i].lower().split())
        for j in range(len(answers)):
            b = set(answers[j].lower().split())
            union = a | b
            row.append(round(len(a & b) / len(union), 3) if union else 1.0)
        matrix.append(row)

    return _wrap(
        {
            "prompt": prompt,
            "models": model_names,
            "matrix": matrix,
            "scoring_method": "jaccard_fallback",
            "interpretation": (
                "Values closer to 1.0 indicate strong lexical agreement between "
                "the row model and the column model on this prompt. Values closer "
                "to 0.0 indicate divergence — useful as Article 14 automation-bias "
                "review evidence material."
            ),
            "generated_at": _now_iso(),
        }
    )


async def _tool_readiness_check(arguments: dict[str, Any]) -> dict[str, Any]:
    """EU AI Act Annex VI readiness gap-analysis on a system description.

    The caller provides a short description of their AI system; we use the
    consensus engine to surface (a) which Annex III risk class the system
    likely falls under, (b) which Articles 9/12/13/14/16 are most relevant,
    (c) what documentation artefacts are typically expected. Advisory only.
    """
    from quorum.core.consensus import consensus

    system_description = str(arguments.get("system_description", "")).strip()
    if not system_description:
        return _wrap({"error": "system_description is required"})

    _call_counter["readiness_check"] = _call_counter.get("readiness_check", 0) + 1

    framed_prompt = (
        "Act as an EU AI Act readiness reviewer. The user describes an AI system "
        "below. Produce a short structured advisory report covering:\n"
        "1. Likely Annex III risk class (if any).\n"
        "2. Which obligations under Articles 9, 12, 13, 14, 16 are most relevant.\n"
        "3. Whether internal (Annex VI) or external (Annex VII) conformity "
        "assessment is likely required, and why.\n"
        "4. Top 3 documentation artefacts the provider should prepare.\n"
        "5. Three highest-risk gaps to fix in the next 30 days.\n\n"
        "Output in bullet form. Be specific. Cite article numbers. End every "
        "section with the words 'ADVISORY — NOT A CONFORMITY ASSESSMENT.'\n\n"
        f"System description:\n{system_description}"
    )

    try:
        result = await consensus(framed_prompt, budget_usd=0.10, timeout_s=90.0)
    except Exception as exc:
        return _wrap({"error": f"readiness_failed: {exc}"})

    return _wrap(
        {
            "system_description": system_description,
            "readiness_report": result.answer,
            "confidence": float(result.confidence),
            "model_count": len(result.models or []),
            "generated_at": _now_iso(),
            "important_note": (
                "This readiness report is advisory technical material produced by "
                "consensus across multiple language models. It is NOT a conformity "
                "assessment, NOT legal advice, and NOT a substitute for engaging "
                "qualified counsel and (where required) a designated Notified Body "
                "under Article 31 of Regulation (EU) 2024/1689."
            ),
        }
    )


async def _tool_evidence_record(arguments: dict[str, Any]) -> dict[str, Any]:
    """Generate a per-query PDF/Markdown evidence record.

    Wraps quorum.hsp.ai_act_cert.generate_cert_pdf — the same code path the
    Pro+ hosted tier uses. Returns the file path; the caller decides what to
    do with it (forward to compliance officer, attach to internal docs file,
    etc.). Advisory only.
    """
    from quorum.core.consensus import consensus
    from quorum.hsp.ai_act_cert import generate_cert_pdf

    prompt = str(arguments.get("prompt", "")).strip()
    if not prompt:
        return _wrap({"error": "prompt is required"})

    _call_counter["evidence_record"] = _call_counter.get("evidence_record", 0) + 1

    try:
        result = await consensus(prompt, budget_usd=0.10, timeout_s=90.0)
    except Exception as exc:
        return _wrap({"error": f"evidence_record_consensus_failed: {exc}"})

    # Materialise the record next to the user's home quorum dir so they can
    # locate it without surprises. .pdf if reportlab is available, .md fallback.
    out_dir = Path.home() / ".quorum" / "evidence_records"
    out_dir.mkdir(parents=True, exist_ok=True)
    query_id = f"mcp-{int(asyncio.get_event_loop().time() * 1000)}"
    out_path = out_dir / f"{query_id}.pdf"

    consensus_dict = {
        "confidence": float(result.confidence),
        "total_cost_usd": float(getattr(result, "total_cost_usd", 0.0)),
        "total_latency_ms": int(getattr(result, "total_latency_ms", 0)),
        "models": [
            {
                "name": str(m.name),
                "weight": float(getattr(m, "weight", 0.0) or 0.0),
                "latency_ms": int(getattr(m, "latency_ms", 0) or 0),
                "cost_usd": float(getattr(m, "cost_usd", 0.0) or 0.0),
                "tokens_in": int(getattr(m, "tokens_in", 0) or 0),
                "tokens_out": int(getattr(m, "tokens_out", 0) or 0),
                "error": str(getattr(m, "error", "") or ""),
            }
            for m in (result.models or [])
        ],
        "disagreements": list(getattr(result, "disagreements", []) or []),
    }
    decision_dict = {
        "approved": True,
        "decision_id": f"mcp-local-{query_id}",
        "reason": "Local evidence record generated via MCP (no HSP webhook).",
        "signed_at": _now_iso(),
        "signature": "0" * 64,
        "audit_trail_url": "",
    }

    try:
        meta = generate_cert_pdf(
            query_id=query_id,
            query_text=prompt,
            consensus_result_dict=consensus_dict,
            hsp_decision_dict=decision_dict,
            output_path=out_path,
        )
    except Exception as exc:
        logger.exception("evidence record generation failed")
        return _wrap({"error": f"evidence_record_render_failed: {exc}"})

    return _wrap(
        {
            "query_id": query_id,
            "file_path": str(meta.get("pdf_path", out_path)),
            "format": str(meta.get("format", "pdf")),
            "sha256": str(meta.get("sha256", "")),
            "generated_at": meta.get("generated_at", _now_iso()),
            "synthesized_answer_preview": (result.answer or "")[:300],
            "model_count": len(result.models or []),
            "important_note": (
                "This evidence record is advisory technical material. It is NOT a "
                "conformity assessment under Regulation (EU) 2024/1689 and Sovereign "
                "Chain Ltd is not a Notified Body under Article 31. Include this "
                "record in your internal Annex VI documentation file at your own "
                "judgement; final conformity assessment remains your responsibility."
            ),
        }
    )


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "quorum_consensus",
        "description": (
            "Run a multi-LLM consensus query: fans the prompt across all configured "
            "providers (Claude, GPT, Gemini, Llama, DeepSeek, Mistral, Qwen, Cohere, "
            "Grok, NVIDIA OSS models), scores semantic agreement, returns the "
            "top-weighted synthesized answer with confidence + per-model breakdown + "
            "disagreement list. Useful for high-stakes decisions where single-model "
            "bias is unacceptable."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The query to fan out across LLMs."},
                "budget_usd": {
                    "type": "number",
                    "description": "Hard cap on total cost per query (USD). Default 0.05.",
                    "default": 0.05,
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Per-query timeout in seconds. Default 60.",
                    "default": 60,
                },
                "user_id": {
                    "type": "string",
                    "description": "Optional user identifier for per-user RLHF + memory.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "quorum_multi",
        "description": (
            "Return RAW per-model answers without synthesizing a winner. Use when "
            "the caller wants to see every model's full response side-by-side "
            "(content authoring, compliance review, manual decision)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The query to fan out."}
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "quorum_verdict",
        "description": (
            "Yes/no verdict on a factual or evaluative claim. Each model votes TRUE/"
            "FALSE/UNCLEAR; the consensus tally + per-model breakdown is returned. "
            "Use for fact-checking, claim verification, or any binary decision where "
            "you want explicit disagreement surfaced."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The claim to evaluate (e.g. 'GDPR applies to AI training data').",
                }
            },
            "required": ["claim"],
        },
    },
    {
        "name": "quorum_disagreement_matrix",
        "description": (
            "Pairwise agreement matrix across every model that responded. Each cell "
            "is the lexical-overlap score between two model answers. Useful as EU "
            "AI Act Article 14 automation-bias evidence material — shows reviewers "
            "exactly where the model panel converged and where it diverged."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The query."}
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "quorum_readiness_check",
        "description": (
            "EU AI Act Annex VI readiness gap-analysis on a free-text description "
            "of an AI system. Returns a structured advisory report: likely Annex "
            "III risk class, relevant Articles 9/12/13/14/16 obligations, internal "
            "(Annex VI) vs external (Annex VII) conformity assessment route, top "
            "documentation artefacts to prepare, top 30-day risk gaps. ADVISORY — "
            "NOT a conformity assessment."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "system_description": {
                    "type": "string",
                    "description": (
                        "Plain-English description of the AI system: what it does, "
                        "who uses it, what data, where deployed."
                    ),
                }
            },
            "required": ["system_description"],
        },
    },
    {
        "name": "quorum_evidence_record",
        "description": (
            "Generate a per-query PDF (or Markdown fallback) evidence record "
            "containing: every model that ran, its weight + answer + cost + "
            "latency, the synthesized consensus, and a SHA-256 hash of the prompt. "
            "File is written to ~/.quorum/evidence_records/. Designed as internal "
            "material referenced by EU AI Act Articles 12 (record-keeping) and 13 "
            "(transparency). ADVISORY — NOT a conformity assessment."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The query to record."}
            },
            "required": ["prompt"],
        },
    },
]


# Lookup table from tool name to implementation function.
_TOOL_IMPL = {
    "quorum_consensus": _tool_consensus,
    "quorum_multi": _tool_multi,
    "quorum_verdict": _tool_verdict,
    "quorum_disagreement_matrix": _tool_disagreement_matrix,
    "quorum_readiness_check": _tool_readiness_check,
    "quorum_evidence_record": _tool_evidence_record,
}


async def _serve() -> None:
    """Run the MCP server on stdio."""
    # Import lazily so a broken MCP install gives a clear error instead of
    # crashing on module import.
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    server: Server = Server("quorum")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        impl = _TOOL_IMPL.get(name)
        if impl is None:
            payload = _wrap({"error": f"unknown_tool: {name}"})
        else:
            try:
                payload = await impl(arguments or {})
            except Exception as exc:
                logger.exception("tool %s raised", name)
                payload = _wrap({"error": f"tool_internal_error: {exc}"})

        return [TextContent(type="text", text=json.dumps(payload, indent=2))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="quorum",
                server_version="0.2.4",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    """Entry point — invoked by `quorum-mcp` console script or `python -m`."""
    log_level = os.environ.get("QUORUM_MCP_LOG", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.WARNING),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
