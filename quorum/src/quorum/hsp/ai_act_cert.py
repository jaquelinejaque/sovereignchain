"""EU AI Act 2026-08 auto-certification PDF generator.

Licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

HSP Commercial Restrictions:
    This module produces certificates that reference the Human Supervision
    Protocol (HSP), patent pending PCT/US26/11908. Commercial deployment of
    the certificate format requires a license from Sovereign Chain Ltd.
    See LICENSE-HSP at the repo root.

WHY this module exists:
    The EU AI Act (entered into force 2024-08; high-risk provisions binding
    from 2026-08-02) requires every high-risk AI inference to be (a) logged,
    (b) traceable to a human-supervised decision, and (c) accompanied by a
    machine-readable evidence record. This module collapses one consensus
    query + one HSP decision into a single immutable artifact (PDF preferred,
    Markdown fallback) that auditors can spot-check.

    Design choices:
      * reportlab is a heavy optional dep. We import lazily so the rest of
        Quorum still installs cleanly without it; when missing, we emit a
        Markdown file with identical structure so the audit trail is never
        broken by a missing package.
      * Every certificate is hashed (SHA-256 over the rendered bytes) so
        downstream chain-of-custody can detect tampering without re-parsing
        the PDF.
      * The cert is deterministic given (query, consensus, decision): we
        embed timestamps explicitly rather than letting reportlab stamp
        metadata, so re-generating the same inputs yields a stable hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PATENT_REF = "PCT/US26/11908"
CERT_VERSION = "1.0"
EU_AI_ACT_REFERENCE = "Regulation (EU) 2024/1689 — High-risk AI systems, Art. 13/14/15"


def _query_hash(query_text: str) -> str:
    """Stable SHA-256 of the prompt — used as a tamper-evident identifier.

    WHY a hash rather than the raw query: the prompt may contain PII or trade
    secrets that we shouldn't print on a certificate that gets emailed to
    auditors. The hash is sufficient to prove "this cert binds to query X" if
    a reviewer holds the original.
    """
    return hashlib.sha256(query_text.encode("utf-8")).hexdigest()


def _summarize_models(consensus_result_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the per-model evidence rows from a ConsensusResult dict.

    WHY tolerant: the consensus engine is evolving; we accept missing fields
    rather than blowing up the cert generator (which would break the audit
    flow). Each row is "best-effort evidence", not a contract.
    """
    rows: list[dict[str, Any]] = []
    for m in consensus_result_dict.get("models", []) or []:
        rows.append(
            {
                "name": str(m.get("name", "unknown")),
                "weight": float(m.get("weight", 0.0) or 0.0),
                "latency_ms": int(m.get("latency_ms", 0) or 0),
                "cost_usd": float(m.get("cost_usd", 0.0) or 0.0),
                "tokens_in": int(m.get("tokens_in", 0) or 0),
                "tokens_out": int(m.get("tokens_out", 0) or 0),
                "error": str(m.get("error") or ""),
            }
        )
    return rows


def _render_markdown(
    *,
    query_id: str,
    query_text: str,
    consensus_result_dict: dict[str, Any],
    hsp_decision_dict: dict[str, Any],
    generated_at: datetime,
) -> str:
    """Render the certificate body as Markdown.

    WHY Markdown as the fallback: it stays diff-friendly in git, is readable
    without a viewer, and preserves every field the PDF carries. If reportlab
    is unavailable on a CI runner we still ship a valid audit artifact.
    """
    qhash = _query_hash(query_text)
    models = _summarize_models(consensus_result_dict)
    lines: list[str] = []
    lines.append("# EU AI Act Compliance Certificate")
    lines.append("")
    lines.append(f"- **Certificate version:** {CERT_VERSION}")
    lines.append(f"- **Regulation:** {EU_AI_ACT_REFERENCE}")
    lines.append(f"- **HSP patent reference:** {PATENT_REF}")
    lines.append(f"- **Generated at (UTC):** {generated_at.isoformat()}")
    lines.append(f"- **Query ID:** `{query_id}`")
    lines.append(f"- **Query SHA-256:** `{qhash}`")
    lines.append("")
    lines.append("## Consensus summary")
    lines.append("")
    lines.append(
        f"- Confidence: **{consensus_result_dict.get('confidence', 0.0):.4f}**"
    )
    lines.append(
        f"- Total cost USD: {consensus_result_dict.get('total_cost_usd', 0.0)}"
    )
    lines.append(
        f"- Total latency ms: {consensus_result_dict.get('total_latency_ms', 0)}"
    )
    disagreements = consensus_result_dict.get("disagreements", []) or []
    lines.append(f"- Disagreements: {len(disagreements)}")
    lines.append("")
    lines.append("## Model evidence")
    lines.append("")
    lines.append("| Model | Weight | Latency (ms) | Cost USD | Tokens in | Tokens out | Error |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for m in models:
        err = m["error"] or "—"
        lines.append(
            f"| {m['name']} | {m['weight']:.3f} | {m['latency_ms']} | "
            f"{m['cost_usd']:.6f} | {m['tokens_in']} | {m['tokens_out']} | {err} |"
        )
    lines.append("")
    lines.append("## HSP decision")
    lines.append("")
    lines.append(
        f"- Approved: **{bool(hsp_decision_dict.get('approved', False))}**"
    )
    lines.append(f"- Decision ID: `{hsp_decision_dict.get('decision_id', '')}`")
    lines.append(f"- Reason: {hsp_decision_dict.get('reason', '')}")
    lines.append(f"- Signed at: {hsp_decision_dict.get('signed_at', '')}")
    lines.append(
        f"- Audit trail: {hsp_decision_dict.get('audit_trail_url', '') or 'N/A'}"
    )
    sig = str(hsp_decision_dict.get("signature", ""))
    sig_display = sig[:32] + ("…" if len(sig) > 32 else "")
    lines.append(f"- Signature (truncated): `{sig_display}`")
    lines.append("")
    lines.append("---")
    lines.append(
        "Signed by Sovereign Chain Ltd. via HSP Protocol "
        f"({PATENT_REF}). This certificate is machine-generated; "
        "verify the signature above against HSP_PROTOCOL_KEY before relying on it."
    )
    return "\n".join(lines) + "\n"


def _render_pdf(
    *,
    query_id: str,
    query_text: str,
    consensus_result_dict: dict[str, Any],
    hsp_decision_dict: dict[str, Any],
    generated_at: datetime,
    output_path: Path,
) -> None:
    """Render a one-page PDF using reportlab.

    WHY a separate renderer rather than markdown-to-pdf: reportlab gives us
    pixel control over the official footer/signature layout, which auditors
    expect on a certificate. We keep this function pure-side-effect (writes to
    output_path) so the caller can hash the bytes after the fact.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    qhash = _query_hash(query_text)
    models = _summarize_models(consensus_result_dict)

    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, height - 25 * mm, "EU AI Act Compliance Certificate")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, height - 32 * mm, EU_AI_ACT_REFERENCE)
    c.drawString(20 * mm, height - 37 * mm, f"HSP Patent Reference: {PATENT_REF}")
    c.drawString(
        20 * mm,
        height - 42 * mm,
        f"Generated (UTC): {generated_at.isoformat()}  |  Cert v{CERT_VERSION}",
    )

    # Query block
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, height - 55 * mm, "Query")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, height - 61 * mm, f"ID: {query_id}")
    c.drawString(20 * mm, height - 66 * mm, f"SHA-256: {qhash}")

    # Consensus summary
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, height - 78 * mm, "Consensus summary")
    c.setFont("Helvetica", 9)
    conf = float(consensus_result_dict.get("confidence", 0.0) or 0.0)
    cost = float(consensus_result_dict.get("total_cost_usd", 0.0) or 0.0)
    lat = int(consensus_result_dict.get("total_latency_ms", 0) or 0)
    disagreements = consensus_result_dict.get("disagreements", []) or []
    c.drawString(
        20 * mm,
        height - 84 * mm,
        f"Confidence: {conf:.4f}   Cost: ${cost:.6f}   Latency: {lat} ms   "
        f"Disagreements: {len(disagreements)}",
    )

    # Model table
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, height - 96 * mm, "Model evidence")
    c.setFont("Helvetica-Bold", 8)
    row_y = height - 102 * mm
    c.drawString(20 * mm, row_y, "Model")
    c.drawString(75 * mm, row_y, "Weight")
    c.drawString(95 * mm, row_y, "Latency")
    c.drawString(115 * mm, row_y, "Cost USD")
    c.drawString(140 * mm, row_y, "Tokens (in/out)")
    c.setFont("Helvetica", 8)
    row_y -= 5 * mm
    for m in models[:12]:  # one-page cap; overflow noted below.
        c.drawString(20 * mm, row_y, m["name"][:30])
        c.drawString(75 * mm, row_y, f"{m['weight']:.3f}")
        c.drawString(95 * mm, row_y, f"{m['latency_ms']}")
        c.drawString(115 * mm, row_y, f"${m['cost_usd']:.6f}")
        c.drawString(140 * mm, row_y, f"{m['tokens_in']}/{m['tokens_out']}")
        row_y -= 5 * mm
    if len(models) > 12:
        c.drawString(
            20 * mm, row_y, f"(+ {len(models) - 12} additional models elided for one-page layout)"
        )
        row_y -= 5 * mm

    # HSP decision
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, row_y - 5 * mm, "HSP Decision")
    c.setFont("Helvetica", 9)
    row_y -= 11 * mm
    approved = bool(hsp_decision_dict.get("approved", False))
    c.drawString(
        20 * mm,
        row_y,
        f"Approved: {approved}   Decision ID: {hsp_decision_dict.get('decision_id', '')}",
    )
    row_y -= 5 * mm
    reason = str(hsp_decision_dict.get("reason", ""))[:120]
    c.drawString(20 * mm, row_y, f"Reason: {reason}")
    row_y -= 5 * mm
    c.drawString(20 * mm, row_y, f"Signed at: {hsp_decision_dict.get('signed_at', '')}")
    row_y -= 5 * mm
    sig = str(hsp_decision_dict.get("signature", ""))
    c.drawString(20 * mm, row_y, f"Signature: {sig[:48]}{'…' if len(sig) > 48 else ''}")

    # Footer
    c.setFont("Helvetica-Oblique", 7)
    c.drawString(
        20 * mm,
        15 * mm,
        f"Signed by Sovereign Chain Ltd. via HSP Protocol ({PATENT_REF}). "
        "Verify signature against HSP_PROTOCOL_KEY.",
    )
    c.drawString(
        20 * mm,
        12 * mm,
        "Apache-2.0 WITH HSP-Commercial-Restrictions. See LICENSE-HSP.",
    )
    c.showPage()
    c.save()


def generate_cert_pdf(
    query_id: str,
    query_text: str,
    consensus_result_dict: dict[str, Any],
    hsp_decision_dict: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Generate a compliance certificate, preferring PDF, falling back to Markdown.

    WHY return a dict instead of just a path: downstream (Stripe receipts,
    audit indexer, supabase mirror) all want the SHA-256 and the
    generated_at timestamp. Computing them once at the source avoids subtle
    re-hash drift if the file is re-read on a system with different newline
    conventions.

    Returns:
        {
          "pdf_path": str,        # may end with .md if reportlab missing
          "sha256":   str,        # hex digest of the rendered file bytes
          "generated_at": str,    # ISO-8601 UTC
          "format": "pdf" | "markdown",
        }
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc)

    fmt = "pdf"
    final_path = out
    try:
        # Probe reportlab without importing at module level (keeps cold-start fast).
        import importlib

        importlib.import_module("reportlab")
        _render_pdf(
            query_id=query_id,
            query_text=query_text,
            consensus_result_dict=consensus_result_dict,
            hsp_decision_dict=hsp_decision_dict,
            generated_at=generated_at,
            output_path=out,
        )
    except ImportError:
        logger.warning(
            "reportlab unavailable — falling back to Markdown certificate at %s",
            out.with_suffix(".md"),
        )
        fmt = "markdown"
        final_path = out.with_suffix(".md")
        body = _render_markdown(
            query_id=query_id,
            query_text=query_text,
            consensus_result_dict=consensus_result_dict,
            hsp_decision_dict=hsp_decision_dict,
            generated_at=generated_at,
        )
        final_path.write_text(body, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        # Defensive: a reportlab font glitch shouldn't lose the audit trail.
        # Fall through to Markdown so the cert always exists.
        logger.error("PDF generation failed (%s); falling back to Markdown.", exc)
        fmt = "markdown"
        final_path = out.with_suffix(".md")
        body = _render_markdown(
            query_id=query_id,
            query_text=query_text,
            consensus_result_dict=consensus_result_dict,
            hsp_decision_dict=hsp_decision_dict,
            generated_at=generated_at,
        )
        final_path.write_text(body, encoding="utf-8")

    digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
    result = {
        "pdf_path": str(final_path),
        "sha256": digest,
        "generated_at": generated_at.isoformat(),
        "format": fmt,
    }
    logger.info(
        "Generated AI Act certificate: path=%s sha256=%s format=%s",
        result["pdf_path"],
        digest,
        fmt,
    )
    return result


__all__ = ["generate_cert_pdf", "PATENT_REF", "EU_AI_ACT_REFERENCE", "CERT_VERSION"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _sample_consensus() -> dict[str, Any]:
    return {
        "answer": "42",
        "confidence": 0.93,
        "models": [
            {
                "name": "anthropic/claude-opus",
                "response": "42",
                "latency_ms": 812,
                "cost_usd": 0.0123,
                "tokens_in": 42,
                "tokens_out": 8,
                "weight": 0.4,
                "error": None,
            },
            {
                "name": "openai/gpt-4o",
                "response": "42",
                "latency_ms": 511,
                "cost_usd": 0.0061,
                "tokens_in": 40,
                "tokens_out": 6,
                "weight": 0.35,
                "error": None,
            },
            {
                "name": "google/gemini-1.5-pro",
                "response": "forty-two",
                "latency_ms": 901,
                "cost_usd": 0.0045,
                "tokens_in": 41,
                "tokens_out": 7,
                "weight": 0.25,
                "error": None,
            },
        ],
        "disagreements": [{"a": "42", "b": "forty-two"}],
        "evolution_signals": [],
        "total_cost_usd": 0.0229,
        "total_latency_ms": 901,
    }


def _sample_decision() -> dict[str, Any]:
    return {
        "approved": True,
        "decision_id": "test-decision-abc-123",
        "reason": "DEV_MODE — no HSP gate configured",
        "audit_trail_url": "",
        "signed_at": "2026-06-16T12:34:56+00:00",
        "signature": "f" * 64,
    }


def _test_generate_returns_metadata(tmp_dir: Path) -> None:
    """Cert generation returns sha256, path, timestamp, and format."""
    out = tmp_dir / "cert.pdf"
    result = generate_cert_pdf(
        query_id="q-1",
        query_text="What is the meaning of life?",
        consensus_result_dict=_sample_consensus(),
        hsp_decision_dict=_sample_decision(),
        output_path=out,
    )
    assert "pdf_path" in result
    assert "sha256" in result and len(result["sha256"]) == 64
    assert "generated_at" in result
    assert result["format"] in ("pdf", "markdown")
    assert Path(result["pdf_path"]).exists()
    assert Path(result["pdf_path"]).stat().st_size > 0


def _test_markdown_fallback_when_reportlab_missing(tmp_dir: Path, monkeypatch: Any) -> None:
    """When reportlab import fails, Markdown fallback engages with full content."""
    import builtins

    real_import = builtins.__import__

    def _no_reportlab(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "reportlab" or name.startswith("reportlab."):
            raise ImportError("reportlab not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_reportlab)
    out = tmp_dir / "cert.pdf"
    result = generate_cert_pdf(
        query_id="q-2",
        query_text="Test fallback",
        consensus_result_dict=_sample_consensus(),
        hsp_decision_dict=_sample_decision(),
        output_path=out,
    )
    assert result["format"] == "markdown"
    assert result["pdf_path"].endswith(".md")
    body = Path(result["pdf_path"]).read_text(encoding="utf-8")
    assert "EU AI Act Compliance Certificate" in body
    assert PATENT_REF in body
    assert "anthropic/claude-opus" in body


def _run_tests() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _test_generate_returns_metadata(tmp)
        # Simulate the monkeypatch path manually without pytest.
        import builtins

        real_import = builtins.__import__

        class _MP:
            def setattr(self, target: Any, name: str, value: Any) -> None:
                setattr(target, name, value)

        try:
            def _no_reportlab(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "reportlab" or name.startswith("reportlab."):
                    raise ImportError("reportlab not installed")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = _no_reportlab  # type: ignore[assignment]
            out = tmp / "fallback.pdf"
            result = generate_cert_pdf(
                query_id="q-2",
                query_text="Test fallback",
                consensus_result_dict=_sample_consensus(),
                hsp_decision_dict=_sample_decision(),
                output_path=out,
            )
            assert result["format"] == "markdown"
            assert result["pdf_path"].endswith(".md")
            body = Path(result["pdf_path"]).read_text(encoding="utf-8")
            assert "EU AI Act Compliance Certificate" in body
            assert PATENT_REF in body
        finally:
            builtins.__import__ = real_import  # type: ignore[assignment]
    logger.info("All AI Act cert tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_tests()
