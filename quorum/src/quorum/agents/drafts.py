"""Draft-generating tools — Quorum-powered, human-approval-gated.

Each function returns a Draft (text + provenance). Nothing is published.
A `sell_quorum()` orchestrator runs all 5 in parallel and writes a bundle
to disk for the user to review and ship.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from quorum.agents.draft_verifier import annotate_draft, find_conflicts
from quorum.agents.fact_sheet import build_fact_sheet, format_as_prompt_block
from quorum.core.consensus import consensus
from quorum.providers.registry import load_default_providers
from quorum.web import search_to_context


# --------- product facts (single source of truth for every draft) ---------

QUORUM_FACTS = """\
QUORUM — open-source multi-LLM consensus engine.
- v0.1.5, Apache 2.0, https://quorum-ai.dev, https://github.com/jaquelinejaque/sovereignchain
- 23 frontier+OSS LLMs in parallel: Claude Sonnet 4.6 / Opus 4.8 / Haiku 4.5, GPT-4.1, GPT-4o-mini, Gemini Flash, Grok-4, Llama 3.3 70B, Llama-4 Maverick, DeepSeek V4, Dracarys 70B, Mistral Large/Codestral/Small, Cohere Command R+/R/A, DeepSeek Chat/Reasoner, NVIDIA OSS, Ollama local.
- Semantic agreement via cosine on embeddings (NOT Jaccard / NOT exact match).
- Per-query observed cost: ~$0.011 (router-selected) to ~$0.07 (all 23 with --web).
- 10/13 self-evolution loops functional (memory, MoE router, RLHF, A/B, synthetic data, Hebbian, meta-learner, ELO competition, self-prompting, adversarial probing). 3 scaffold (distillation, NAS, federated) — disclosed in README.
- HSP Gate patent PCT/US26/11908 — async approval webhook for high-stakes calls (opt-in).
- EU AI Act Art. 12/13 helper: per-query SHA-256 hash-chained audit log (PDF on demand).
- VS Code extension live: sovereignchain.quorum-vscode (currently 16 installs in 36h, 0 paying customers).
- Pro tier £49/mo · Free OSS forever · Self-host unlimited.

REAL CASE STUDY produced 2026-06-17: asked "Should I quit £80k UK corporate job to start a SaaS in 2026?" → 86% consensus, 23/25 OK, $0.028, 14.4s, 0 dissenters. Verdict: depends, validate first, 18-24 months runway. Screenshot at ~/Desktop/quorum-saas-vs-job.png.

COMPETITIVE LANDSCAPE June 2026:
- OpenRouter Fusion went GA mid-June: 300+ models, 5.5% credit fee, judge model HIDES per-model dissent.
- OrcaRouter Fusion: #2 RouterArena, $0 routing markup, also hides dissent.
- Quorum's wedge: EXPOSES every model's vote + dissent — open source, no black box.

FOUNDER: Jaqueline Martins, solo, UK, Sovereign Chain Ltd.
"""


@dataclass
class Draft:
    """A single generated draft (text + provenance)."""

    kind: str  # 'linkedin' | 'twitter' | 'show_hn' | 'email' | 'vscode_listing'
    content: str
    confidence: float
    models_ok: int
    models_total: int
    cost_usd: float
    latency_ms: float
    verification_attempts: int = 1
    unresolved_conflicts: int = 0
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DraftBundle:
    """Result of `sell_quorum()` — all 5 drafts + paths on disk."""

    output_dir: str
    drafts: List[Draft]
    total_cost_usd: float
    total_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "drafts": [d.to_dict() for d in self.drafts],
            "total_cost_usd": self.total_cost_usd,
            "total_latency_ms": self.total_latency_ms,
        }


# --------- public retry harness (used by tests; thin dict wrapper around _ask_quorum_all) ---------

async def regenerate_loop(
    prompt: str,
    fact_sheet: dict,
    *,
    max_attempts: int = 3,
) -> dict:
    """Public wrapper: run _ask_quorum_all with verification retry.

    Returns a dict with at least {text, attempts, unresolved} for
    consumer/test ergonomics; full bookkeeping (cost, latency) is also
    included so callers don't have to re-derive them.
    """
    answer, confidence, ok, total, cost, latency, attempts, unresolved = (
        await _ask_quorum_all(
            prompt,
            use_web=False,
            max_attempts=max_attempts,
            fact_sheet=fact_sheet,
        )
    )
    return {
        "text": answer,
        "attempts": attempts,
        "unresolved": unresolved,
        "confidence": confidence,
        "models_ok": ok,
        "models_total": total,
        "cost_usd": cost,
        "latency_ms": latency,
    }


# --------- helpers ---------

async def _ask_quorum_all(
    prompt: str,
    *,
    use_web: bool = False,
    timeout_s: float = 60.0,
    max_attempts: int = 3,
    fact_sheet: dict | None = None,
) -> tuple[str, float, int, int, float, float, int, int]:
    """Run consensus across all configured providers with a verification retry loop.

    The loop calls `consensus`, then `find_conflicts(answer, fact_sheet)`. If
    conflicts are found and we still have attempts left, we prepend a RETRY
    block instructing the models to regenerate without the invented claims and
    try again. After exhausting attempts with conflicts still present, the
    final answer is annotated via `annotate_draft`.

    Returns
    -------
    (answer, confidence, ok, total, cost_usd, latency_ms,
     verification_attempts, unresolved_conflicts)

    `cost_usd` is accumulated across all attempts. `latency_ms` is wall-clock
    of all attempts. `confidence`, `ok`, `total` reflect the FINAL attempt.
    """
    if use_web:
        ctx = search_to_context(prompt[:300], n=4)
        prompt = f"{ctx}\n\nTASK:\n{prompt}"

    if fact_sheet is None:
        fact_sheet = build_fact_sheet()

    providers = load_default_providers()

    loop_start = asyncio.get_event_loop().time()
    accumulated_cost = 0.0
    current_prompt = prompt
    answer = ""
    confidence = 0.0
    ok = 0
    total = 0
    conflicts: list[dict] = []
    attempts = 0

    for attempts in range(1, max_attempts + 1):
        result = await consensus(
            current_prompt,
            providers=providers,
            budget_usd=10.0,
            route=False,
            timeout_s=timeout_s,
        )
        answer = result.answer
        confidence = result.confidence
        ok = sum(1 for m in result.models if not m.error)
        total = len(result.models)
        accumulated_cost += result.total_cost_usd

        conflicts = find_conflicts(answer, fact_sheet)
        if not conflicts or attempts >= max_attempts:
            break

        retry_lines = ["RETRY -- your previous draft contained invented claims:"]
        for c in conflicts:
            claim = c.get("claim_text", "")
            expected = c.get("expected_value")
            if expected is not None:
                retry_lines.append(
                    f"  - '{claim}' -- DO NOT use; expected value: {expected}"
                )
            else:
                retry_lines.append(
                    f"  - '{claim}' -- DO NOT use; expected value: (not in fact_sheet)"
                )
        retry_lines.append("Regenerate without these inventions.")
        retry_block = "\n".join(retry_lines)
        current_prompt = f"{retry_block}\n\n{prompt}"

    if conflicts:
        answer = annotate_draft(answer, conflicts)

    latency_ms = (asyncio.get_event_loop().time() - loop_start) * 1000.0
    unresolved_conflicts = len(conflicts)

    return (
        answer,
        confidence,
        ok,
        total,
        accumulated_cost,
        latency_ms,
        attempts,
        unresolved_conflicts,
    )


def _facts_block() -> str:
    return QUORUM_FACTS


# --------- 5 draft generators ---------

async def draft_linkedin_post(*, use_web: bool = True, fact_sheet: dict | None = None) -> Draft:
    """Generate a LinkedIn post draft. Tone: founder-honest, factual, no fluff."""
    fact_sheet = fact_sheet if fact_sheet is not None else build_fact_sheet()
    prompt = f"""You are writing a LinkedIn post for Jaqueline Martins, solo UK founder, to ship Quorum to her network.

Constraints:
- 180-260 words.
- First line MUST be a hook that earns the next click (story / specific number / honest stake).
- Embed the REAL case study (£80k SaaS question) with concrete numbers.
- Attack OpenRouter Fusion's hidden-judge directly — that is the differentiator.
- Honest scorecard line: 10/13 loops functional, 3 scaffold, link to README.
- End with one CTA: try it at quorum-ai.dev OR install VS Code extension.
- No emojis. No "🚀". No "in this post we will explore". No "blessed". Plain founder voice.
- No hallucinated quotes, no fake testimonials, no "we are excited to announce" patterns.

Product facts (single source of truth — do NOT invent any numbers beyond these):
{_facts_block()}

{format_as_prompt_block(fact_sheet)}

Write the post body ONLY. No preamble like "Here is the post:". No closing meta-commentary."""
    answer, conf, ok, total, cost, latency, attempts, conflicts = await _ask_quorum_all(
        prompt, use_web=use_web, fact_sheet=fact_sheet
    )
    return Draft(
        kind="linkedin",
        content=answer.strip(),
        confidence=conf,
        models_ok=ok,
        models_total=total,
        cost_usd=cost,
        latency_ms=latency,
        verification_attempts=attempts,
        unresolved_conflicts=conflicts,
    )


async def draft_twitter_thread(*, use_web: bool = True, fact_sheet: dict | None = None) -> Draft:
    """Generate an 8-tweet Twitter/X thread."""
    fact_sheet = fact_sheet if fact_sheet is not None else build_fact_sheet()
    prompt = f"""Write an 8-tweet Twitter/X thread (numbered 1/8 ... 8/8) for Jaqueline Martins to ship Quorum.

Constraints:
- Tweet 1 MUST be a hook with a specific number or stake (no generic intros).
- Each tweet ≤ 270 chars. Threading must read naturally — not 8 disconnected facts.
- Embed REAL case study (£80k SaaS question, 86% consensus, $0.028, 14.4s).
- Attack OpenRouter Fusion's hidden-judge directly in tweet 5 or 6.
- Honest scorecard line (10/13 functional) — do NOT hide what's scaffold.
- Final tweet: CTA + repo link + screenshot mention.
- No emojis except 🧵 on tweet 1 if natural. No threads-of-affirmations.
- No "BREAKING:", no "1/", use "1/8" format.

Product facts (do NOT invent beyond these):
{_facts_block()}

{format_as_prompt_block(fact_sheet)}

Output: just the 8 numbered tweets separated by blank lines. No preamble."""
    answer, conf, ok, total, cost, latency, attempts, conflicts = await _ask_quorum_all(
        prompt, use_web=use_web, fact_sheet=fact_sheet
    )
    return Draft(
        kind="twitter",
        content=answer.strip(),
        confidence=conf,
        models_ok=ok,
        models_total=total,
        cost_usd=cost,
        latency_ms=latency,
        verification_attempts=attempts,
        unresolved_conflicts=conflicts,
    )


async def draft_show_hn(*, use_web: bool = True, fact_sheet: dict | None = None) -> Draft:
    """Generate a Show HN post (title + body)."""
    fact_sheet = fact_sheet if fact_sheet is not None else build_fact_sheet()
    prompt = f"""Write a Show HN submission for Quorum.

Format:
TITLE: Show HN: ... (max 80 chars, ONE specific hook — not generic)
URL: https://quorum-ai.dev
BODY: 180-300 words, plain founder voice, honest, no emojis.

The body must:
- Open with what Quorum IS in one sentence — a 7-year-old should understand.
- Explain WHY it matters with a real failure mode (e.g. "single-LLM silent failure" — well-known dev pain).
- Show the architecture in 3 lines: 23 LLMs parallel → cosine on embeddings → top-weighted answer + dissent.
- Include real case study with numbers (86% conf, $0.028, 14.4s on £80k question).
- Honest scorecard (10/13 loops functional, 3 scaffold, README links).
- Address the OpenRouter Fusion / OrcaRouter question head-on: yes they exist, here is the wedge.
- Close with: open source, self-host, Pro £49/mo for managed cloud, looking for design partners.
- Anticipate the top HN comment (cost? latency? what about local?) and pre-answer it.
- NO "we are excited", NO "revolutionary", NO emojis. Be the founder, not the marketer.

Product facts:
{_facts_block()}

{format_as_prompt_block(fact_sheet)}

Output: TITLE on first line, blank line, URL on next line, blank line, then BODY."""
    answer, conf, ok, total, cost, latency, attempts, conflicts = await _ask_quorum_all(
        prompt, use_web=use_web, fact_sheet=fact_sheet
    )
    return Draft(
        kind="show_hn",
        content=answer.strip(),
        confidence=conf,
        models_ok=ok,
        models_total=total,
        cost_usd=cost,
        latency_ms=latency,
        verification_attempts=attempts,
        unresolved_conflicts=conflicts,
    )


async def draft_email_outreach(*, use_web: bool = False, fact_sheet: dict | None = None) -> Draft:
    """Generate a cold email outreach to a hypothetical AI dev tooling founder/lead."""
    fact_sheet = fact_sheet if fact_sheet is not None else build_fact_sheet()
    prompt = f"""Write a cold-outreach email to send to founders of AI dev tooling companies (LiteLLM/Berri AI, Helicone, Portkey, Continue.dev, Cline) — asking for honest 15-min feedback call on Quorum.

Constraints:
- 90-130 words MAX. Cold email — short or dead.
- Subject line: 4-7 words, curiosity not desperation.
- First sentence: respect for their work (specific reference — not "love your product").
- Sentence 2: what Quorum is, ONE sentence.
- Sentence 3: the ASK — 15 min, no pitch, just brutal feedback.
- Sentence 4: cheap "no" offer (reply DECLINE and that's it).
- Sign: Jaqueline, founder, Sovereign Chain Ltd.
- NO emojis. NO "synergy", "leverage", "circle back". Plain text.

Product facts:
{_facts_block()}

{format_as_prompt_block(fact_sheet)}

Output: SUBJECT: ... on first line, blank line, then email body."""
    answer, conf, ok, total, cost, latency, attempts, conflicts = await _ask_quorum_all(
        prompt, use_web=use_web, fact_sheet=fact_sheet
    )
    return Draft(
        kind="email",
        content=answer.strip(),
        confidence=conf,
        models_ok=ok,
        models_total=total,
        cost_usd=cost,
        latency_ms=latency,
        verification_attempts=attempts,
        unresolved_conflicts=conflicts,
    )


async def draft_vscode_listing(*, use_web: bool = True, fact_sheet: dict | None = None) -> Draft:
    """Generate an upgraded VS Code Marketplace listing description."""
    fact_sheet = fact_sheet if fact_sheet is not None else build_fact_sheet()
    prompt = f"""Rewrite the VS Code Marketplace listing description for `sovereignchain.quorum-vscode`.

The CURRENT listing is generic and has only 16 installs in 36 hours despite #1 SEO ranking for "multi-llm consensus". Recon shows top extensions emphasize SHORT 1st-line hooks + GIFs/screenshots + concrete features.

Constraints:
- First line: ONE-sentence hook with a specific reason a dev clicks "Install" (not "boost your productivity").
- Lines 2-4: 3-bullet feature list. Each bullet = ONE concrete thing the user gets in their editor.
- 1 paragraph (≤ 60 words) explaining the consensus engine architecture.
- "How it works" → 3 numbered steps, code-comment style, in the editor.
- "Why" → ONE sentence comparing to OpenRouter Fusion's hidden judge.
- Footer: open source, Apache 2.0, link to quorum-ai.dev.
- NO emojis except 1 sparingly in the hook if natural. NO "feel the power", NO "supercharge".
- Markdown that renders well on marketplace.visualstudio.com.

Product facts:
{_facts_block()}

{format_as_prompt_block(fact_sheet)}

Output: the markdown body of the listing. No preamble."""
    answer, conf, ok, total, cost, latency, attempts, conflicts = await _ask_quorum_all(
        prompt, use_web=use_web, fact_sheet=fact_sheet
    )
    return Draft(
        kind="vscode_listing",
        content=answer.strip(),
        confidence=conf,
        models_ok=ok,
        models_total=total,
        cost_usd=cost,
        latency_ms=latency,
        verification_attempts=attempts,
        unresolved_conflicts=conflicts,
    )


# --------- orchestrator ---------

DRAFT_FNS = {
    "linkedin": draft_linkedin_post,
    "twitter": draft_twitter_thread,
    "show_hn": draft_show_hn,
    "email": draft_email_outreach,
    "vscode_listing": draft_vscode_listing,
}


async def sell_quorum(
    *,
    output_dir: str | None = None,
    only: list[str] | None = None,
    use_web: bool = True,
) -> DraftBundle:
    """Generate ALL 5 drafts (or a subset) in parallel; write to disk."""
    out = Path(output_dir or os.path.expanduser("~/Desktop/quorum-drafts"))
    out.mkdir(parents=True, exist_ok=True)

    keys = only or list(DRAFT_FNS.keys())
    # Build the fact-sheet ONCE and reuse across all 5 drafts to avoid redundant
    # rebuilds (and to guarantee every draft sees identical APPROVED FACTS).
    shared_fact_sheet = build_fact_sheet()
    tasks = [DRAFT_FNS[k](use_web=use_web, fact_sheet=shared_fact_sheet) for k in keys]
    drafts = await asyncio.gather(*tasks, return_exceptions=True)

    final: list[Draft] = []
    total_cost = 0.0
    total_latency = 0.0
    for k, d in zip(keys, drafts):
        if isinstance(d, Exception):
            err_text = f"[draft {k} failed: {type(d).__name__}: {d}]"
            final.append(
                Draft(
                    kind=k,
                    content=err_text,
                    confidence=0.0,
                    models_ok=0,
                    models_total=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                )
            )
            continue
        final.append(d)
        total_cost += d.cost_usd
        total_latency = max(total_latency, d.latency_ms)
        path = out / f"{k}.md"
        path.write_text(
            f"# Quorum draft — {k}\n\n"
            f"_Generated_: {d.generated_at}\n"
            f"_Confidence_: {d.confidence:.0%}  · "
            f"_Models OK_: {d.models_ok}/{d.models_total}  · "
            f"_Cost_: ${d.cost_usd:.4f}  · "
            f"_Latency_: {d.latency_ms:.0f}ms\n\n"
            f"---\n\n{d.content}\n"
        )

    # Write README index
    (out / "README.md").write_text(
        f"# Quorum drafts — {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Total cost (parallel run): ${total_cost:.4f}  · "
        f"Wall-clock: {total_latency:.0f}ms\n\n"
        + "\n".join(f"- [{k}]({k}.md) ({d.models_ok}/{d.models_total} OK, "
                   f"{d.confidence:.0%} conf, ${d.cost_usd:.4f}, "
                   f"{d.unresolved_conflicts} unresolved conflicts after "
                   f"{d.verification_attempts} attempts)"
                   for k, d in zip(keys, final))
        + "\n\nReview each draft. Approve, edit, or regenerate before publishing.\n"
        + "Nothing has been published. All drafts are local.\n"
    )

    return DraftBundle(
        output_dir=str(out),
        drafts=final,
        total_cost_usd=total_cost,
        total_latency_ms=total_latency,
    )
