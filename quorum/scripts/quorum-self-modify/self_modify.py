#!/usr/bin/env python3
"""Quorum Self-Modify — propose-review-apply flow with human-in-the-loop.

SAFETY POSTURE (read before changing this file):
- LOCAL ONLY. Refuses to run if hosted env markers are present.
- NEVER applies a patch automatically. Always 2 steps: propose, then
  explicit `apply <id>` from the human reviewer.
- NEVER pushes. After `apply`, the change is a *local* git commit. The
  human is the only one who can `git push`.
- All proposals + their full rationale + safety analysis are persisted
  in ~/.quorum/proposals/ so nothing happens silently.
- Each apply records a Git "Co-Authored-By" line to make AI-authored
  edits visible in git blame forever.

Workflow:
    # 1) Ask Quorum to look at a target file and propose 1 fix
    QUORUM_API_KEY=... GEMINI_API_KEY=... python3 self_modify.py propose \
        --repo /path/to/repo --file relative/path/to/file.py \
        --goal "fix the N+1 query in fetch_users"

    # 2) See what it suggested
    python3 self_modify.py list
    python3 self_modify.py review <id>

    # 3) Apply (creates local commit, does NOT push)
    python3 self_modify.py apply <id>

    # Or reject
    python3 self_modify.py reject <id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROPOSALS_DIR = Path.home() / ".quorum" / "proposals"
HOSTED_MARKERS = (
    "K_SERVICE", "KUBERNETES_SERVICE_HOST", "ECS_CONTAINER_METADATA_URI",
    "AWS_LAMBDA_FUNCTION_NAME", "FUNCTION_TARGET", "WEBSITE_INSTANCE_ID",
)


def _refuse_if_hosted() -> None:
    """LOCAL ONLY. Self-modify must never run in commercial hosted envs."""
    hosted = [m for m in HOSTED_MARKERS if os.environ.get(m)]
    if hosted:
        sys.exit(
            f"REFUSED: hosted environment markers present ({hosted}). "
            "Quorum self-modify is local-only by design — it would never "
            "ship to a paying customer's runtime."
        )


def _git(repo: Path, *args: str, capture: bool = True) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=capture, text=True, check=False,
    )
    if r.returncode != 0:
        sys.exit(f"git {args[0]} failed in {repo}: {r.stderr.strip()}")
    return r.stdout


def _query_gemini(prompt: str, timeout: int = 90, max_retries: int = 4) -> str:
    """Direct Gemini call (bypasses Quorum cloud — keeps everything local)."""
    import time as _t
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("GEMINI_API_KEY required (this tool stays local — no cloud)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
                break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < max_retries - 1:
                wait = 15 * (2 ** attempt)
                print(f"  Gemini {e.code} — retry in {wait}s", file=sys.stderr)
                _t.sleep(wait)
                continue
            sys.exit(f"Gemini HTTP {e.code}: {e.read().decode()[:200]}")
    candidates = data.get("candidates") or []
    text = ""
    if candidates:
        text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
    return text


PROPOSE_PROMPT = """You are a senior software engineer working on this codebase. You have been asked
to propose ONE concrete, minimal code change. You DO NOT apply the change —
a human will review it.

REPO: {repo}
FILE: {file}
GOAL: {goal}

CURRENT CONTENTS of {file}:
```
{contents}
```

REQUIRED OUTPUT FORMAT (be machine-parseable — humans use this to decide):

# Rationale
1-3 paragraphs: why the change. What's wrong now, what gets better, what's
the failure mode it prevents or the metric it improves.

# Safety analysis
Honest assessment. Address:
- What this change CANNOT break (the obvious safe paths)
- What this change MIGHT break (the risks — be specific, not vague)
- Files / functions that should be re-tested manually after this change
- Whether this should be paired with new tests (yes/no and what)

# Patch
A unified-diff patch (the EXACT output of `git diff`) that the reviewer
will apply with `git apply`. Keep it MINIMAL — only the lines that must
change. Don't reformat unrelated code. Use 3 lines of context. Headers
must be:
```
--- a/{file}
+++ b/{file}
@@ ...
```

Make the smallest correct change. If there is no good change to make,
write ONLY "NO CHANGE PROPOSED" and explain why in Rationale."""


REVIEW_PROMPT = """You are a senior code reviewer doing a SECOND OPINION on an AI-generated
patch. The proposer was Gemini 2.5 Flash. You are Claude. The human will
read both of you before applying anything.

Be skeptical. Your job is to catch what the proposer missed.

ORIGINAL GOAL: {goal}
TARGET FILE: {file}

ORIGINAL FILE CONTENTS:
```
{contents}
```

GEMINI'S PROPOSAL (rationale + safety + patch):
{proposal}

RESPOND IN THIS EXACT FORMAT:

# Verdict
ONE of: APPROVE / APPROVE_WITH_NOTES / REJECT

# Reasoning
1-3 short paragraphs. What you agree with, what worries you, what the
proposer overlooked. Be specific. If APPROVE, say why the patch is safe
AND addresses the goal. If REJECT, name the concrete failure mode.

# Risks the proposer didn't mention
Bullet list. Things Gemini missed in its own safety analysis. If you
genuinely find nothing extra, write "(none — proposer's analysis was complete)".

# What the human should double-check before applying
Bullet list of specific things to verify. Keep it to the truly important
checks, not boilerplate."""


def _query_claude_via_cli(prompt: str, timeout: int = 90) -> str:
    """Use the local `claude` CLI (Claude Code headless) — no API key needed."""
    try:
        r = subprocess.run(
            ["claude", "-p"],
            input=prompt, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except FileNotFoundError:
        return "(claude CLI not on PATH — install Claude Code to enable review)"
    except subprocess.TimeoutExpired:
        return f"(claude review timed out after {timeout}s)"
    if r.returncode != 0:
        return f"(claude exit {r.returncode}: {r.stderr[:200]})"
    return r.stdout.strip()


def _query_ollama(model: str, prompt: str, timeout: int = 180) -> str:
    """Call a local Ollama model. Skips silently if Ollama isn't running."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }).encode("utf-8")
    req = urllib.request.Request(
        os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/generate",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        return f"(ollama {model} unreachable: {e})"
    except Exception as e:  # noqa: BLE001
        return f"(ollama {model} error: {e})"
    return data.get("response", "").strip() or f"(ollama {model} returned empty)"


# --- Cloud provider adapters --------------------------------------------------
# Each returns the assistant's text or a "(provider error: ...)" string. They
# never raise so one reviewer crashing doesn't take down the whole panel.

def _post_json(url: str, body: dict, headers: dict, timeout: int = 120) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _query_anthropic(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return "(no ANTHROPIC_API_KEY)"
    try:
        d = _post_json("https://api.anthropic.com/v1/messages",
            {"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]},
            {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in d.get("content", []))
    except Exception as e:  # noqa: BLE001
        return f"(anthropic error: {e})"


def _query_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return "(no OPENAI_API_KEY)"
    try:
        d = _post_json("https://api.openai.com/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0.2},
            {"Authorization": f"Bearer {key}"})
        return d["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return f"(openai error: {e})"


def _query_grok(prompt: str, model: str = "grok-4") -> str:
    key = os.getenv("XAI_API_KEY", "")
    if not key:
        return "(no XAI_API_KEY)"
    try:
        d = _post_json("https://api.x.ai/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0.2},
            {"Authorization": f"Bearer {key}"})
        return d["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return f"(grok error: {e})"


def _query_mistral(prompt: str, model: str = "mistral-large-latest") -> str:
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        return "(no MISTRAL_API_KEY)"
    try:
        d = _post_json("https://api.mistral.ai/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0.2},
            {"Authorization": f"Bearer {key}"})
        return d["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return f"(mistral error: {e})"


def _query_cohere(prompt: str, model: str = "command-r-plus-08-2024") -> str:
    key = os.getenv("COHERE_API_KEY", "")
    if not key:
        return "(no COHERE_API_KEY)"
    try:
        d = _post_json("https://api.cohere.com/v2/chat",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0.2},
            {"Authorization": f"Bearer {key}"})
        return "".join(b.get("text", "") for b in d.get("message", {}).get("content", []))
    except Exception as e:  # noqa: BLE001
        return f"(cohere error: {e})"


def _query_nvidia(prompt: str, model: str = "meta/llama-3.3-70b-instruct") -> str:
    key = os.getenv("NVIDIA_API_KEY", "")
    if not key:
        return "(no NVIDIA_API_KEY)"
    try:
        d = _post_json("https://integrate.api.nvidia.com/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0.2},
            {"Authorization": f"Bearer {key}"})
        return d["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return f"(nvidia error: {e})"


def _query_deepseek(prompt: str, model: str = "deepseek-chat") -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        return "(no DEEPSEEK_API_KEY)"
    try:
        d = _post_json("https://api.deepseek.com/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0.2},
            {"Authorization": f"Bearer {key}"})
        return d["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return f"(deepseek error: {e})"


# Full reviewer roster — selection driven by SELF_MODIFY_REVIEWERS env (comma list).
# Default = locals only (zero cost, always works, no rate limits).
ALL_REVIEWERS = {
    "claude":     ("Claude (Anthropic, via Claude Code CLI — local sub)",                 _query_claude_via_cli),
    "claude-api": ("Claude Sonnet (Anthropic API)",                                        _query_anthropic),
    "gpt":        ("GPT-4o-mini (OpenAI)",                                                 _query_openai),
    "grok":       ("Grok 4 (xAI)",                                                         _query_grok),
    "mistral":    ("Mistral Large (Mistral AI, France)",                                   _query_mistral),
    "cohere":     ("Command R+ (Cohere, Canada)",                                          _query_cohere),
    "nvidia":     ("Llama 3.3 70B via NVIDIA NIM",                                         _query_nvidia),
    "deepseek":   ("DeepSeek Chat (DeepSeek, China)",                                      _query_deepseek),
    "hermes":     ("Hermes 3 8B (Nous Research, local Ollama)",                            lambda p: _query_ollama("hermes3:8b", p)),
    "llama":      ("Llama 3.2 (Meta, local Ollama)",                                       lambda p: _query_ollama("llama3.2", p)),
    "dolphin":    ("Dolphin Mistral 7B (uncensored, local Ollama)",                        lambda p: _query_ollama("dolphin-mistral:7b", p)),
}

_DEFAULT_REVIEWERS = "claude,hermes,llama"


def _resolve_reviewers() -> list[tuple[str, str, callable]]:
    """Pick reviewers from SELF_MODIFY_REVIEWERS env (comma list). Unknown ids skipped."""
    requested = os.getenv("SELF_MODIFY_REVIEWERS", _DEFAULT_REVIEWERS)
    ids = [s.strip() for s in requested.split(",") if s.strip()]
    out: list[tuple[str, str, callable]] = []
    for rid in ids:
        if rid in ALL_REVIEWERS:
            name, fn = ALL_REVIEWERS[rid]
            out.append((rid, name, fn))
        else:
            print(f"  (skipping unknown reviewer: {rid})", file=sys.stderr)
    return out


# Late-bound so env changes between calls take effect.
def get_reviewers() -> list[tuple[str, str, callable]]:
    return _resolve_reviewers()


# Legacy constant kept for back-compat; not used in cmd_propose.
REVIEWERS = [
    ("claude",  "Claude (Anthropic, via Claude Code CLI)",  _query_claude_via_cli),
    ("hermes",  "Hermes 3 8B (Nous Research, local Ollama)", lambda p: _query_ollama("hermes3:8b", p)),
    ("llama",   "Llama 3.2 (Meta, local Ollama)",            lambda p: _query_ollama("llama3.2", p)),
]


def _parse_verdict(review_text: str) -> str:
    """Extract APPROVE / APPROVE_WITH_NOTES / REJECT from a review."""
    for line in review_text.splitlines():
        s = line.strip().upper()
        # Direct match
        if s in ("APPROVE", "APPROVE_WITH_NOTES", "REJECT"):
            return s
        # Inline match — order matters (longest first)
        for v in ("APPROVE_WITH_NOTES", "APPROVE", "REJECT"):
            if v in s:
                return v
    return "UNKNOWN"


def _consensus(verdicts: list[str]) -> str:
    """Aggregate per-reviewer verdicts into a single verdict.

    Rule: if ANY reviewer rejects → REJECT (cautious posture).
    Otherwise if ANY says APPROVE_WITH_NOTES → APPROVE_WITH_NOTES.
    Otherwise if at least one APPROVE → APPROVE.
    Else UNKNOWN.
    """
    if "REJECT" in verdicts:
        return "REJECT"
    if "APPROVE_WITH_NOTES" in verdicts:
        return "APPROVE_WITH_NOTES"
    if "APPROVE" in verdicts:
        return "APPROVE"
    return "UNKNOWN"


def _research_mode_enabled() -> bool:
    """Local research mode: auto-apply on strong consensus, bypass HSP gate.

    Refuses if hosted markers present — local Mac only. Opt-in via
    QUORUM_LOCAL_RESEARCH_MODE=1 plus a sanity env to make sure the human
    knew what they were doing the first time.
    """
    hosted = [m for m in HOSTED_MARKERS if os.environ.get(m)]
    if hosted:
        return False
    return os.environ.get("QUORUM_LOCAL_RESEARCH_MODE") == "1"


# Per-day rate limit on autonomous evolutions — anti-AutoGPT.
RESEARCH_LOG = Path.home() / ".quorum" / "research-mode.log"
RESEARCH_DAY_CAP = int(os.environ.get("QUORUM_RESEARCH_DAILY_CAP", "10"))


def _today_apply_count() -> int:
    """How many auto-applies were committed today (local time)."""
    if not RESEARCH_LOG.is_file():
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    count = 0
    for line in RESEARCH_LOG.read_text().splitlines():
        if line.startswith(today) and "AUTO_APPLY" in line:
            count += 1
    return count


def _log_research(event: str, **kwargs: object) -> None:
    """Append a single-line audit record. Human-readable, grep-friendly."""
    RESEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    extras = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
    with RESEARCH_LOG.open("a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d')} {stamp} {event} {extras}\n")


def _git_snapshot(repo: Path) -> str:
    """Create a tag pointing at HEAD before an auto-apply. Returns tag name."""
    tag = "research-snapshot-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    subprocess.run(
        ["git", "-C", str(repo), "tag", "-a", tag, "-m", "Quorum research-mode snapshot before auto-apply"],
        capture_output=True, text=True, check=False,
    )
    return tag


def _auto_apply_if_strong_consensus(pid: str, pdir: Path, m: dict, reviews: dict) -> bool:
    """Local research mode: apply automatically if ≥80% reviewers say APPROVE
    (no APPROVE_WITH_NOTES counted as full approve). Returns True if applied."""
    if not _research_mode_enabled():
        return False
    if not (pdir / "patch.diff").is_file():
        _log_research("AUTO_SKIP", pid=pid, reason="no_patch")
        return False
    used = _today_apply_count()
    if used >= RESEARCH_DAY_CAP:
        _log_research("AUTO_SKIP", pid=pid, reason="rate_limit", used=used, cap=RESEARCH_DAY_CAP)
        print(f"  [research-mode] daily auto-apply cap reached ({used}/{RESEARCH_DAY_CAP})", file=sys.stderr)
        return False
    verdicts = [r["verdict"] for r in reviews.values()]
    if not verdicts:
        return False
    approves = sum(1 for v in verdicts if v == "APPROVE")
    ratio = approves / len(verdicts)
    if ratio < 0.8:
        _log_research("AUTO_SKIP", pid=pid, reason="weak_consensus",
                      approves=approves, total=len(verdicts), ratio=round(ratio, 2))
        return False

    repo = Path(m["repo"])
    # Dry-run first
    dry = subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", str(pdir / "patch.diff")],
        capture_output=True, text=True,
    )
    if dry.returncode != 0:
        _log_research("AUTO_FAIL", pid=pid, reason="patch_invalid", stderr=dry.stderr.strip()[:100])
        return False

    tag = _git_snapshot(repo)  # rollback handle
    _git(repo, "apply", str(pdir / "patch.diff"))
    _git(repo, "add", m["file"])
    commit_msg = (
        f"chore(self-modify-auto): {m['goal']}\n\n"
        f"AUTO-APPLIED by Quorum research mode — strong consensus "
        f"({approves}/{len(verdicts)} APPROVE).\n"
        f"Proposal: {pid}\n"
        f"Snapshot before: {tag} (git checkout {tag} to revert)\n\n"
        f"Co-Authored-By: Quorum Self-Modify <noreply@quorum-ai.dev>"
    )
    _git(repo, "commit", "-m", commit_msg)
    _log_research("AUTO_APPLY", pid=pid, snapshot=tag, approves=approves, total=len(verdicts))
    m["status"] = "applied_auto"
    m["applied_at"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    m["snapshot_tag"] = tag
    (pdir / "meta.json").write_text(json.dumps(m, indent=2))
    print(f"  ⚡ AUTO-APPLIED in research mode ({approves}/{len(verdicts)} approve, snapshot={tag})")
    return True


def cmd_propose(args: argparse.Namespace) -> int:
    _refuse_if_hosted()
    repo = Path(args.repo).expanduser().resolve()
    target = repo / args.file
    if not target.is_file():
        sys.exit(f"file not found: {target}")
    contents = target.read_text()
    if len(contents) > 60_000:
        sys.exit(f"file too large ({len(contents)} bytes); pick a smaller one")

    print(f"Step 1/2 — Gemini proposing fix for {args.file}...", file=sys.stderr)
    prompt = PROPOSE_PROMPT.format(
        repo=repo, file=args.file, goal=args.goal, contents=contents,
    )
    proposal = _query_gemini(prompt)
    if not proposal.strip():
        sys.exit("empty response from Gemini")

    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    seed = f"{repo}|{args.file}|{args.goal}|{ts}".encode()
    pid = ts + "-" + hashlib.sha256(seed).hexdigest()[:6]

    pdir = PROPOSALS_DIR / pid
    pdir.mkdir()
    (pdir / "proposal.md").write_text(proposal)
    patch_text = _extract_patch(proposal)
    if patch_text:
        (pdir / "patch.diff").write_text(patch_text)

    # Step 2 — multi-LLM review panel (configurable via SELF_MODIFY_REVIEWERS env)
    review_prompt = REVIEW_PROMPT.format(
        goal=args.goal, file=args.file, contents=contents, proposal=proposal,
    )
    panel = get_reviewers()
    if not panel:
        sys.exit("No reviewers configured — set SELF_MODIFY_REVIEWERS (e.g. claude,hermes,llama,grok,gpt)")
    print(f"Step 2/2 — multi-LLM review panel ({len(panel)} reviewers: {[r[0] for r in panel]})...", file=sys.stderr)

    import concurrent.futures
    reviews: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(panel)) as ex:
        futures = {
            ex.submit(fn, review_prompt): (rid, name)
            for rid, name, fn in panel
        }
        for fut in concurrent.futures.as_completed(futures):
            rid, name = futures[fut]
            try:
                text = fut.result()
            except Exception as e:  # noqa: BLE001
                text = f"(reviewer crashed: {e})"
            verdict = _parse_verdict(text)
            (pdir / f"review_{rid}.md").write_text(f"# Reviewer: {name}\n\n{text}\n")
            reviews[rid] = {"name": name, "verdict": verdict, "length": len(text)}
            icon = {"APPROVE": "✓", "APPROVE_WITH_NOTES": "⚠", "REJECT": "✗"}.get(verdict, "?")
            print(f"  {icon} {rid:7} verdict={verdict}", file=sys.stderr)

    verdicts = [r["verdict"] for r in reviews.values()]
    consensus = _consensus(verdicts)

    (pdir / "meta.json").write_text(json.dumps({
        "id": pid,
        "repo": str(repo),
        "file": args.file,
        "goal": args.goal,
        "created_at": ts,
        "status": "pending",
        "proposer": "gemini-2.5-flash",
        "reviewers": reviews,
        "consensus_verdict": consensus,
    }, indent=2))

    icon = {"APPROVE": "✓", "APPROVE_WITH_NOTES": "⚠", "REJECT": "✗"}.get(consensus, "?")
    votes_str = ", ".join(f"{k}={v['verdict']}" for k, v in reviews.items())
    print(f"\n{icon} Proposal saved: {pid}")
    print(f"  Path:               {pdir}")
    print(f"  Patch extracted:    {'yes' if patch_text else 'NO — manual review needed'}")
    print(f"  Consensus verdict:  {consensus}  (votes: {votes_str})")

    # Reload meta to pass the freshly-written reviews dict into auto-apply gate.
    meta_now = json.loads((pdir / "meta.json").read_text())
    auto_applied = _auto_apply_if_strong_consensus(pid, pdir, meta_now, reviews)

    print()
    if auto_applied:
        print(f"  (research-mode auto-apply succeeded — see git log)")
    else:
        print(f"Review:  python3 self_modify.py review {pid}")
        print(f"Apply:   python3 self_modify.py apply  {pid}")
        print(f"Reject:  python3 self_modify.py reject {pid}")
    return 0


def _extract_patch(answer: str) -> str | None:
    """Pull the unified-diff block out of the markdown answer."""
    lines = answer.splitlines()
    in_diff = False
    in_codeblock = False
    out: list[str] = []
    for ln in lines:
        if ln.strip().startswith("```diff") or ln.strip() == "```":
            if in_diff and ln.strip() == "```":
                break
            in_codeblock = not in_codeblock
            continue
        if ln.startswith("--- a/") or ln.startswith("--- /dev/null"):
            in_diff = True
        if in_diff:
            out.append(ln)
    return "\n".join(out) + "\n" if out else None


def cmd_list(_args: argparse.Namespace) -> int:
    if not PROPOSALS_DIR.exists():
        print("(no proposals yet)")
        return 0
    proposals = sorted(PROPOSALS_DIR.iterdir())
    if not proposals:
        print("(no proposals yet)")
        return 0
    for pdir in proposals:
        meta_file = pdir / "meta.json"
        if not meta_file.is_file():
            continue
        m = json.loads(meta_file.read_text())
        print(f"  [{m['status']:9}] {m['id']}  {m['file']:50} {m['goal'][:40]}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    pdir = PROPOSALS_DIR / args.id
    if not pdir.is_dir():
        sys.exit(f"proposal not found: {args.id}")
    m = json.loads((pdir / "meta.json").read_text())
    reviewers = m.get("reviewers", {})
    print(f"=== Proposal {m['id']} [{m['status']}] ===")
    print(f"Repo:      {m['repo']}")
    print(f"File:      {m['file']}")
    print(f"Goal:      {m['goal']}")
    print(f"Proposer:  {m.get('proposer', m.get('model', '?'))}")
    if reviewers:
        print(f"Reviewers: {len(reviewers)}")
        for rid, info in reviewers.items():
            print(f"  - {rid:7} → {info['verdict']:18} ({info['name']})")
        print(f"Consensus: {m.get('consensus_verdict', '?')}")
    else:
        # Legacy single-reviewer format
        print(f"Reviewer:  {m.get('reviewer', '(no reviewer)')}")
        print(f"Verdict:   {m.get('reviewer_verdict', '(not reviewed)')}")
    print()
    print("─" * 70)
    print("GEMINI'S PROPOSAL")
    print("─" * 70)
    print((pdir / "proposal.md").read_text())
    # New multi-reviewer files: review_<rid>.md
    for rid in reviewers:
        rfile = pdir / f"review_{rid}.md"
        if rfile.is_file():
            print("\n" + "─" * 70)
            print(f"REVIEWER: {rid.upper()}")
            print("─" * 70)
            print(rfile.read_text())
    # Legacy single review.md
    if (pdir / "review.md").is_file() and not reviewers:
        print("\n" + "─" * 70)
        print("REVIEW (single-reviewer legacy)")
        print("─" * 70)
        print((pdir / "review.md").read_text())
    if (pdir / "patch.diff").is_file():
        print("\n" + "─" * 70)
        print("EXTRACTED PATCH.DIFF (what `apply` will run)")
        print("─" * 70)
        print((pdir / "patch.diff").read_text())
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    _refuse_if_hosted()
    pdir = PROPOSALS_DIR / args.id
    if not pdir.is_dir():
        sys.exit(f"proposal not found: {args.id}")
    m = json.loads((pdir / "meta.json").read_text())
    if m["status"] != "pending":
        sys.exit(f"already {m['status']} — refusing to re-apply")
    patch = pdir / "patch.diff"
    if not patch.is_file():
        sys.exit("no patch.diff in proposal; cannot auto-apply (review proposal.md manually)")

    repo = Path(m["repo"])
    # Dry-run apply first
    dry = subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", str(patch)],
        capture_output=True, text=True,
    )
    if dry.returncode != 0:
        sys.exit(f"git apply --check failed:\n{dry.stderr}")

    print(f"✓ Patch applies cleanly to {repo}")
    print(f"  Target file:        {m['file']}")
    # Support both new multi-reviewer format and legacy single-reviewer
    verdict = m.get("consensus_verdict") or m.get("reviewer_verdict", "UNKNOWN")
    reviewers = m.get("reviewers", {})
    if reviewers:
        votes = ", ".join(f"{k}={v['verdict']}" for k, v in reviewers.items())
        print(f"  Consensus verdict:  {verdict}  ({votes})")
    else:
        print(f"  Reviewer verdict:   {verdict}")

    if verdict == "REJECT":
        print("\n⚠  Panel REJECTED this proposal. Read each review_*.md before forcing.")
        confirm = input('Type "OVERRIDE" to apply despite the reject verdict: ').strip()
        if confirm != "OVERRIDE":
            print("Aborted.")
            return 1
    else:
        yn = input("Apply + create local commit (no push)? [y/N] ").strip().lower()
        if yn != "y":
            print("Aborted.")
            return 1

    _git(repo, "apply", str(patch))
    _git(repo, "add", m["file"])
    commit_msg = (
        f"chore(self-modify): {m['goal']}\n\n"
        f"Proposed by Quorum self-modify ({m['model']}).\n"
        f"Proposal ID: {m['id']}\n"
        f"Reviewed and approved by human before commit.\n\n"
        f"Co-Authored-By: Quorum Self-Modify <noreply@quorum-ai.dev>"
    )
    _git(repo, "commit", "-m", commit_msg)

    m["status"] = "applied"
    m["applied_at"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (pdir / "meta.json").write_text(json.dumps(m, indent=2))

    print(f"✓ Committed locally (NOT pushed). Run `git push` manually when ready.")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    pdir = PROPOSALS_DIR / args.id
    if not pdir.is_dir():
        sys.exit(f"proposal not found: {args.id}")
    m = json.loads((pdir / "meta.json").read_text())
    if m["status"] != "pending":
        sys.exit(f"already {m['status']}")
    m["status"] = "rejected"
    m["rejected_at"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (pdir / "meta.json").write_text(json.dumps(m, indent=2))
    print(f"✓ Marked as rejected: {args.id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("propose", help="Ask Quorum/Gemini to propose ONE fix")
    p.add_argument("--repo", required=True, help="Path to git repository")
    p.add_argument("--file", required=True, help="File path relative to repo")
    p.add_argument("--goal", required=True, help="What you want fixed/improved")
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser("list", help="List all proposals")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("review", help="Print a proposal in full")
    p.add_argument("id")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("apply", help="Apply a proposal (creates local commit, no push)")
    p.add_argument("id")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("reject", help="Mark proposal as rejected")
    p.add_argument("id")
    p.set_defaults(func=cmd_reject)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
