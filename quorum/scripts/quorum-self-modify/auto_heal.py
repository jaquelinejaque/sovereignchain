#!/usr/bin/env python3
"""Quorum Auto-Heal — close the loop on the propose/apply flow.

Reads Cloud Run error logs, identifies a target file from the traceback,
runs self_modify.py propose, then self_modify.py apply --yes if the
multi-LLM review panel reaches APPROVE/APPROVE_WITH_NOTES consensus.

By owner decision (2026-06-23) this STILL stops at the local commit —
never pushes, never deploys. Owner runs `git push && gcloud run deploy`
when they want the heal to reach production.

Scheduled by ~/Library/LaunchAgents/dev.quorum-ai.auto-heal.plist
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELF_MODIFY = Path(__file__).resolve().parent / "self_modify.py"
STATE_DIR = Path.home() / ".quorum" / "auto-heal"
STATE_DIR.mkdir(parents=True, exist_ok=True)
SEEN_FILE = STATE_DIR / "seen-errors.json"
RUN_LOG = STATE_DIR / "run.log"

# Files auto_heal will NEVER touch without owner. Per [[feedback_no_cli_py_commit]]
# and the owner's licence/billing risk profile.
SKIP_FILES = {
    "src/quorum/cli.py",
    "src/quorum/_license.py",
    "src/quorum/server/main.py",  # contains Stripe webhook + signup
}

CLOUD_RUN_SERVICE = "quorum-api"
CLOUD_RUN_REGION = "europe-west1"

# Local stderr logs the LaunchAgents write to. Auto-heal was blind to these
# before — observed 2026-06-28: 112 EMFILE traces accumulated 26h before
# anyone noticed because every run reported "fetched 0 error logs" (it was
# only ever querying gcloud, never the Mac).
LOCAL_LOG_DIR = Path.home() / ".quorum"
LOCAL_LOG_GLOBS = (
    "launchagent-stderr.log",
    "*-stderr.log",
    "continuous-stderr.log",
    "auto-heal/launchd.err",
)
# Cap how much of each file we scan. Stderr can balloon (the EMFILE log hit
# 1.6 MB before truncation); rich tracebacks are <2 KB so the last 200 KB
# easily covers the most recent dozen failures without OOM risk in cron.
LOCAL_LOG_TAIL_BYTES = 200_000

# Owner's personal Gemini key for the LOCAL cron only. Never leaves this Mac.
# Customers use BYOK via /v1/customer/keys — they never see this key.
GEMINI_KEY_FILE = Path.home() / ".config" / "quorum" / "gemini.key"


def _load_owner_gemini_key() -> str | None:
    if not GEMINI_KEY_FILE.is_file():
        return None
    key = GEMINI_KEY_FILE.read_text().strip()
    return key or None


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}"
    print(line, file=sys.stderr)
    with RUN_LOG.open("a") as fh:
        fh.write(line + "\n")


def load_seen() -> set[str]:
    if not SEEN_FILE.is_file():
        return set()
    return set(json.loads(SEEN_FILE.read_text()))


def save_seen(seen: set[str]) -> None:
    SEEN_FILE.write_text(json.dumps(sorted(seen)))


def fetch_recent_errors(freshness: str = "2h", limit: int = 50) -> list[dict]:
    """Pull error logs from Cloud Run via gcloud."""
    cmd = [
        "gcloud", "logging", "read",
        f'resource.type="cloud_run_revision" AND '
        f'resource.labels.service_name="{CLOUD_RUN_SERVICE}" AND '
        f'severity>=ERROR',
        f"--limit={limit}",
        "--format=json",
        f"--freshness={freshness}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        log(f"gcloud failed: {r.stderr[:200]}")
        return []
    try:
        return json.loads(r.stdout) or []
    except json.JSONDecodeError:
        return []


def _tail_bytes(path: Path, n: int = LOCAL_LOG_TAIL_BYTES) -> str:
    """Read up to ``n`` trailing bytes of ``path`` as utf-8 (errors=replace)."""
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size == 0:
        return ""
    with path.open("rb") as fh:
        if size > n:
            fh.seek(size - n)
        data = fh.read()
    return data.decode("utf-8", errors="replace")


def _split_tracebacks(text: str) -> list[str]:
    """Split a log blob into individual Python traceback blocks.

    Returns each block from a ``Traceback (most recent call last):`` line up
    to (and including) the terminating ``ExceptionType: message`` line.
    Tolerates the rich.traceback output the launchd jobs produce (boxed
    frames with ``│`` chars and ``❱`` arrows) — those still contain the
    canonical ``File "..."`` and ``ExceptionType: ...`` lines that
    :func:`parse_traceback` keys off, so we can pass them through unchanged.
    """
    blocks: list[str] = []
    in_block = False
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Plain or rich-boxed traceback header.
        if "Traceback (most recent call last)" in stripped:
            if in_block and current:
                blocks.append("\n".join(current))
            current = [line]
            in_block = True
            continue
        if not in_block:
            continue
        current.append(line)
        # Heuristic terminator: a bare "ExcType: message" line at the left
        # margin (not inside a rich box) closes the frame.
        if re.match(r"^[A-Z][A-Za-z_]*(Error|Exception):", stripped) and "│" not in line:
            blocks.append("\n".join(current))
            current = []
            in_block = False
    if in_block and current:
        blocks.append("\n".join(current))
    return blocks


def fetch_local_errors() -> list[dict]:
    """Pull tracebacks from the launchd stderr logs on this Mac.

    Returns the same shape as :func:`fetch_recent_errors` so the main
    loop can concatenate the two transparently. A ``_source: "local"``
    marker is added to each record so downstream logging shows where the
    error came from.
    """
    out: list[dict] = []
    if not LOCAL_LOG_DIR.is_dir():
        return out
    seen_paths: set[Path] = set()
    for pattern in LOCAL_LOG_GLOBS:
        for p in LOCAL_LOG_DIR.glob(pattern):
            if p in seen_paths or not p.is_file():
                continue
            seen_paths.add(p)
            text = _tail_bytes(p)
            if not text:
                continue
            for block in _split_tracebacks(text):
                out.append({
                    "textPayload": block,
                    "_source": "local",
                    "_path": str(p),
                })
    return out


_FILE_FRAME_PATTERNS = (
    # Container site-packages (Cloud Run): /opt/venv/.../site-packages/quorum/x.py
    re.compile(
        r'File "(?:/opt/venv/lib/python3\.\d+/site-packages/)(quorum/[^"]+\.py)",'
        r' line (\d+)'
    ),
    # System / framework site-packages (local Mac):
    # /Library/Frameworks/Python.framework/Versions/3.X/lib/python3.X/site-packages/quorum/x.py
    # /Users/.../.venv/lib/python3.X/site-packages/quorum/x.py
    re.compile(
        r'File "[^"]*site-packages/(quorum/[^"]+\.py)", line (\d+)'
    ),
    # Editable install / source tree (local Mac): /Users/.../sovereignchain/quorum/src/quorum/x.py
    re.compile(
        r'File "[^"]*/sovereignchain/quorum/src/(quorum/[^"]+\.py)",'
        r' line (\d+)'
    ),
)


def parse_traceback(text: str) -> dict | None:
    """Extract (file, line, exc_type, exc_msg) from a Python traceback.

    Handles three frame-path shapes:
      1. ``/opt/venv/.../site-packages/quorum/...`` — Cloud Run container
      2. ``…/site-packages/quorum/...``             — local Python install
      3. ``…/sovereignchain/quorum/src/quorum/...`` — editable / source tree

    Returns the deepest matching frame (last entry in the traceback).
    """
    file_matches: list[tuple[str, str]] = []
    for pat in _FILE_FRAME_PATTERNS:
        file_matches.extend(pat.findall(text))
    if not file_matches:
        return None
    file_rel, lineno = file_matches[-1]
    # Site-packages "quorum/..." → repo "src/quorum/...". The source-tree
    # pattern already strips the prefix down to "quorum/..." via its
    # capture group, so this prefix is correct in all three cases.
    repo_rel = f"src/{file_rel}"

    exc_match = re.search(r'^(\w+Error|\w+Exception): (.+)$', text, re.MULTILINE)
    exc_type = exc_match.group(1) if exc_match else "UnknownError"
    exc_msg = exc_match.group(2).strip()[:200] if exc_match else ""

    return {
        "file": repo_rel,
        "line": int(lineno),
        "exc_type": exc_type,
        "exc_msg": exc_msg,
        "fingerprint": f"{repo_rel}:{lineno}:{exc_type}:{exc_msg[:60]}",
    }


def already_fixed_in_main(error: dict) -> bool:
    """Skip if the deployed version is behind the local fix.

    Cheap heuristic: if git log mentions the exception type since the deploy
    timestamp recorded in state, the fix may already exist on disk awaiting deploy.
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "log", "--since=14.days", "--oneline", "--all"],
            capture_output=True, text=True, timeout=10,
        )
        recent_log = r.stdout.lower()
        return error["exc_type"].lower() in recent_log or error["exc_msg"][:30].lower() in recent_log
    except Exception:
        return False


def propose(target_file: str, goal: str) -> str | None:
    """Run self_modify.py propose and return the proposal ID, or None."""
    log(f"propose: {target_file} | {goal[:80]}")
    r = subprocess.run(
        [sys.executable, str(SELF_MODIFY), "propose",
         "--repo", str(REPO),
         "--file", target_file,
         "--goal", goal],
        capture_output=True, text=True, timeout=300,
    )
    output = r.stdout + r.stderr
    if r.returncode != 0:
        log(f"propose failed (exit {r.returncode}): {output[-300:]}")
        return None
    # Output line: "✓ Proposal saved: <id>"
    m = re.search(r"Proposal saved: (\S+)", output)
    return m.group(1) if m else None


def apply_if_approved(proposal_id: str) -> bool:
    """Apply the proposal with --yes. Returns True if commit was created."""
    pdir = Path.home() / ".quorum" / "proposals" / proposal_id
    meta = json.loads((pdir / "meta.json").read_text())
    verdict = meta.get("consensus_verdict")
    log(f"verdict for {proposal_id}: {verdict}")
    if verdict not in ("APPROVE", "APPROVE_WITH_NOTES"):
        log(f"skip apply: verdict={verdict}")
        return False
    if not (pdir / "patch.diff").is_file():
        log("skip apply: no patch.diff extracted")
        return False
    r = subprocess.run(
        [sys.executable, str(SELF_MODIFY), "apply", proposal_id, "--yes"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        log(f"apply failed: {r.stdout[-200:]} | {r.stderr[-200:]}")
        return False
    log(f"applied: {proposal_id}")
    return True


def main() -> int:
    log("=== run start ===")
    owner_key = _load_owner_gemini_key()
    if owner_key:
        os.environ["GEMINI_API_KEY"] = owner_key
        log(f"loaded owner gemini key from {GEMINI_KEY_FILE} (len={len(owner_key)})")
    elif not os.environ.get("GEMINI_API_KEY"):
        log(f"no key at {GEMINI_KEY_FILE} and GEMINI_API_KEY not in env — propose will fail")
    seen = load_seen()
    freshness = os.environ.get("AUTO_HEAL_FRESHNESS", "6h")
    cloud_errors = fetch_recent_errors(freshness=freshness)
    local_errors = fetch_local_errors()
    log(f"fetched {len(cloud_errors)} cloud + {len(local_errors)} local error blocks")

    parsed: list[dict] = []
    for r in cloud_errors + local_errors:
        text = r.get("textPayload") or json.dumps(r.get("jsonPayload", {}))
        info = parse_traceback(text)
        if not info:
            continue
        # Tag the source so logs make it obvious whether the heal target
        # came from Cloud Run or the Mac. Fingerprint stays source-agnostic
        # so the same fix doesn't get re-attempted once for each source.
        info["source"] = r.get("_source", "cloud")
        if r.get("_path"):
            info["source_path"] = r["_path"]
        parsed.append(info)

    log(f"parsed {len(parsed)} python tracebacks")

    healed = 0
    skipped_seen = 0
    skipped_skiplist = 0
    skipped_already_fixed = 0

    for info in parsed:
        fp = info["fingerprint"]
        if fp in seen:
            skipped_seen += 1
            continue
        if info["file"] in SKIP_FILES:
            log(f"SKIP_FILES: {info['file']}")
            skipped_skiplist += 1
            seen.add(fp)
            continue
        if already_fixed_in_main(info):
            log(f"already_fixed_in_main: {fp}")
            skipped_already_fixed += 1
            seen.add(fp)
            continue
        if not (REPO / info["file"]).is_file():
            log(f"target_missing: {info['file']}")
            seen.add(fp)
            continue

        source = info.get("source", "cloud")
        origin = (
            "Cloud Run production logs" if source == "cloud"
            else f"local launchd stderr ({info.get('source_path', 'unknown')})"
        )
        goal = (
            f"Fix the {info['exc_type']} at line {info['line']}: {info['exc_msg']}. "
            f"This error was observed in {origin}. "
            f"Make the minimal change that prevents this exception."
        )
        pid = propose(info["file"], goal)
        if not pid:
            continue
        if apply_if_approved(pid):
            healed += 1
        seen.add(fp)
        # Rate limit: max 3 heals per run
        if healed >= 3:
            log("hit per-run cap of 3 heals")
            break

    save_seen(seen)
    log(f"=== run end: healed={healed} skipped_seen={skipped_seen} "
        f"skipped_skiplist={skipped_skiplist} skipped_already_fixed={skipped_already_fixed} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
