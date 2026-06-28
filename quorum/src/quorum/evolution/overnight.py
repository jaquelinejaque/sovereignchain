"""Overnight learning runner — autonomous topic harvest with rate-limit defenses.

Reads a topic list from a file, harvests one topic at a time with
randomized backoff between calls, detects DuckDuckGo throttling and
sleeps longer when it happens, and writes progress + summary to disk.

Designed to run for hours unattended via:

    nohup quorum overnight \\
        --topics-file quorum_overnight_topics.txt \\
        --max-hours 8 \\
        --log ~/Desktop/quorum-overnight.log &

Stop signal: create the file pointed to by --stop-file (default
``$QUORUM_DATA_DIR/STOP_OVERNIGHT``) and the loop exits at the next
topic boundary.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from quorum.evolution.web_learner import harvest, stats

logger = logging.getLogger("quorum.evolution.overnight")


# Opt-in: when set, the overnight loop routes every harvest through the
# multi-source + LLM-oracle fallback chain in kb_harvest_fallback.
# Default off so a regression here can't silently change the primary path.
_USE_FALLBACK = os.environ.get("QUORUM_KB_FALLBACK", "0") == "1"
_FALLBACK_AFTER_DRY = int(os.environ.get("QUORUM_FALLBACK_DRY_STREAK", "2"))


def _data_dir() -> Path:
    p = Path(os.environ.get("QUORUM_DATA_DIR", str(Path.home() / ".quorum")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_topics(path: str) -> List[str]:
    topics: List[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        topics.append(line)
    return topics


async def run_overnight(
    *,
    topics_file: str,
    max_hours: float = 8.0,
    base_delay_s: float = 120.0,
    jitter_s: float = 60.0,
    n_search: int = 4,
    max_chunks: int = 4,
    log_path: str | None = None,
    stop_file: str | None = None,
    shuffle: bool = True,
) -> dict:
    """Run the overnight loop.

    Args:
        topics_file: Path to a newline-delimited topic list (# = comment).
        max_hours: Wall-clock budget; loop stops when exceeded.
        base_delay_s: Mean delay between harvests (DDG-safe).
        jitter_s: +/- jitter around the mean.
        n_search, max_chunks: Forwarded to ``harvest``.
        log_path: Where to append plaintext progress.
        stop_file: Touch this path to stop the loop cleanly.
    """
    topics = _load_topics(topics_file)
    if shuffle:
        # Use a deterministic-ish shuffle from time mod (no Random in workflow scripts elsewhere)
        seed = int(time.time()) & 0xFFFF
        rnd = random.Random(seed)
        rnd.shuffle(topics)

    log_p = Path(log_path or (_data_dir() / f"overnight-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"))
    stop_p = Path(stop_file or (_data_dir() / "STOP_OVERNIGHT"))
    stop_p.unlink(missing_ok=True)  # clear any stale signal

    started = time.time()
    deadline = started + max_hours * 3600
    done: List[dict] = []
    dry_streak = 0  # consecutive harvests that stored 0 facts (likely rate-limited)

    def _log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        with open(log_p, "a") as f:
            f.write(line + "\n")

    _log(f"Starting overnight learner. topics={len(topics)}  max_hours={max_hours}  log={log_p}  stop_file={stop_p}")
    _log(f"Initial KB: {stats()}")

    for i, topic in enumerate(topics, 1):
        if stop_p.exists():
            _log("STOP signal detected — exiting cleanly.")
            break
        if time.time() > deadline:
            _log(f"Wall-clock budget reached ({max_hours}h) — exiting cleanly.")
            break

        elapsed_h = (time.time() - started) / 3600
        _log(f"[{i}/{len(topics)}  elapsed={elapsed_h:.1f}h] HARVEST: {topic!r}")
        # Route to fallback chain once dry_streak crosses the threshold, so
        # we only pay the multi-source / oracle cost when DDG is actually
        # throttling. The first 1–2 dry hits could just be unlucky topics.
        use_fallback = _USE_FALLBACK and dry_streak >= _FALLBACK_AFTER_DRY
        try:
            if use_fallback:
                from quorum.evolution.kb_harvest_fallback import harvest_with_fallback
                r = await harvest_with_fallback(
                    topic, n_search=n_search, max_chunks_per_source=max_chunks,
                )
            else:
                r = await harvest(topic, n_search=n_search, max_chunks_per_source=max_chunks)
        except Exception as e:  # noqa: BLE001
            _log(f"  ERROR on {topic!r}: {type(e).__name__}: {e}")
            r = {"topic": topic, "error": str(e), "stored": 0}

        stored = r.get("stored", 0)
        fb = r.get("fallback_used")
        fb_note = f"  fallback={fb}" if fb else ""
        _log(f"  → stored={stored}  fetched={r.get('fetched_sources', 0)}  candidate_chunks={r.get('candidate_chunks', 0)}{fb_note}")
        done.append({"topic": topic, **r})

        # Detect DDG throttle: consecutive zero-stored runs → long sleep
        if stored == 0:
            dry_streak += 1
            if dry_streak >= 3:
                cool = 1200 + random.randint(0, 300)  # 20–25 min
                _log(f"  Dry streak {dry_streak} → DDG likely throttling. Sleeping {cool}s.")
                await asyncio.sleep(cool)
                dry_streak = 0
                continue
        else:
            dry_streak = 0

        delay = base_delay_s + random.uniform(-jitter_s, jitter_s)
        _log(f"  Sleeping {delay:.0f}s before next topic.")
        await asyncio.sleep(max(15, delay))

    final_stats = stats()
    summary = {
        "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "ended": datetime.now(timezone.utc).isoformat(),
        "wall_hours": (time.time() - started) / 3600,
        "topics_attempted": len(done),
        "topics_with_facts": sum(1 for d in done if d.get("stored", 0) > 0),
        "total_facts_stored": sum(d.get("stored", 0) for d in done),
        "final_kb": final_stats,
        "log_path": str(log_p),
    }
    _log(f"FINAL: {summary}")
    return summary


__all__ = ["run_overnight"]
