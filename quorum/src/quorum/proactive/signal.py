# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Data shapes that flow through the proactive pipeline.

Three core records:

  Signal   — raw item from an ingest source (tweet, email, RSS item, ...)
  Draft    — LLM-proposed action attached to a signal (DM/email/post text)
  Action   — owner's decision on a draft (approve/reject/edit + executed?)

Everything is a dataclass — no pydantic on the hot path. The store
serialises to JSON for SQLite persistence.

Hash chain: every record carries ``prev_hash`` so the store can verify
the chain end-to-end (HSP audit compatibility).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(payload: dict[str, Any], prev: str = "") -> str:
    """Stable SHA-256 of (prev_hash || canonical JSON of payload)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Signal                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class Signal:
    """One raw item ingested from any source.

    Source can be: ``twitter`` / ``rss`` / ``gmail`` / ``linkedin`` /
    ``site`` / ``manual``. Source-specific metadata lives in ``extra``.

    ``dedupe_key`` is a stable hash of (source + external_id) — used to
    drop duplicates without re-running the full pipeline.
    """

    source: str
    external_id: str  # source-specific id (tweet id, email message-id, url, ...)
    author: str  # display name / handle / sender
    title: str  # subject / tweet text first line / RSS title
    body: str  # full text
    url: str  # canonical link back to the item
    fetched_at: str = field(default_factory=_now)
    extra: dict[str, Any] = field(default_factory=dict)
    # Filled in by the pipeline:
    id: str = ""  # SHA-256 of payload (chain-friendly)
    prev_hash: str = ""  # previous record's hash in the audit chain

    @property
    def dedupe_key(self) -> str:
        return hashlib.sha256(
            f"{self.source}::{self.external_id}".encode("utf-8")
        ).hexdigest()

    def seal(self, prev_hash: str = "") -> None:
        """Compute id from current contents. Call after all fields are set."""
        self.prev_hash = prev_hash
        payload = {k: v for k, v in asdict(self).items() if k not in ("id",)}
        self.id = _hash(payload, prev_hash)

    def to_row(self) -> dict[str, Any]:
        return asdict(self) | {"dedupe_key": self.dedupe_key}


# --------------------------------------------------------------------------- #
# Draft                                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class Draft:
    """An LLM-proposed action attached to a signal.

    ``kind`` is the medium: ``dm`` / ``email_reply`` / ``post`` / ``note``.
    ``intent_score`` is the consensus score (0..1) on "is this signal
    worth acting on?". ``draft_score`` is the consensus score on "is this
    draft good enough to send?".
    """

    signal_id: str  # FK → Signal.id
    kind: str  # dm | email_reply | post | note
    target: str  # who/where it goes (handle, email address, channel)
    subject: str = ""  # for emails / posts (titles)
    body: str = ""
    intent_score: float = 0.0
    draft_score: float = 0.0
    rationale: str = ""  # short LLM explanation: why this draft
    consensus_models: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    id: str = ""
    prev_hash: str = ""

    def seal(self, prev_hash: str = "") -> None:
        self.prev_hash = prev_hash
        payload = {k: v for k, v in asdict(self).items() if k not in ("id",)}
        self.id = _hash(payload, prev_hash)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Action                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class Action:
    """The owner's decision on a draft.

    ``decision`` is one of: ``approved`` / ``rejected`` / ``edited`` /
    ``pending``. ``executed_at`` is filled only when an approved action
    is actually sent to the world; ``execution_result`` captures the
    side-effect (tweet id, message id, error).
    """

    draft_id: str  # FK → Draft.id
    decision: str = "pending"  # pending | approved | rejected | edited
    edited_body: str = ""  # if owner tweaked the draft
    decided_at: str = ""
    executed_at: str = ""
    execution_result: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    prev_hash: str = ""

    def seal(self, prev_hash: str = "") -> None:
        self.prev_hash = prev_hash
        payload = {k: v for k, v in asdict(self).items() if k not in ("id",)}
        self.id = _hash(payload, prev_hash)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["Signal", "Draft", "Action"]
