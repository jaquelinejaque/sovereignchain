# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Quorum Proactive — autonomous monitoring + draft generation layer.

Pipeline:
    Ingest (Twitter / RSS / Gmail / LinkedIn / sites)
        → Signal (raw item with provenance + dedupe hash)
        → Score (4-LLM consensus rates "is this worth acting on?")
        → Draft (LLM writes proposed action: DM, email, post)
        → Notify (email digest to owner with approve/reject links)
        → Execute (ONLY after owner clicks approve)

Hard rules:
* Never auto-execute anything that costs money or speaks publicly in
  the owner's name. Every external action is owner-gated.
* Every signal/draft/decision is logged in HSP-compatible hash chain
  for full auditability.
* Privacy: opt-in per source. Gmail ingestion warns at install.

This module is **brand new** (2026-06-22) and intentionally minimal —
the MVP works first, optimisations come after first real signal lands.
"""

from quorum.proactive.signal import Signal, Draft, Action

__all__ = ["Signal", "Draft", "Action"]
