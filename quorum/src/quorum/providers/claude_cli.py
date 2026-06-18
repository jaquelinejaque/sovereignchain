"""Claude Code CLI provider — uses the locally-authenticated `claude` binary
instead of the Anthropic billed API.

WHY THIS EXISTS
---------------
The hosted Anthropic API charges per-token from a credit balance. Most
developers running Quorum locally already pay for a Claude Pro/Max
subscription via `claude` (the Claude Code CLI), which is a separate
billing relationship from the API.

This provider lets the local Quorum CLI include Claude in its consensus
panel by shelling out to `claude -p` (headless mode), using the user's
existing CLI authentication — no API key, no credit balance.

CONSTRAINTS
-----------
* LOCAL ONLY. The hosted Quorum API explicitly excludes this provider
  via the env-marker check in registry.py — a Cloud Run container has
  no `claude` binary and no logged-in session.
* Requires `claude` on PATH (install Claude Code).
* Cost is reported as 0.0 because the user pays via their subscription,
  not per-token. This is honest for the user's bookkeeping; Quorum is
  not the one being billed.
* Latency is variable (CLI subprocess + Claude inference).
"""

from __future__ import annotations

import asyncio
import os
import shutil

from quorum.providers.base import ModelResponse, Provider


class ClaudeCLIProvider(Provider):
    """Calls the local `claude -p` binary. No API key required."""

    name = "claude-cli"

    def __init__(self, timeout_s: float = 90.0):
        self.timeout_s = timeout_s

    async def complete(
        self, prompt: str, *, max_tokens: int = 800, **kwargs: object
    ) -> ModelResponse:
        if shutil.which("claude") is None:
            return ModelResponse(
                name=self.name, response="",
                error="claude_cli_not_installed",
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ},
            )
        except Exception as e:  # noqa: BLE001
            return ModelResponse(name=self.name, response="", error=f"spawn_failed: {e}")
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
            return ModelResponse(
                name=self.name, response="",
                error=f"timeout after {self.timeout_s}s",
            )
        if proc.returncode != 0:
            return ModelResponse(
                name=self.name, response="",
                error=f"exit_{proc.returncode}: {stderr.decode(errors='replace')[:200]}",
            )
        text = stdout.decode(errors="replace").strip()
        # Cost is 0.0 — subscription pays for it, not Quorum
        return ModelResponse(
            name=self.name, response=text, tokens_in=0, tokens_out=0, cost_usd=0.0,
        )
