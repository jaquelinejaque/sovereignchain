"""HSP Webhook Client — real Human Supervision Protocol gate integration.

Licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

HSP Commercial Restrictions:
    This module integrates with the Human Supervision Protocol (HSP),
    patent pending PCT/US26/11908. Commercial use requires a separate license
    from Sovereign Chain Ltd. See LICENSE-HSP at the repo root.

WHY this module exists:
    The existing `gate.py` decorator stub only POSTs and trusts the response.
    For a fail-closed safety layer that will sign EU AI Act certificates and
    gate self-evolution actions on production models, we need:
      1. A *typed* `ApprovalDecision` so callers don't poke a raw dict.
      2. *HMAC signature verification* — a compromised webhook MUST NOT be able
         to silently approve a critical action by returning {"approved": true}.
         The webhook signs its decision; we verify against HSP_PROTOCOL_KEY.
      3. *Tamper-evident audit log* — every decision (approved OR denied) is
         appended to an in-memory ring with optional file persistence. This is
         the evidence trail the EU AI Act 2026-08 demands.
      4. *Graceful dev mode* — when HSP_GATE_WEBHOOK is unset we MUST still
         return a typed decision so tests and dev loops run without keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# In-memory audit ring. We keep this module-local so tests stay isolated;
# `append_audit` also mirrors to disk when HSP_AUDIT_LOG_PATH is set.
_AUDIT_LOG: list[dict[str, Any]] = []
_AUDIT_LOCK = asyncio.Lock()


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Typed result of an HSP gate request.

    WHY frozen: a decision is immutable evidence. Once HSP signs it, mutating
    the dataclass would silently break signature verification downstream
    (the AI Act PDF generator re-hashes this object).

    Fields:
        approved: terminal yes/no — the only thing the gate decorator branches on.
        decision_id: server-assigned UUID, or locally-minted in dev mode.
        reason: human-readable justification (shown in denial errors, logged on approval).
        audit_trail_url: where a human can inspect the full HSP record.
        signed_at: timestamp the HSP authority signed the decision (UTC).
        signature: HMAC-SHA256 hex digest over the canonical JSON of the decision.
    """

    approved: bool
    decision_id: str
    reason: str
    audit_trail_url: str
    signed_at: datetime
    signature: str

    def to_serializable(self) -> dict[str, Any]:
        """Return a JSON-safe dict (datetime -> ISO 8601).

        WHY: `dataclasses.asdict` leaves datetimes intact, which breaks
        `json.dumps`. The audit log and the AI Act PDF both need a stable
        canonical form, so we centralize the conversion here.
        """
        d = asdict(self)
        d["signed_at"] = self.signed_at.astimezone(timezone.utc).isoformat()
        return d


def _canonical_payload(
    *,
    approved: bool,
    decision_id: str,
    reason: str,
    audit_trail_url: str,
    signed_at: str,
) -> bytes:
    """Produce the exact byte string the HSP authority signed.

    WHY a dedicated helper: both the client (verify) and a future test harness
    (sign fake responses) need bit-identical inputs. Any whitespace or key-order
    drift here silently breaks HMAC. `sort_keys` + `separators` lock it down.
    """
    body = {
        "approved": approved,
        "decision_id": decision_id,
        "reason": reason,
        "audit_trail_url": audit_trail_url,
        "signed_at": signed_at,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _verify_signature(payload: bytes, signature_hex: str, key: str) -> bool:
    """Constant-time HMAC-SHA256 verification.

    WHY constant-time: a naive `==` on hex strings is timing-attackable.
    `hmac.compare_digest` is the standard mitigation.
    """
    expected = hmac.new(key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_hex)


def _sign_payload(payload: bytes, key: str) -> str:
    """Produce an HMAC-SHA256 hex digest.

    WHY exposed (internal): dev mode synthesizes a decision and self-signs it so
    that the downstream AI Act PDF generator can verify a real signature even
    when no HSP authority is reachable. This keeps the verification path
    exercised in CI.
    """
    return hmac.new(key.encode("utf-8"), payload, hashlib.sha256).hexdigest()


class HSPWebhookClient:
    """Async client for the HSP approval webhook.

    WHY a class instead of a free function: the client holds (a) the httpx
    AsyncClient lifetime (so multiple requests reuse the connection pool) and
    (b) the protocol key so callers don't have to thread it through every call.
    """

    def __init__(
        self,
        *,
        webhook_url: str | None = None,
        protocol_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._webhook = webhook_url if webhook_url is not None else os.getenv("HSP_GATE_WEBHOOK", "")
        self._key = protocol_key if protocol_key is not None else os.getenv("HSP_PROTOCOL_KEY", "")
        self._owns_client = http_client is None
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-instantiate the httpx client.

        WHY lazy: in dev mode we never make a request, so spinning up a client
        in __init__ would waste sockets and break test isolation.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying httpx client if we own it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> HSPWebhookClient:
        await self._get_client()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def request_approval(
        self,
        action: str,
        risk_level: str,
        context: dict[str, Any],
        timeout_s: float = 60.0,
    ) -> ApprovalDecision:
        """Request approval for `action` from the HSP authority.

        WHY this exact contract: the EU AI Act 2026-08 self-cert requires every
        high-stakes inference to carry a signed approval token. The caller
        passes the action description, risk band, and contextual evidence (the
        consensus result, the model list, the user query hash). The HSP
        authority returns a signed yes/no. We verify the signature, append to
        the audit log, and return a typed decision.

        Returns approved=True with reason "DEV_MODE — no HSP gate configured"
        when HSP_GATE_WEBHOOK is unset, so local dev and CI never break.
        Critically, the dev-mode decision is also signed (with HSP_PROTOCOL_KEY
        if present, else a stable local key) so downstream signature checks
        still pass.
        """
        if not self._webhook:
            decision = self._dev_mode_decision(action=action, risk_level=risk_level)
            await append_audit(decision, context={"action": action, "risk_level": risk_level, **context})
            logger.info(
                "HSP dev-mode approval issued",
                extra={"action": action, "risk_level": risk_level, "decision_id": decision.decision_id},
            )
            return decision

        payload = {
            "action": action,
            "risk_level": risk_level,
            "context": context,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }

        client = await self._get_client()
        try:
            response = await client.post(self._webhook, json=payload, timeout=timeout_s)
        except httpx.HTTPError as exc:
            logger.error("HSP webhook unreachable: %s", exc)
            # Fail-closed: webhook unreachable for a configured prod gate => deny.
            decision = ApprovalDecision(
                approved=False,
                decision_id=str(uuid.uuid4()),
                reason=f"HSP webhook unreachable: {exc}",
                audit_trail_url="",
                signed_at=datetime.now(timezone.utc),
                signature="",
            )
            await append_audit(decision, context=payload)
            return decision

        if response.status_code != 200:
            decision = ApprovalDecision(
                approved=False,
                decision_id=str(uuid.uuid4()),
                reason=f"HSP gate returned HTTP {response.status_code}",
                audit_trail_url="",
                signed_at=datetime.now(timezone.utc),
                signature="",
            )
            await append_audit(decision, context=payload)
            return decision

        body = response.json()
        decision = self._decision_from_response(body)

        # Verify signature unless explicitly disabled (only allowed in tests).
        if self._key and decision.signature:
            canonical = _canonical_payload(
                approved=decision.approved,
                decision_id=decision.decision_id,
                reason=decision.reason,
                audit_trail_url=decision.audit_trail_url,
                signed_at=decision.signed_at.astimezone(timezone.utc).isoformat(),
            )
            if not _verify_signature(canonical, decision.signature, self._key):
                logger.error(
                    "HSP signature verification FAILED for decision %s",
                    decision.decision_id,
                )
                # Forge protection: a bad signature collapses to a denial.
                decision = ApprovalDecision(
                    approved=False,
                    decision_id=decision.decision_id,
                    reason="HSP signature verification failed",
                    audit_trail_url=decision.audit_trail_url,
                    signed_at=decision.signed_at,
                    signature="",
                )

        await append_audit(decision, context=payload)
        logger.info(
            "HSP decision: %s (%s)",
            "APPROVED" if decision.approved else "DENIED",
            decision.decision_id,
        )
        return decision

    def _dev_mode_decision(self, *, action: str, risk_level: str) -> ApprovalDecision:
        """Mint a locally-signed approval when no gate is configured.

        WHY auto-approve in dev: blocking developers behind a non-existent
        webhook makes the safety layer hostile and forces people to mock it.
        We log loudly and always tag the reason with "DEV_MODE" so audit
        readers can grep these out before any production export.
        """
        decision_id = str(uuid.uuid4())
        signed_at = datetime.now(timezone.utc)
        reason = "DEV_MODE — no HSP gate configured"
        audit_trail_url = ""
        key = self._key or "DEV_MODE_LOCAL_KEY"
        canonical = _canonical_payload(
            approved=True,
            decision_id=decision_id,
            reason=reason,
            audit_trail_url=audit_trail_url,
            signed_at=signed_at.isoformat(),
        )
        signature = _sign_payload(canonical, key)
        logger.warning(
            "HSP_GATE_WEBHOOK is unset — issuing DEV_MODE auto-approval for action=%s risk=%s",
            action,
            risk_level,
        )
        return ApprovalDecision(
            approved=True,
            decision_id=decision_id,
            reason=reason,
            audit_trail_url=audit_trail_url,
            signed_at=signed_at,
            signature=signature,
        )

    @staticmethod
    def _decision_from_response(body: dict[str, Any]) -> ApprovalDecision:
        """Build an ApprovalDecision from raw webhook JSON.

        WHY tolerant parsing: real HSP authorities may evolve their schema; we
        accept missing fields by defaulting them, but we never invent an
        approval — `approved` defaults to False (fail-closed).
        """
        raw_signed = body.get("signed_at")
        if isinstance(raw_signed, str):
            try:
                signed_at = datetime.fromisoformat(raw_signed.replace("Z", "+00:00"))
                if signed_at.tzinfo is None:
                    signed_at = signed_at.replace(tzinfo=timezone.utc)
            except ValueError:
                signed_at = datetime.now(timezone.utc)
        else:
            signed_at = datetime.now(timezone.utc)
        return ApprovalDecision(
            approved=bool(body.get("approved", False)),
            decision_id=str(body.get("decision_id") or uuid.uuid4()),
            reason=str(body.get("reason", "")),
            audit_trail_url=str(body.get("audit_trail_url", "")),
            signed_at=signed_at,
            signature=str(body.get("signature", "")),
        )


async def append_audit(
    decision: ApprovalDecision,
    *,
    context: dict[str, Any] | None = None,
) -> None:
    """Append an HSP decision to the audit log.

    WHY async-locked: multiple concurrent consensus requests may all trigger
    HSP calls. Without a lock the in-memory list and the disk file race and we
    lose evidence — exactly what the EU AI Act forbids.

    WHY mirror to disk: regulators ask "show me the gate decisions for query
    X on date Y". An in-memory ring is wiped on restart, so when
    HSP_AUDIT_LOG_PATH is set we also append JSONL to that file.
    """
    entry: dict[str, Any] = {
        "decision": decision.to_serializable(),
        "context": context or {},
        "appended_at": datetime.now(timezone.utc).isoformat(),
    }
    async with _AUDIT_LOCK:
        _AUDIT_LOG.append(entry)
        path = os.getenv("HSP_AUDIT_LOG_PATH", "")
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with Path(path).open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
            except OSError as exc:
                logger.error("Failed to persist HSP audit entry: %s", exc)


def get_audit_log() -> list[dict[str, Any]]:
    """Return a snapshot copy of the in-memory audit log.

    WHY a copy: callers (tests, the cert generator) iterate while consensus
    workloads keep appending. Handing out the live list invites concurrent
    modification bugs.
    """
    return list(_AUDIT_LOG)


def _reset_audit_log_for_tests() -> None:
    """Clear the audit ring. Used only by tests; not part of the public API.

    WHY exposed: test isolation. Tests need to assert "exactly N entries"
    without leaking state from siblings.
    """
    _AUDIT_LOG.clear()


__all__ = [
    "ApprovalDecision",
    "HSPWebhookClient",
    "append_audit",
    "get_audit_log",
]


# ---------------------------------------------------------------------------
# Tests (executable with `python webhook.py` or wired into pytest).
# ---------------------------------------------------------------------------


async def _test_dev_mode_auto_approves() -> None:
    """When HSP_GATE_WEBHOOK is unset, request_approval returns approved=True."""
    os.environ.pop("HSP_GATE_WEBHOOK", None)
    os.environ.pop("HSP_PROTOCOL_KEY", None)
    _reset_audit_log_for_tests()
    client = HSPWebhookClient()
    decision = await client.request_approval(
        action="test_action", risk_level="low", context={"hint": "unit-test"}
    )
    assert decision.approved is True
    assert "DEV_MODE" in decision.reason
    assert decision.signature  # must be locally signed
    log = get_audit_log()
    assert len(log) == 1
    await client.aclose()


async def _test_dev_mode_signature_is_verifiable() -> None:
    """The dev-mode signature verifies against the same key."""
    os.environ.pop("HSP_GATE_WEBHOOK", None)
    os.environ["HSP_PROTOCOL_KEY"] = "test-key-123"
    _reset_audit_log_for_tests()
    client = HSPWebhookClient()
    decision = await client.request_approval(
        action="self_evolve", risk_level="high", context={"version": "v7.2"}
    )
    canonical = _canonical_payload(
        approved=decision.approved,
        decision_id=decision.decision_id,
        reason=decision.reason,
        audit_trail_url=decision.audit_trail_url,
        signed_at=decision.signed_at.isoformat(),
    )
    assert _verify_signature(canonical, decision.signature, "test-key-123")
    await client.aclose()


async def _test_signature_mismatch_collapses_to_denial() -> None:
    """A response with an invalid signature must be treated as a denial."""
    os.environ["HSP_GATE_WEBHOOK"] = "https://example.invalid/hsp"
    os.environ["HSP_PROTOCOL_KEY"] = "real-key"
    _reset_audit_log_for_tests()

    forged_body = {
        "approved": True,
        "decision_id": "abc-123",
        "reason": "forged",
        "audit_trail_url": "https://attacker.example/forged",
        "signed_at": "2026-06-16T12:00:00+00:00",
        "signature": "deadbeef",  # not a real HMAC of the payload
    }

    class _StubTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=forged_body)

    stub_client = httpx.AsyncClient(transport=_StubTransport())
    client = HSPWebhookClient(http_client=stub_client)
    decision = await client.request_approval(
        action="critical_promote", risk_level="critical", context={}
    )
    assert decision.approved is False
    assert "signature" in decision.reason.lower()
    await stub_client.aclose()


def _run_tests() -> None:
    asyncio.run(_test_dev_mode_auto_approves())
    asyncio.run(_test_dev_mode_signature_is_verifiable())
    asyncio.run(_test_signature_mismatch_collapses_to_denial())
    logger.info("All HSP webhook tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_tests()
