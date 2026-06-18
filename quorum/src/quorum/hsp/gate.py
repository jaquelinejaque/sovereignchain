"""HSP Gate — fail-closed approval before high-stakes evolution actions.

Patent Pending: PCT/US26/11908.
Commercial use requires HSP license. See LICENSE-HSP.

The gate is a decorator that intercepts a function call and requires a human
(or HSP-certified webhook) to approve before the function executes.

DEFAULT IS FAIL-CLOSED: if HSP_GATE_WEBHOOK is unset, the gate DENIES the
action. To run without a webhook (development only), explicitly set
HSP_GATE_DEV_MODE=1. This matches the marketed behavior — "fail-closed
execution layer" — instead of silently passing in production.
"""

from __future__ import annotations

import asyncio
import os
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

import httpx

T = TypeVar("T")


class HSPGateDenied(RuntimeError):
    """Raised when an HSP gate denies an action."""


def requires_hsp_approval(
    *,
    action: str,
    risk_level: str = "high",
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that gates an async function behind HSP approval.

    Args:
        action: Short string describing what the function does
                (e.g. "promote_llama_checkpoint", "deploy_new_router_policy").
        risk_level: "low" | "medium" | "high" | "critical".

    Behavior:
        - DEFAULT: if HSP_GATE_WEBHOOK is unset, the gate DENIES (fail-closed).
        - If HSP_GATE_DEV_MODE=1 is explicitly set, no webhook required and
          the call passes through — for local development only.
        - If HSP_GATE_WEBHOOK is set, the gate POSTs the action context to
          the webhook and waits for {approved: bool, reason?: str}.
        - On approval, the wrapped function runs.
        - On denial / no webhook / unreachable webhook, raises HSPGateDenied.

    Example:
        @requires_hsp_approval(action="promote_llama_lora", risk_level="high")
        async def promote_checkpoint(version: str) -> None: ...
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            webhook = os.getenv("HSP_GATE_WEBHOOK", "")
            if not webhook:
                # Fail-closed by default. Explicit opt-out only via env.
                if os.getenv("HSP_GATE_DEV_MODE") == "1":
                    return await fn(*args, **kwargs)
                raise HSPGateDenied(
                    f"HSP gate denied action '{action}': HSP_GATE_WEBHOOK is "
                    "not configured (fail-closed default). Configure a webhook "
                    "or set HSP_GATE_DEV_MODE=1 for local development."
                )

            ctx = {
                "action": action,
                "risk_level": risk_level,
                "function": fn.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    r = await client.post(webhook, json=ctx)
                except Exception as e:  # noqa: BLE001
                    raise HSPGateDenied(f"HSP webhook unreachable: {e}") from e

            if r.status_code != 200:
                raise HSPGateDenied(f"HSP gate returned HTTP {r.status_code}")

            decision = r.json()
            if not decision.get("approved", False):
                reason = decision.get("reason", "no reason given")
                raise HSPGateDenied(f"HSP denied action '{action}': {reason}")

            return await fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = ["requires_hsp_approval", "HSPGateDenied"]
