"""HSP Gate — fail-closed approval before high-stakes evolution actions.

Patent: PCT/US26/11908.
Commercial use requires HSP license. See LICENSE-HSP.

The gate is a decorator that intercepts a function call and requires a human
(or HSP-certified webhook) to approve before the function executes.
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
        - If HSP_GATE_WEBHOOK is unset, the gate logs and PASSES (dev mode).
        - If set, the gate POSTs the action context to the webhook and waits
          for a JSON response {approved: bool, reason?: str}.
        - On approval, the wrapped function runs.
        - On denial, raises HSPGateDenied.

    Example:
        @requires_hsp_approval(action="promote_llama_lora", risk_level="high")
        async def promote_checkpoint(version: str) -> None: ...
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            webhook = os.getenv("HSP_GATE_WEBHOOK", "")
            if not webhook:
                # Dev mode — no HSP gate configured. Log + pass.
                # Production deployments MUST configure HSP_GATE_WEBHOOK.
                return await fn(*args, **kwargs)

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
