"""HSP Gate — fail-closed approval before high-stakes evolution actions.

Patent Pending: PCT/US26/11908.
Commercial use requires HSP license. See LICENSE-HSP.

The gate is a decorator that intercepts a function call and requires a human
(or HSP-certified webhook) to approve before the function executes.

POSTURE (intentional split):
* HOSTED / commercial (Cloud Run, Kubernetes, ECS, paying customers):
  ALWAYS fail-closed. HSP_GATE_DEV_MODE is *ignored* here so a tenant
  cannot just flip an env var and turn off the supervision layer they
  pay for. Detection is by infra-provided env (K_SERVICE for Cloud Run,
  KUBERNETES_SERVICE_HOST for k8s, ECS_CONTAINER_METADATA_URI for ECS).
* LOCAL CLI / self-host researcher: HSP_GATE_DEV_MODE=1 unlocks
  unrestricted evolution. This is the path Jaqueline uses to iterate on
  loops without round-tripping every weight nudge through a webhook.
  The CLI entry point sets DEV_MODE=1 automatically when invoked
  outside a hosted environment.

The gate fails closed by default — both paths above are explicit
opt-ins. A misconfigured deployment denies, never silently passes.
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


# Hosted-environment markers. If ANY of these env vars are present, the gate
# treats this process as commercial/hosted and ignores HSP_GATE_DEV_MODE.
# A real webhook is then mandatory.
_HOSTED_MARKERS = (
    "K_SERVICE",                  # Cloud Run (Google)
    "KUBERNETES_SERVICE_HOST",    # any k8s cluster
    "ECS_CONTAINER_METADATA_URI", # AWS ECS
    "ECS_CONTAINER_METADATA_URI_V4",
    "AWS_LAMBDA_FUNCTION_NAME",   # Lambda
    "FUNCTION_TARGET",            # Google Cloud Functions
    "WEBSITE_INSTANCE_ID",        # Azure App Service
)


def _is_hosted() -> bool:
    """True when this process is running in a commercial hosting environment."""
    return any(os.environ.get(k) for k in _HOSTED_MARKERS)


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
        - HOSTED ENV (Cloud Run / k8s / ECS / Lambda / Functions / App
          Service): HSP_GATE_DEV_MODE is *ignored*. Only a real webhook
          unlocks the call. This is the commercial guarantee — a tenant
          cannot flip an env var to disable supervision they pay for.
        - LOCAL ENV (no hosted markers present): HSP_GATE_DEV_MODE=1
          unlocks the call without a webhook. This is the local-research
          path the CLI uses.
        - If HSP_GATE_WEBHOOK is set, the gate POSTs the action context
          to the webhook and waits for {approved: bool, reason?: str}.
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
                # No webhook → only local + explicit DEV_MODE may pass.
                hosted = _is_hosted()
                if not hosted and os.getenv("HSP_GATE_DEV_MODE") == "1":
                    return await fn(*args, **kwargs)
                # Hosted env or no DEV_MODE → deny.
                hint = (
                    "Hosted environment detected — DEV_MODE is ignored, "
                    "configure HSP_GATE_WEBHOOK with a real approver."
                    if hosted else
                    "Configure HSP_GATE_WEBHOOK with an approver webhook, "
                    "or set HSP_GATE_DEV_MODE=1 for local development."
                )
                raise HSPGateDenied(
                    f"HSP gate denied action '{action}' (fail-closed). {hint}"
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
