"""Auto-enable HSP_GATE_DEV_MODE when running outside hosted environments.

Why this exists
---------------
The HSP gate is fail-closed by design. In commercial / hosted environments
(Cloud Run, k8s, ECS, Lambda, Cloud Functions, App Service) it MUST stay
that way — a paying tenant cannot flip a flag and disable supervision.

But the local researcher path (Jaqueline iterating on loops on her own
machine, or anyone doing `pip install quorum-ai && quorum learn ...`)
should not have to spin up an approver webhook just to evolve weights.

Import-time side effect: if NO hosted markers are present and HSP_GATE_DEV_MODE
isn't already set, set it to "1" so the local CLI Just Works. If hosted
markers are present, this module touches nothing — the gate stays armed.

This file is imported by quorum/__init__.py so the side effect runs before
any consensus / evolution code reads the env var.
"""

from __future__ import annotations

import os

# Mirror of quorum.hsp.gate._HOSTED_MARKERS — duplicated to avoid an
# import cycle (gate.py is imported by evolution modules which are imported
# transitively from consensus.py).
_HOSTED_MARKERS = (
    "K_SERVICE",
    "KUBERNETES_SERVICE_HOST",
    "ECS_CONTAINER_METADATA_URI",
    "ECS_CONTAINER_METADATA_URI_V4",
    "AWS_LAMBDA_FUNCTION_NAME",
    "FUNCTION_TARGET",
    "WEBSITE_INSTANCE_ID",
)


def _auto_enable_dev_mode() -> None:
    hosted = any(os.environ.get(k) for k in _HOSTED_MARKERS)
    if hosted:
        return
    if os.environ.get("HSP_GATE_DEV_MODE") is not None:
        return  # user already chose; respect them
    if os.environ.get("HSP_GATE_WEBHOOK"):
        return  # webhook configured; don't shadow it
    os.environ["HSP_GATE_DEV_MODE"] = "1"


_auto_enable_dev_mode()
