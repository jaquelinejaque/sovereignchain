"""Quorum 0.2.0+ FSL license gate — self-hosted / CLI / library use.

Why this module
---------------
Quorum 0.2.0 switched from Apache 2.0 to FSL-1.1 (Functional Source License,
Apache-2.0-future). Source remains visible for compliance audit, but commercial
use requires a Pro license key from https://quorum-ai.dev (£149/mo, 7-day trial).

The hosted SaaS at api.quorum-ai.dev is exempt — quota / Stripe handle that.
This gate fires only for self-hosted / CLI / library import paths where
the developer is running Quorum on their own infrastructure.

Set ``QUORUM_LICENSE_KEY`` env var to your Pro key. Trial keys auto-issue on
first email signup (`POST /v1/signup` with email).

Bypass for legitimate non-commercial use
----------------------------------------
- ``QUORUM_DEV_MODE=1``: local development on the same machine that owns the
  source. Honour-system bypass for self-improvers / contributors / academic.
- Running inside Quorum's own hosted API container (auto-detected via
  ``QUORUM_HOSTED=1`` set by Cloud Run service deployment).
- Test runs (``PYTEST_CURRENT_TEST`` env var set by pytest).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("quorum.license")

_LICENSE_VALIDATE_URL = os.getenv(
    "QUORUM_LICENSE_VALIDATE_URL",
    "https://api.quorum-ai.dev/v1/license/validate",
)
_CACHE_PATH = Path.home() / ".quorum" / "license_cache.json"
_CACHE_TTL_S = 24 * 3600  # 24h online check; cached otherwise so airgapped works

_LICENSE_REQUIRED_MESSAGE = """
================================================================================
Quorum 0.2.0+ requires a paid Pro license key.

No trial. No free tier. Source-available for compliance audit, commercial
use is paid.

Buy a Pro license (£149/mo) at:

    https://quorum-ai.dev

After purchase, your license key arrives by email. Set it as:

    export QUORUM_LICENSE_KEY=quorum_xxxxxxxxxxxxxxxx

Existing customer needs a re-send? https://quorum-ai.dev/account

For non-commercial / academic / contributor use, set:

    export QUORUM_DEV_MODE=1

(Honour-system bypass — see FSL-1.1 License terms in repo /LICENSE.)
================================================================================
"""


def _is_hosted_quorum() -> bool:
    """True when running inside Quorum's own hosted API container."""
    return os.getenv("QUORUM_HOSTED", "").strip() == "1"


def _is_test_run() -> bool:
    """True during pytest test runs."""
    return "PYTEST_CURRENT_TEST" in os.environ


def _is_dev_mode() -> bool:
    """True when honour-system dev bypass is set."""
    return os.getenv("QUORUM_DEV_MODE", "").strip() == "1"


def _cache_get(key: str) -> dict | None:
    try:
        if not _CACHE_PATH.exists():
            return None
        data = json.loads(_CACHE_PATH.read_text())
        rec = data.get(key)
        if rec and time.time() - rec.get("validated_at", 0) < _CACHE_TTL_S:
            return rec
    except Exception:  # noqa: BLE001
        return None
    return None


def _cache_set(key: str, valid: bool, plan: str = "unknown") -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        if _CACHE_PATH.exists():
            try:
                data = json.loads(_CACHE_PATH.read_text())
            except Exception:  # noqa: BLE001
                data = {}
        data[key] = {"valid": valid, "plan": plan, "validated_at": time.time()}
        _CACHE_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:  # noqa: BLE001
        logger.debug("license cache write failed: %s", e)


def _validate_remote(key: str) -> tuple[bool, str]:
    """Query the license validation endpoint. Returns (valid, plan)."""
    try:
        req = urllib.request.Request(
            f"{_LICENSE_VALIDATE_URL}?key={key}",
            headers={"User-Agent": "quorum-cli-license-check/0.2.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read())
            return bool(payload.get("valid")), str(payload.get("plan", "unknown"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, "invalid"
        # Network/server transient errors — fall back to cache or fail-open
        # so legitimate users aren't grounded by an outage.
        logger.warning("license validate HTTP %s; using cache or fail-open", e.code)
        return _validate_cached_or_fail_open(key)
    except Exception as e:  # noqa: BLE001
        logger.warning("license validate exception (%s); using cache or fail-open", e)
        return _validate_cached_or_fail_open(key)


def _validate_cached_or_fail_open(key: str) -> tuple[bool, str]:
    """When network is down, honour the last known cache up to 7 days."""
    try:
        if _CACHE_PATH.exists():
            data = json.loads(_CACHE_PATH.read_text())
            rec = data.get(key)
            if rec and time.time() - rec.get("validated_at", 0) < 7 * 24 * 3600:
                return bool(rec.get("valid")), str(rec.get("plan", "unknown"))
    except Exception:  # noqa: BLE001
        pass
    # No cache — fail open in offline edge cases to avoid grounding paid users.
    # Hosted SaaS callers are gated separately and don't reach this code path.
    return True, "offline_grace"


def check_license() -> None:
    """Gate import; raise SystemExit with a clear message if unauthorised.

    Order of checks (cheapest to most expensive):
      1. Hosted Quorum container → bypass
      2. Pytest test run → bypass
      3. Dev mode honour bypass → bypass
      4. QUORUM_LICENSE_KEY env set → validate (cached / remote)
      5. None of the above → block with onboarding message
    """
    if _is_hosted_quorum() or _is_test_run() or _is_dev_mode():
        return

    key = os.getenv("QUORUM_LICENSE_KEY", "").strip()
    if not key:
        sys.stderr.write(_LICENSE_REQUIRED_MESSAGE)
        raise SystemExit(2)

    # Fast path: cached and within 24h.
    cached = _cache_get(key)
    if cached:
        if cached.get("valid"):
            return
        sys.stderr.write(
            f"License key is INVALID (cached). Reason: {cached.get('plan', 'invalid')}.\n"
            "Re-issue: https://quorum-ai.dev/account\n"
        )
        raise SystemExit(3)

    # Slow path: hit the validation endpoint, cache for 24h.
    valid, plan = _validate_remote(key)
    _cache_set(key, valid, plan)
    if not valid:
        sys.stderr.write(
            f"License key rejected by server (plan={plan}).\n"
            "Re-issue: https://quorum-ai.dev/account\n"
        )
        raise SystemExit(3)


__all__ = ["check_license"]
