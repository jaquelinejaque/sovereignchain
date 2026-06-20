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

import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("quorum.license")

_DEFAULT_LICENSE_VALIDATE_URL = "https://api.quorum-ai.dev/v1/license/validate"
_LICENSE_HOSTNAME_ALLOWLIST = frozenset({
    "api.quorum-ai.dev",
    "api-staging.quorum-ai.dev",
    "localhost",
    "127.0.0.1",
})


def _load_license_validate_url() -> str:
    """Load the license validate URL from env, enforcing hostname allowlist.

    Prevents SSRF / exfiltration of license keys to attacker-controlled hosts
    via env injection. Falls back to default if env value targets a hostname
    outside the allowlist.
    """
    override = os.getenv("QUORUM_LICENSE_VALIDATE_URL", "").strip()
    if not override:
        return _DEFAULT_LICENSE_VALIDATE_URL
    try:
        host = (urlparse(override).hostname or "").lower()
    except Exception:  # noqa: BLE001
        host = ""
    if host in _LICENSE_HOSTNAME_ALLOWLIST:
        return override
    logger.warning(
        "QUORUM_LICENSE_VALIDATE_URL hostname %r not in allowlist; "
        "falling back to default %s",
        host or override,
        _DEFAULT_LICENSE_VALIDATE_URL,
    )
    return _DEFAULT_LICENSE_VALIDATE_URL


_LICENSE_VALIDATE_URL = _load_license_validate_url()
_CACHE_PATH = Path.home() / ".quorum" / "license_cache.json"
_INSTALL_ID_PATH = Path.home() / ".quorum" / "install_id"
_CACHE_TTL_S = 24 * 3600  # 24h online check; cached otherwise so airgapped works

_INSTALL_SECRET: bytes | None = None


def _cache_secret() -> bytes:
    """Return per-install HMAC secret, creating it on first use.

    The secret lives at ~/.quorum/install_id. It binds cache records to this
    machine so an attacker can't hand-craft a cache file claiming valid=True
    without also reading the secret off disk. Cached in a module-level var so
    we only touch disk once per process.
    """
    global _INSTALL_SECRET
    if _INSTALL_SECRET is not None:
        return _INSTALL_SECRET
    try:
        _INSTALL_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _INSTALL_ID_PATH.exists():
            secret_hex = _INSTALL_ID_PATH.read_text().strip()
            if not secret_hex:
                raise ValueError("empty install_id")
        else:
            secret_hex = secrets.token_hex(32)
            _INSTALL_ID_PATH.write_text(secret_hex)
            try:
                os.chmod(_INSTALL_ID_PATH, 0o600)
            except Exception:  # noqa: BLE001
                pass
        _INSTALL_SECRET = bytes.fromhex(secret_hex)
    except Exception as e:  # noqa: BLE001
        logger.debug("install_id read/create failed (%s); using ephemeral secret", e)
        _INSTALL_SECRET = secrets.token_bytes(32)
    return _INSTALL_SECRET


def _cache_signature(key: str, valid: bool, plan: str, validated_at: float) -> str:
    """HMAC-SHA256 over the cache record fields."""
    msg = f"{key}|{valid}|{plan}|{validated_at}".encode()
    return hmac.new(_cache_secret(), msg, hashlib.sha256).hexdigest()


def _verify_record(key: str, rec: dict) -> bool:
    """Recompute the signature on a cache record and compare in constant time."""
    sig = rec.get("signature")
    if not isinstance(sig, str):
        return False
    expected = _cache_signature(
        key,
        bool(rec.get("valid")),
        str(rec.get("plan", "unknown")),
        float(rec.get("validated_at", 0)),
    )
    return hmac.compare_digest(sig, expected)

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


_HOSTED_ATTESTED: bool | None = None


def _is_hosted_quorum() -> bool:
    """True when running inside Quorum's own hosted API container.

    Requires BOTH:
      1. ``QUORUM_HOSTED=1`` env var (set by Cloud Run service deployment).
      2. GCP metadata server attestation — proves we really are inside a
         Google compute environment, not a malicious local shell that just
         exported the env var to bypass licensing.

    Attestation is cached at module level so the metadata call happens once
    per process.
    """
    global _HOSTED_ATTESTED
    if os.getenv("QUORUM_HOSTED", "").strip() != "1":
        return False
    if _HOSTED_ATTESTED is not None:
        return _HOSTED_ATTESTED
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            _HOSTED_ATTESTED = resp.status == 200
    except Exception:  # noqa: BLE001
        _HOSTED_ATTESTED = False
    return _HOSTED_ATTESTED


def _is_test_run() -> bool:
    """True during pytest test runs."""
    return os.environ.get("_QUORUM_TEST_BYPASS_INTERNAL") == "1"


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
            if not _verify_record(key, rec):
                logger.warning("license cache signature mismatch; ignoring record")
                return None
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
        validated_at = time.time()
        signature = _cache_signature(key, valid, plan, validated_at)
        data[key] = {
            "valid": valid,
            "plan": plan,
            "validated_at": validated_at,
            "signature": signature,
        }
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
                if not _verify_record(key, rec):
                    logger.warning(
                        "license cache signature mismatch in offline path; "
                        "ignoring record"
                    )
                else:
                    return bool(rec.get("valid")), str(rec.get("plan", "unknown"))
    except Exception:  # noqa: BLE001
        pass
    # No cache — fail open in offline edge cases to avoid grounding paid users.
    # Hosted SaaS callers are gated separately and don't reach this code path.
    return False, "offline_no_cache_fail_secure"


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
