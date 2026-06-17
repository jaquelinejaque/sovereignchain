"""Firestore-backed persistent storage — drop-in replacement for the SQLite stores.

Why this exists
---------------
The original SQLite stores live in $QUORUM_DATA_DIR (default /tmp/quorum on
Cloud Run). That directory is destroyed on every Cloud Run revision rollout,
which means EVERY paying customer loses their API key, registered BYOK
provider keys, billing tier, and quota usage the moment a new image deploys.
Validated empirically 2026-06-17 with the qk_RhZ2... key created in
revision 00014-7ml and gone in 00017-w2m.

These Firestore-backed equivalents keep the same async API as the SQLite
classes so the rest of the server doesn't need to know which backend it's
talking to. Activated by env var ``QUORUM_USE_FIRESTORE=1`` (any truthy
value). When the flag isn't set, the legacy SQLite path stays in place,
which means tests and self-host dev never need a Firestore emulator
running.

Collection naming uses a ``quorum_`` prefix because the Sovereign Chain
project's Firestore database is shared with the Keratin Pro Mastery
e-commerce app — the prefix keeps the two apps from accidentally
shadowing each other's collection names.

Auth path: relies on Application Default Credentials picked up from the
Cloud Run service account (``86770458722-compute@developer.gserviceaccount.com``
needs ``roles/datastore.user`` on the project). On local dev,
``gcloud auth application-default login`` is the path; for CI, set
``GOOGLE_APPLICATION_CREDENTIALS`` to a service-account JSON.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("quorum.firestore_stores")

_FIRESTORE_PREFIX = "quorum_"
_COL_API_KEYS = f"{_FIRESTORE_PREFIX}api_keys"
_COL_CUSTOMER_KEYS = f"{_FIRESTORE_PREFIX}customer_keys"
_COL_CUSTOMERS = f"{_FIRESTORE_PREFIX}customers"
_COL_USAGE = f"{_FIRESTORE_PREFIX}usage"


def use_firestore() -> bool:
    """Feature flag: any non-empty/non-zero value of QUORUM_USE_FIRESTORE."""
    v = os.environ.get("QUORUM_USE_FIRESTORE", "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


def _client():
    """Lazy import + lazy instantiation so the module is safe to import even
    when google-cloud-firestore isn't installed (e.g. self-host wheels)."""
    try:
        from google.cloud import firestore  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-firestore not installed; "
            "pip install google-cloud-firestore"
        ) from e
    return firestore.Client()


# ---------------------------------------------------------------------------
# Pydantic-shaped record so the server stays decoupled from this module
# ---------------------------------------------------------------------------


class _APIKeyRecord:
    """Mirror of server.main.APIKeyRecord so we can return one without
    importing from server (avoids circular import)."""

    def __init__(self, *, user_id: str, tier: str,
                 created_at: datetime, revoked_at: Optional[datetime]):
        self.user_id = user_id
        self.tier = tier
        self.created_at = created_at
        self.revoked_at = revoked_at


# ---------------------------------------------------------------------------
# 1. API keys
# ---------------------------------------------------------------------------


class FirestoreAPIKeyStore:
    """Drop-in replacement for ``server.main.APIKeyStore``.

    Doc layout in ``quorum_api_keys`` collection:
      doc_id    = sha256(plaintext) hex
      fields    = { user_id, tier, created_at (float epoch),
                    revoked_at (float epoch | None) }
    """

    def __init__(self):
        self._col_name = _COL_API_KEYS
        # Lazy client init in each thread-pool call rather than once at
        # construction so we can survive a Firestore brown-out at startup.

    @staticmethod
    def _hash(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    async def issue(self, user_id: str, tier: str = "free"):
        plaintext = f"qk_{secrets.token_urlsafe(32)}"
        rec = await asyncio.to_thread(self._issue_sync, plaintext, user_id, tier)
        return plaintext, rec

    def _issue_sync(self, plaintext: str, user_id: str, tier: str) -> _APIKeyRecord:
        now = time.time()
        key_hash = self._hash(plaintext)
        c = _client()
        c.collection(self._col_name).document(key_hash).set({
            "user_id": user_id,
            "tier": tier,
            "created_at": now,
            "revoked_at": None,
        })
        return _APIKeyRecord(
            user_id=user_id,
            tier=tier,
            created_at=datetime.fromtimestamp(now, tz=timezone.utc),
            revoked_at=None,
        )

    async def lookup(self, plaintext: str) -> Optional[_APIKeyRecord]:
        if not plaintext:
            return None
        return await asyncio.to_thread(self._lookup_sync, plaintext)

    def _lookup_sync(self, plaintext: str) -> Optional[_APIKeyRecord]:
        key_hash = self._hash(plaintext)
        c = _client()
        snap = c.collection(self._col_name).document(key_hash).get()
        if not snap.exists:
            return None
        d = snap.to_dict() or {}
        if d.get("revoked_at") is not None:
            return None
        return _APIKeyRecord(
            user_id=str(d.get("user_id", "")),
            tier=str(d.get("tier", "free")),
            created_at=datetime.fromtimestamp(float(d.get("created_at", 0)), tz=timezone.utc),
            revoked_at=None,
        )

    async def revoke(self, plaintext: str) -> bool:
        return await asyncio.to_thread(self._revoke_sync, plaintext)

    def _revoke_sync(self, plaintext: str) -> bool:
        key_hash = self._hash(plaintext)
        c = _client()
        ref = c.collection(self._col_name).document(key_hash)
        snap = ref.get()
        if not snap.exists:
            return False
        d = snap.to_dict() or {}
        if d.get("revoked_at") is not None:
            return False
        ref.update({"revoked_at": time.time()})
        return True


# ---------------------------------------------------------------------------
# 2. Customer-registered BYOK keys
# ---------------------------------------------------------------------------


class FirestoreCustomerKeyStore:
    """Drop-in replacement for ``customer_keys.CustomerKeyStore``.

    Doc layout in ``quorum_customer_keys`` collection:
      doc_id    = base64url(user_id)__provider   (so emails with @ work)
      fields    = { user_id, provider, key_encrypted (bytes->b64 string),
                    updated_at (float epoch) }
    The same Fernet KEK from CUSTOMER_KEYS_ENCRYPTION_KEY is used for
    at-rest encryption — Firestore itself encrypts at-rest server-side but
    application-layer encryption means a Firestore-only breach (without
    KEK access) still doesn't leak customer provider keys.
    """

    def __init__(self):
        # Reuse the existing SUPPORTED_PROVIDERS set so the validation
        # surface stays identical to the SQLite store.
        from quorum.customer_keys import SUPPORTED_PROVIDERS, _fernet
        self.SUPPORTED_PROVIDERS = SUPPORTED_PROVIDERS
        self._fernet = _fernet

    @staticmethod
    def _doc_id(user_id: str, provider: str) -> str:
        # Email or other user_id may contain '@', '/', etc. Firestore
        # doc IDs disallow '/' and prefer short URL-safe strings, so we
        # base64url the user_id and join with the provider name.
        uid = base64.urlsafe_b64encode(user_id.encode("utf-8")).decode("ascii").rstrip("=")
        return f"{uid}__{provider}"

    def set(self, user_id: str, provider: str, api_key: str) -> None:
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"unsupported provider: {provider}")
        if not api_key or not api_key.strip():
            raise ValueError("api_key cannot be empty")
        token = self._fernet().encrypt(api_key.encode("utf-8"))
        c = _client()
        c.collection(_COL_CUSTOMER_KEYS).document(self._doc_id(user_id, provider)).set({
            "user_id": user_id,
            "provider": provider,
            "key_encrypted": base64.b64encode(token).decode("ascii"),
            "updated_at": time.time(),
        })

    def get(self, user_id: str, provider: str) -> Optional[str]:
        c = _client()
        snap = c.collection(_COL_CUSTOMER_KEYS).document(self._doc_id(user_id, provider)).get()
        if not snap.exists:
            return None
        d = snap.to_dict() or {}
        try:
            token = base64.b64decode(d["key_encrypted"])
            return self._fernet().decrypt(token).decode("utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("decrypt failed for %s/%s: %s", user_id, provider, e)
            return None

    def get_all(self, user_id: str) -> dict[str, str]:
        c = _client()
        # Indexed query — needs an "ASC user_id" composite index if you
        # add more filters later. Single field is auto-indexed by Firestore.
        snaps = c.collection(_COL_CUSTOMER_KEYS).where("user_id", "==", user_id).stream()
        out: dict[str, str] = {}
        f = self._fernet()
        for snap in snaps:
            d = snap.to_dict() or {}
            try:
                token = base64.b64decode(d["key_encrypted"])
                out[d["provider"]] = f.decrypt(token).decode("utf-8")
            except Exception as e:  # noqa: BLE001
                logger.warning("decrypt failed for %s/%s: %s", user_id, d.get("provider"), e)
        return out

    def list_providers(self, user_id: str) -> list[dict[str, Any]]:
        c = _client()
        snaps = c.collection(_COL_CUSTOMER_KEYS).where("user_id", "==", user_id).stream()
        out: list[dict[str, Any]] = []
        for snap in snaps:
            d = snap.to_dict() or {}
            out.append({
                "provider": d.get("provider", ""),
                "updated_at": d.get("updated_at", 0),
                "ciphertext_bytes": len(d.get("key_encrypted", "")),
            })
        out.sort(key=lambda x: str(x["provider"]))
        return out

    def delete(self, user_id: str, provider: str) -> bool:
        c = _client()
        ref = c.collection(_COL_CUSTOMER_KEYS).document(self._doc_id(user_id, provider))
        snap = ref.get()
        if not snap.exists:
            return False
        ref.delete()
        return True

    def delete_all(self, user_id: str) -> int:
        c = _client()
        snaps = list(c.collection(_COL_CUSTOMER_KEYS).where("user_id", "==", user_id).stream())
        for snap in snaps:
            snap.reference.delete()
        return len(snaps)


# ---------------------------------------------------------------------------
# 3. Billing customer tier + usage
# ---------------------------------------------------------------------------


class FirestoreBillingCache:
    """Drop-in for the SQLite-backed customers + usage tables in stripe_billing.

    Stores tier under ``quorum_customers/<customer_id>`` and per-period
    usage counters under ``quorum_usage/<customer_id>__<period_key>``.

    Only the methods actually called by stripe_billing.BillingClient are
    implemented — adding more is a one-liner if needed.
    """

    def __init__(self):
        pass

    def set_tier(self, customer_id: str, tier: str,
                 *, email: Optional[str] = None, subscription_id: Optional[str] = None) -> None:
        c = _client()
        c.collection(_COL_CUSTOMERS).document(customer_id).set({
            "customer_id": customer_id,
            "tier": tier,
            "email": email,
            "subscription_id": subscription_id,
            "updated_at": time.time(),
        }, merge=True)

    def get_tier(self, customer_id: str) -> Optional[str]:
        c = _client()
        snap = c.collection(_COL_CUSTOMERS).document(customer_id).get()
        if not snap.exists:
            return None
        return str((snap.to_dict() or {}).get("tier", ""))

    def get_customer(self, customer_id: str) -> Optional[dict]:
        c = _client()
        snap = c.collection(_COL_CUSTOMERS).document(customer_id).get()
        if not snap.exists:
            return None
        return snap.to_dict()

    @staticmethod
    def _usage_doc_id(customer_id: str, period_key: str) -> str:
        cid = base64.urlsafe_b64encode(customer_id.encode("utf-8")).decode("ascii").rstrip("=")
        return f"{cid}__{period_key}"

    def increment_usage(self, customer_id: str, period_key: str, n: int = 1) -> int:
        """Atomic increment via Firestore transaction. Returns new count."""
        from google.cloud import firestore  # type: ignore
        c = _client()
        ref = c.collection(_COL_USAGE).document(self._usage_doc_id(customer_id, period_key))

        # Firestore .Increment is atomic; the .set(...,merge=True) creates
        # the doc on first hit. Read-back is a second round-trip, but
        # usage record is fire-and-forget for the hot path so latency
        # impact is bounded by the check_quota call upstream.
        ref.set({
            "customer_id": customer_id,
            "period_key": period_key,
            "count": firestore.Increment(n),
            "updated_at": time.time(),
        }, merge=True)
        snap = ref.get()
        return int((snap.to_dict() or {}).get("count", n))

    def get_usage(self, customer_id: str, period_key: str) -> int:
        c = _client()
        snap = c.collection(_COL_USAGE).document(self._usage_doc_id(customer_id, period_key)).get()
        if not snap.exists:
            return 0
        return int((snap.to_dict() or {}).get("count", 0))
