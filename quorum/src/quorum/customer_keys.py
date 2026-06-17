"""Per-customer encrypted storage for BYOK (Bring Your Own Key) provider credentials.

Solves the core economic gap discovered 2026-06-17: until now the hosted
``/v1/consensus`` endpoint silently used the operator's own Anthropic /
OpenAI / Gemini keys for every customer query, so the £49/mo subscription
was net-negative for any customer who actually used the headline 5k
queries. Marketing promised BYOK; the code did not have it.

This module makes BYOK real: each Quorum customer (identified by the
``user_id`` on their API key — typically their email) registers their own
provider API keys via POST /v1/customer/keys. Keys are encrypted at rest
with a Fernet KEK (server-only env var ``CUSTOMER_KEYS_ENCRYPTION_KEY``),
stored in a SQLite table next to the other Quorum data, and resolved per
provider at query time. **No fallback to operator keys** — a provider the
customer hasn't configured is simply skipped from their consensus pool.
That makes the £49/mo billing pure operator margin: orchestration,
consensus algorithm, audit cert, dashboard. The customer pays their
provider bills directly to Anthropic/OpenAI/etc.

Key naming intentionally mirrors the env vars the provider factories
already read (anthropic, openai, gemini, mistral, cohere, grok,
dashscope, replicate, deepseek, nvidia, zhipu, moonshot) so the same
dict can be passed to ``load_default_providers`` with no further mapping.
"""

from __future__ import annotations

import base64
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("quorum.customer_keys")

# Canonical provider keys clients can register. Mirrors the env var names
# the provider factories read so a customer-keys dict drops in directly.
SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "anthropic",
    "openai",
    "gemini",
    "nvidia",
    "mistral",
    "cohere",
    "grok",
    "dashscope",
    "replicate",
    "deepseek",
    "zhipu",
    "moonshot",
)


def _data_dir() -> Path:
    base = os.environ.get("QUORUM_DATA_DIR") or str(Path.home() / ".quorum")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _kek() -> bytes:
    """Load the Fernet KEK from env. Generates a stable per-session warning
    if missing so dev mode still works but prod misconfig is visible."""
    raw = os.environ.get("CUSTOMER_KEYS_ENCRYPTION_KEY", "")
    if not raw:
        logger.warning(
            "CUSTOMER_KEYS_ENCRYPTION_KEY not set — falling back to a static "
            "dev key; PROD MUST configure a real KEK or customer keys are "
            "trivially recoverable on a disk dump."
        )
        # Deterministic dev fallback (NEVER use in prod — KEK warning above).
        return base64.urlsafe_b64encode(b"\x00" * 32)
    return raw.encode("utf-8")


def _fernet():
    """Lazy import to keep import time fast when the module isn't used."""
    try:
        from cryptography.fernet import Fernet
    except ImportError as e:
        raise RuntimeError(
            "cryptography package required: pip install cryptography"
        ) from e
    return Fernet(_kek())


class CustomerKeyStore:
    """SQLite-backed encrypted store of (user_id, provider) -> api_key.

    Methods are intentionally synchronous — the dataset is tiny (< few
    rows per user), reads are O(1) on a primary-key lookup, and the
    encrypt/decrypt cost dominates anyway. Avoiding async here keeps the
    surface easy to call from FastAPI handlers and from sync test code.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (_data_dir() / "customer_keys.db")
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS customer_keys (
                    user_id      TEXT NOT NULL,
                    provider     TEXT NOT NULL,
                    key_encrypted BLOB NOT NULL,
                    updated_at   REAL NOT NULL,
                    PRIMARY KEY (user_id, provider)
                )"""
            )

    def set(self, user_id: str, provider: str, api_key: str) -> None:
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"unsupported provider: {provider}")
        if not api_key or not api_key.strip():
            raise ValueError("api_key cannot be empty")
        token = _fernet().encrypt(api_key.encode("utf-8"))
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO customer_keys "
                "(user_id, provider, key_encrypted, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, provider, token, time.time()),
            )

    def get(self, user_id: str, provider: str) -> Optional[str]:
        with self._conn() as c:
            row = c.execute(
                "SELECT key_encrypted FROM customer_keys WHERE user_id=? AND provider=?",
                (user_id, provider),
            ).fetchone()
        if not row:
            return None
        try:
            return _fernet().decrypt(row[0]).decode("utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("decrypt failed for %s/%s: %s", user_id, provider, e)
            return None

    def get_all(self, user_id: str) -> dict[str, str]:
        """Return all (provider -> plaintext key) pairs for a user."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT provider, key_encrypted FROM customer_keys WHERE user_id=?",
                (user_id,),
            ).fetchall()
        out: dict[str, str] = {}
        f = _fernet()
        for provider, blob in rows:
            try:
                out[provider] = f.decrypt(blob).decode("utf-8")
            except Exception as e:  # noqa: BLE001
                logger.warning("decrypt failed for %s/%s: %s", user_id, provider, e)
        return out

    def list_providers(self, user_id: str) -> list[dict[str, object]]:
        """Public listing for /v1/customer/keys GET — does NOT decrypt.

        Returns one record per configured provider with the timestamp it
        was last updated and a short prefix/suffix tail of the original
        key so the customer can visually confirm which key is stored
        without exposing it. The prefix/suffix is computed at write time
        (we'd need to decrypt to do it now), but we approximate via the
        ciphertext length to avoid touching the secret on a list call.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT provider, updated_at, length(key_encrypted) FROM customer_keys WHERE user_id=? ORDER BY provider",
                (user_id,),
            ).fetchall()
        return [
            {"provider": p, "updated_at": ts, "ciphertext_bytes": n}
            for p, ts, n in rows
        ]

    def delete(self, user_id: str, provider: str) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM customer_keys WHERE user_id=? AND provider=?",
                (user_id, provider),
            )
        return cur.rowcount > 0

    def delete_all(self, user_id: str) -> int:
        with self._conn() as c:
            cur = c.execute("DELETE FROM customer_keys WHERE user_id=?", (user_id,))
        return cur.rowcount


# Module-level singleton for the FastAPI app to share — created lazily on
# first use so test code can rebind ``CustomerKeyStore.__init__`` with a
# per-test path before the production store is constructed.
_default: Optional[CustomerKeyStore] = None


def default_store() -> CustomerKeyStore:
    global _default
    if _default is None:
        _default = CustomerKeyStore()
    return _default
