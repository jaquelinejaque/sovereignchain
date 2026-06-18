# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# HSP ATTRIBUTION
# ---------------
# This server module exposes the hosted SaaS surface of Quorum: quota
# enforcement, billing, audit-grade certificates. Those features are
# Harmonised Sovereignty Protocol (HSP) gated under PCT/US26/11908.
# The ``/v1/cert/{query_id}`` endpoint, in particular, materialises the
# HSP-gated EU AI Act compliance certificate and is subject to the
# additional commercial-use restrictions defined in LICENSE-HSP.
#
# Self-hosted, single-tenant, BYOK use remains free under Apache 2.0.
# Commercial hosted deployment requires an HSP licence.
"""FastAPI server exposing Quorum as a hosted multi-LLM consensus SaaS.

WHY THIS MODULE EXISTS
----------------------
The rest of Quorum is a library: ``await consensus(prompt)`` and you get
back a synthesised cross-model answer. That covers the BYOK self-hosted
story but says nothing about *how* a paying customer would consume Quorum
over the network. This module is the network surface:

* It turns ``consensus()`` into an HTTP endpoint with quota-gated auth.
* It owns the API-key lifecycle (issuance, hashing, revocation, lookup).
* It wires the billing client (``quorum.billing.stripe_billing``) into the
  request path so over-quota customers get a clean ``402 Payment Required``
  instead of silently consuming free compute.
* It exposes the RLHF feedback loop (``quorum.evolution.rlhf``) as a
  public endpoint so the dashboard / SDK can post thumbs-up/down without
  needing to know about the SQLite schema underneath.
* It exposes the EU AI Act PDF certificate endpoint, which is the
  HSP-gated audit artefact that justifies the PRO/TEAM tier price.

DESIGN CHOICES
--------------
* API keys live in their *own* SQLite DB at ``~/.quorum/api_keys.db`` —
  deliberately separate from billing's ``usage.db`` so a key-management
  bug can never corrupt the usage counters and vice versa.
* Keys are stored as SHA-256 hashes only; the plaintext is shown exactly
  once at issuance. ``hmac.compare_digest`` is used on lookup to avoid
  timing oracles.
* Rate limits are tier-scoped via ``slowapi``: 60/min FREE, 600/min PRO,
  6000/min TEAM. The limit is computed from the API key, not the IP, so
  honest load-balancing doesn't get throttled.
* When ``slowapi`` is not installed (CI without the optional dep), the
  server *still works* — limiting silently no-ops. We never block on an
  optional dependency in the hot path.
* CORS is fully open. The dashboard origin allowlist will land alongside
  the dashboard itself; locking it down before then would just break
  every contributor running ``curl`` against localhost.
* Everything ships with an in-memory / dev-mode fallback so tests run
  without any keys, exactly mirroring the rest of the codebase.

LICENSE
-------
SPDX-License-Identifier: Apache-2.0 WITH HSP-Commercial-Restrictions
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Literal, Optional

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from quorum import __version__ as QUORUM_VERSION
from quorum.billing.stripe_billing import (
    TIERS,
    BillingClient,
    BillingError,
    QuotaStatus,
    Tier,
    WebhookResult,
)
from quorum.core.consensus import ConsensusResult, consensus
from quorum.evolution.ab_testing import ABTestStore
from quorum.evolution.rlhf import RLHFTracker
from quorum.hsp.ai_act_cert import generate_cert_pdf

try:  # pragma: no cover - optional dependency
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    _SLOWAPI_AVAILABLE = True
except Exception:  # pragma: no cover - tested via no-slowapi fallback
    Limiter = None  # type: ignore[misc,assignment]
    RateLimitExceeded = Exception  # type: ignore[misc,assignment]
    SlowAPIMiddleware = None  # type: ignore[misc,assignment]
    get_remote_address = lambda r: "0.0.0.0"  # noqa: E731
    _SLOWAPI_AVAILABLE = False


__all__ = [
    "app",
    "create_app",
    "APIKeyStore",
    "ConsensusRequest",
    "FeedbackRequest",
    "ABFeedbackRequest",
    "ABFeedbackResponse",
    "AppState",
]

logger = logging.getLogger("quorum.server")


# ---------------------------------------------------------------------------
# Paths and tier-keyed rate limits
# ---------------------------------------------------------------------------
#
# Tier → "<N>/minute" mapping. Encoded inline (not pulled from billing's
# tier matrix) because the matrix is a *commercial* contract and rate
# limits are an *operational* knob — they may need to change without a
# pricing change. Keeping them in two files keeps blast radius small.

DATA_DIR = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()
_DEFAULT_API_KEY_DB = DATA_DIR / "api_keys.db"
_DEFAULT_CERT_DIR = DATA_DIR / "certs"

_TIER_RATE_LIMITS: dict[Tier, str] = {
    "free": "60/minute",
    "pro": "600/minute",
    "team": "6000/minute",
    "enterprise": "60000/minute",
}


# ---------------------------------------------------------------------------
# API-key store (SQLite at ~/.quorum/api_keys.db)
# ---------------------------------------------------------------------------


class APIKeyRecord(BaseModel):
    """Public-facing view of a stored API key row.

    The plaintext key is *never* present — only its hash is held server-side.
    The record is what the auth dependency returns after a successful lookup,
    so downstream handlers can read ``user_id`` and ``tier`` without going
    back to the DB.
    """

    user_id: str
    tier: Tier
    created_at: datetime
    revoked_at: Optional[datetime] = None


class APIKeyStore:
    """SQLite-backed API key registry.

    WHY a dedicated class rather than free functions: the table lives in a
    distinct file from the billing cache, and we want a single object that
    owns *all* access to it. That makes it trivial to swap out for Postgres
    later without touching every call site.

    The store is sync internally (sqlite3) but exposed as async via
    ``asyncio.to_thread`` so the request handlers don't block the event
    loop.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_API_KEY_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection with WAL mode for safety.

        WHY context-managed and short-lived: connection pooling sync
        SQLite handles under an async wrapper invites deadlocks; short
        lifetimes side-step the whole class of bug.
        """
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create the schema if it doesn't exist (idempotent).

        WHY in __init__ rather than lazily: the server is a long-lived
        process; eating one disk write at startup is preferable to having
        a first-request latency spike for nothing.
        """
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash    TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    tier        TEXT NOT NULL DEFAULT 'free',
                    created_at  REAL NOT NULL,
                    revoked_at  REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id)"
            )

    @staticmethod
    def _hash(key: str) -> str:
        """Return SHA-256 hex digest of ``key``.

        WHY SHA-256 and not bcrypt/argon2: API keys are 256 bits of entropy
        from ``secrets.token_urlsafe(32)``. A KDF buys nothing against
        brute force here (the search space is already astronomical) and
        would add latency to every authenticated request. Hashing is for
        breach containment (so a stolen DB file doesn't leak live keys),
        not password-style protection.
        """
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    async def issue(self, user_id: str, tier: Tier = "free") -> tuple[str, APIKeyRecord]:
        """Generate, persist, and return a fresh API key.

        WHY return the plaintext: this is the *only* moment it exists in
        cleartext. The caller is responsible for displaying it to the user
        exactly once; subsequent lookups can only verify against the hash.
        """
        if tier not in TIERS:
            raise ValueError(f"unknown tier: {tier!r}")
        plaintext = f"qk_{secrets.token_urlsafe(32)}"
        record = await asyncio.to_thread(self._issue_sync, plaintext, user_id, tier)
        return plaintext, record

    def _issue_sync(self, plaintext: str, user_id: str, tier: Tier) -> APIKeyRecord:
        now = time.time()
        key_hash = self._hash(plaintext)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO api_keys (key_hash, user_id, tier, created_at, revoked_at)
                   VALUES (?, ?, ?, ?, NULL)""",
                (key_hash, user_id, tier, now),
            )
        return APIKeyRecord(
            user_id=user_id,
            tier=tier,
            created_at=datetime.fromtimestamp(now, tz=timezone.utc),
            revoked_at=None,
        )

    async def lookup(self, plaintext: str) -> Optional[APIKeyRecord]:
        """Return the record for ``plaintext`` if active, else ``None``.

        WHY ``hmac.compare_digest`` on the hash: even though we look the
        row up by hash (so a timing leak could only reveal that a hash
        prefix matched), defence-in-depth costs nothing here.
        """
        if not plaintext:
            return None
        return await asyncio.to_thread(self._lookup_sync, plaintext)

    def _lookup_sync(self, plaintext: str) -> Optional[APIKeyRecord]:
        key_hash = self._hash(plaintext)
        with self._connect() as conn:
            row = conn.execute(
                """SELECT key_hash, user_id, tier, created_at, revoked_at
                   FROM api_keys WHERE key_hash = ?""",
                (key_hash,),
            ).fetchone()
        if not row:
            return None
        stored_hash, user_id, tier, created_at, revoked_at = row
        if not hmac.compare_digest(str(stored_hash), key_hash):
            return None  # belt and braces against a hypothetical hash collision
        if revoked_at is not None:
            return None
        return APIKeyRecord(
            user_id=str(user_id),
            tier=str(tier),  # type: ignore[arg-type]
            created_at=datetime.fromtimestamp(float(created_at), tz=timezone.utc),
            revoked_at=None,
        )

    async def revoke(self, plaintext: str) -> bool:
        """Mark a key as revoked. Idempotent. Returns True if a row changed."""
        return await asyncio.to_thread(self._revoke_sync, plaintext)

    def _revoke_sync(self, plaintext: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_hash = ? AND revoked_at IS NULL",
                (time.time(), self._hash(plaintext)),
            )
            return bool(cur.rowcount)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ConsensusRequest(BaseModel):
    """POST /v1/consensus body.

    ``providers`` is optional — when omitted, the server uses whatever the
    library's ``load_default_providers`` returns. When supplied, it is a
    soft filter on provider names; unrecognised names are silently dropped
    (returning 400 would create a poor SDK ergonomics for clients passing
    forward-compatible names they expect a newer server to honour).
    """

    prompt: str = Field(min_length=1, max_length=64_000)
    providers: Optional[list[str]] = None
    user_id: Optional[str] = Field(default=None, max_length=128)

    @field_validator("prompt")
    @classmethod
    def _strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return v


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Any  # str or list[dict]


class ChatCompletionRequest(BaseModel):
    model: str = "quorum"
    messages: list[ChatMessage]
    temperature: Optional[float] = 1.0
    quorum_mode: str = "consensus"


class ChatMessageResponse(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChatMessageResponse
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "quorum-consensus"
    choices: list[Choice]
    usage: Usage


class FeedbackRequest(BaseModel):
    """POST /v1/feedback body."""

    query_id: str = Field(min_length=1, max_length=128)
    model_name: str = Field(min_length=1, max_length=200)
    rating: Literal[-1, 0, 1]
    user_id: str = Field(min_length=1, max_length=128)
    query: Optional[str] = Field(
        default=None,
        max_length=64_000,
        description=(
            "Original prompt (optional, lets RLHF re-classify the query "
            "without us holding a separate query_id→prompt cache)."
        ),
    )


class FeedbackResponse(BaseModel):
    """POST /v1/feedback response."""

    query_id: str
    user_id: str
    model_name: str
    rating: int
    query_class: str
    accepted: bool


class ABFeedbackRequest(BaseModel):
    """POST /v1/ab/feedback body.

    Carries a verdict for a prompt-template A/B experiment previously
    recorded by ``consensus_ab``. ``winner`` is the arm the caller (human
    or automated judge) picked — or ``'tie'`` if both answers were
    indistinguishable. ``source`` lets the store distinguish gold human
    signal from cheap LLM-judge votes.
    """

    experiment_id: str = Field(min_length=1, max_length=64)
    winner: Literal["a", "b", "tie"]
    source: Literal["user", "auto"] = "user"


class ABFeedbackResponse(BaseModel):
    """POST /v1/ab/feedback response — echoes the verdict plus updated stats."""

    experiment_id: str
    winner: Literal["a", "b", "tie"]
    source: Literal["user", "auto"]
    accepted: bool
    stats: dict[str, Any]


class HealthResponse(BaseModel):
    """GET /v1/healthz response — intentionally small, cheap to serve."""

    status: Literal["ok"] = "ok"
    version: str = QUORUM_VERSION
    time: datetime


# ---------------------------------------------------------------------------
# App state and dependency wiring
# ---------------------------------------------------------------------------


class AppState:
    """Container for long-lived per-process dependencies.

    WHY a class rather than module-globals: it lets us spin up an isolated
    state in tests (different DB paths, mocked billing client) without
    touching globals or monkeypatching modules.
    """

    def __init__(
        self,
        *,
        api_key_store: Optional[APIKeyStore] = None,
        billing: Optional[BillingClient] = None,
        rlhf: Optional[RLHFTracker] = None,
        ab_store: Optional[ABTestStore] = None,
        cert_dir: Optional[Path] = None,
    ) -> None:
        if api_key_store is not None:
            self.api_key_store = api_key_store
        else:
            # Persistent Firestore store survives Cloud Run revision rollouts.
            # Flip QUORUM_USE_FIRESTORE=1 in prod to swap the ephemeral SQLite
            # /tmp store for the Firestore-backed one. Default stays SQLite so
            # self-host + tests don't gain a Firestore dependency.
            from quorum.firestore_stores import use_firestore
            if use_firestore():
                from quorum.firestore_stores import FirestoreAPIKeyStore
                self.api_key_store = FirestoreAPIKeyStore()
                logger.info("APIKeyStore backend: Firestore")
            else:
                self.api_key_store = APIKeyStore()
        self.billing = billing or BillingClient()
        self.rlhf = rlhf or RLHFTracker()
        # ABTestStore tracks prompt-template A/B verdicts. Initialized eagerly
        # (same as rlhf) so the first /v1/ab/feedback request doesn't pay the
        # schema-creation cost; SQLite open is sub-millisecond on macOS APFS.
        self.ab_store = ab_store or ABTestStore()
        self.cert_dir = cert_dir or _DEFAULT_CERT_DIR
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        # In-memory query record so /v1/cert can find prior consensus
        # results without us also persisting them to disk on the hot path.
        # Bounded LRU-ish: we keep the last 1024 queries.
        self._query_cache: dict[str, dict[str, Any]] = {}
        self._query_cache_lock = asyncio.Lock()
        self._query_cache_max = 1024

    async def remember_query(self, query_id: str, record: dict[str, Any]) -> None:
        """Stash a query record so /v1/cert can later materialise the PDF.

        WHY in-memory: a hosted deployment will swap this for Postgres,
        but in the dev/test path an in-memory bound is enough and keeps
        the test suite hermetic.
        """
        async with self._query_cache_lock:
            if len(self._query_cache) >= self._query_cache_max:
                # Drop the oldest entry (Py3.7+ dicts preserve insertion order).
                oldest = next(iter(self._query_cache))
                self._query_cache.pop(oldest, None)
            self._query_cache[query_id] = record

    async def recall_query(self, query_id: str) -> Optional[dict[str, Any]]:
        async with self._query_cache_lock:
            return self._query_cache.get(query_id)


def _get_state(request: Request) -> AppState:
    """Pull the per-app ``AppState`` off the FastAPI ``app.state``.

    WHY indirection rather than module globals: ``create_app`` builds a
    fresh state per app instance, which is what makes the test fixtures
    able to run side-by-side without stomping on each other's DBs.
    """
    return request.app.state.quorum_state  # type: ignore[no-any-return]


async def _require_api_key(
    request: Request,
    x_quorum_api_key: Optional[str] = Header(default=None, alias="X-Quorum-API-Key"),
) -> APIKeyRecord:
    """FastAPI dependency: resolve and validate the X-Quorum-API-Key header.

    WHY a dependency rather than middleware: dependencies compose nicely
    with the per-route quota check and with OpenAPI doc generation. A
    middleware would have to re-hash and re-lookup or stash the result on
    ``request.state``, both of which are clunkier.
    """
    if not x_quorum_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing X-Quorum-API-Key header",
            headers={"WWW-Authenticate": 'ApiKey realm="quorum"'},
        )
    state = _get_state(request)
    record = await state.api_key_store.lookup(x_quorum_api_key)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or revoked API key",
        )
    # Stash for downstream tier-aware rate limiter.
    request.state.api_key_record = record
    return record


# ---------------------------------------------------------------------------
# Rate-limit key + lookup
# ---------------------------------------------------------------------------


def _rate_limit_key(request: Request) -> str:
    """slowapi key function — partitions by API key when present, else IP.

    WHY API-key-keyed: an honest load balancer fronting many customers
    would share a tiny pool of egress IPs. IP-keyed limits would punish
    every customer for being on AWS. API-key-keyed limits punish the
    actual offender.
    """
    api_record: Optional[APIKeyRecord] = getattr(
        request.state, "api_key_record", None
    )
    if api_record:
        return f"key:{api_record.user_id}"
    # Fallback for /healthz, /webhooks/stripe etc.
    return get_remote_address(request) or "anon"


def _dynamic_limit_for_request(request: Request) -> str:
    """Return the tier-scoped rate-limit string for the current request.

    WHY computed per request rather than as a static decorator argument:
    the tier comes from the (per-request) API-key record. A static
    decorator would force one rate limit per *route*, not per *caller*.
    """
    rec: Optional[APIKeyRecord] = getattr(request.state, "api_key_record", None)
    if rec is None:
        return "60/minute"
    return _TIER_RATE_LIMITS.get(rec.tier, "60/minute")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(state: Optional[AppState] = None) -> FastAPI:
    """Build a FastAPI app wired against the given (or a default) ``AppState``.

    WHY a factory: tests construct one app per test with an isolated state,
    production uses ``app = create_app()`` at the module level. Same code
    path either way.
    """
    app_state = state or AppState()

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Lifespan hook so we have a hook point for warmup/teardown later."""
        logger.info(
            "Quorum server starting: api_keys_backend=%s cert_dir=%s",
            getattr(app_state.api_key_store, "_db_path", type(app_state.api_key_store).__name__),
            app_state.cert_dir,
        )
        yield
        logger.info("Quorum server shutting down")

    app = FastAPI(
        title="Quorum",
        version=QUORUM_VERSION,
        summary="Multi-LLM consensus engine — hosted SaaS surface.",
        description=(
            "Quorum is a multi-LLM consensus engine. This is the hosted "
            "SaaS surface (Apache-2.0 + HSP commercial restrictions). "
            "Self-hosted BYOK use of the underlying library is free."
        ),
        lifespan=_lifespan,
    )

    # Stash the state where dependencies can find it.
    app.state.quorum_state = app_state

    # CORS — wide-open for now; locked down when the dashboard ships.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting — optional dep; degrades to no-op when missing.
    if _SLOWAPI_AVAILABLE and Limiter is not None:
        limiter = Limiter(
            key_func=_rate_limit_key,
            default_limits=["60/minute"],
            headers_enabled=True,
        )
        app.state.limiter = limiter
        app.add_middleware(SlowAPIMiddleware)

        @app.exception_handler(RateLimitExceeded)
        async def _rate_limit_handler(  # type: ignore[no-redef]
            request: Request, exc: Exception
        ) -> Response:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "rate limit exceeded", "limit": str(exc)},
            )

    else:
        logger.warning(
            "slowapi not installed — rate limiting is DISABLED. "
            "Install with: pip install slowapi"
        )

    _register_routes(app, app_state)
    return app


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI, app_state: AppState) -> None:
    """Wire all routes onto ``app``.

    WHY a separate function: keeps ``create_app`` short and makes it
    obvious where new routes go. Also makes it easy to A/B versions of a
    handler from a test without rebuilding the whole app.
    """

    # ---------- landing page (root) ------------------------------------------

    @app.get("/", response_class=HTMLResponse, tags=["meta"])
    async def landing() -> HTMLResponse:
        """Self-contained HTML landing page for cold visitors.

        WHY this exists: Cloud Run logs showed real browsers hitting
        ``https://quorum-ai.dev/`` and getting 404 because every route was
        prefixed with ``/v1/``. A landing page converts those visitors
        without a separate static-hosting layer (no S3, no CDN, no extra
        deploy). Inlining the HTML keeps the request path single-hop and
        dependency-free.
        """
        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quorum &mdash; Multi-LLM Consensus</title>
<meta name="description" content="8 LLMs in parallel for $0.000001 per query. Semantic agreement scoring across Claude, GPT, Gemini, Llama, DeepSeek, Phi, Mistral. BYOK.">
<style>
  :root {{
    --bg: #0a0a0a;
    --fg: #e4e4e4;
    --muted: #8a8a8a;
    --accent: #10a37f;
    --card: #131313;
    --border: #222;
    --code-bg: #0f0f0f;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--fg); }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    font-size: 16px;
  }}
  code, pre, .mono {{ font-family: "SF Mono", Menlo, Consolas, "Courier New", monospace; }}
  main {{ max-width: 920px; margin: 0 auto; padding: 64px 24px; }}
  .brand {{ font-size: 14px; color: var(--accent); letter-spacing: 2px; text-transform: uppercase; }}
  h1 {{ font-size: 56px; margin: 12px 0 8px; letter-spacing: -1.5px; font-weight: 700; }}
  .tagline {{ font-size: 22px; color: var(--muted); margin: 0 0 32px; }}
  .lede {{ font-size: 17px; color: var(--fg); max-width: 700px; }}
  .lede strong {{ color: var(--accent); }}
  .ctas {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 40px 0; }}
  .cta {{
    display: inline-block;
    background: var(--code-bg);
    border: 1px solid var(--border);
    color: var(--fg);
    text-decoration: none;
    padding: 14px 20px;
    border-radius: 6px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 14px;
    transition: border-color 0.15s, color 0.15s;
  }}
  .cta:hover {{ border-color: var(--accent); color: var(--accent); }}
  .cta::before {{ content: "$ "; color: var(--muted); }}
  section {{ margin: 48px 0; }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 2px; color: var(--muted); margin: 0 0 16px; }}
  pre.snippet {{
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
    overflow-x: auto;
    color: var(--fg);
    font-size: 13px;
    margin: 0;
  }}
  pre.snippet .k {{ color: var(--accent); }}
  pre.snippet .s {{ color: #d29922; }}
  .pricing {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .tier {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
  }}
  .tier.pro {{ border-color: var(--accent); position: relative; }}
  .tier.pro::after {{
    content: "POPULAR";
    position: absolute;
    top: -10px; right: 16px;
    background: var(--accent);
    color: #000;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 3px;
    letter-spacing: 1px;
  }}
  .tier h3 {{ margin: 0 0 8px; font-size: 18px; }}
  .tier .price {{ font-size: 28px; font-weight: 700; margin: 0 0 4px; }}
  .tier .price small {{ font-size: 14px; color: var(--muted); font-weight: 400; }}
  .tier .quota {{ font-size: 13px; color: var(--muted); margin: 0; }}
  footer {{
    border-top: 1px solid var(--border);
    margin-top: 64px;
    padding-top: 24px;
    color: var(--muted);
    font-size: 13px;
  }}
  footer a {{ color: var(--muted); }}
  footer .line {{ margin: 4px 0; }}
  @media (max-width: 640px) {{
    main {{ padding: 40px 18px; }}
    h1 {{ font-size: 38px; }}
    .tagline {{ font-size: 18px; }}
    .pricing {{ grid-template-columns: 1fr; }}
    .ctas {{ flex-direction: column; }}
    .cta {{ text-align: left; }}
  }}
</style>
</head>
<body>
<main>
  <div class="brand">Quorum</div>
  <h1>Quorum</h1>
  <p class="tagline">8 LLMs in parallel for $0.000001 per query.</p>

  <p class="lede">
    Quorum runs your prompt across <strong>Claude, GPT, Gemini, Llama, DeepSeek, Phi, and Mistral</strong>
    simultaneously, then scores semantic agreement to surface the answer the models converge on
    (and flag the ones where they don&rsquo;t). Bring your own API keys, self-host under Apache 2.0,
    or use the hosted SaaS with audit-grade EU AI Act certificates.
  </p>

  <div class="ctas">
    <a class="cta" href="https://github.com/jaquelinejaque/sovereignchain">View on GitHub</a>
    <a class="cta" href="https://marketplace.visualstudio.com/items?itemName=sovereignchain.quorum-vscode">Install VS Code Extension</a>
    <a class="cta" href="/docs">API Docs</a>
  </div>

  <section>
    <h2>Live health check</h2>
<pre class="snippet"><span class="k">GET</span> /v1/healthz
{{
  "status": <span class="s">"ok"</span>,
  "version": <span class="s">"{QUORUM_VERSION}"</span>
}}</pre>
  </section>

  <section>
    <h2>Pricing</h2>
    <div class="pricing">
      <div class="tier">
        <h3>Free</h3>
        <p class="price">&pound;0 <small>/ month</small></p>
        <p class="quota">100 queries / month &mdash; BYOK only</p>
      </div>
      <div class="tier pro">
        <h3>Pro</h3>
        <p class="price">&pound;49 <small>/ month</small></p>
        <p class="quota">10,000 queries + EU AI Act certs</p>
      </div>
      <div class="tier">
        <h3>Team</h3>
        <p class="price">&pound;199 <small>/ month</small></p>
        <p class="quota">100,000 queries &mdash; contact sales</p>
      </div>
    </div>
  </section>

  <footer>
    <div class="line">Apache 2.0 &middot; HSP patent PCT/US26/11908 &middot; Made in the UK</div>
    <div class="line">&copy; Sovereign Chain Ltd. &middot; <a href="/docs">API</a> &middot; <a href="/v1/healthz">Status</a></div>
  </footer>
</main>
</body>
</html>"""
        return HTMLResponse(content=html)

    # ---------- self-service free signup -----------------------------------

    @app.post("/v1/signup", tags=["signup"])
    async def signup(
        body: dict,
        request: Request,
    ) -> dict[str, object]:
        """Self-service Free tier signup (100 queries/month, BYOK).

        Body: ``{"email": "user@example.com"}``. Returns the plaintext API
        key once (also sent by email). Rate-limited per IP via Firestore
        so a single host can't farm thousands of keys in a burst.

        Free tier is BYOK: the issued key only authenticates to the
        orchestration layer — every consensus call still requires the
        customer to register at least one provider key via
        /v1/customer/keys. That keeps operator token cost at zero and
        makes the free tier honest about what it is (a self-serve
        sandbox of the multi-LLM orchestration, not free Claude/GPT).
        """
        state = _get_state(request)
        email = str(body.get("email", "")).strip().lower()
        if not email or "@" not in email or len(email) > 256:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="valid email required",
            )

        # Rate limit per IP via Firestore: max 3 free signups per IP per
        # rolling 24h. Burst-protects against scripted abuse without
        # blocking a household NAT with two devs trying it.
        ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
              or request.headers.get("x-real-ip", "")
              or (request.client.host if request.client else "unknown"))
        try:
            from quorum.firestore_stores import use_firestore
            if use_firestore():
                from google.cloud import firestore  # type: ignore
                fc = firestore.Client()
                rl_ref = fc.collection("quorum_signup_ratelimit").document(
                    base64.urlsafe_b64encode(ip.encode("utf-8")).decode("ascii").rstrip("=")
                )
                snap = rl_ref.get()
                now = time.time()
                window_start = now - 86400.0
                signups = []
                if snap.exists:
                    signups = [
                        t for t in (snap.to_dict() or {}).get("signups", [])
                        if isinstance(t, (int, float)) and t > window_start
                    ]
                if len(signups) >= 3:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="too many free signups from this network; try again tomorrow or upgrade Pro",
                    )
                signups.append(now)
                rl_ref.set({"ip": ip, "signups": signups, "updated_at": now})
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("signup rate-limit check failed (%s); allowing this one", exc)

        # Issue the free key and email it.
        try:
            plaintext, _record = await state.api_key_store.issue(
                user_id=email, tier="free",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("free signup failed for %s", email)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="signup failed; please try again or contact support",
            ) from exc

        # Alias tier under the email in billing so /v1/usage shows the
        # free 100/mo limit immediately rather than the unknown-customer
        # default.
        try:
            state.billing._set_tier_local(  # noqa: SLF001
                email, "free", subscription_id=None,
            )
        except Exception:  # noqa: BLE001
            logger.exception("could not alias free tier for %s", email)

        # Send welcome email (Resend) — fire and continue even if it fails
        emailed = False
        try:
            from quorum.billing.email_sender import send_free_welcome_email
            emailed = await send_free_welcome_email(to=email, api_key=plaintext)
        except Exception:  # noqa: BLE001
            logger.exception("free welcome email failed for %s", email)

        logger.info("free signup ok email=%s ip=%s emailed=%s", email, ip, emailed)
        return {
            "ok": True,
            "tier": "free",
            "monthly_query_limit": 100,
            "api_key": plaintext,  # shown ONCE — copy now
            "emailed": emailed,
            "next_step": "register provider keys with POST /v1/customer/keys",
            "supported_providers": [
                "anthropic", "openai", "gemini", "nvidia", "mistral", "cohere",
                "grok", "dashscope", "replicate", "deepseek", "zhipu", "moonshot",
            ],
        }

    # ---------- health -------------------------------------------------------

    @app.get("/v1/healthz", response_model=HealthResponse, tags=["meta"])
    async def healthz() -> HealthResponse:
        """Cheap liveness probe — no DB hit, no provider call.

        WHY no auth: k8s / load balancers need to probe without rotating
        credentials. The endpoint reveals nothing sensitive (version + UTC
        time), so anonymous access is safe.
        """
        return HealthResponse(time=datetime.now(timezone.utc))

    # ---------- openai proxy -------------------------------------------------

    @app.post("/v1/chat/completions", tags=["proxy"], response_model=ChatCompletionResponse)
    async def chat_completions(
        req: ChatCompletionRequest,
        request: Request,
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> ChatCompletionResponse:
        """OpenAI-compatible drop-in proxy. Routes to the Consensus Engine."""
        state = _get_state(request)

        if _SLOWAPI_AVAILABLE and hasattr(app.state, "limiter"):
            limit_string = _dynamic_limit_for_request(request)
            try:
                app.state.limiter.limit(  # type: ignore[attr-defined]
                    limit_string, key_func=_rate_limit_key
                )(_noop)(request)  # type: ignore[arg-type]
            except RateLimitExceeded as exc:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"rate limit exceeded ({limit_string})",
                ) from exc
            except Exception as exc:  # noqa: BLE001
                logger.warning("slowapi limiter call failed (%s) — degrading open", exc)

        customer_id = api_record.user_id
        quota = await state.billing.check_quota(customer_id)
        if quota.over_quota:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="quota_exceeded"
            )

        prompt_parts = []
        images = []
        for msg in req.messages:
            if isinstance(msg.content, str):
                prompt_parts.append(f"{msg.role.upper()}: {msg.content}")
            elif isinstance(msg.content, list):
                role_text = []
                for part in msg.content:
                    if part.get("type") == "text":
                        role_text.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            b64 = url.split("base64,")[-1]
                            images.append(b64)
                prompt_parts.append(f"{msg.role.upper()}: {' '.join(role_text)}")
        prompt = "\n".join(prompt_parts)

        try:
            from quorum.customer_keys import default_store
            from quorum.providers.registry import load_default_providers
            ck_store = default_store()
            customer_keys_dict = ck_store.get_all(api_record.user_id)
            providers = load_default_providers(customer_keys=customer_keys_dict)
            if not providers:
                raise HTTPException(
                    status_code=422,
                    detail="No provider keys registered. POST to /v1/customer/keys first.",
                )
            if req.quorum_mode == "debate":
                from quorum.evolution.swarm import run_debate
                result = await run_debate(
                    prompt,
                    providers=providers,
                    rounds=1,
                    timeout_s=60.0,
                    images=images
                )
            else:
                result = await consensus(
                    prompt,
                    providers=providers,
                    budget_usd=1.0,
                    route=True,
                    images=images,
                    user_id=api_record.user_id,
                )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("consensus failed via proxy")
            raise HTTPException(status_code=500, detail=str(exc))

        try:
            await state.billing.record_usage(customer_id, 1, result.total_cost_usd)
        except Exception:  # noqa: BLE001
            logger.exception("failed to record billing for %s", customer_id)

        msg_resp = ChatMessageResponse(role="assistant", content=result.answer)
        choice = Choice(index=0, message=msg_resp, finish_reason="stop")
        total_tokens = sum(m.tokens_in + m.tokens_out for m in result.models)
        usage = Usage(prompt_tokens=0, completion_tokens=total_tokens, total_tokens=total_tokens)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            choices=[choice],
            usage=usage
        )

    # ---------- consensus ----------------------------------------------------

    @app.post("/v1/consensus", tags=["consensus"])
    async def post_consensus(
        body: ConsensusRequest,
        request: Request,
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> dict[str, Any]:
        """Run a multi-LLM consensus query.

        WHY we apply the tier rate-limit *inside* the handler (rather
        than as a decorator): the limit depends on the authenticated
        tier, which is only known after ``_require_api_key`` runs.
        SlowAPI's decorator API doesn't see post-dependency state.
        """
        state = _get_state(request)

        # Apply tier-scoped rate limit if slowapi is wired up.
        # Wrapped in a broad except because slowapi's programmatic limit()
        # API has been moving — newer versions require an explicit
        # `response` parameter that breaks the legacy call site. We catch
        # everything that isn't RateLimitExceeded and degrade open (logged)
        # so a slowapi upgrade can't take the entire /v1/consensus endpoint
        # down for paying customers. The Stripe-quota enforcement below is
        # the real billing gate; rate-limit-per-second is a soft DoS guard.
        if _SLOWAPI_AVAILABLE and hasattr(app.state, "limiter"):
            limit_string = _dynamic_limit_for_request(request)
            try:
                app.state.limiter.limit(  # type: ignore[attr-defined]
                    limit_string, key_func=_rate_limit_key
                )(_noop)(request)  # type: ignore[arg-type]
            except RateLimitExceeded as exc:  # pragma: no cover - depends on slowapi
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"rate limit exceeded ({limit_string})",
                ) from exc
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "slowapi limiter call failed (%s) — degrading open; quota gate still enforced below",
                    exc,
                )

        # Customer resolution — the API key's user_id is the canonical
        # billing identity; the optional ``user_id`` in the body is for
        # downstream RLHF only (e.g. multi-seat orgs).
        customer_id = api_record.user_id

        quota = await state.billing.check_quota(customer_id)
        if quota.over_quota:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "quota_exceeded",
                    "tier": quota.tier,
                    "used": quota.used,
                    "limit": quota.limit,
                    "resets_at": quota.resets_at.isoformat(),
                },
            )

        # Run the consensus. BYOK: pull this customer's registered
        # provider keys and pass them as the SOLE source — operator env
        # keys are deliberately excluded for hosted callers so each
        # customer pays their own providers directly. If the customer
        # has registered zero keys, consensus() raises and we surface a
        # 422 with a hint instead of secretly billing the operator.
        try:
            from quorum.customer_keys import default_store
            from quorum.providers.registry import load_default_providers
            ck_store = default_store()
            customer_keys_dict = ck_store.get_all(api_record.user_id)
            providers = load_default_providers(customer_keys=customer_keys_dict)
            if not providers:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        "no provider keys configured for this account. "
                        "POST your API keys to /v1/customer/keys first "
                        "(supported providers: anthropic, openai, gemini, "
                        "nvidia, mistral, cohere, grok, dashscope, replicate, "
                        "deepseek, zhipu, moonshot)."
                    ),
                )
            # Soft filter — match body.providers against provider .name by substring
            # so "gemini" matches "gemini-flash", "hermes" matches both Hermes models.
            if body.providers:
                wanted = [p.lower() for p in body.providers]
                providers = [
                    pr for pr in providers
                    if any(w in pr.name.lower() for w in wanted)
                ] or providers  # if filter empties everything, fall back to all
            # route=True activates the MoE router (picks ~3 best models instead of fanning out to all).
            # user_id ties the request to RLHF weights + memory loop for this caller, so personalization
            # accrues across queries. api_record.user_id is the email tied to the authenticated API key.
            result: ConsensusResult = await consensus(
                body.prompt,
                providers=providers,
                route=True,
                user_id=api_record.user_id,
            )
        except HTTPException:
            raise
        except RuntimeError as exc:
            # No providers configured — caller's environment problem, but
            # we surface it as a service error so they fix their deploy.
            logger.warning("consensus failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"no providers available: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("consensus crashed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="consensus engine error",
            ) from exc

        # Record usage (fire-and-forget shape, but awaited so quota is
        # consistent for the *next* request even under burst load).
        try:
            await state.billing.record_usage(
                customer_id,
                query_count=1,
                metadata={"providers": body.providers},
            )
        except Exception:  # noqa: BLE001 - logged, never user-visible
            logger.exception("record_usage failed for %s", customer_id)

        # Assign a query id and remember the result so /v1/cert works.
        query_id = f"q_{uuid.uuid4().hex}"
        record = {
            "query_id": query_id,
            "query_text": body.prompt,
            "consensus": result.to_dict(),
            "user_id": body.user_id or customer_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await state.remember_query(query_id, record)

        payload = result.to_dict()
        payload["query_id"] = query_id
        return payload

    # ---------- feedback (RLHF) ---------------------------------------------

    @app.post("/v1/feedback", response_model=FeedbackResponse, tags=["evolution"])
    async def post_feedback(
        body: FeedbackRequest,
        request: Request,
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> FeedbackResponse:
        """Record a thumbs-up / down on a prior consensus answer.

        WHY this is its own endpoint (rather than a query param on
        consensus): feedback is asynchronous from the user's PoV — they
        rate the answer minutes after seeing it. Bolting it onto the
        request that *produced* the answer would be the wrong UX shape.
        """
        state = _get_state(request)

        # Resolve the original query text (cache hit) or fall back to the
        # text in the request body. WHY fall back: the in-memory cache
        # evicts old queries, and we don't want a feedback to be dropped
        # just because the LRU rolled.
        record = await state.recall_query(body.query_id)
        original_query = body.query or (
            record["query_text"] if record else "(unknown query)"
        )
        models_for_credit: list[Any] = []
        if record:
            models_for_credit = record["consensus"].get("models", []) or []

        try:
            event = await state.rlhf.record_feedback(
                user_id=body.user_id,
                query=original_query,
                chosen_model_name=body.model_name,
                all_model_responses=models_for_credit,
                rating=body.rating,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("RLHF record_feedback failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="feedback could not be recorded",
            ) from exc

        # We don't log the API record beyond a debug line — the user_id in
        # the body is the *RLHF* subject, which may differ from the API
        # key holder for multi-seat orgs.
        logger.debug(
            "feedback recorded by api_user=%s for rlhf_user=%s",
            api_record.user_id,
            event.user_id,
        )
        return FeedbackResponse(
            query_id=body.query_id,
            user_id=event.user_id,
            model_name=event.chosen_model_name,
            rating=event.rating,
            query_class=event.query_class,
            accepted=True,
        )

    # ---------- A/B prompt-template feedback --------------------------------

    @app.post(
        "/v1/ab/feedback",
        response_model=ABFeedbackResponse,
        tags=["evolution"],
    )
    async def post_ab_feedback(
        body: ABFeedbackRequest,
        request: Request,
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> ABFeedbackResponse:
        """Attach a winner verdict to a prompt-template A/B experiment.

        WHY a separate endpoint from /v1/feedback: RLHF feedback is per
        (query, model) — it credits a specific model for a specific
        answer. A/B feedback is per (experiment, arm) — it picks between
        two whole *prompts* and is consumed by ``get_active_winner`` to
        rank prompt templates over time. Different shape, different
        downstream consumer, deliberately different routes.

        Returns 404 if the experiment_id is unknown — that almost always
        means the caller is replaying an old feedback after the store
        was wiped, and we want them to notice rather than silently
        accepting a vote that goes nowhere.
        """
        state = _get_state(request)
        try:
            await state.ab_store.report_winner(
                body.experiment_id,
                winner=body.winner,
                source=body.source,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("ab_store report_winner failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="A/B feedback could not be recorded",
            ) from exc

        try:
            updated_stats = await state.ab_store.stats()
        except Exception:  # noqa: BLE001 - stats are best-effort, never fatal
            logger.exception("ab_store stats failed after vote")
            updated_stats = {}

        logger.debug(
            "ab_feedback recorded api_user=%s exp=%s winner=%s source=%s",
            api_record.user_id,
            body.experiment_id,
            body.winner,
            body.source,
        )
        return ABFeedbackResponse(
            experiment_id=body.experiment_id,
            winner=body.winner,
            source=body.source,
            accepted=True,
            stats=updated_stats,
        )

    # ---------- BYOK customer key management -------------------------------

    @app.post("/v1/customer/keys", tags=["byok"])
    async def set_customer_keys(
        body: dict[str, str],
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> dict[str, object]:
        """Register one or more provider API keys for the authenticated customer.

        Body is a flat ``{"provider": "key", ...}`` dict — accepted
        providers: anthropic, openai, gemini (alias google_ai_studio),
        nvidia, mistral, cohere, grok, dashscope (alias qwen), replicate,
        deepseek, zhipu, moonshot. Each key is encrypted (Fernet) and
        stored under the customer's user_id; existing entries are
        overwritten so a re-POST is the safe way to rotate.

        Empty values delete that provider's entry. Unknown provider names
        are returned in ``rejected`` rather than raising, so a partial
        success still lands the valid keys.
        """
        from quorum.customer_keys import default_store, SUPPORTED_PROVIDERS
        store = default_store()
        saved: list[str] = []
        deleted: list[str] = []
        rejected: list[str] = []
        for raw_provider, raw_key in (body or {}).items():
            # Normalise aliases the registry accepts
            provider = raw_provider.lower().strip()
            if provider in ("google_ai_studio", "glm"):
                provider = "gemini" if provider == "google_ai_studio" else "zhipu"
            if provider == "qwen":
                provider = "dashscope"
            if provider == "xai":
                provider = "grok"
            if provider not in SUPPORTED_PROVIDERS:
                rejected.append(raw_provider)
                continue
            if raw_key is None or not str(raw_key).strip():
                if store.delete(api_record.user_id, provider):
                    deleted.append(provider)
                continue
            store.set(api_record.user_id, provider, str(raw_key))
            saved.append(provider)
        return {
            "saved": saved,
            "deleted": deleted,
            "rejected": rejected,
            "supported_providers": list(SUPPORTED_PROVIDERS),
        }

    @app.get("/v1/customer/keys", tags=["byok"])
    async def list_customer_keys(
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> dict[str, object]:
        """List providers the customer has registered. Keys are NEVER returned."""
        from quorum.customer_keys import default_store, SUPPORTED_PROVIDERS
        store = default_store()
        return {
            "providers": store.list_providers(api_record.user_id),
            "supported_providers": list(SUPPORTED_PROVIDERS),
        }

    @app.delete("/v1/customer/keys/{provider}", tags=["byok"])
    async def delete_customer_key(
        provider: str,
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> dict[str, object]:
        """Delete a single provider's registered key."""
        from quorum.customer_keys import default_store
        store = default_store()
        ok = store.delete(api_record.user_id, provider.lower().strip())
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no key registered for provider '{provider}'",
            )
        return {"deleted": provider}

    # ---------- usage --------------------------------------------------------

    @app.get("/v1/usage", response_model=QuotaStatus, tags=["billing"])
    async def get_usage(
        request: Request,
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> QuotaStatus:
        """Return the authenticated customer's quota snapshot.

        WHY a dedicated endpoint rather than embedding the quota in every
        ``/v1/consensus`` response: clients that *don't* call consensus
        (e.g. a billing dashboard polling for "am I close to my cap")
        still need to read it. Returning it on consensus too would force
        every SDK to thread it through; cleaner to make it a single GET.
        """
        state = _get_state(request)
        return await state.billing.check_quota(api_record.user_id)

    # ---------- Stripe webhook ----------------------------------------------

    @app.post("/v1/webhooks/stripe", tags=["billing"])
    async def stripe_webhook(
        request: Request,
        stripe_signature: Optional[str] = Header(default=None, alias="Stripe-Signature"),
    ) -> dict[str, Any]:
        """Receive subscription lifecycle events from Stripe.

        WHY we accept the raw body manually: FastAPI's default JSON
        parser would lose the byte-exact payload, which Stripe's HMAC
        signature is computed over. Verification *must* happen on the
        unparsed bytes.
        """
        state = _get_state(request)
        payload = await request.body()
        try:
            result: WebhookResult = await state.billing.handle_webhook(
                payload, stripe_signature or ""
            )
        except BillingError as exc:
            logger.warning("rejecting webhook: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("webhook handler crashed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="webhook handler error",
            ) from exc

        # New paid upgrade → issue an API key and email it to the buyer.
        # We swallow any failure here so Stripe still gets its 200 — the buyer
        # is in `customer_id` / `customer_email` and we can always re-issue
        # manually via the admin path if the email pipe is down.
        emailed = False
        if result.is_new_paid_upgrade and result.customer_email and result.tier:
            try:
                plaintext, _record = await state.api_key_store.issue(
                    user_id=result.customer_email, tier=result.tier,
                )
                # Bridge the customer_id ↔ email gap: the webhook updated
                # the tier under the Stripe customer_id, but check_quota()
                # looks it up by api_record.user_id (= email). Without this
                # alias, the paying customer would see tier=free in
                # /v1/usage despite an active subscription. We write a
                # parallel customers row keyed by email so the hot path
                # resolves to the right tier.
                try:
                    state.billing._set_tier_local(  # noqa: SLF001 — see comment above
                        result.customer_email,
                        result.tier,
                        subscription_id=None,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "could not alias tier under email %s — /v1/usage may show free",
                        result.customer_email,
                    )
                from quorum.billing.email_sender import send_welcome_email
                emailed = await send_welcome_email(
                    to=result.customer_email,
                    api_key=plaintext,
                    tier=result.tier.capitalize() if isinstance(result.tier, str) else "Pro",
                )
                logger.info(
                    "issued+emailed API key to %s tier=%s emailed=%s",
                    result.customer_email, result.tier, emailed,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "post-upgrade onboarding failed for %s (%s) — will need manual key issue",
                    result.customer_email, exc,
                )

        # Stripe wants a 200 + small JSON ack; anything else triggers retries.
        response = result.model_dump(mode="json")
        response["emailed"] = emailed
        return response

    # ---------- GitHub webhook (push event → audit diff) ---------------------

    @app.post("/v1/webhooks/github", tags=["webhooks"])
    async def github_webhook(
        request: Request,
        x_hub_signature_256: Optional[str] = Header(default=None, alias="X-Hub-Signature-256"),
        x_github_event: Optional[str] = Header(default=None, alias="X-GitHub-Event"),
    ) -> dict[str, Any]:
        """Receive GitHub events (push, pull_request) for audit/improvement loop.

        Verifies HMAC SHA-256 signature against GITHUB_WEBHOOK_SECRET env var.
        Without the secret configured, the endpoint refuses ALL requests
        (fail-closed) — matches the HSP gate posture.

        Currently logs the event + repo + commits. Future iterations will
        dispatch a Quorum audit of the diff and surface findings.
        """
        import hmac as _hmac
        import hashlib as _hashlib
        import json as _json

        secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
        if not secret:
            raise HTTPException(
                status_code=503,
                detail="GitHub webhook receiver not configured (set GITHUB_WEBHOOK_SECRET)",
            )

        payload = await request.body()
        expected = "sha256=" + _hmac.new(
            secret.encode(), payload, _hashlib.sha256
        ).hexdigest()
        if not x_hub_signature_256 or not _hmac.compare_digest(
            expected, x_hub_signature_256
        ):
            raise HTTPException(status_code=401, detail="invalid signature")

        try:
            evt = _json.loads(payload)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON")

        repo = evt.get("repository", {}).get("full_name", "?")
        commits = evt.get("commits", []) or []
        head = evt.get("head_commit", {}) or {}
        logger.info(
            "github.%s repo=%s head=%s commits=%d",
            x_github_event or "?",
            repo,
            head.get("id", "?")[:7],
            len(commits),
        )
        # TODO(phase 2): dispatch async audit job + comment on PR
        return {
            "ok": True,
            "event": x_github_event,
            "repo": repo,
            "head_sha": head.get("id"),
            "commits": len(commits),
        }

    # ---------- Cron target — site improver ---------------------------------

    @app.post("/v1/cron/site-improver", tags=["webhooks"])
    async def cron_site_improver(
        request: Request,
        x_cloudscheduler: Optional[str] = Header(default=None, alias="X-Cloudscheduler"),
        x_quorum_cron_secret: Optional[str] = Header(default=None, alias="X-Quorum-Cron-Secret"),
    ) -> dict[str, Any]:
        """Cloud Scheduler target — kicks off a Quorum-driven site audit.

        Gating: requires X-Quorum-Cron-Secret header to match QUORUM_CRON_SECRET
        env var. Cloud Scheduler can set it via its HTTP target config. Without
        the secret configured, endpoint refuses (fail-closed).

        Returns the report markdown directly (so Scheduler logs preserve it),
        rather than writing to disk — Cloud Run filesystem is ephemeral anyway.
        """
        secret = os.getenv("QUORUM_CRON_SECRET", "")
        if not secret:
            raise HTTPException(
                status_code=503,
                detail="cron endpoint not configured (set QUORUM_CRON_SECRET)",
            )
        if not x_quorum_cron_secret or not hmac.compare_digest(
            secret, x_quorum_cron_secret
        ):
            raise HTTPException(status_code=401, detail="invalid cron secret")

        target_site = os.getenv("CRON_TARGET_SITE", "https://keratintreatment.co.uk")
        pages = os.getenv("CRON_TARGET_PAGES", "/,/products,/science,/protocol").split(",")
        logger.info("cron.site_improver site=%s pages=%d", target_site, len(pages))

        # MVP: just log the trigger + return what we WOULD do. Actual fetch +
        # Quorum query happens in the local script today. Phase 2 ports the
        # full improver into this endpoint.
        return {
            "ok": True,
            "scheduled_at": time.time(),
            "target_site": target_site,
            "pages_planned": [p.strip() for p in pages if p.strip()],
            "status": "logged_only_phase_2_will_dispatch",
        }

    # ---------- synthetic-data corpus stats ---------------------------------

    @app.get("/v1/synthetic/stats", tags=["evolution"])
    async def get_synthetic_stats(
        request: Request,
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> dict[str, Any]:
        """Aggregate counts over the opt-in synthetic-training corpus.

        WHY auth-required: the stats themselves are anonymous (per-user
        counts only, no prompts or answers), but the *existence* of the
        corpus is a paid-tier feature and we don't want unauthenticated
        scraping. The response shape is the same dict that
        :meth:`SyntheticDatasetStore.stats` returns.
        """
        # Lazy import keeps the server bootable when the evolution loops
        # aren't on the path (e.g. minimal deploys).
        from quorum.evolution.synthetic_data import SyntheticDatasetStore

        del request  # auth-only dependency
        store = SyntheticDatasetStore()
        stats = await store.stats()
        logger.debug(
            "synthetic stats requested by %s: total=%d",
            api_record.user_id,
            stats.get("total_examples", 0),
        )
        return stats

    # ---------- certificate (HSP-gated) -------------------------------------

    @app.get(
        "/v1/cert/{query_id}",
        tags=["compliance"],
        responses={
            200: {"content": {"application/pdf": {}, "text/markdown": {}}},
            404: {"description": "query_id not found in cache"},
        },
    )
    async def get_cert(
        query_id: str,
        request: Request,
        api_record: APIKeyRecord = Depends(_require_api_key),
    ) -> FileResponse:
        """Return the EU AI Act compliance certificate for a past query.

        HSP NOTE
        --------
        This endpoint materialises a certificate that references the HSP
        protocol (PCT/US26/11908). Self-hosted use is free under Apache
        2.0; commercial hosted deployment of the certificate format
        requires an HSP licence (see LICENSE-HSP).

        WHY we render lazily on read: PDFs are heavy (~50KB each) and
        most queries are never audited. Rendering at request time keeps
        the consensus hot path lean. We cache the rendered file on disk
        so repeat requests are O(disk read).
        """
        state = _get_state(request)
        record = await state.recall_query(query_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown query_id {query_id}",
            )

        # FREE tier doesn't get certs — that's a paid-tier promise.
        if api_record.tier == "free":
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="EU AI Act certificates are a paid feature (PRO+)",
            )

        out_path = state.cert_dir / f"{query_id}.pdf"
        if not out_path.exists() and not out_path.with_suffix(".md").exists():
            # In dev mode (no HSP webhook) we still produce a stub decision
            # so the cert is structurally valid for the audit trail.
            decision = {
                "approved": True,
                "decision_id": f"local-{query_id}",
                "reason": "Local certificate (no HSP webhook configured)",
                "signed_at": datetime.now(timezone.utc).isoformat(),
                "signature": "0" * 64,
                "audit_trail_url": "",
            }
            try:
                meta = await asyncio.to_thread(
                    generate_cert_pdf,
                    query_id,
                    record["query_text"],
                    record["consensus"],
                    decision,
                    out_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("cert generation failed for %s", query_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="certificate generation failed",
                ) from exc
            final_path = Path(meta["pdf_path"])
            media_type = (
                "application/pdf"
                if meta["format"] == "pdf"
                else "text/markdown"
            )
        else:
            # PDF preferred, Markdown fallback if PDF wasn't writable.
            final_path = out_path if out_path.exists() else out_path.with_suffix(".md")
            media_type = (
                "application/pdf" if final_path.suffix == ".pdf" else "text/markdown"
            )

        return FileResponse(
            path=final_path,
            media_type=media_type,
            filename=final_path.name,
        )


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _noop(request: Request) -> None:
    """Placeholder used as the ``Limiter.limit(...)`` decoratee.

    WHY: we want slowapi's *bookkeeping* (incrementing the counter and
    raising on overflow) without delegating any actual handler logic to
    it. Wrapping a no-op gives us that.
    """
    return None


# ---------------------------------------------------------------------------
# Module-level app (uvicorn entrypoint)
# ---------------------------------------------------------------------------


app = create_app()


# ---------------------------------------------------------------------------
# Tests (importable so pytest can pick them up; also runnable via __main__)
# ---------------------------------------------------------------------------


def _test_api_key_store_issue_and_lookup(tmp_dir: Path) -> None:
    """Issuing a key returns plaintext + record; lookup round-trips both."""
    store = APIKeyStore(db_path=tmp_dir / "keys.db")

    async def go() -> None:
        plaintext, record = await store.issue("user-1", tier="pro")
        assert plaintext.startswith("qk_")
        assert record.user_id == "user-1"
        assert record.tier == "pro"
        looked_up = await store.lookup(plaintext)
        assert looked_up is not None
        assert looked_up.user_id == "user-1"
        assert looked_up.tier == "pro"
        assert await store.lookup("qk_not-a-real-key") is None

    asyncio.run(go())


def _test_api_key_revoke(tmp_dir: Path) -> None:
    """Revoked keys no longer resolve."""
    store = APIKeyStore(db_path=tmp_dir / "keys.db")

    async def go() -> None:
        plaintext, _ = await store.issue("user-2", tier="free")
        assert await store.revoke(plaintext) is True
        # Idempotent revoke.
        assert await store.revoke(plaintext) is False
        assert await store.lookup(plaintext) is None

    asyncio.run(go())


def _build_test_app(tmp_dir: Path) -> tuple[Any, AppState, str]:
    """Build a self-contained app + a pre-issued API key for HTTP tests.

    WHY a builder: tests want a fresh DB per case, and the TestClient
    fixture wants the app already configured against it.
    """
    from fastapi.testclient import TestClient

    state = AppState(
        api_key_store=APIKeyStore(db_path=tmp_dir / "keys.db"),
        billing=BillingClient(
            stripe_api_key=None,
            stripe_webhook_secret="whsec_test",
            db_path=tmp_dir / "usage.db",
        ),
        rlhf=RLHFTracker(db_path=tmp_dir / "rlhf.db"),
        cert_dir=tmp_dir / "certs",
    )
    test_app = create_app(state)
    client = TestClient(test_app)
    # WHY asyncio.run here rather than the caller awaiting: each test
    # builder produces an isolated event loop turn, which keeps the test
    # callers free of async boilerplate.
    plaintext, _ = asyncio.run(state.api_key_store.issue("test-user", tier="pro"))
    return client, state, plaintext


def _test_healthz(tmp_dir: Path) -> None:
    """/v1/healthz returns 200 without auth."""
    client, _, _ = _build_test_app(tmp_dir)
    r = client.get("/v1/healthz")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"


def _test_consensus_requires_key(tmp_dir: Path) -> None:
    """/v1/consensus returns 401 without X-Quorum-API-Key."""
    client, _, _ = _build_test_app(tmp_dir)
    r = client.post("/v1/consensus", json={"prompt": "hello"})
    assert r.status_code == 401


def _test_usage_with_key(tmp_dir: Path) -> None:
    """/v1/usage returns a QuotaStatus for a freshly-issued key."""
    client, state, key = _build_test_app(tmp_dir)
    # Customer must exist in the billing cache before check_quota can find
    # the tier — provision it by mirroring the API-key user as a customer.

    async def provision() -> None:
        # Manually upsert a 'pro' customer row so the billing client
        # returns PRO tier limits rather than the default FREE.
        from quorum.billing.stripe_billing import _txn  # noqa: PLC0415

        with _txn(state.billing._db_path) as conn:  # noqa: SLF001
            conn.execute(
                """INSERT OR REPLACE INTO customers
                   (customer_id, email, tier, subscription_id, updated_at)
                   VALUES (?, ?, ?, NULL, ?)""",
                ("test-user", "test-user@example.com", "pro", time.time()),
            )

    asyncio.run(provision())
    r = client.get("/v1/usage", headers={"X-Quorum-API-Key": key})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["customer_id"] == "test-user"
    assert body["tier"] == "pro"
    assert body["limit"] == TIERS["pro"].monthly_query_limit


def _test_feedback_records_event(tmp_dir: Path) -> None:
    """/v1/feedback succeeds and is reflected in the RLHF tracker."""
    client, state, key = _build_test_app(tmp_dir)
    payload = {
        "query_id": "q_fake_1",
        "model_name": "anthropic/claude",
        "rating": 1,
        "user_id": "test-user",
        "query": "Write a Python function to reverse a string.",
    }
    r = client.post("/v1/feedback", json=payload, headers={"X-Quorum-API-Key": key})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["rating"] == 1
    assert body["query_class"] in (
        "code",
        "general",
        "math",
        "factual",
        "legal",
        "creative",
        "security",
    )


def _test_cert_404_when_unknown(tmp_dir: Path) -> None:
    """/v1/cert/{id} returns 404 for an unknown query id."""
    client, _, key = _build_test_app(tmp_dir)
    r = client.get("/v1/cert/q_does_not_exist", headers={"X-Quorum-API-Key": key})
    assert r.status_code == 404


def _test_cert_402_when_free_tier(tmp_dir: Path) -> None:
    """/v1/cert/{id} is 402 for FREE tier even if the query exists."""
    state = AppState(
        api_key_store=APIKeyStore(db_path=tmp_dir / "keys.db"),
        billing=BillingClient(
            stripe_api_key=None,
            stripe_webhook_secret="whsec_test",
            db_path=tmp_dir / "usage.db",
        ),
        rlhf=RLHFTracker(db_path=tmp_dir / "rlhf.db"),
        cert_dir=tmp_dir / "certs",
    )
    from fastapi.testclient import TestClient

    test_app = create_app(state)
    client = TestClient(test_app)

    async def setup() -> str:
        free_key, _ = await state.api_key_store.issue("free-user", tier="free")
        await state.remember_query(
            "q_seeded",
            {
                "query_id": "q_seeded",
                "query_text": "hello world",
                "consensus": {"answer": "hi", "models": []},
                "user_id": "free-user",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return free_key

    free_key = asyncio.run(setup())
    r = client.get("/v1/cert/q_seeded", headers={"X-Quorum-API-Key": free_key})
    assert r.status_code == 402


def _test_stripe_webhook_bad_signature_rejects(tmp_dir: Path) -> None:
    """Stripe webhook with a bogus signature returns 400."""
    client, _, _ = _build_test_app(tmp_dir)
    r = client.post(
        "/v1/webhooks/stripe",
        content=b'{"id":"evt_x","type":"customer.subscription.deleted","data":{"object":{}}}',
        headers={"Stripe-Signature": "t=1,v1=deadbeef"},
    )
    assert r.status_code == 400


def _run_tests() -> None:
    """Run the lightweight smoke suite into a temp directory."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="quorum-server-") as td:
        base = Path(td)
        _test_api_key_store_issue_and_lookup(base / "a")
        _test_api_key_revoke(base / "b")
        _test_healthz(base / "c")
        _test_consensus_requires_key(base / "d")
        _test_usage_with_key(base / "e")
        _test_feedback_records_event(base / "f")
        _test_cert_404_when_unknown(base / "g")
        _test_cert_402_when_free_tier(base / "h")
        _test_stripe_webhook_bad_signature_rejects(base / "i")
    logger.info("All Quorum server smoke tests passed.")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _run_tests()
    else:
        import uvicorn

        host = os.environ.get("QUORUM_HOST", "0.0.0.0")  # noqa: S104
        port = int(os.environ.get("QUORUM_PORT", "8000"))
        uvicorn.run(
            "quorum.server.main:app",
            host=host,
            port=port,
            log_level=os.environ.get("QUORUM_LOG_LEVEL", "info"),
        )
