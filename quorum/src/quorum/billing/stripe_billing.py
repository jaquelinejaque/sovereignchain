# Copyright 2026 Jaqueline Martins / Sovereign Chain.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This module is part of Quorum (multi-LLM consensus engine).
#
# HSP ATTRIBUTION
# ---------------
# The hosted SaaS surface (quotas, SSO, audit log, dashboards) that this
# billing module monetises is a Harmonised Sovereignty Protocol (HSP) gated
# feature. HSP is filed as PCT/US26/11908. Self-hosted, single-tenant,
# bring-your-own-key (BYOK) use of Quorum remains free under Apache 2.0.
# Commercial hosted deployment of this billing module requires an HSP licence.
# See LICENSE-HSP in the repository root.
"""Stripe billing client for the Quorum hosted SaaS surface.

WHY THIS MODULE EXISTS
----------------------
Quorum is dual-licensed: solo BYOK self-host is free under Apache 2.0, hosted
SaaS access (with quotas, SSO, dashboards, audit log) is paid. This module is
the *single source of truth* for that paid surface:

* It defines the tier matrix (PRO / FREE / TEAM / ENTERPRISE / COMPLIANCE).
  PRO £49/mo is the default self-serve, headline product; TEAM / ENTERPRISE
  / COMPLIANCE are contact-sales (no self-serve Stripe Checkout).
* It talks to Stripe for customer + subscription lifecycle.
* It enforces per-customer monthly quotas via a *local* SQLite cache so the
  hot path (one ``check_quota`` call per Quorum query) never touches the
  Stripe API. Hammering Stripe per query would both blow up our latency
  budget and trigger Stripe rate limits.
* It exposes a webhook handler so Stripe can push subscription lifecycle
  events (created / cancelled / payment_failed) and we can update the
  local cache in O(1).

DEV MODE
--------
When ``STRIPE_SECRET_KEY`` is not set in the environment, the client
silently degrades into a pure-Python in-memory implementation. Every
customer is treated as FREE tier, network calls are stubbed, and webhook
verification is skipped. This is what makes the test-suite runnable
without any real Stripe keys and what lets contributors hack locally
without an account.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field

try:  # pragma: no cover - optional dependency
    import stripe as _stripe_sdk  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - tested via dev-mode fallback
    _stripe_sdk = None  # type: ignore[assignment]


__all__ = [
    "Tier",
    "TIERS",
    "TierConfig",
    "DEFAULT_TIER",
    "get_default_tier",
    "list_tiers",
    "QuotaStatus",
    "WebhookResult",
    "BillingClient",
    "BillingError",
]

logger = logging.getLogger("quorum.billing.stripe")


# ---------------------------------------------------------------------------
# Tier matrix
# ---------------------------------------------------------------------------
#
# The tier matrix is intentionally encoded as plain constants (not loaded from
# config) because pricing is a *commercial contract* with our users: it must
# be auditable in version control, not silently overridable by an env var.
# If pricing ever changes, that change must show up in a git diff.


Tier = Literal["pro", "free", "team", "enterprise", "compliance"]


@dataclass(frozen=True)
class TierConfig:
    """Static description of one pricing tier.

    Frozen dataclass (not pydantic) so it can be hashed and used as a dict
    key, and so accidental mutation at runtime raises immediately.

    WHY ``amount_pence`` exists alongside ``price_gbp_monthly``: Stripe's
    API speaks minor units (pence), so the canonical machine-readable
    price lives in ``amount_pence``. ``price_gbp_monthly`` is a derived,
    human-friendly integer kept for backward compatibility with anything
    that already read the field. They must stay in sync.

    WHY ``contact_sales``: tiers that require a human conversation
    (TEAM / ENTERPRISE / COMPLIANCE) should NOT be reachable via the
    self-serve Stripe Checkout flow. Marking them here keeps the API
    surface honest — front-ends can simply hide the "Subscribe" button
    and show "Contact Sales" instead.
    """

    name: Tier
    price_gbp_monthly: Optional[int]  # None == "contact sales"
    amount_pence: Optional[int]  # canonical Stripe minor-unit amount; None == contact sales
    currency: str  # ISO 4217 lower-case ("gbp")
    interval: Literal["month", "year", "none"]  # billing cadence; "none" for free / contact-sales
    monthly_query_limit: int  # 0 == unlimited (enterprise, usage-billed)
    byok: bool
    sso: bool
    audit_log: bool
    dashboard: bool
    semantic_scoring: bool
    contact_sales: bool  # True == not self-serve; route to humans
    stripe_price_env: Optional[str]
    # Env var holding the Stripe Price ID. Kept as an env-var *name* rather
    # than the price ID itself so the matrix can be committed publicly.


# ``TIERS`` is intentionally ordered Pro-first. Python 3.7+ guarantees
# insertion-order iteration over ``dict``, so anything that does
# ``next(iter(TIERS))`` or ``list(TIERS)[0]`` gets PRO — our headline
# self-serve product.
TIERS: dict[Tier, TierConfig] = {
    "pro": TierConfig(
        name="pro",
        price_gbp_monthly=49,
        amount_pence=4900,
        currency="gbp",
        interval="month",
        monthly_query_limit=5_000,
        byok=True,
        sso=False,
        audit_log=False,
        dashboard=True,
        semantic_scoring=True,
        contact_sales=False,
        stripe_price_env="STRIPE_PRICE_PRO",
    ),
    "free": TierConfig(
        name="free",
        price_gbp_monthly=0,
        amount_pence=0,
        currency="gbp",
        interval="none",
        monthly_query_limit=100,
        byok=True,
        sso=False,
        audit_log=False,
        dashboard=False,
        semantic_scoring=False,
        contact_sales=False,
        stripe_price_env=None,
    ),
    "team": TierConfig(
        name="team",
        price_gbp_monthly=199,
        amount_pence=19_900,
        currency="gbp",
        interval="month",
        monthly_query_limit=50_000,
        byok=True,
        sso=True,
        audit_log=True,
        dashboard=True,
        semantic_scoring=True,
        contact_sales=True,
        stripe_price_env="STRIPE_PRICE_TEAM",
    ),
    "enterprise": TierConfig(
        name="enterprise",
        price_gbp_monthly=None,
        amount_pence=None,
        currency="gbp",
        interval="none",
        monthly_query_limit=0,  # usage-billed, no hard cap
        byok=True,
        sso=True,
        audit_log=True,
        dashboard=True,
        semantic_scoring=True,
        contact_sales=True,
        stripe_price_env="STRIPE_PRICE_ENTERPRISE_USAGE",
    ),
    "compliance": TierConfig(
        name="compliance",
        price_gbp_monthly=None,
        amount_pence=None,
        currency="gbp",
        interval="none",
        monthly_query_limit=0,  # bespoke, usage- or seat-billed per contract
        byok=True,
        sso=True,
        audit_log=True,
        dashboard=True,
        semantic_scoring=True,
        contact_sales=True,
        stripe_price_env="STRIPE_PRICE_COMPLIANCE",
    ),
}


# The default / headline self-serve product is PRO. This constant exists so
# call sites don't have to hard-code the string in three places.
DEFAULT_TIER: Tier = "pro"


def get_default_tier() -> TierConfig:
    """Return the default / headline self-serve tier (PRO £49/mo).

    WHY this exists: marketing pages, API ``/pricing`` endpoints, and the
    CLI all want one canonical answer to "what should we show first?".
    Centralising it means a future re-launch only flips one constant.
    """
    return TIERS[DEFAULT_TIER]


def list_tiers(*, self_serve_only: bool = False) -> list[TierConfig]:
    """Return tiers in display order, PRO first.

    Args:
        self_serve_only: If True, omit ``contact_sales`` tiers
            (TEAM / ENTERPRISE / COMPLIANCE) — useful for rendering the
            Stripe Checkout picker that should never let a user click
            "Subscribe" on a tier we sell via humans.

    WHY list (not dict): order matters for UI rendering and dict ordering
    is an implementation detail callers shouldn't depend on.
    """
    tiers = list(TIERS.values())
    if self_serve_only:
        tiers = [t for t in tiers if not t.contact_sales]
    return tiers


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------


class QuotaStatus(BaseModel):
    """Snapshot of a customer's current month quota.

    Returned by ``BillingClient.check_quota`` and meant to be cheap to
    produce (SQLite-backed) so the consensus hot path can call it per-query.
    """

    customer_id: str
    tier: Tier
    used: int = Field(ge=0)
    limit: int = Field(ge=0, description="0 means unlimited (enterprise).")
    remaining: int = Field(ge=0)
    resets_at: datetime
    over_quota: bool = False


class WebhookResult(BaseModel):
    """Outcome of processing a Stripe webhook event.

    We return a structured result (rather than raising) for everything that
    isn't a signature failure, because Stripe expects ``200 OK`` for any
    event we *received* successfully — even ones we don't care about — to
    avoid retries hammering us.
    """

    event_id: str
    event_type: str
    handled: bool
    customer_id: Optional[str] = None
    tier: Optional[Tier] = None
    message: str = ""
    # Set on checkout.session.completed so the server-side webhook endpoint
    # can issue an API key and email it. Stripe puts the buyer email in
    # data.object.customer_details.email — we pull it once here so the
    # endpoint doesn't have to re-parse the raw payload.
    customer_email: Optional[str] = None
    is_new_paid_upgrade: bool = False


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BillingError(Exception):
    """Raised for any unrecoverable billing failure.

    Recoverable conditions (over-quota, dev mode, missing customer) are
    expressed as return values; only bugs and integrity violations raise.
    """


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------


def _current_period_key(now: Optional[datetime] = None) -> str:
    """Return a ``YYYY-MM`` bucket for monthly quota windows.

    WHY: We bucket usage by calendar month rather than billing-cycle anchor
    on the FREE tier because FREE users have no Stripe subscription anchor
    to align against. For paid tiers this is "close enough" — the webhook
    handler resets usage on ``customer.subscription.updated`` if the real
    anchor drifts.
    """
    now = now or datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def _period_reset_at(period_key: str) -> datetime:
    """Compute the UTC moment a monthly bucket rolls over.

    WHY: ``QuotaStatus.resets_at`` is shown to end users in dashboards, so
    it must be a real timestamp, not a vague "next month". We compute the
    first instant of the following month in UTC.
    """
    year, month = (int(p) for p in period_key.split("-"))
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    return datetime(year, month, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------
#
# Schema:
#   customers(customer_id PRIMARY KEY, email, tier, subscription_id,
#             updated_at)
#   usage(customer_id, period_key, count, PRIMARY KEY(customer_id, period_key))
#
# We keep the cache *authoritative* for quota enforcement. Stripe is the
# authority for *tier*, and the webhook handler is the only thing that
# mutates the ``tier`` column in the steady state.


DATA_DIR = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()
_DEFAULT_DB_PATH = DATA_DIR / "usage.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open (and migrate, if needed) the SQLite usage cache.

    WHY a private helper: connections are short-lived per operation so
    every call site benefits from the same migration check. SQLite's
    ``CREATE TABLE IF NOT EXISTS`` makes this idempotent and free.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id   TEXT PRIMARY KEY,
            email         TEXT NOT NULL,
            tier          TEXT NOT NULL DEFAULT 'free',
            subscription_id TEXT,
            updated_at    REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage (
            customer_id   TEXT NOT NULL,
            period_key    TEXT NOT NULL,
            count         INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (customer_id, period_key)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)")
    return conn


@contextmanager
def _txn(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Context-managed SQLite transaction.

    WHY: SQLite + threading + isolation_level=None is fiddly; centralising
    BEGIN/COMMIT/ROLLBACK here means call sites stay readable and we never
    leak a half-open transaction on exception paths.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# BillingClient
# ---------------------------------------------------------------------------


class BillingClient:
    """High-level Stripe + local-cache facade for Quorum hosted SaaS.

    Designed so that the *only* place the rest of the codebase needs to
    know about billing is this class. Everything else (consensus engine,
    CLI, API) should depend on the small surface defined here.

    The client is intentionally cheap to construct: heavy work (Stripe
    network calls) happens lazily on the first method call, and the SQLite
    file is opened per-operation rather than held open.
    """

    def __init__(
        self,
        *,
        stripe_api_key: Optional[str] = None,
        stripe_webhook_secret: Optional[str] = None,
        db_path: Optional[Path] = None,
        success_url: str = "https://quorum.ai/billing/success",
        cancel_url: str = "https://quorum.ai/billing/cancel",
    ) -> None:
        self._api_key = stripe_api_key or os.environ.get("STRIPE_SECRET_KEY")
        self._webhook_secret = stripe_webhook_secret or os.environ.get(
            "STRIPE_WEBHOOK_SECRET"
        )
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._success_url = success_url
        self._cancel_url = cancel_url

        # Dev mode is decided once at construction. Toggling it mid-run
        # would create heisenbugs ("why did Stripe suddenly start charging
        # in tests?"), so we freeze it here.
        # Persistent customer + usage cache via Firestore when the flag is
        # set in prod. Without this, every Cloud Run revision rollout wipes
        # paying customers' tier/quota (they revert to free + 100 limit),
        # forcing manual reissue. Self-host stays on the ephemeral SQLite
        # path so a dev clone has no Firestore dependency.
        self._fs_store = None
        try:
            from quorum.firestore_stores import use_firestore, FirestoreBillingCache
            if use_firestore():
                self._fs_store = FirestoreBillingCache()
                logger.info("BillingClient cache backend: Firestore")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Firestore billing cache unavailable (%s); using SQLite at %s", e, self._db_path,
            )

        self._dev_mode = not self._api_key or _stripe_sdk is None
        if self._dev_mode:
            logger.warning(
                "BillingClient running in DEV MODE (no STRIPE_SECRET_KEY or "
                "stripe SDK unavailable). All customers treated as FREE tier; "
                "no real charges will be made."
            )
        else:
            # The stripe SDK is module-global state; we set the key once.
            _stripe_sdk.api_key = self._api_key  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Customer lifecycle
    # ------------------------------------------------------------------

    async def get_or_create_customer(self, email: str) -> str:
        """Return a stable Stripe customer id for ``email``.

        WHY async: even though Stripe's SDK is synchronous, we wrap it in
        ``asyncio.to_thread`` so calling code in our async stack doesn't
        block the event loop. WHY get-or-create: avoids duplicate Stripe
        customers when a user signs up twice (e.g. retry after a failed
        confirmation email).
        """
        email = email.strip().lower()
        if not email or "@" not in email:
            raise BillingError(f"invalid email: {email!r}")

        # Fast path: cache hit.
        with _txn(self._db_path) as conn:
            row = conn.execute(
                "SELECT customer_id FROM customers WHERE email = ?", (email,)
            ).fetchone()
            if row:
                logger.debug("customer cache hit", extra={"email": email})
                return str(row[0])

        if self._dev_mode:
            cust_id = f"dev_cus_{uuid.uuid4().hex[:16]}"
            logger.info("dev-mode: created synthetic customer %s", cust_id)
        else:
            # Stripe's `list(email=...)` is the documented dedupe pattern.
            customer = await asyncio.to_thread(self._stripe_customer_get_or_create, email)
            cust_id = str(customer["id"])

        with _txn(self._db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO customers
                   (customer_id, email, tier, subscription_id, updated_at)
                   VALUES (?, ?, 'free', NULL, ?)""",
                (cust_id, email, time.time()),
            )
        return cust_id

    def _stripe_customer_get_or_create(self, email: str) -> dict[str, Any]:
        """Sync Stripe call, isolated so ``to_thread`` has a clean target.

        WHY: keeping the sync surface tiny (one function per Stripe call)
        makes it trivial to mock in tests and gives us a single place to
        add retries/backoff later.
        """
        assert _stripe_sdk is not None  # invariant: dev_mode guards this
        existing = _stripe_sdk.Customer.list(email=email, limit=1)
        if existing.data:
            return dict(existing.data[0])
        # Stripe is at-least-once on client retries; without an idempotency
        # key a transient connection reset between request send and response
        # receive can create two customers with the same email (the
        # ``list(email=...)`` dedupe above is a TOCTOU window). Hash the
        # lowercased email so retries collapse to the same key.
        idem_key = hashlib.sha256(email.lower().encode()).hexdigest()[:32]
        created = _stripe_sdk.Customer.create(
            email=email, idempotency_key=idem_key
        )
        return dict(created)

    # ------------------------------------------------------------------
    # Subscription lifecycle
    # ------------------------------------------------------------------

    async def create_subscription(self, customer_id: str, tier: str) -> str:
        """Start a subscription for ``customer_id`` on the given ``tier``.

        For paid tiers (pro/team) we return a Stripe Checkout URL the user
        must complete in their browser; that's how Stripe wants SCA-bound
        flows handled in 2026. For ``free`` we just update the local
        cache. For ``enterprise`` we return a sentinel + log because that
        flow is human-driven (contact sales).

        WHY return a *string* rather than a structured object: callers
        usually need exactly one piece of information (the URL to redirect
        to, or the subscription id). Pydantic-wrapping it would force every
        caller to introspect a field they don't care about.
        """
        if tier not in TIERS:
            raise BillingError(f"unknown tier: {tier!r}")
        tier_typed: Tier = tier  # type: ignore[assignment]
        cfg = TIERS[tier_typed]

        if tier_typed == "free":
            self._set_tier_local(customer_id, "free", subscription_id=None)
            return f"free:{customer_id}"

        # Any contact-sales tier (TEAM / ENTERPRISE / COMPLIANCE) is routed
        # to humans rather than self-serve Stripe Checkout. We log and
        # return a stable sentinel so the API surface can render a "Talk to
        # sales" CTA instead of a Stripe URL.
        if cfg.contact_sales:
            logger.info(
                "%s subscription requested for %s — routing to sales",
                tier_typed,
                customer_id,
            )
            return "contact-sales"

        if self._dev_mode:
            fake_sub = f"dev_sub_{uuid.uuid4().hex[:16]}"
            self._set_tier_local(customer_id, tier_typed, subscription_id=fake_sub)
            logger.info("dev-mode: created synthetic subscription %s", fake_sub)
            return f"https://quorum.local/dev/checkout/{fake_sub}"

        price_env = cfg.stripe_price_env
        assert price_env is not None  # mypy: pro/team always have one
        price_id = os.environ.get(price_env)
        if not price_id:
            raise BillingError(
                f"{price_env} is not configured; cannot create {tier} subscription"
            )

        session = await asyncio.to_thread(
            self._stripe_create_checkout_session, customer_id, price_id
        )
        return str(session["url"])

    def _stripe_create_checkout_session(
        self, customer_id: str, price_id: str
    ) -> dict[str, Any]:
        """Create a Stripe Checkout Session for a subscription.

        WHY Checkout (vs raw PaymentIntent): Checkout owns SCA, tax,
        receipts, dunning, and currency conversion. Re-implementing those
        ourselves would be a multi-quarter project and a constant source
        of compliance bugs.
        """
        assert _stripe_sdk is not None
        # Idempotency: scope to (customer, price, UTC day) so a network retry
        # within the same day collapses to one Checkout Session, but the user
        # can still legitimately start a new flow tomorrow. Prevents Stripe
        # analytics pollution from duplicate Session creates.
        day_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        idem_raw = f"checkout:{customer_id}:{price_id}:{day_bucket}"
        idem_key = hashlib.sha256(idem_raw.encode()).hexdigest()[:32]
        session = _stripe_sdk.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=self._success_url,
            cancel_url=self._cancel_url,
            allow_promotion_codes=True,
            idempotency_key=idem_key,
        )
        return dict(session)

    # ------------------------------------------------------------------
    # Usage recording
    # ------------------------------------------------------------------

    async def record_usage(
        self,
        customer_id: str,
        query_count: int = 1,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Increment ``customer_id``'s usage counter by ``query_count``.

        Always writes to the local cache (the source of truth for quota
        enforcement). Additionally reports usage to Stripe for enterprise
        usage-based subscriptions, where Stripe is the source of truth for
        the invoice.

        WHY not block on Stripe in the hot path: usage reporting to Stripe
        is fire-and-forget — we log failures but never propagate them to
        the caller, because losing a single usage record is much less bad
        than losing a successful Quorum query.
        """
        if query_count <= 0:
            return
        period_key = _current_period_key()

        if self._fs_store is not None:
            self._fs_store.increment_usage(customer_id, period_key, query_count)
            cust = self._fs_store.get_customer(customer_id) or {}
            tier: Tier = cust.get("tier", "free")  # type: ignore[assignment]
            subscription_id: Optional[str] = cust.get("subscription_id")
        else:
            with _txn(self._db_path) as conn:
                conn.execute(
                    """INSERT INTO usage (customer_id, period_key, count)
                       VALUES (?, ?, ?)
                       ON CONFLICT(customer_id, period_key)
                       DO UPDATE SET count = count + excluded.count""",
                    (customer_id, period_key, query_count),
                )
                row = conn.execute(
                    "SELECT tier, subscription_id FROM customers WHERE customer_id = ?",
                    (customer_id,),
                ).fetchone()
            tier: Tier = (row[0] if row else "free")  # type: ignore[assignment]
            subscription_id: Optional[str] = row[1] if row else None

        if metadata:
            logger.debug("usage metadata for %s: %s", customer_id, metadata)

        if tier == "enterprise" and not self._dev_mode and subscription_id:
            # Generate the idempotency key BEFORE the (potentially retried)
            # Stripe call so any retry sees the same key and Stripe collapses
            # it server-side. A fresh UUID per logical invocation guarantees
            # distinct legitimate increments stay distinct.
            usage_idem_key = uuid.uuid4().hex
            try:
                await asyncio.to_thread(
                    self._stripe_report_usage,
                    subscription_id,
                    query_count,
                    usage_idem_key,
                )
            except Exception as exc:  # noqa: BLE001 - intentional swallow
                logger.warning(
                    "failed to report usage to Stripe for %s: %s",
                    customer_id,
                    exc,
                )

    def _stripe_report_usage(
        self,
        subscription_id: str,
        quantity: int,
        idempotency_key: str,
    ) -> None:
        """Send a usage record to Stripe for an enterprise subscription.

        WHY a dedicated method: usage-billing item resolution (subscription
        item id vs. subscription id) is fiddly and version-dependent;
        isolating it here means we update *one* place when Stripe changes
        the API shape (again).

        WHY ``idempotency_key`` is REQUIRED: enterprise customers are
        usage-billed. Without it, a retry from the ``asyncio.to_thread``
        wrapper, an SDK-internal retry on a network blip, or an asyncio
        cancellation the SDK sees as a hung connection would double-bill
        the customer. The swallowed exception at the caller means we'd
        never see it. The caller MUST pass a key it generated BEFORE the
        call so a retry reuses the same value.
        """
        assert _stripe_sdk is not None
        sub = _stripe_sdk.Subscription.retrieve(subscription_id)
        items = sub["items"]["data"]
        if not items:
            raise BillingError(f"subscription {subscription_id} has no items")
        item_id = items[0]["id"]
        _stripe_sdk.SubscriptionItem.create_usage_record(
            item_id,
            quantity=quantity,
            timestamp=int(time.time()),
            action="increment",
            idempotency_key=idempotency_key,
        )

    # ------------------------------------------------------------------
    # Quota check
    # ------------------------------------------------------------------

    async def check_quota(self, customer_id: str) -> QuotaStatus:
        """Return the current quota status, reading only the SQLite cache.

        WHY async even though it never awaits: keeps the public API
        uniform with the rest of ``BillingClient`` so callers don't have
        to remember which methods are sync. If we later add an async
        Stripe verification path here, we won't have to change call sites.
        """
        period_key = _current_period_key()
        if self._fs_store is not None:
            tier_str = self._fs_store.get_tier(customer_id) or "free"
            tier: Tier = tier_str  # type: ignore[assignment]
            used = self._fs_store.get_usage(customer_id, period_key)
        else:
            with _txn(self._db_path) as conn:
                row = conn.execute(
                    "SELECT tier FROM customers WHERE customer_id = ?",
                    (customer_id,),
                ).fetchone()
                used_row = conn.execute(
                    "SELECT count FROM usage WHERE customer_id = ? AND period_key = ?",
                    (customer_id, period_key),
                ).fetchone()
            tier: Tier = (row[0] if row else "free")  # type: ignore[assignment]
            used = int(used_row[0]) if used_row else 0
        limit = TIERS[tier].monthly_query_limit
        if limit == 0:  # unlimited (enterprise)
            remaining = 2**31 - 1
            over = False
        else:
            remaining = max(0, limit - used)
            over = used >= limit
        return QuotaStatus(
            customer_id=customer_id,
            tier=tier,
            used=used,
            limit=limit,
            remaining=remaining,
            resets_at=_period_reset_at(period_key),
            over_quota=over,
        )

    # ------------------------------------------------------------------
    # Webhook handling
    # ------------------------------------------------------------------

    async def handle_webhook(self, payload: bytes, signature: str) -> WebhookResult:
        """Verify and process a Stripe webhook event.

        WHY verify even in dev mode (when secret is set): a misconfigured
        dev environment that accepts unsigned webhooks would silently
        accept malicious ones in staging too. So we verify whenever a
        secret is available, regardless of mode.
        """
        event = self._parse_and_verify(payload, signature)
        etype = str(event.get("type", "unknown"))
        eid = str(event.get("id", f"evt_{uuid.uuid4().hex[:12]}"))
        data_object = event.get("data", {}).get("object", {}) or {}
        customer_id = data_object.get("customer")

        handled = False
        new_tier: Optional[Tier] = None
        message = ""
        customer_email: Optional[str] = None
        is_new_paid_upgrade = False

        if etype == "checkout.session.completed":
            sub_id = data_object.get("subscription")
            new_tier = self._infer_tier_from_price(data_object)
            # Stripe puts the buyer email in customer_details.email after the
            # session completes. Fall back to data_object.customer_email
            # (older API versions) and finally None if the buyer was a
            # guest checkout with no email at all.
            details = data_object.get("customer_details") or {}
            customer_email = details.get("email") or data_object.get("customer_email")
            if customer_id and new_tier:
                self._set_tier_local(customer_id, new_tier, subscription_id=sub_id)
                handled = True
                # Flag for the server endpoint: this is a NEW paying upgrade
                # that needs an API key + welcome email. We only set this
                # on checkout.session.completed (not on subscription.updated)
                # so existing subscribers updating their plan don't get a
                # second welcome email.
                is_new_paid_upgrade = new_tier != "free"
                message = f"upgraded {customer_id} to {new_tier}"

        elif etype == "customer.subscription.updated":
            new_tier = self._infer_tier_from_subscription(data_object)
            if customer_id and new_tier:
                self._set_tier_local(
                    customer_id, new_tier, subscription_id=data_object.get("id")
                )
                handled = True
                message = f"updated {customer_id} to {new_tier}"

        elif etype in ("customer.subscription.deleted", "customer.subscription.canceled"):
            if customer_id:
                self._set_tier_local(customer_id, "free", subscription_id=None)
                new_tier = "free"
                handled = True
                message = f"downgraded {customer_id} to free (subscription ended)"

        elif etype == "invoice.payment_failed":
            # We don't downgrade immediately on a single failure — Stripe
            # has its own dunning retries. We just log and let the eventual
            # subscription.deleted event do the work.
            handled = True
            message = f"payment failed for {customer_id}, awaiting Stripe dunning"
            logger.warning(message)

        else:
            message = f"unhandled event type {etype}"
            logger.info(message)

        return WebhookResult(
            event_id=eid,
            event_type=etype,
            handled=handled,
            customer_id=customer_id,
            tier=new_tier,
            message=message,
            customer_email=customer_email,
            is_new_paid_upgrade=is_new_paid_upgrade,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_and_verify(self, payload: bytes, signature: str) -> dict[str, Any]:
        """Parse a webhook payload and verify the Stripe-Signature header.

        WHY ``stripe.Webhook.construct_event`` is preferred: it is the
        Stripe-maintained reference implementation of the v1 signing
        scheme, including the constant-time comparison and the 5-minute
        replay tolerance. Hand-rolling our own verifier is a footgun —
        any divergence from Stripe's behaviour (new ``v2`` scheme,
        timestamp rules, etc.) becomes a silent security regression on
        our side, not theirs. We delegate when the SDK is available.

        WHY a fallback still exists: when the Stripe SDK is not
        installed (dev mode, contributor laptops, CI without the extra),
        we still want the webhook code path to be exercisable. The
        fallback uses ``hmac.compare_digest`` + a 5-minute replay window
        — the same algorithm the SDK uses — and is only reachable when
        ``_stripe_sdk is None``.
        """
        if not self._webhook_secret:
            if self._dev_mode:
                logger.warning(
                    "dev-mode webhook: STRIPE_WEBHOOK_SECRET missing, skipping verify"
                )
                return json.loads(payload.decode("utf-8"))
            raise BillingError("STRIPE_WEBHOOK_SECRET is required to verify webhooks")

        if not signature:
            raise BillingError("missing Stripe-Signature header")

        if _stripe_sdk is not None:
            # Canonical path: Stripe SDK does signature + replay verification.
            try:
                event = _stripe_sdk.Webhook.construct_event(
                    payload=payload,
                    sig_header=signature,
                    secret=self._webhook_secret,
                )
            except Exception as exc:  # noqa: BLE001 - SDK surfaces multiple exception classes
                # Covers SignatureVerificationError, ValueError (bad JSON), etc.
                raise BillingError(f"invalid Stripe-Signature: {exc}") from exc
            # ``construct_event`` returns a ``stripe.Event`` (a StripeObject
            # subclass). ``dict(event)`` triggers ``__iter__`` which Stripe
            # implements as positional access, raising ``KeyError: 0``.
            # Use ``to_dict_recursive`` when available; otherwise re-parse
            # the already-verified payload (it's the same bytes Stripe signed).
            if isinstance(event, dict):
                return event
            if hasattr(event, "to_dict_recursive"):
                return event.to_dict_recursive()
            return json.loads(payload.decode("utf-8"))

        # Fallback (no SDK available): mirror Stripe's v1 algorithm.
        timestamp, sig = self._parse_signature_header(signature)
        signed_payload = f"{timestamp}.{payload.decode('utf-8')}".encode("utf-8")
        expected = hmac.new(
            self._webhook_secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, sig):
            raise BillingError("invalid Stripe-Signature")

        # Replay-window check: reject events older than 5 minutes.
        try:
            ts_int = int(timestamp)
            if abs(time.time() - ts_int) > 300:
                raise BillingError("webhook timestamp outside tolerance")
        except ValueError:
            raise BillingError("malformed webhook timestamp") from None

        return json.loads(payload.decode("utf-8"))

    @staticmethod
    def _parse_signature_header(header: str) -> tuple[str, str]:
        """Pull (timestamp, v1 signature) out of a Stripe-Signature header.

        Stripe sends ``t=<ts>,v1=<hex>``; we tolerate extra fields (``v0``)
        because Stripe has historically added new schemes without notice.
        """
        parts = dict(
            kv.split("=", 1) for kv in header.split(",") if "=" in kv
        )
        if "t" not in parts or "v1" not in parts:
            raise BillingError("malformed Stripe-Signature header")
        return parts["t"], parts["v1"]

    def _infer_tier_from_price(self, session_obj: dict[str, Any]) -> Optional[Tier]:
        """Best-effort tier inference from a checkout.session.completed.

        WHY best-effort: checkout sessions don't always inline the line
        items, and we'd rather treat an unknown tier as "leave alone" than
        accidentally downgrade a paying customer.
        """
        amount = session_obj.get("amount_total")
        if amount is None:
            return None
        # Stripe amounts are minor units (pence). Thresholds are anchored
        # to the canonical ``amount_pence`` values in TIERS so a future
        # price change only has to touch the matrix above.
        team_pence = TIERS["team"].amount_pence or 19_900
        pro_pence = TIERS["pro"].amount_pence or 4_900
        if amount >= team_pence:
            return "team"
        if amount >= pro_pence:
            return "pro"
        return None

    def _infer_tier_from_subscription(
        self, sub_obj: dict[str, Any]
    ) -> Optional[Tier]:
        """Map a subscription object's price id back to one of our tiers."""
        items = (sub_obj.get("items") or {}).get("data") or []
        if not items:
            return None
        price_id = items[0].get("price", {}).get("id")
        if not price_id:
            return None
        for tier_name, cfg in TIERS.items():
            if cfg.stripe_price_env and os.environ.get(cfg.stripe_price_env) == price_id:
                return tier_name
        return None

    def _set_tier_local(
        self,
        customer_id: str,
        tier: Tier,
        *,
        subscription_id: Optional[str],
    ) -> None:
        """Persist a tier change. Firestore-backed in hosted mode, SQLite otherwise.

        WHY UPSERT rather than UPDATE: webhooks can arrive for a customer
        whose row we never created locally (e.g. they paid via a flow that
        bypassed ``get_or_create_customer``). We still want to honour the
        upgrade.
        """
        if self._fs_store is not None:
            self._fs_store.set_tier(
                customer_id, tier,
                email=f"{customer_id}@unknown.local",
                subscription_id=subscription_id,
            )
            return
        with _txn(self._db_path) as conn:
            conn.execute(
                """INSERT INTO customers
                       (customer_id, email, tier, subscription_id, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(customer_id) DO UPDATE SET
                       tier = excluded.tier,
                       subscription_id = excluded.subscription_id,
                       updated_at = excluded.updated_at""",
                (customer_id, f"{customer_id}@unknown.local", tier, subscription_id, time.time()),
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
#
# These run under ``pytest`` (preferred) when imported by the suite, and also
# directly via ``python -m quorum.billing.stripe_billing`` for quick smoke
# testing without pytest installed. They exercise dev-mode only — real Stripe
# integration tests live elsewhere and require live keys.


def _fresh_client(tmp_dir: Path) -> BillingClient:
    """Build a dev-mode client backed by a private SQLite file.

    WHY a helper: every test wants an isolated DB, and forgetting to pass
    ``db_path`` would silently scribble on the user's real ~/.quorum/usage.db.
    """
    return BillingClient(
        stripe_api_key=None,
        stripe_webhook_secret="whsec_test_secret",
        db_path=tmp_dir / "usage.db",
    )


async def _test_dev_mode_customer_and_quota(tmp_dir: Path) -> None:
    """Free tier defaults to 100/month, increments on record_usage."""
    bc = _fresh_client(tmp_dir)
    cust = await bc.get_or_create_customer("Test@Example.COM")
    assert cust.startswith("dev_cus_")
    again = await bc.get_or_create_customer("test@example.com")
    assert again == cust, "email normalisation must dedupe"

    status = await bc.check_quota(cust)
    assert status.tier == "free"
    assert status.used == 0
    assert status.limit == 100
    assert status.remaining == 100
    assert status.over_quota is False

    await bc.record_usage(cust, query_count=10)
    status = await bc.check_quota(cust)
    assert status.used == 10
    assert status.remaining == 90


async def _test_dev_mode_subscription_upgrade(tmp_dir: Path) -> None:
    """create_subscription on PRO/TEAM upgrades the local tier in dev mode."""
    bc = _fresh_client(tmp_dir)
    cust = await bc.get_or_create_customer("pro@example.com")
    url = await bc.create_subscription(cust, "pro")
    assert url.startswith("https://quorum.local/dev/checkout/")
    status = await bc.check_quota(cust)
    assert status.tier == "pro"
    assert status.limit == 5_000


async def _test_quota_exhaustion(tmp_dir: Path) -> None:
    """over_quota flips true once usage >= limit on FREE."""
    bc = _fresh_client(tmp_dir)
    cust = await bc.get_or_create_customer("exhaust@example.com")
    await bc.record_usage(cust, query_count=100)
    status = await bc.check_quota(cust)
    assert status.used == 100
    assert status.remaining == 0
    assert status.over_quota is True


def _sign(payload: bytes, secret: str, ts: Optional[int] = None) -> str:
    """Build a valid Stripe-Signature header for tests."""
    ts = ts or int(time.time())
    signed = f"{ts}.{payload.decode('utf-8')}".encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


async def _test_webhook_checkout_completed_upgrades(tmp_dir: Path) -> None:
    """checkout.session.completed with PRO-level amount upgrades the customer."""
    bc = _fresh_client(tmp_dir)
    cust = await bc.get_or_create_customer("hook@example.com")
    payload = json.dumps(
        {
            "id": "evt_test_1",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "customer": cust,
                    "subscription": "sub_test_1",
                    # Anchored to PRO's canonical amount_pence (£49 = 4_900)
                    # so the inference threshold and this fixture stay in sync.
                    "amount_total": TIERS["pro"].amount_pence,
                }
            },
        }
    ).encode("utf-8")
    sig = _sign(payload, "whsec_test_secret")
    result = await bc.handle_webhook(payload, sig)
    assert result.handled is True
    assert result.tier == "pro"
    status = await bc.check_quota(cust)
    assert status.tier == "pro"


async def _test_webhook_bad_signature_rejects(tmp_dir: Path) -> None:
    """Tampered signatures must raise BillingError."""
    bc = _fresh_client(tmp_dir)
    payload = b'{"id":"evt_x","type":"customer.subscription.deleted","data":{"object":{}}}'
    bad_sig = "t=1,v1=deadbeef"
    raised = False
    try:
        await bc.handle_webhook(payload, bad_sig)
    except BillingError:
        raised = True
    assert raised, "bad signature must raise"


async def _test_webhook_subscription_deleted_downgrades(tmp_dir: Path) -> None:
    """subscription.deleted moves customer back to FREE."""
    bc = _fresh_client(tmp_dir)
    cust = await bc.get_or_create_customer("downgrade@example.com")
    await bc.create_subscription(cust, "pro")
    assert (await bc.check_quota(cust)).tier == "pro"
    payload = json.dumps(
        {
            "id": "evt_test_2",
            "type": "customer.subscription.deleted",
            "data": {"object": {"customer": cust}},
        }
    ).encode("utf-8")
    sig = _sign(payload, "whsec_test_secret")
    result = await bc.handle_webhook(payload, sig)
    assert result.handled is True
    assert result.tier == "free"
    assert (await bc.check_quota(cust)).tier == "free"


def _test_pro_is_default_tier() -> None:
    """PRO £49/mo is the canonical default + first in list_tiers()."""
    pro = get_default_tier()
    assert pro.name == "pro"
    assert pro.price_gbp_monthly == 49
    assert pro.amount_pence == 4_900
    assert pro.currency == "gbp"
    assert pro.interval == "month"
    assert pro.contact_sales is False
    # Insertion order: PRO must be first when iterating TIERS / list_tiers.
    assert list(TIERS.keys())[0] == "pro"
    assert list_tiers()[0].name == "pro"
    # Self-serve filter must drop TEAM / ENTERPRISE / COMPLIANCE.
    self_serve = list_tiers(self_serve_only=True)
    self_serve_names = {t.name for t in self_serve}
    assert "pro" in self_serve_names
    assert "free" in self_serve_names
    assert self_serve_names.isdisjoint({"team", "enterprise", "compliance"})


def _test_contact_sales_flags() -> None:
    """TEAM / ENTERPRISE / COMPLIANCE must still exist and be contact_sales."""
    for tier_name in ("team", "enterprise", "compliance"):
        cfg = TIERS[tier_name]  # type: ignore[index]
        assert cfg.contact_sales is True, f"{tier_name} must be contact_sales"
    # FREE and PRO remain self-serve.
    assert TIERS["free"].contact_sales is False
    assert TIERS["pro"].contact_sales is False


async def _test_contact_sales_routes_to_sales(tmp_dir: Path) -> None:
    """create_subscription on a contact_sales tier returns the sentinel."""
    bc = _fresh_client(tmp_dir)
    cust = await bc.get_or_create_customer("team@example.com")
    for tier_name in ("team", "enterprise", "compliance"):
        result = await bc.create_subscription(cust, tier_name)
        assert result == "contact-sales", f"{tier_name} must route to sales"


async def _run_all_tests() -> None:
    """Run the dev-mode test suite into a temp directory."""
    import tempfile

    _test_pro_is_default_tier()
    _test_contact_sales_flags()
    with tempfile.TemporaryDirectory(prefix="quorum-billing-") as td:
        tmp = Path(td)
        await _test_dev_mode_customer_and_quota(tmp / "a")
        await _test_dev_mode_subscription_upgrade(tmp / "b")
        await _test_quota_exhaustion(tmp / "c")
        await _test_webhook_checkout_completed_upgrades(tmp / "d")
        await _test_webhook_bad_signature_rejects(tmp / "e")
        await _test_webhook_subscription_deleted_downgrades(tmp / "f")
        await _test_contact_sales_routes_to_sales(tmp / "g")
    logger.info("all dev-mode billing tests passed")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    asyncio.run(_run_all_tests())
