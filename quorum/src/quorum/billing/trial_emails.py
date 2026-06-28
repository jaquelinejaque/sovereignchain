"""Trial-flow transactional emails — Quorum Pro 7-day free trial.

Three templates, one per stage of the trial:

  * Day 0 — Trial started      → :func:`trial_start_html`     / :func:`send_trial_start_email`
  * Day 5 — 2 days remaining   → :func:`trial_day5_html`      / :func:`send_trial_day5_email`
  * Day 8 — Trial ended        → :func:`trial_ended_html`     / :func:`send_trial_ended_email`

Why separate from :mod:`email_sender`
-------------------------------------
The paid-welcome template is already 200+ lines and is on the critical
hot path (sent inside the Stripe webhook handler). Trial emails are
fire-and-forget on a separate scheduler, so keeping them in their own
module lets the welcome path stay tiny and lets the trial schedule
evolve without touching the proven webhook code.

All three reuse :func:`quorum.billing.email_sender.send_email` so the
Resend client, retry logic, and ``RESEND_FROM_EMAIL`` env handling stay
in one place.
"""

from __future__ import annotations

import logging
from typing import Optional

from quorum.billing.email_sender import send_email

logger = logging.getLogger("quorum.billing.trial_emails")


# Shared visual palette — identical to email_sender so the trial flow
# doesn't feel like a different product than the paid welcome email.
_BG = "#0a0a0c"
_CARD = "#111114"
_BORDER = "#1f1f24"
_TEXT = "#e8e8ea"
_TEXT_DIM = "#d8d8dc"
_TEXT_FAINT = "#9a9aa0"
_TEXT_GHOST = "#7a7a80"
_ACCENT = "#7fb8e8"
_GOLD = "#caac7d"
_DARK = "#050507"


_FOOTER = f"""
        <tr><td style="padding:18px 36px 28px;border-top:1px solid {_BORDER};">
          <p style="font-size:12px;color:{_TEXT_GHOST};margin:0;line-height:1.6;">
            Reply to this email for support or to cancel.<br>
            Manage subscription: <a href="https://billing.stripe.com/p/login" style="color:{_ACCENT};">Stripe customer portal</a><br>
            Open-source self-host (free, FSL-1.1): <a href="https://github.com/jaquelinejaque/sovereignchain" style="color:{_ACCENT};">github.com/jaquelinejaque/sovereignchain</a>
          </p>
        </td></tr>

      </table>
      <div style="font-size:11px;color:#5a5a60;margin-top:18px;">Sovereign Chain Ltd · UK · PCT/US26/11908</div>
    </td></tr>
  </table>
</body>
</html>"""


def _wrap(title: str, header_kicker: str, body_inner: str) -> str:
    """Shared HTML shell so the three trial emails look like one product.

    ``body_inner`` is the per-stage content; everything around it
    (background, card, brand stamp, footer) is identical to the paid
    welcome email so trial users see the same brand on Day 0 → Day 8.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
</head>
<body style="margin:0;padding:0;background:{_BG};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;color:{_TEXT};">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{_BG};padding:48px 24px;">
    <tr><td align="center">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;background:{_CARD};border:1px solid {_BORDER};border-radius:10px;">

        <tr><td style="padding:36px 36px 12px;text-align:center;">
          <div style="font-family:'Cormorant Garamond',Georgia,serif;font-size:28px;color:{_ACCENT};letter-spacing:3px;">QUORUM</div>
          <div style="color:#8b8b90;font-size:11px;letter-spacing:3px;text-transform:uppercase;margin-top:6px;">{header_kicker}</div>
        </td></tr>

{body_inner}

{_FOOTER}"""


# ---------------------------------------------------------------------------
# Day 0 — Trial started
# ---------------------------------------------------------------------------


def trial_start_html(api_key: str, email: str = "", trial_end_date: str = "in 7 days") -> str:
    """Day-0 trial-start email. Sent inside the Stripe checkout-session-completed
    webhook when the subscription is in ``trialing`` status.

    The user has not been charged. They have a working API key, full Pro
    access, and a deadline. Tone: practical, no hard sell, surface the
    three concrete actions they can take in the first session.
    """
    safe_email = email or "you"
    key_preview = api_key[:14] + "..."
    body = f"""
        <tr><td style="padding:0 36px 24px;">
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">Welcome, {safe_email}.</p>
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">
            Your 7-day Pro trial is active. No card has been charged yet. Your trial ends <strong>{trial_end_date}</strong>, after which the £15/month subscription you authorised begins — or you can cancel any time from the Stripe portal at the bottom of this email.
          </p>
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">
            Here is your Pro API key. <strong>Copy it now — we store only a hash and cannot show it again</strong>:
          </p>

          <div style="margin:20px 0;padding:18px 20px;background:{_DARK};border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,Consolas,monospace;font-size:14px;color:{_ACCENT};word-break:break-all;">{api_key}</div>

          <p style="font-size:13px;color:{_TEXT_FAINT};margin-top:8px;">If you lose it, reply to this email and we'll issue a replacement and revoke this one.</p>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="font-size:13px;color:#8b8b90;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">Try these three things in your first session</div>

          <div style="margin-bottom:18px;padding:16px;background:#1a1107;border:1px solid {_GOLD};border-radius:6px;">
            <div style="font-size:14px;color:{_GOLD};margin-bottom:8px;"><strong>1 · Register your provider keys (BYOK, 30 seconds)</strong></div>
            <p style="font-size:13px;color:{_TEXT_DIM};line-height:1.6;margin:0 0 10px;">
              Quorum is BYOK — you bring keys for Claude/GPT/Gemini/etc. and Quorum orchestrates the consensus. Your providers, your bills. Quorum charges only £15/mo for the orchestration, refusal filter, hallucination guard, and audit chain.
            </p>
            <pre style="margin:0;padding:12px;background:{_DARK};border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,monospace;font-size:11px;color:{_GOLD};overflow-x:auto;">curl -X POST https://api.quorum-ai.dev/v1/customer/keys \\
  -H "X-Quorum-API-Key: {key_preview}" \\
  -H "Content-Type: application/json" \\
  -d '{{"anthropic":"sk-ant-...","openai":"sk-...","gemini":"..."}}'</pre>
          </div>

          <div style="margin-bottom:18px;">
            <div style="font-size:14px;color:{_TEXT};margin-bottom:6px;"><strong>2 · Run your first consensus query</strong></div>
            <pre style="margin:0;padding:14px;background:{_DARK};border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,monospace;font-size:12px;color:{_GOLD};overflow-x:auto;">curl -X POST https://api.quorum-ai.dev/v1/consensus \\
  -H "X-Quorum-API-Key: {key_preview}" \\
  -d '{{"prompt":"Should I use sqlite or postgres for 100 paying users?"}}'</pre>
            <p style="font-size:12px;color:{_TEXT_FAINT};margin:8px 0 0;">
              You'll get back the canonical answer, the per-model confidence breakdown, and the disagreement matrix. If any sub-model recused itself, the refusal filter excludes it from the score automatically.
            </p>
          </div>

          <div>
            <div style="font-size:14px;color:{_TEXT};margin-bottom:6px;"><strong>3 · See the hallucination guard in action</strong></div>
            <p style="font-size:13px;color:{_TEXT_DIM};line-height:1.6;margin:0 0 10px;">
              Ask something in a long-tail regulated domain (UK FCA rules, EU AI Act articles, US tax §, etc.). When 70%+ of the models agree confidently in a fabricated answer, the convergent-hallucination guard fires and downgrades the score in the response. We shipped this because the engine demonstrated the failure mode to us first — full story in the public repo.
            </p>
          </div>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="font-size:13px;color:#8b8b90;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">Pro tier includes</div>
          <ul style="font-size:14px;line-height:1.8;color:{_TEXT_DIM};padding-left:20px;margin:0;">
            <li>5,000 consensus queries / month</li>
            <li>14+ LLM pool — Anthropic, OpenAI, Gemini, Mistral, Cohere, Grok, Qwen, NVIDIA, local Llama</li>
            <li>BYOK across every provider</li>
            <li>Convergent-hallucination guard (new — caught its own 5/5 fabrication on 2026-06-28)</li>
            <li>Refusal filter (sub-models that recuse don't pollute the score)</li>
            <li>Disagreement matrix exposed — see exactly where the models split</li>
            <li>Tamper-evident audit chain (hash-linked, EU AI Act Annex VI ready)</li>
          </ul>
        </td></tr>
"""
    return _wrap(
        title="Your Quorum Pro trial has started",
        header_kicker="Multi-LLM Consensus Engine · 7-day Pro trial",
        body_inner=body,
    )


async def send_trial_start_email(*, to: str, api_key: str, trial_end_date: str = "in 7 days") -> bool:
    """Day-0 send. Returns True on Resend 200, False otherwise.

    Failure is logged but never raised — losing this email is bad UX
    but losing it inside the Stripe webhook handler would also lose the
    user a working subscription, which is worse.
    """
    html = trial_start_html(api_key=api_key, email=to, trial_end_date=trial_end_date)
    return await send_email(
        to=to,
        subject="Your Quorum Pro trial has started — your API key inside",
        html=html,
    )


# ---------------------------------------------------------------------------
# Day 5 — 2 days remaining
# ---------------------------------------------------------------------------


def trial_day5_html(email: str = "", days_used: int = 5, queries_used: int = 0) -> str:
    """Day-5 reminder. Sent by a scheduled job (cron / Cloud Scheduler)
    that walks the customers table for ``status='trialing'`` and
    ``trial_started_at`` between 4.5 and 5.5 days ago.

    Tone: helpful, surfaces what they HAVE done so far (queries_used)
    so the value is concrete, then states the upcoming charge clearly.
    No fake urgency. No "limited time offer." The trial is a trial.
    """
    safe_email = email or "you"
    usage_line = (
        f"You've run <strong>{queries_used}</strong> consensus queries so far. "
        if queries_used > 0
        else "You haven't run a consensus query yet — here's the curl to get started: "
    )

    body = f"""
        <tr><td style="padding:0 36px 24px;">
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">Hi {safe_email},</p>
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">
            Your Quorum Pro trial is on day {days_used} of 7. <strong>2 days remaining.</strong>
          </p>
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">
            {usage_line}On day 8 the £15/month subscription you authorised begins — unless you cancel, which you can do from the Stripe portal link below in one click.
          </p>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="font-size:13px;color:#8b8b90;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">Three Pro features worth trying before day 8</div>

          <div style="margin-bottom:14px;">
            <div style="font-size:14px;color:{_TEXT};margin-bottom:4px;"><strong>· Disagreement matrix</strong></div>
            <p style="font-size:13px;color:{_TEXT_DIM};line-height:1.6;margin:0;">
              When the models split (which is the interesting case), Quorum surfaces a pairwise matrix of who disagreed with whom. Useful for "which model do I trust on this kind of question?" — Pro tier only.
            </p>
          </div>

          <div style="margin-bottom:14px;">
            <div style="font-size:14px;color:{_TEXT};margin-bottom:4px;"><strong>· Hallucination risk on every response</strong></div>
            <p style="font-size:13px;color:{_TEXT_DIM};line-height:1.6;margin:0;">
              The ``hallucination_risk`` field on every Pro response tells you whether convergent fabrication patterns were detected (UK regulatory terms, dated legal citations, exact monetary figures, SOC/NACE/ICD codes, versioned documents). Low/elevated/high, with the per-flag evidence.
            </p>
          </div>

          <div>
            <div style="font-size:14px;color:{_TEXT};margin-bottom:4px;"><strong>· Audit chain receipt (PDF)</strong></div>
            <p style="font-size:13px;color:{_TEXT_DIM};line-height:1.6;margin:0;">
              <code style="background:#1a1a1f;padding:2px 6px;border-radius:3px;color:{_GOLD};">GET /v1/receipt/{{query_id}}</code> returns a hash-linked PDF receipt suitable for EU AI Act Annex VI evidence. Sovereign Chain is not a Notified Body — this is advisory evidence, not certification.
            </p>
          </div>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="font-size:13px;color:{_TEXT_FAINT};line-height:1.6;">
            Already decided? Reply to this email with "cancel" and I'll process it before day 8 — no card form, no friction.
          </div>
        </td></tr>
"""
    return _wrap(
        title="2 days left on your Quorum Pro trial",
        header_kicker="Day 5 of 7 · Trial reminder",
        body_inner=body,
    )


async def send_trial_day5_email(*, to: str, queries_used: int = 0) -> bool:
    html = trial_day5_html(email=to, queries_used=queries_used)
    return await send_email(
        to=to,
        subject="2 days left on your Quorum Pro trial",
        html=html,
    )


# ---------------------------------------------------------------------------
# Day 8 — Trial ended, billing started (or cancelled)
# ---------------------------------------------------------------------------


def trial_ended_html(email: str = "", subscription_active: bool = True) -> str:
    """Day-8 confirmation. Sent by the Stripe webhook on
    ``customer.subscription.updated`` when the subscription transitions
    from ``trialing`` to ``active`` (paid) — or from ``trialing`` to
    ``canceled`` (cancelled before charge).

    Branches on ``subscription_active`` so we send the right tone:

      * Active   → "you're a paid Pro now, here's what changes" (nothing
                    practical — same API key, same features, just billing
                    flipped on; the value is reassurance not action).
      * Cancelled → "thanks for trying, here's how to come back" (low-
                    pressure, links to FREE tier so we don't lose them
                    entirely to a competitor).
    """
    safe_email = email or "you"
    if subscription_active:
        body = f"""
        <tr><td style="padding:0 36px 24px;">
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">Hi {safe_email},</p>
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">
            Your 7-day trial ended and your Quorum Pro subscription is now active. <strong>£15</strong> was charged via Stripe and your monthly billing cycle has started.
          </p>
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">
            <strong>Nothing practical changes.</strong> Same API key, same Pro features, same 5,000 queries/month allowance — just the billing flag flipped from "trialing" to "active" on the Stripe side.
          </p>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="font-size:13px;color:#8b8b90;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">A few things worth knowing</div>
          <ul style="font-size:14px;line-height:1.7;color:{_TEXT_DIM};padding-left:20px;margin:0;">
            <li>Your queries reset on the 1st of each month (UTC). Unused queries don't roll over.</li>
            <li>If you cross 5,000 queries in a month, Quorum returns a 429 with a clear "upgrade or wait" message — no surprise overage charges.</li>
            <li>To upgrade to Team (£49/mo, 50,000 queries, audit log, SSO), reply to this email — that tier is contact-sales only because of the SSO setup.</li>
            <li>To cancel any time: <a href="https://billing.stripe.com/p/login" style="color:{_ACCENT};">Stripe customer portal</a> · one click, no questions.</li>
          </ul>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <p style="font-size:13px;color:{_TEXT_FAINT};line-height:1.6;margin:0;">
            Thanks for taking the trial. If you hit anything weird in the first paid month, reply to this email — it goes to a human, not a queue.
          </p>
        </td></tr>
"""
        title = "Your Quorum Pro subscription is active"
        kicker = "Trial ended · Subscription active"
        subject_status = "Welcome to Quorum Pro — your subscription is active"
    else:
        body = f"""
        <tr><td style="padding:0 36px 24px;">
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">Hi {safe_email},</p>
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">
            Your Quorum Pro trial has ended and your subscription was cancelled before any charge was made. <strong>£0 was billed.</strong> Your Pro API key has been deactivated.
          </p>
          <p style="font-size:15px;line-height:1.6;color:{_TEXT_DIM};">
            No follow-up sales emails are coming. This is the only one.
          </p>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="font-size:13px;color:#8b8b90;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">If you still want to use Quorum</div>

          <div style="margin-bottom:14px;">
            <div style="font-size:14px;color:{_TEXT};margin-bottom:4px;"><strong>Free tier — 100 queries/month</strong></div>
            <p style="font-size:13px;color:{_TEXT_DIM};line-height:1.6;margin:0;">
              Same consensus engine, just rate-limited and without the disagreement matrix / audit chain receipts. Reply "free tier" to this email and I'll issue you a free API key — takes 30 seconds, no card.
            </p>
          </div>

          <div>
            <div style="font-size:14px;color:{_TEXT};margin-bottom:4px;"><strong>Self-host — free, FSL-1.1 source</strong></div>
            <p style="font-size:13px;color:{_TEXT_DIM};line-height:1.6;margin:0;">
              <a href="https://github.com/jaquelinejaque/sovereignchain" style="color:{_ACCENT};">github.com/jaquelinejaque/sovereignchain</a> — clone, set <code style="background:#1a1a1f;padding:2px 6px;border-radius:3px;color:{_GOLD};">QUORUM_DEV_MODE=1</code>, you have unlimited consensus locally. No license required for honour-system non-commercial use.
            </p>
          </div>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <p style="font-size:13px;color:{_TEXT_FAINT};line-height:1.6;margin:0;">
            If something specific drove the cancel — a missing model, a price point, a missing feature — I'd genuinely value the one-line feedback. Reply to this email; it goes to a human.
          </p>
        </td></tr>
"""
        title = "Your Quorum Pro trial ended — no charge"
        kicker = "Trial ended · Cancelled"
        subject_status = "Your Quorum Pro trial ended — no charge"

    return _wrap(title=title, header_kicker=kicker, body_inner=body)


async def send_trial_ended_email(
    *, to: str, subscription_active: bool = True,
) -> bool:
    html = trial_ended_html(email=to, subscription_active=subscription_active)
    subject = (
        "Welcome to Quorum Pro — your subscription is active"
        if subscription_active
        else "Your Quorum Pro trial ended — no charge"
    )
    return await send_email(to=to, subject=subject, html=html)


__all__ = [
    "trial_start_html",
    "send_trial_start_email",
    "trial_day5_html",
    "send_trial_day5_email",
    "trial_ended_html",
    "send_trial_ended_email",
]
