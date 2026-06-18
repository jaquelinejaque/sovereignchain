"""Transactional email — Resend HTTP API client.

Sends the welcome-with-API-key email after Stripe webhook confirms a paid
upgrade. Pure HTTP via httpx so we keep the dependency tree thin (no Resend
SDK), and so test mode degrades gracefully when RESEND_API_KEY is missing.

Required env vars in production:
  RESEND_API_KEY     — from https://resend.com/api-keys (shared with Keratin)
  RESEND_FROM_EMAIL  — verified sender, e.g.
                       "Quorum <onboarding@quorum-ai.dev>"
                       Default falls back to the Keratin verified sender so
                       prod still sends even before the quorum-ai.dev domain
                       is verified on Resend.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("quorum.billing.email_sender")

_RESEND_ENDPOINT = "https://api.resend.com/emails"

# Sender configurable; default uses Keratin's already-verified domain so the
# first welcome emails land before quorum-ai.dev sender verification ships.
_DEFAULT_FROM = "Quorum <onboarding@keratintreatment.co.uk>"
_DEFAULT_SUPPORT = "facecomercce1@gmail.com"


def _api_key() -> str | None:
    return os.environ.get("RESEND_API_KEY")


def _from_email() -> str:
    return os.environ.get("RESEND_FROM_EMAIL") or _DEFAULT_FROM


def _support_email() -> str:
    return os.environ.get("QUORUM_SUPPORT_EMAIL") or _DEFAULT_SUPPORT


async def send_email(
    *,
    to: str,
    subject: str,
    html: str,
    reply_to: str | None = None,
) -> dict[str, Any] | None:
    """Send via Resend. Returns the API response dict, or None if degraded.

    Never raises — a missing key or 4xx/5xx logs and returns None so the
    webhook handler always 200s back to Stripe (else Stripe retries forever).
    """
    key = _api_key()
    if not key:
        logger.warning("RESEND_API_KEY not set; skipping email to %s", to)
        return None

    payload = {
        "from": _from_email(),
        "to": [to],
        "subject": subject,
        "html": html,
        "reply_to": reply_to or _support_email(),
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(_RESEND_ENDPOINT, json=payload, headers=headers)
        if r.status_code >= 400:
            body = r.text[:300]
            logger.warning("Resend %s for %s: %s", r.status_code, to, body)
            return None
        return r.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Resend call failed for %s: %s", to, exc)
        return None


def welcome_html(api_key: str, tier: str = "Pro", email: str = "") -> str:
    """Render the welcome email body for a freshly-paid customer.

    Sends the plaintext API key exactly once (Quorum stores only a hash —
    we can't regenerate this). The instructions cover the three most likely
    first uses: curl, VS Code extension, Python client.
    """
    safe_email = email or "you"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Welcome to Quorum {tier}</title>
</head>
<body style="margin:0;padding:0;background:#0a0a0c;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;color:#e8e8ea;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0c;padding:48px 24px;">
    <tr><td align="center">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;background:#111114;border:1px solid #1f1f24;border-radius:10px;">

        <tr><td style="padding:36px 36px 12px;text-align:center;">
          <div style="font-family:'Cormorant Garamond',Georgia,serif;font-size:28px;color:#7fb8e8;letter-spacing:3px;">QUORUM</div>
          <div style="color:#8b8b90;font-size:11px;letter-spacing:3px;text-transform:uppercase;margin-top:6px;">Multi-LLM Consensus Engine · {tier} tier</div>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <p style="font-size:15px;line-height:1.6;color:#d8d8dc;">Welcome, {safe_email}.</p>
          <p style="font-size:15px;line-height:1.6;color:#d8d8dc;">Your subscription is active. Here is your API key — <strong>copy it now, we don't store it in plaintext and cannot show it again</strong>:</p>

          <div style="margin:20px 0;padding:18px 20px;background:#050507;border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,Consolas,monospace;font-size:14px;color:#7fb8e8;word-break:break-all;">{api_key}</div>

          <p style="font-size:13px;color:#9a9aa0;margin-top:8px;">If you lose it, reply to this email and I'll issue a replacement and revoke this one.</p>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="font-size:13px;color:#8b8b90;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">Three ways to use it</div>

          <div style="margin-bottom:18px;padding:16px;background:#1a1107;border:1px solid #caac7d;border-radius:6px;">
            <div style="font-size:14px;color:#caac7d;margin-bottom:8px;"><strong>⚠️ FIRST: register your provider keys (BYOK)</strong></div>
            <p style="font-size:13px;color:#d8d8dc;line-height:1.6;margin:0 0 10px;">
              Quorum is BYOK — you bring keys for Claude/GPT/Gemini/etc. and Quorum
              orchestrates the consensus across them. Your providers, your bills.
              Quorum charges only the £49/mo for orchestration + audit + dashboard.
            </p>
            <pre style="margin:0;padding:12px;background:#050507;border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,monospace;font-size:11px;color:#caac7d;overflow-x:auto;">curl -X POST https://api.quorum-ai.dev/v1/customer/keys \\
  -H "X-Quorum-API-Key: {api_key[:14]}..." \\
  -H "Content-Type: application/json" \\
  -d '{{"anthropic":"sk-ant-...","openai":"sk-...","gemini":"..."}}'</pre>
            <p style="font-size:12px;color:#9a9aa0;margin:8px 0 0;">
              Supported: anthropic, openai, gemini, nvidia, mistral, cohere, grok,
              dashscope, replicate, deepseek, zhipu, moonshot. Add as many as you
              want — providers without keys are simply excluded from your pool.
              Encrypted at rest (Fernet, server-side KEK).
            </p>
          </div>

          <div style="margin-bottom:18px;">
            <div style="font-size:14px;color:#e8e8ea;margin-bottom:6px;"><strong>1. curl (after keys are registered)</strong></div>
            <pre style="margin:0;padding:14px;background:#050507;border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,monospace;font-size:12px;color:#caac7d;overflow-x:auto;">curl -X POST https://api.quorum-ai.dev/v1/consensus \\
  -H "X-Quorum-API-Key: {api_key[:14]}..." \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt":"Should I use sqlite or postgres for 100 paying users?"}}'</pre>
          </div>

          <div style="margin-bottom:18px;">
            <div style="font-size:14px;color:#e8e8ea;margin-bottom:6px;"><strong>2. VS Code extension</strong></div>
            <p style="font-size:13px;color:#b8b8be;line-height:1.6;margin:0;">Install <a href="https://marketplace.visualstudio.com/items?itemName=sovereignchain.quorum-vscode" style="color:#7fb8e8;">sovereignchain.quorum-vscode</a>, open Settings → Quorum, paste the key in <code style="background:#1a1a1f;padding:2px 6px;border-radius:3px;color:#caac7d;">quorum.apiKey</code>.</p>
          </div>

          <div>
            <div style="font-size:14px;color:#e8e8ea;margin-bottom:6px;"><strong>3. Python</strong></div>
            <pre style="margin:0;padding:14px;background:#050507;border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,monospace;font-size:12px;color:#caac7d;overflow-x:auto;">pip install quorum-client  # coming soon — for now use httpx
import httpx
r = httpx.post(
    "https://api.quorum-ai.dev/v1/consensus",
    headers={{"X-Quorum-API-Key": "{api_key[:14]}..."}},
    json={{"prompt": "..."}}, timeout=60,
)
print(r.json()["answer"])</pre>
          </div>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="font-size:13px;color:#8b8b90;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">{tier} tier includes</div>
          <ul style="font-size:14px;line-height:1.8;color:#d8d8dc;padding-left:20px;margin:0;">
            <li>5,000 consensus queries / month</li>
            <li>14+ LLM pool: Anthropic, OpenAI, Gemini, Mistral, Cohere, Grok, Qwen, NVIDIA, local Llama</li>
            <li>BYOK — bring your own backend keys if you prefer</li>
            <li>EU AI Act Article 12/13 audit certificates per query (SHA-256 hash-chained PDF)</li>
            <li>Disagreement matrix exposed — see exactly where the models split</li>
          </ul>
        </td></tr>

        <tr><td style="padding:18px 36px 28px;border-top:1px solid #1f1f24;">
          <p style="font-size:12px;color:#7a7a80;margin:0;line-height:1.6;">
            Reply to this email for support, billing changes, or to request a higher tier.<br>
            Manage your subscription: <a href="https://billing.stripe.com/p/login" style="color:#7fb8e8;">Stripe customer portal</a><br>
            Open-source self-host (free, Apache 2.0): <a href="https://github.com/jaquelinejaque/sovereignchain" style="color:#7fb8e8;">github.com/jaquelinejaque/sovereignchain</a>
          </p>
        </td></tr>

      </table>
      <div style="font-size:11px;color:#5a5a60;margin-top:18px;">Sovereign Chain Ltd · UK · PCT/US26/11908</div>
    </td></tr>
  </table>
</body>
</html>"""


async def send_welcome_email(*, to: str, api_key: str, tier: str = "Pro") -> bool:
    """Convenience wrapper for the welcome-with-key flow. Returns True if sent."""
    subject = f"Welcome to Quorum {tier} — your API key inside"
    html = welcome_html(api_key=api_key, tier=tier, email=to)
    result = await send_email(to=to, subject=subject, html=html)
    return result is not None


def free_welcome_html(api_key: str, email: str = "") -> str:
    """Welcome email for the self-service Free tier (100 queries/month, BYOK).

    Different copy than the paid welcome: emphasises that the customer
    brings their own provider keys (we just orchestrate), that quorum
    works better with MORE keys registered, and that Pro unlocks 5,000
    queries plus the dashboard / audit cert features when they're ready.
    """
    safe_email = email or "there"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Welcome to Quorum Free</title>
</head>
<body style="margin:0;padding:0;background:#0a0a0c;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;color:#e8e8ea;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0c;padding:48px 24px;">
    <tr><td align="center">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;background:#111114;border:1px solid #1f1f24;border-radius:10px;">

        <tr><td style="padding:36px 36px 12px;text-align:center;">
          <div style="font-family:'Cormorant Garamond',Georgia,serif;font-size:28px;color:#7fb8e8;letter-spacing:3px;">QUORUM</div>
          <div style="color:#8b8b90;font-size:11px;letter-spacing:3px;text-transform:uppercase;margin-top:6px;">Multi-LLM Consensus Engine · Free tier</div>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <p style="font-size:15px;line-height:1.6;color:#d8d8dc;">Welcome, {safe_email}. You've got <strong>100 free consensus queries per month</strong> to see how multi-LLM consensus actually feels on your own work — no card on file.</p>
          <p style="font-size:15px;line-height:1.6;color:#d8d8dc;">Your API key — <strong>copy it now, we don't store it in plaintext and cannot show it again</strong>:</p>

          <div style="margin:20px 0;padding:18px 20px;background:#050507;border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,Consolas,monospace;font-size:14px;color:#7fb8e8;word-break:break-all;">{api_key}</div>

          <p style="font-size:13px;color:#9a9aa0;margin-top:8px;">If you lose it, reply to this email and I'll issue a replacement and revoke this one.</p>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="margin-bottom:18px;padding:16px;background:#1a1107;border:1px solid #caac7d;border-radius:6px;">
            <div style="font-size:14px;color:#caac7d;margin-bottom:8px;"><strong>STEP 1 — register your provider keys (BYOK)</strong></div>
            <p style="font-size:13px;color:#d8d8dc;line-height:1.6;margin:0 0 10px;">
              Quorum is BYOK: you bring keys for Claude / GPT / Gemini / Mistral / etc. and we orchestrate the consensus across them. <strong>The more providers you register, the richer the consensus</strong> — a single key still works, but real divergence (and the "aha, the models actually disagree on this") only shows up with 3+.
            </p>
            <pre style="margin:0;padding:12px;background:#050507;border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,monospace;font-size:11px;color:#caac7d;overflow-x:auto;">curl -X POST https://api.quorum-ai.dev/v1/customer/keys \\
  -H "X-Quorum-API-Key: {api_key[:14]}..." \\
  -H "Content-Type: application/json" \\
  -d '{{"anthropic":"sk-ant-...","openai":"sk-...","gemini":"...","mistral":"..."}}'</pre>
            <p style="font-size:12px;color:#9a9aa0;margin:8px 0 0;">
              Supported: anthropic, openai, gemini, nvidia, mistral, cohere, grok, dashscope, replicate, deepseek, zhipu, moonshot. Each key is encrypted at rest (Fernet + server-side KEK). You pay your providers directly — Quorum charges only for orchestration. NVIDIA AI Foundation has a free tier with 6 OSS models if you want to start at zero cost.
            </p>
          </div>

          <div style="margin-bottom:18px;">
            <div style="font-size:14px;color:#e8e8ea;margin-bottom:6px;"><strong>STEP 2 — run a consensus query</strong></div>
            <pre style="margin:0;padding:14px;background:#050507;border:1px solid #2a2a30;border-radius:6px;font-family:'SF Mono',Menlo,monospace;font-size:12px;color:#caac7d;overflow-x:auto;">curl -X POST https://api.quorum-ai.dev/v1/consensus \\
  -H "X-Quorum-API-Key: {api_key[:14]}..." \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt":"Should I use sqlite or postgres for 100 paying users?"}}'</pre>
          </div>

          <div>
            <div style="font-size:14px;color:#e8e8ea;margin-bottom:6px;"><strong>STEP 3 — see the disagreement</strong></div>
            <p style="font-size:13px;color:#b8b8be;line-height:1.6;margin:0;">Check <code style="background:#1a1a1f;padding:2px 6px;border-radius:3px;color:#caac7d;">/v1/usage</code> to see remaining queries. The response payload shows every model's answer + which ones agreed/disagreed — that divergence is the signal you're paying for, not the headline answer.</p>
          </div>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="font-size:13px;color:#8b8b90;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">Free tier limits</div>
          <ul style="font-size:14px;line-height:1.7;color:#d8d8dc;padding-left:20px;margin:0;">
            <li><strong>100 consensus queries / month</strong> (resets 1st of each month)</li>
            <li>BYOK only — you bring your provider keys, we orchestrate</li>
            <li>Up to 12 providers in your pool (one of each: anthropic / openai / gemini / nvidia / mistral / cohere / grok / qwen / replicate / deepseek / zhipu / moonshot)</li>
            <li>No audit certificate, no dashboard, no SSO — those are Pro</li>
          </ul>
        </td></tr>

        <tr><td style="padding:0 36px 24px;">
          <div style="background:#0a1820;border:1px solid #1f3a4a;border-radius:6px;padding:18px;">
            <div style="font-size:13px;color:#7fe8c8;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Bonus — your queries help train the engine</div>
            <p style="font-size:13px;line-height:1.6;color:#d8d8dc;margin:0;">
              Every consensus query you run feeds Quorum's evolution loops — the memory recall, the MoE router, the model-vs-model ELO rater, the prompt rewriter. The more solo devs use the free tier, the smarter the consensus engine becomes for everyone. You're not just trying a product — you're part of how it learns.
            </p>
          </div>
        </td></tr>

        <tr><td style="padding:0 36px 28px;">
          <div style="background:#0f1419;border:1px solid #1f2937;border-radius:6px;padding:18px;">
            <div style="font-size:13px;color:#7fb8e8;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">For companies — Pro tier</div>
            <p style="font-size:14px;line-height:1.6;color:#d8d8dc;margin:0;">
              Need EU AI Act audit certificates per query + the disagreement-matrix dashboard + 5,000 queries/month + SLA? <strong>Pro is £49/month</strong>, designed for teams shipping AI in regulated industries (legal, fintech, health). Reply to this email or visit <a href="https://quorum-ai.dev#pricing" style="color:#7fb8e8;">quorum-ai.dev#pricing</a>.
            </p>
          </div>
        </td></tr>

        <tr><td style="padding:18px 36px 28px;border-top:1px solid #1f1f24;">
          <p style="font-size:12px;color:#7a7a80;margin:0;line-height:1.6;">
            Reply to this email for support, billing, or to suggest a provider we haven't added yet.<br>
            Open-source self-host (free, Apache 2.0): <a href="https://github.com/jaquelinejaque/sovereignchain" style="color:#7fb8e8;">github.com/jaquelinejaque/sovereignchain</a>
          </p>
        </td></tr>

      </table>
      <div style="font-size:11px;color:#5a5a60;margin-top:18px;">Sovereign Chain Ltd · UK · PCT/US26/11908</div>
    </td></tr>
  </table>
</body>
</html>"""


async def send_free_welcome_email(*, to: str, api_key: str) -> bool:
    """Send the Free-tier welcome email."""
    subject = "Welcome to Quorum — your free API key inside"
    html = free_welcome_html(api_key=api_key, email=to)
    result = await send_email(to=to, subject=subject, html=html)
    return result is not None
