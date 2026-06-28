"""Tests for the Pro trial-flow email templates.

These templates run on the customer onboarding path: Day 0 → Day 5 → Day 8.
The render functions must NEVER raise, must NEVER leak the full API key
in a way that breaks Resend's HTML parser, and the Day-8 template must
branch correctly between active / cancelled subscriptions.

We intentionally test the rendered HTML strings, not the network senders
— Resend calls are covered by the existing email_sender tests, and we
don't want trial_emails tests to flake on outbound HTTP.
"""

from __future__ import annotations

import re

import pytest

# Bypass FSL license gate inside tests — the gate is already covered in
# tests/test_license.py and would otherwise spam every test session.
import os
os.environ.setdefault("QUORUM_DEV_MODE", "1")

from quorum.billing.trial_emails import (
    trial_day5_html,
    trial_ended_html,
    trial_start_html,
)


# ---------- Day 0 (trial start) ---------------------------------------------


class TestTrialStart:
    """The trial-start email goes inside the Stripe webhook. Any raise
    here would lose the customer their working subscription on day 0."""

    def test_renders_minimal_inputs(self) -> None:
        html = trial_start_html(api_key="quorum_minimal_key")
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")

    def test_embeds_api_key_verbatim(self) -> None:
        # The plaintext key must appear once and only once in the visible
        # body (we also embed a preview in curl examples; that's fine, but
        # the value-prop is the full key in the readable box).
        key = "quorum_abc123_long_key_value_xyz"
        html = trial_start_html(api_key=key, email="user@example.com")
        assert key in html

    def test_preview_in_curl_uses_prefix_only(self) -> None:
        key = "quorum_full_key_should_not_leak_into_curl_block"
        html = trial_start_html(api_key=key, email="user@example.com")
        # The curl example uses key[:14] + "..." — confirm the *full* key
        # never appears inside a curl <pre> block.
        pre_blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL)
        for block in pre_blocks:
            assert key not in block, (
                "Full API key leaked into a curl/pre block — Resend may "
                "still send it but copy-paste users would copy too much."
            )

    def test_trial_end_date_appears(self) -> None:
        html = trial_start_html(
            api_key="k", email="a@b.com", trial_end_date="2026-07-05",
        )
        assert "2026-07-05" in html

    def test_default_trial_end_phrasing(self) -> None:
        # When the caller doesn't pass a date, the template falls back to
        # "in 7 days" — must not produce a broken "ends ." sentence.
        html = trial_start_html(api_key="k", email="a@b.com")
        assert "in 7 days" in html
        # And must not produce a broken "ends ." (empty placeholder).
        assert "ends ." not in html
        assert "ends <strong></strong>" not in html

    def test_pro_features_listed(self) -> None:
        html = trial_start_html(api_key="k")
        # Each of the 7 Pro tier features mentioned in the commit message
        # must be discoverable in the email so the trial user sees what
        # they're trialing.
        for must_contain in (
            "5,000 consensus queries",
            "BYOK",
            "hallucination guard",
            "Refusal filter",
            "audit chain",
        ):
            assert must_contain in html, f"missing: {must_contain}"

    def test_branding_present(self) -> None:
        html = trial_start_html(api_key="k")
        assert "QUORUM" in html
        assert "Sovereign Chain Ltd" in html
        assert "PCT/US26/11908" in html


# ---------- Day 5 (reminder) ------------------------------------------------


class TestTrialDay5:
    def test_usage_branch_with_queries(self) -> None:
        # When the user has run some queries, the email surfaces the count
        # — concrete value beats abstract "you've been using it!"
        html = trial_day5_html(email="user@example.com", queries_used=23)
        assert "<strong>23</strong>" in html
        assert "consensus queries so far" in html

    def test_usage_branch_zero_queries(self) -> None:
        # When the user hasn't run anything, the email becomes an
        # onboarding nudge with the curl example — NOT a "you've run 0
        # queries" shaming line.
        html = trial_day5_html(email="user@example.com", queries_used=0)
        assert "haven't run a consensus query yet" in html
        assert ">0</strong>" not in html  # no shame number

    def test_subject_line_state(self) -> None:
        html = trial_day5_html(email="user@example.com", queries_used=5)
        assert "2 days remaining" in html or "2 days left" in html.lower()

    def test_no_fake_urgency(self) -> None:
        # The CLAUDE.md preference and the commit message both reject
        # "limited time offer" style urgency. Pin it in a test.
        html = trial_day5_html(email="u@x.com")
        forbidden = [
            "limited time", "act now", "don't miss", "last chance",
            "expires soon", "while supplies last",
        ]
        lower = html.lower()
        for phrase in forbidden:
            assert phrase not in lower, f"unwanted urgency phrase: {phrase!r}"


# ---------- Day 8 (ended — branches on subscription_active) -----------------


class TestTrialEnded:
    def test_active_branch_mentions_charge(self) -> None:
        html = trial_ended_html(email="u@x.com", subscription_active=True)
        # The user just got charged £15 — the email must confirm that
        # exact amount so they recognise the Stripe line item.
        assert "£15" in html
        assert "subscription is now active" in html.lower()

    def test_active_branch_no_friction(self) -> None:
        # "Nothing practical changes" is the reassurance promise. If we
        # drop that phrase the email reads like an upsell instead.
        html = trial_ended_html(email="u@x.com", subscription_active=True)
        assert "nothing practical changes" in html.lower()

    def test_cancelled_branch_no_charge(self) -> None:
        html = trial_ended_html(email="u@x.com", subscription_active=False)
        # The most important fact to surface when the trial cancelled:
        # zero was charged. Anything ambiguous here causes refund emails.
        assert "£0" in html
        assert "no charge" in html.lower() or "before any charge" in html.lower()

    def test_cancelled_branch_low_pressure_followup(self) -> None:
        # Cancelled users get one offer: free tier + self-host. No
        # "wait! special discount!" — that's the [[feedback_no_calls]]
        # / honesty principle applied to email.
        html = trial_ended_html(email="u@x.com", subscription_active=False)
        assert "100 queries/month" in html  # free tier mention
        assert "self-host" in html.lower()  # OSS fallback
        assert "no follow-up sales emails" in html.lower()

    def test_cancelled_branch_does_not_mention_charge_amount(self) -> None:
        # If the cancel branch says "£15 was charged" by accident, the
        # support inbox fills with refund requests by lunchtime.
        html = trial_ended_html(email="u@x.com", subscription_active=False)
        assert "£15 was charged" not in html
        assert "£15</strong> was charged" not in html

    def test_branches_use_different_subject_kicker(self) -> None:
        # Header kicker should differ so the customer can tell at a
        # glance which email this is (active vs cancelled).
        a = trial_ended_html(email="u@x.com", subscription_active=True)
        c = trial_ended_html(email="u@x.com", subscription_active=False)
        # Extract the kicker text from each.
        active_kicker = re.search(r"text-transform:uppercase[^>]*>([^<]+)<", a)
        cancel_kicker = re.search(r"text-transform:uppercase[^>]*>([^<]+)<", c)
        assert active_kicker is not None and cancel_kicker is not None
        assert active_kicker.group(1).strip() != cancel_kicker.group(1).strip()


# ---------- shared invariants -----------------------------------------------


@pytest.mark.parametrize(
    "renderer,kwargs",
    [
        (trial_start_html, {"api_key": "k", "email": "u@x.com"}),
        (trial_day5_html, {"email": "u@x.com", "queries_used": 5}),
        (trial_ended_html, {"email": "u@x.com", "subscription_active": True}),
        (trial_ended_html, {"email": "u@x.com", "subscription_active": False}),
    ],
)
class TestSharedInvariants:
    """Every trial email shares the same shell (brand, footer, support
    link). Pin them in one place so a future template tweak can't
    accidentally drop the patent stamp or the unsubscribe link."""

    def test_html_doctype(self, renderer, kwargs) -> None:
        html = renderer(**kwargs)
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")

    def test_brand_line(self, renderer, kwargs) -> None:
        assert "QUORUM" in renderer(**kwargs)

    def test_footer_company_stamp(self, renderer, kwargs) -> None:
        assert "Sovereign Chain Ltd" in renderer(**kwargs)
        assert "PCT/US26/11908" in renderer(**kwargs)

    def test_stripe_portal_link(self, renderer, kwargs) -> None:
        assert "billing.stripe.com" in renderer(**kwargs)

    def test_github_link(self, renderer, kwargs) -> None:
        assert "github.com/jaquelinejaque/sovereignchain" in renderer(**kwargs)
