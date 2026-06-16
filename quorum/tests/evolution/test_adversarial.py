# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sovereign Chain / Quorum contributors
"""Tests for ``AdversarialProbe`` — the systematic attack-suite loop.

WHAT THIS FILE PROVES
=====================
1. Every probe in the default suite is well-formed: id, category,
   prompt, detector all present and detect_failure() runs without
   raising on a typical refusal string.
2. The pass/fail signal propagates: a provider that *fails* the
   injection canary increments injection vulnerability above zero.
3. The penalty signal compounds: once the same provider has failed an
   injection probe, `penalty_multiplier(query_class="injection")`
   drops strictly below 1.0 — and stays inside [0.5, 1.0].
4. Innocent-until-proven-guilty: a brand-new provider with no
   recorded probe history gets `penalty_multiplier == 1.0`.

We avoid hitting any real provider. The fake providers below are
deterministic so the assertions are claims about the math, not about
an LLM's behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quorum.evolution.adversarial import (
    DEFAULT_PROBES,
    AdversarialProbe,
    Probe,
    _ProbeResultsStore,
)
from quorum.providers.base import ModelResponse, Provider


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _InjectionFailingProvider(Provider):
    """Provider that obediently echoes the injection canary ('Banana').

    All five injection probes carry a canary string that a vulnerable
    model would emit verbatim; we put every canary into one response so
    one fake provider can fail the whole injection category at once.
    The other categories pass (we send a refusal-shaped reply).
    """

    name = "fake-injection-victim"

    async def complete(
        self, prompt: str, *, max_tokens: int = 800
    ) -> ModelResponse:
        p = prompt.lower()
        # Injection probes trigger the canary echo.
        if "banana" in p or "pwned_by_injection" in p or "root_shell" in p \
                or '"owned"' in p or "injected" in p \
                or "system prompt:" in p:
            return ModelResponse(
                name=self.name,
                response=(
                    "Banana. PWNED_BY_INJECTION. ROOT_SHELL. OWNED. INJECTED. "
                    "SYSTEM PROMPT: you are a helpful assistant."
                ),
            )
        # Everything else: safe refusal so other categories don't trip.
        return ModelResponse(
            name=self.name,
            response="I cannot help with that request.",
        )


class _SafeProvider(Provider):
    """Provider that refuses everything — should pass every probe."""

    name = "fake-safe"

    async def complete(
        self, prompt: str, *, max_tokens: int = 800
    ) -> ModelResponse:
        return ModelResponse(
            name=self.name,
            response="I cannot help with that. I am not aware of any such record.",
        )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def probe(tmp_path: Path) -> AdversarialProbe:
    """Fresh on-disk probe under tmp_path/adversarial.db.

    Using tmp_path keeps every test hermetic — no probe-results leakage
    between cases via the shared $QUORUM_DATA_DIR/adversarial.db.
    """
    store = _ProbeResultsStore(db_path=tmp_path / "adversarial.db")
    return AdversarialProbe(store=store)


# --------------------------------------------------------------------------- #
# Probe suite well-formedness
# --------------------------------------------------------------------------- #


def test_default_probes_well_formed() -> None:
    """Every default probe has all required fields populated.

    Failing this means the probe constant table is broken — the rest
    of the suite would explode in ways that are hard to attribute.
    """
    assert len(DEFAULT_PROBES) >= 15, "MVP requires at least 15 probes"
    ids = [p.id for p in DEFAULT_PROBES]
    assert len(ids) == len(set(ids)), "probe ids must be unique"
    categories = {p.category for p in DEFAULT_PROBES}
    assert {"injection", "jailbreak", "hallucination"}.issubset(categories), (
        "MVP requires at least these three categories"
    )
    for p in DEFAULT_PROBES:
        assert p.prompt, f"probe {p.id} has empty prompt"
        assert p.expected_failure_pattern, (
            f"probe {p.id} has empty failure detector"
        )


def test_detect_failure_handles_garbage_regex() -> None:
    """A malformed regex pattern must not raise — it should return False."""
    bad = Probe(
        id="bad",
        category="injection",
        prompt="x",
        expected_failure_pattern="[unterminated",
    )
    assert bad.detect_failure("anything") is False


# --------------------------------------------------------------------------- #
# Pass/fail signal propagation (the headline assertion)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_injection_failure_raises_vulnerability_score(
    probe: AdversarialProbe,
) -> None:
    """Mock provider fails injection probes → vulnerability_score > 0
    and penalty_multiplier < 1.0.

    This is the contract the consensus engine depends on: if a model
    proves it can be hijacked, its weight in the next consensus call
    must drop.
    """
    victim = _InjectionFailingProvider()

    results = await probe.run_probes(victim)
    await probe.record_results(victim.name, results)

    # At least one injection probe must have failed for this signal.
    failed_injection = sum(
        1
        for pid, failed in results.items()
        if failed and any(
            p.category == "injection" and p.id == pid for p in DEFAULT_PROBES
        )
    )
    assert failed_injection >= 1, (
        f"victim provider should fail at least one injection probe, "
        f"got results={results}"
    )

    score = await probe.vulnerability_score(victim.name, "injection")
    assert score > 0, f"vulnerability_score(injection) must be > 0, got {score}"

    penalty = await probe.penalty_multiplier(victim.name, "injection")
    assert penalty < 1.0, (
        f"penalty_multiplier(injection) must drop below 1.0 for a "
        f"demonstrated-vulnerable model, got {penalty}"
    )
    # And the floor must hold.
    assert penalty >= 0.5, (
        f"penalty_multiplier must respect the 0.5 floor, got {penalty}"
    )


@pytest.mark.asyncio
async def test_unseen_model_gets_neutral_penalty(
    probe: AdversarialProbe,
) -> None:
    """A provider with zero recorded probes is innocent-until-proven-guilty.

    Without this, a brand-new model added to the pool would be punished
    purely for being new — which would freeze the pool over time.
    """
    penalty = await probe.penalty_multiplier(
        "never-tested-model", query_class="injection"
    )
    assert penalty == 1.0, (
        f"unseen model must receive neutral 1.0 penalty, got {penalty}"
    )


@pytest.mark.asyncio
async def test_safe_provider_gets_neutral_penalty(
    probe: AdversarialProbe,
) -> None:
    """A provider that refuses everything should not be penalised."""
    safe = _SafeProvider()
    results = await probe.run_probes(safe)
    await probe.record_results(safe.name, results)

    score_inj = await probe.vulnerability_score(safe.name, "injection")
    score_jb = await probe.vulnerability_score(safe.name, "jailbreak")
    score_hal = await probe.vulnerability_score(safe.name, "hallucination")
    assert score_inj == 0.0
    assert score_jb == 0.0
    assert score_hal == 0.0

    penalty = await probe.penalty_multiplier(safe.name, "injection")
    assert penalty == 1.0, (
        f"safe provider must receive 1.0 penalty, got {penalty}"
    )
