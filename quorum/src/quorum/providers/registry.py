"""Auto-discover available providers based on configured API keys.

Two execution paths:

* **Self-host / operator mode** (``customer_keys`` is None): provider
  selection is driven by the OS env vars. This is what the open-source
  CLI does and what test/dev scripts hit. A provider is included iff
  its env var is set.

* **Hosted BYOK mode** (``customer_keys`` is a dict): provider selection
  is driven by what the *customer* has registered via
  ``POST /v1/customer/keys``. The operator's own env keys are NOT used
  as a fallback — a provider the customer hasn't registered is simply
  excluded from their consensus pool. This is what makes the hosted
  £49/mo Pro tier sustainable: the customer pays Anthropic / OpenAI /
  Gemini directly, the operator pockets the £49 as pure orchestration
  margin. Marketing has always said "BYOK"; this is where it becomes
  true at the API layer.
"""

from __future__ import annotations

import os
from typing import Optional

from quorum.providers.base import Provider


def load_default_providers(
    customer_keys: Optional[dict[str, str]] = None,
) -> list[Provider]:
    """Return the providers configured for this call.

    When ``customer_keys`` is provided, it is the SOLE source of API
    keys — operator env keys are ignored. Provider modules accept
    ``api_key=`` overrides on their factories; passing the customer key
    there means no env-fallback happens at provider construction time
    either, so a typo or missing key surfaces as a clean per-provider
    "no_api_key" error rather than a stealth charge on the operator.

    When ``customer_keys`` is None, fall back to the historical
    env-var-driven behaviour for CLI / self-host / tests.
    """
    providers: list[Provider] = []
    byok = customer_keys is not None
    ck = customer_keys or {}

    def _key(*names: str) -> Optional[str]:
        """Resolve a key from customer_keys (preferred) or env (operator mode).

        ``names`` is the list of acceptable aliases (e.g. for Gemini we
        accept both ``gemini`` and ``google_ai_studio``). The first one
        with a value wins. In BYOK mode env is NEVER consulted.
        """
        for n in names:
            v = ck.get(n)
            if v:
                return v
        if byok:
            return None
        for n in names:
            v = os.environ.get(n.upper() + "_API_KEY") or os.environ.get(n.upper() + "_API_TOKEN")
            if v:
                return v
        return None

    # Anthropic — Claude
    k = _key("anthropic")
    if k:
        from quorum.providers import anthropic as an
        providers.append(an.claude_sonnet() if not byok else an.AnthropicProvider(model="claude-sonnet-4-6", api_key=k))
        providers.append(an.claude_opus() if not byok else an.AnthropicProvider(model="claude-opus-4-8", api_key=k))
        providers.append(an.claude_haiku() if not byok else an.AnthropicProvider(model="claude-haiku-4-5", api_key=k))

    # OpenAI — GPT
    k = _key("openai")
    if k:
        from quorum.providers import openai as oa
        providers.append(oa.gpt_4_1() if not byok else oa.OpenAIProvider(model="gpt-4.1", api_key=k))
        providers.append(oa.gpt_4o_mini() if not byok else oa.OpenAIProvider(model="gpt-4o-mini", api_key=k))

    # Google Gemini
    k = _key("gemini", "google_ai_studio")
    if k:
        from quorum.providers.gemini import GeminiProvider
        providers.append(GeminiProvider(api_key=k))

    # Replicate (Llama, Qwen, DeepSeek, Hermes via Replicate)
    k = _key("replicate")
    if k:
        from quorum.providers import replicate as r
        providers.append(r.llama_3_3() if not byok else r.ReplicateProvider(model_slug="meta/llama-3.3-70b-instruct", api_token=k))
        providers.append(r.deepseek_v3() if not byok else r.ReplicateProvider(model_slug="deepseek-ai/deepseek-v3", api_token=k))
        providers.append(r.hermes_3_70b() if not byok else r.ReplicateProvider(model_slug="nousresearch/hermes-3-llama-3.1-70b", name="hermes-3-llama-3.1-70b", api_token=k))
        providers.append(r.hermes_3_405b() if not byok else r.ReplicateProvider(model_slug="nousresearch/hermes-3-llama-3.1-405b", name="hermes-3-llama-3.1-405b", api_token=k))

    # NVIDIA AI Foundation — multiple OSS models on one key
    k = _key("nvidia")
    if k:
        from quorum.providers import nvidia as nv
        providers.append(nv.llama_3_2_3b_nvidia() if not byok else nv.NvidiaProvider(model="meta/llama-3.2-3b-instruct", api_key=k))
        providers.append(nv.llama_3_1_8b_nvidia() if not byok else nv.NvidiaProvider(model="meta/llama-3.1-8b-instruct", api_key=k))
        providers.append(nv.llama_4_maverick_nvidia() if not byok else nv.NvidiaProvider(model="meta/llama-4-maverick-17b-128e-instruct", api_key=k))
        providers.append(nv.deepseek_v4_nvidia() if not byok else nv.NvidiaProvider(model="deepseek-ai/deepseek-v4-flash", api_key=k))
        providers.append(nv.dracarys_70b_nvidia() if not byok else nv.NvidiaProvider(model="abacusai/dracarys-llama-3.1-70b-instruct", api_key=k))
        providers.append(nv.llama_3_3_nvidia() if not byok else nv.NvidiaProvider(model="meta/llama-3.3-70b-instruct", api_key=k))

    # DeepSeek direct
    k = _key("deepseek")
    if k:
        from quorum.providers import deepseek as ds
        providers.append(ds.deepseek_chat() if not byok else ds.DeepSeekProvider(model="deepseek-chat", api_key=k))
        providers.append(ds.deepseek_reasoner() if not byok else ds.DeepSeekProvider(model="deepseek-reasoner", api_key=k))

    # Mistral
    k = _key("mistral")
    if k:
        from quorum.providers import mistral as ms
        providers.append(ms.MistralProvider(model="mistral-large-latest", api_key=k))
        providers.append(ms.MistralProvider(model="codestral-latest", api_key=k))
        providers.append(ms.MistralProvider(model="mistral-small-latest", api_key=k))

    # Cohere — uses dated model names that survive deprecations
    k = _key("cohere")
    if k:
        from quorum.providers import cohere as co
        providers.append(co.CohereProvider(model="command-r-plus-08-2024", api_key=k))
        providers.append(co.CohereProvider(model="command-r-08-2024", api_key=k))
        providers.append(co.CohereProvider(model="command-a-03-2025", api_key=k))

    # xAI Grok
    k = _key("grok", "xai")
    if k:
        from quorum.providers import grok as gk
        providers.append(gk.grok_4() if not byok else gk.GrokProvider(model="grok-4", api_key=k))
        providers.append(gk.grok_4_20_chat() if not byok else gk.GrokProvider(model="grok-4-20-chat", api_key=k))

    # ---- Chinese frontier pool ----

    k = _key("zhipu", "glm")
    if k:
        from quorum.providers import zhipu as zp
        providers.append(zp.ZhipuProvider(model="glm-5.2", api_key=k))
        providers.append(zp.ZhipuProvider(model="glm-5.2-air", api_key=k))
        providers.append(zp.ZhipuProvider(model="glm-4.6", api_key=k))

    k = _key("moonshot")
    if k:
        from quorum.providers import moonshot as mn
        providers.append(mn.MoonshotProvider(model="kimi-k2.6", api_key=k))
        providers.append(mn.MoonshotProvider(model="kimi-k2-turbo", api_key=k))

    k = _key("dashscope", "qwen")
    if k:
        from quorum.providers import qwen as qw
        providers.append(qw.QwenProvider(model="qwen3.7-max", api_key=k))
        providers.append(qw.QwenProvider(model="qwen3-max", api_key=k))
        providers.append(qw.QwenProvider(model="qwen3-coder-plus", api_key=k))
        providers.append(qw.QwenProvider(model="qwen-plus", api_key=k))

    # Local Ollama is always tried in self-host mode (no key) but skipped in
    # hosted BYOK because the hosted server can't reach the customer's local
    # machine. Self-host customers keep this for free.
    if not byok:
        try:
            from quorum.providers.ollama import OllamaProvider
            providers.append(OllamaProvider())  # default llama3.2
            # Hermes 3 (Nous Research) — same family, low-RLHF, good for audit/agentic.
            # Silently skipped if not pulled (provider returns ollama_unreachable error).
            providers.append(OllamaProvider(model="hermes3:8b"))
        except Exception:  # noqa: BLE001
            pass

        # Claude Code CLI — uses the local user's Claude Pro/Max subscription
        # instead of the billed Anthropic API. Skipped silently if `claude`
        # isn't on PATH (the provider also self-checks and returns a clean
        # error so it doesn't break consensus). Hosted mode (byok=True) never
        # reaches this branch — Cloud Run has no Claude CLI installed.
        try:
            from quorum.providers.claude_cli import ClaudeCLIProvider
            providers.append(ClaudeCLIProvider())
        except Exception:  # noqa: BLE001
            pass

    return providers
