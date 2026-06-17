"""Auto-discover available providers based on environment variables."""

from __future__ import annotations

import os

from quorum.providers.base import Provider


def load_default_providers() -> list[Provider]:
    """Return all providers whose API keys are configured.

    Falls back gracefully if a provider's key is missing.
    """
    providers: list[Provider] = []

    # Paid — only if key is set
    if os.getenv("ANTHROPIC_API_KEY"):
        from quorum.providers import anthropic as an
        providers.append(an.claude_sonnet())
        providers.append(an.claude_opus())
        providers.append(an.claude_haiku())

    if os.getenv("OPENAI_API_KEY"):
        from quorum.providers import openai as oa
        providers.append(oa.gpt_4_1())
        providers.append(oa.gpt_4o_mini())
        # gpt-5 family included but heavy — uncomment to enable
        # providers.append(oa.gpt_5())
        # providers.append(oa.gpt_5_mini())

    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_STUDIO_KEY"):
        from quorum.providers.gemini import GeminiProvider
        providers.append(GeminiProvider())

    # Open-source via Replicate — one key, many models
    if os.getenv("REPLICATE_API_TOKEN"):
        from quorum.providers import replicate as r
        providers.append(r.llama_3_3())
        providers.append(r.deepseek_v3())
        # Mistral, Qwen, Phi available — uncomment to add to default pool
        # providers.append(r.mistral_large())
        # providers.append(r.qwen_2_5())
        # providers.append(r.phi_4())

    # NVIDIA AI Foundation — one key, multiple frontier OSS models (free tier)
    if os.getenv("NVIDIA_API_KEY"):
        from quorum.providers import nvidia as nv
        providers.append(nv.llama_3_2_3b_nvidia())
        providers.append(nv.llama_3_1_8b_nvidia())
        providers.append(nv.llama_4_maverick_nvidia())
        providers.append(nv.deepseek_v4_nvidia())
        providers.append(nv.dracarys_70b_nvidia())
        providers.append(nv.llama_3_3_nvidia())

    if os.getenv("DEEPSEEK_API_KEY"):
        from quorum.providers import deepseek as ds
        providers.append(ds.deepseek_chat())
        providers.append(ds.deepseek_reasoner())

    if os.getenv("MISTRAL_API_KEY"):
        from quorum.providers import mistral as ms
        providers.append(ms.mistral_large())
        providers.append(ms.codestral())
        providers.append(ms.mistral_small())

    if os.getenv("COHERE_API_KEY"):
        from quorum.providers import cohere as co
        providers.append(co.command_r_plus())
        providers.append(co.command_r())
        providers.append(co.command_a())

    if os.getenv("XAI_API_KEY"):
        from quorum.providers import grok as gk
        providers.append(gk.grok_4())
        providers.append(gk.grok_4_20_chat())

    # ---- Chinese frontier pool (opt-in; see provider modules for data-residency notes) ----

    if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
        from quorum.providers import zhipu as zp
        providers.append(zp.glm_5_2())
        providers.append(zp.glm_5_2_air())
        providers.append(zp.glm_4_6())

    if os.getenv("MOONSHOT_API_KEY"):
        from quorum.providers import moonshot as mn
        providers.append(mn.kimi_k2_6())
        providers.append(mn.kimi_k2_turbo())

    if os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"):
        from quorum.providers import qwen as qw
        providers.append(qw.qwen3_7_max())  # 2026-06 release; PAI MaaS workspaces
        providers.append(qw.qwen3_max())
        providers.append(qw.qwen3_coder_plus())
        providers.append(qw.qwen_plus())

    # Always try local Ollama (free, runs on user's Mac) — best effort
    try:
        from quorum.providers.ollama import OllamaProvider
        providers.append(OllamaProvider())
    except Exception:  # noqa: BLE001
        pass

    return providers
