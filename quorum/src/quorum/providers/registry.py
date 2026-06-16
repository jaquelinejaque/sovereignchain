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
        from quorum.providers.anthropic import AnthropicProvider
        providers.append(AnthropicProvider())

    if os.getenv("OPENAI_API_KEY"):
        from quorum.providers.openai import OpenAIProvider
        providers.append(OpenAIProvider())

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
        providers.append(gk.grok_4_mini())

    # Always try local Ollama (free, runs on user's Mac) — best effort
    try:
        from quorum.providers.ollama import OllamaProvider
        providers.append(OllamaProvider())
    except Exception:  # noqa: BLE001
        pass

    return providers
