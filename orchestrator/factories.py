from __future__ import annotations

import os
from typing import Optional

from orchestrator.settings import Settings

# Qwen3 bills thinking tokens as output tokens, which alone can exceed the
# per-model output-tokens-per-minute budget on Groq's free tier. "none" turns
# thinking off. Only qwen takes this param, so it is never sent to the
# gpt-oss/llama models.
QWEN_REASONING_EFFORT = os.getenv("QWEN_REASONING_EFFORT", "none")


def _reasoning_kwargs(model: str) -> dict:
    if "qwen" not in (model or "").lower():
        return {}
    return {"reasoning_effort": QWEN_REASONING_EFFORT}


def build_chat_groq(settings: Settings, *, model: Optional[str] = None, temperature: float = 0.2):
    """Single place every ChatGroq client is built, so the qwen thinking
    override applies to the router, the agents and the SQL agent alike."""
    from langchain_groq import ChatGroq

    m = model or settings.llm_model
    if not settings.groq_api_key:
        raise ValueError("Missing GROQ_API_KEY. Set it in your environment or .env.")

    extra = _reasoning_kwargs(m)
    base = dict(groq_api_key=settings.groq_api_key, model=m, temperature=temperature)
    try:
        return ChatGroq(**base, **extra)
    except (TypeError, ValueError):
        # Older langchain-groq has no reasoning_effort field; pass it straight
        # through to the API instead.
        return ChatGroq(**base, model_kwargs=extra)


def get_llm(settings: Settings, *, model: Optional[str] = None, temperature: float = 0.2):
    # We use Groq in your stack (same as project 1).
    # If you want OpenAI later, you can add a get_openai_llm here.
    return build_chat_groq(settings, model=model, temperature=temperature)
