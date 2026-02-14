from __future__ import annotations
from typing import Optional

from orchestrator.settings import Settings

def get_llm(settings: Settings, *, model: Optional[str] = None, temperature: float = 0.2):
    # We use Groq in your stack (same as project 1).
    # If you want OpenAI later, you can add a get_openai_llm here.
    from langchain_groq import ChatGroq

    m = model or settings.llm_model
    if not settings.groq_api_key:
        raise ValueError("Missing GROQ_API_KEY. Set it in your environment or .env.")
    return ChatGroq(groq_api_key=settings.groq_api_key, model=m, temperature=temperature)
