"""Chat-model factory. The only module that imports provider packages."""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

# Check https://console.groq.com/docs/models — Groq rotates models often.
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"  # preview model; tool use + reasoning
DEFAULT_MAX_TOKENS = 2048


class LLMConfigError(RuntimeError):
    pass


def get_chat_model(temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS) -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise LLMConfigError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com/keys."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or DEFAULT_GROQ_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            # Keep <think> text out of message content; "raw" is rejected with tool use.
            reasoning_format="parsed",
        )

    if provider == "anthropic":
        # Paid provider: explicit opt-in only. No default model, no fallback.
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMConfigError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        if not model:
            raise LLMConfigError("LLM_PROVIDER=anthropic requires LLM_MODEL to be set explicitly.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, max_tokens=max_tokens)

    raise LLMConfigError(f"Unknown LLM_PROVIDER {provider!r}; expected 'groq' or 'anthropic'.")
