"""Chat-model factory. The only module that imports provider packages."""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

load_dotenv()

Role = Literal["router", "fetcher", "analyst", "judge"]

# Groq free-tier rate limits are per model, so giving each role its own model multiplies the
# budget. The judge is a different model family from the analyst it grades, which reduces
# self-preference bias. Check https://console.groq.com/docs/models — Groq rotates models often.
GROQ_DEFAULTS: dict[Role, str] = {
    "router": "openai/gpt-oss-20b",
    "fetcher": "qwen/qwen3.8-27b",  # makes parallel tool calls; gpt-oss-20b calls one at a time
    "analyst": "openai/gpt-oss-120b",
    "judge": "qwen/qwen3.8-27b",
}
DEFAULT_MAX_TOKENS = 4096  # reasoning tokens count toward this on reasoning models
# Free-tier output-tokens-per-minute caps: Groq rejects (not queues) any request whose
# max_tokens exceeds them, so requests are clamped under the cap.
GROQ_OUTPUT_TPM = {"qwen/qwen3.8-27b": 1000}


class LLMConfigError(RuntimeError):
    pass


def _env(name: str) -> str:
    # `.env` lines like `LLM_PROVIDER=` load as empty strings; treat those as unset.
    return os.getenv(name, "").strip()


def get_chat_model(
    role: Role = "analyst",
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning: bool = True,
) -> BaseChatModel:
    """Model for a graph role. Override per role with LLM_MODEL_<ROLE>, or all roles with
    LLM_MODEL. `reasoning=False` turns off hidden reasoning where the model supports it (saves
    output tokens, which are the scarce budget on some free-tier models)."""
    provider = (_env("LLM_PROVIDER") or "groq").lower()
    model = _env(f"LLM_MODEL_{role.upper()}") or _env("LLM_MODEL")

    if provider == "groq":
        if not _env("GROQ_API_KEY"):
            raise LLMConfigError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com/keys."
            )
        from langchain_groq import ChatGroq

        model = model or GROQ_DEFAULTS[role]
        extra = {}
        if model.startswith("qwen/"):
            # Keep <think> text out of message content; "raw" is rejected with tool use.
            extra["reasoning_format"] = "parsed"
            if not reasoning:
                extra["reasoning_effort"] = "none"
        if model in GROQ_OUTPUT_TPM:
            max_tokens = min(max_tokens, GROQ_OUTPUT_TPM[model] - 50)
        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens, **extra)

    if provider == "anthropic":
        # Paid provider: explicit opt-in only. No default model, no fallback.
        if not _env("ANTHROPIC_API_KEY"):
            raise LLMConfigError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        if not model:
            raise LLMConfigError("LLM_PROVIDER=anthropic requires LLM_MODEL to be set explicitly.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, max_tokens=max_tokens)

    raise LLMConfigError(f"Unknown LLM_PROVIDER {provider!r}; expected 'groq' or 'anthropic'.")


def with_schema(model: BaseChatModel, schema: type[BaseModel]) -> Runnable:
    """Structured output using the most reliable method for the model's provider."""
    from langchain_groq import ChatGroq

    if isinstance(model, ChatGroq):
        # Groq's tool-call route fails hard when the model answers in prose instead
        # ("Tool choice is required, but model did not call a tool"); JSON-schema mode
        # constrains the output itself.
        return model.with_structured_output(schema, method="json_schema")
    return model.with_structured_output(schema)
