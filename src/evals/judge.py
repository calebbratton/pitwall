"""DeepEval judge backed by the project's "judge" role (Qwen on Groq by default).

A different model family from the analyst it grades (gpt-oss), which reduces self-preference
bias. Goes through src/llm so the provider rules hold: Groq by default, Anthropic only if
LLM_PROVIDER=anthropic is set explicitly.
"""

import os
import time

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from src.llm.factory import get_chat_model, with_schema

RATE_LIMIT_WAIT_S = 30
RETRIES = 4


def _rate_limited(error: Exception) -> bool:
    text = str(error).lower()
    return "rate limit" in text or "429" in text or "rate_limit" in text


def call_with_backoff(fn, *args):
    """Groq's free tier allows ~8K tokens/minute per model: wait and retry on 429s."""
    for attempt in range(RETRIES):
        try:
            return fn(*args)
        except Exception as e:
            if not _rate_limited(e) or attempt == RETRIES - 1:
                raise
            time.sleep(RATE_LIMIT_WAIT_S * (attempt + 1))
    raise AssertionError("unreachable")


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        # Qwen on Groq's free tier allows 1000 output tokens a minute; reasoning would
        # spend them all on one verdict.
        self._chat = get_chat_model("judge", reasoning=False)
        super().__init__(getattr(self._chat, "model_name", "judge"))

    def load_model(self):
        return self._chat

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is not None:
            try:
                return call_with_backoff(with_schema(self._chat, schema).invoke, prompt)
            except Exception as e:
                if _rate_limited(e):
                    raise
                # Schema mode rejected (e.g. an unsupported JSON-schema feature): fall back to
                # plain text, which DeepEval parses as JSON itself.
        return call_with_backoff(
            self._chat.invoke, prompt + "\n\nReply with the JSON object only, no prose."
        ).content

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.name
