"""LLM phrasing for strategy alerts: the engine decides, the LLM only says it well.

Each alert (src/livetiming/alerts.py) goes out at once with its templated text; this adds a
one-line strategist-style version as an in-place update. The prompt carries only the alert's own
facts, and any line that introduces a number not in those facts is rejected (the template stays),
so the stream can't make things up. Off when PITWALL_ALERT_LLM=0 (tests) or without a Groq key.
"""

import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

PROMPT = """You are a Formula 1 race strategist talking to a fan watching the race.
Rewrite this pit-wall alert as ONE short sentence (at most 24 words), in plain race language.
Use only the facts given. Do not add numbers, names, laps, percentages or claims that aren't in
the facts. A call or a chance is a possibility, not something that has happened: never say a car
"is pitting" or "comes out ahead" unless the facts say it already did, and keep any percentage.
No preamble, no quotes.

Facts: {headline}. {detail}"""
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def enabled() -> bool:
    return os.getenv("PITWALL_ALERT_LLM", "1") != "0" and bool(
        os.getenv("GROQ_API_KEY", "").strip()
    )


def _faithful(text: str, alert: dict[str, Any]) -> bool:
    """Every number in the line must appear in the alert's own text."""
    source = f"{alert.get('headline', '')} {alert.get('detail', '')}"
    allowed = set(NUMBER.findall(source))
    if alert.get("p") is not None:
        allowed.add(str(round(alert["p"] * 100)))
    return all(n in allowed for n in NUMBER.findall(text))


def phrase(alert: dict[str, Any], model=None) -> str | None:
    """A strategist's one-liner for the alert, or None (template stays)."""
    try:
        if model is None:
            from src.llm.factory import get_chat_model

            model = get_chat_model("judge", max_tokens=120, reasoning=False)
        reply = model.invoke(
            PROMPT.format(headline=alert["headline"], detail=alert.get("detail", ""))
        )
        text = str(getattr(reply, "content", reply)).strip().strip('"')
    except Exception as e:  # noqa: BLE001 — phrasing is a nicety; never break the stream
        log.info("alert phrasing failed: %s", e)
        return None
    if not text or len(text) > 220 or not _faithful(text, alert):
        return None
    # A prediction must keep its probability, so it can't read as a done deal.
    if alert.get("p") is not None and f"{round(alert['p'] * 100)}%" not in text:
        return None
    return text
