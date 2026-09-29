"""Team radio transcription with Whisper on Groq's free tier.

Free-tier limits (whisper-large-v3-turbo): 20 requests/min, 7,200 audio seconds/hour. A race has
~100-200 short clips, so a small concurrency cap plus the SDK's 429 retries is enough. Results are
cached on disk by clip URL, so replaying a race never re-transcribes it.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

log = logging.getLogger(__name__)
load_dotenv()  # this module is used without the chat-model factory (e.g. src/live.py)

MODEL = "whisper-large-v3-turbo"
# Bump the version when transcription settings change, so stale transcripts aren't reused.
DEFAULT_CACHE = Path("data/livetiming/radio_transcripts.v3.json")


# Whisper's prompt biases spelling. A natural sentence with the speaking driver's full name
# measured best on real clips; a vocabulary list got recited back on noisy audio
# ("Safety car, VSC, red flag...") and a full grid of names produced name lists.
def radio_prompt(driver_name: str | None) -> str:
    who = f" with {driver_name}" if driver_name else ""
    return f"Formula 1 team radio{who} and the race engineer."


# Standard Whisper quality gates, applied per segment: repetitive loops, low confidence, silence.
MAX_COMPRESSION_RATIO = 2.4
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH_PROB = 0.6
_REPEATED_WORD = re.compile(r"\b(\S+)(?:[\s,.]+\1\b){3,}", re.IGNORECASE)
_NON_WORD_RUN = re.compile(r"\S{25,}")
# Whisper invents these on silence or noise.
_HALLUCINATIONS = {"", ".", "you", "thank you.", "thanks for watching!", "thank you for watching."}


class RadioTranscriber:
    def __init__(
        self,
        cache_path: Path | None = DEFAULT_CACHE,
        max_concurrency: int = 2,
        model: str = MODEL,
    ) -> None:
        from groq import AsyncGroq

        self._client = AsyncGroq(max_retries=4)
        self._http = httpx.AsyncClient(timeout=30)
        self._limit = asyncio.Semaphore(max_concurrency)
        self._model = model
        self._cache_path = cache_path
        self._cache: dict[str, str] = (
            json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
        )

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("GROQ_API_KEY", "").strip())

    def cached(self, url: str) -> str | None:
        return self._cache.get(url)

    async def transcribe(self, url: str, names: list[str] = ()) -> str | None:
        """Transcript text ("" when there's no speech), or None if transcription failed."""
        if url in self._cache:
            return self._cache[url]
        prompt = radio_prompt(names[0] if names else None)
        async with self._limit:
            try:
                audio = (await self._http.get(url)).raise_for_status().content
                result = await self._client.audio.transcriptions.create(
                    file=(url.rsplit("/", 1)[-1], audio),
                    model=self._model,
                    language="en",
                    prompt=prompt,
                    temperature=0.0,
                    response_format="verbose_json",
                )
            except Exception:
                log.warning("transcription failed for %s", url, exc_info=True)
                return None
        text = clean_transcript(getattr(result, "segments", None) or [], result.text)
        self._cache[url] = text
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=1))
        return text

    async def aclose(self) -> None:
        await self._http.aclose()
        await self._client.close()


def _field(segment, name: str, default: float) -> float:
    value = segment.get(name) if isinstance(segment, dict) else getattr(segment, name, None)
    return default if value is None else float(value)


def clean_transcript(segments: list, fallback_text: str) -> str:
    """Keep confident speech segments; drop noise loops and hallucinated filler."""
    if segments:
        kept = [
            (s.get("text") if isinstance(s, dict) else s.text).strip()
            for s in segments
            if _field(s, "compression_ratio", 0) <= MAX_COMPRESSION_RATIO
            and _field(s, "avg_logprob", 0) >= MIN_AVG_LOGPROB
            and _field(s, "no_speech_prob", 0) <= MAX_NO_SPEECH_PROB
        ]
        text = " ".join(t for t in kept if t)
    else:
        text = fallback_text.strip()
    text = _dedupe_sentences(text)
    text = _REPEATED_WORD.sub(r"\1", text)
    text = _NON_WORD_RUN.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return "" if text.casefold() in _HALLUCINATIONS else text


_SENTENCE = re.compile(r"[^.!?]+[.!?]*")


def _dedupe_sentences(text: str) -> str:
    """Whisper loops whole phrases on noise: keep each sentence once."""
    seen, kept = set(), []
    for sentence in _SENTENCE.findall(text):
        key = re.sub(r"\W+", " ", sentence.casefold()).strip()
        if key and key not in seen:
            seen.add(key)
            kept.append(sentence.strip())
    return " ".join(kept)
