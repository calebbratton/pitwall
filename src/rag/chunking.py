"""Split Sporting Regulations text into clause-level chunks.

Two document formats exist:
- "classic" (2023-2025): articles `30) SUPPLY OF TYRES ...`, clauses `30.5 Use of Tyres`.
- "section_b" (2026+): `ARTICLE B6: TYRE LIMITATIONS`, sections `B6.3 Use & Return of Tyres`,
  clauses `B6.3.6 Unless they have used ...`.
Each clause becomes one chunk (long clauses are split into parts) so retrieval results can cite
an exact article number.
"""

import re
import unicodedata
from dataclasses import dataclass

from src.rag.sources import RegSource

MAX_CHUNK_CHARS = 1800
MIN_TAIL_CHARS = 300  # shorter leftovers are merged into the previous part


@dataclass(frozen=True)
class RegChunk:
    chunk_id: str
    article: str  # most specific number, e.g. "30.5" or "B6.3.6"
    article_title: str  # enclosing article (classic) or section (section_b) title
    part: int
    text: str
    season: int
    issue: int

    @property
    def citation(self) -> str:
        return f"{self.season} Sporting Regulations (Issue {self.issue}), Article {self.article}"

    def embedding_text(self) -> str:
        return f"Article {self.article} ({self.article_title.title()}): {self.text}"


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts, current = [], ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > MAX_CHUNK_CHARS:
            parts.append(current)
            current = ""
        current += line
    if current.strip() and parts and len(current) < MIN_TAIL_CHARS:
        parts[-1] += current
    elif current.strip():
        parts.append(current)
    return parts


def _make_chunks(spans: list[tuple[str, str, str]], source: RegSource) -> list[RegChunk]:
    """spans: (article number, title, text)."""
    chunks = []
    for article, title, text in spans:
        text = text.strip()
        if not text or text.upper() == "VOID":
            continue
        for part, piece in enumerate(_split_long(text), start=1):
            chunks.append(
                RegChunk(
                    chunk_id=f"{source.season}-iss{source.issue}-{article}-{part}",
                    article=article,
                    article_title=title,
                    part=part,
                    text=piece.strip(),
                    season=source.season,
                    issue=source.issue,
                )
            )
    return chunks


_MID_SENTENCE = re.compile(
    r"(?:\b(?:Articles?|and|or|of|to|in|with|under|see)|,)\s*$", re.IGNORECASE
)


def _is_wrapped_reference(text: str, m: re.Match) -> bool:
    """True if the line before `m` ends mid-sentence ("...see Article" / "...B5.3.3 and"), meaning
    `m` is a cross-reference that wrapped onto a new line rather than a new clause."""
    previous = text[: m.start()].rstrip().rsplit("\n", 1)[-1]
    return bool(_MID_SENTENCE.search(previous))


def _sequential(matches, key, max_gap: int = 3) -> list[re.Match]:
    """Keep matches that plausibly follow the previous clause: a slightly higher number in the
    same section, or clause 1 of a nearby section. Keys are (section, clause) tuples.

    Cross-references that wrap onto a new line ("...see Article\\n10.1 which...") otherwise look
    like clause starts, and a single forward-jumping one would swallow the real clauses after it.
    """
    kept, (ps, pc) = [], (0, 0)
    for m in matches:
        cs, cc = key(m)
        first_in_article = not kept and cc == 1
        if (
            first_in_article
            or (cs == ps and pc < cc <= pc + max_gap)
            or (ps < cs <= ps + max_gap and cc == 1)
        ):
            kept.append(m)
            ps, pc = cs, cc
    return kept


def _slice(text: str, matches: list[re.Match], i: int, end: int) -> str:
    return text[matches[i].start() : matches[i + 1].start() if i + 1 < len(matches) else end]


# --- classic (2023-2025) -------------------------------------------------------------------

_C_ARTICLE = re.compile(r"^\s*(\d{1,2})\)\s+([A-Z][A-Z ,\-/()&'’É]+?)\s*$", re.MULTILINE)
_C_CLAUSE = re.compile(r"^\s*(\d{1,2})\.(\d{1,2})\s", re.MULTILINE)
_C_APPENDIX = re.compile(r"^\s*APPENDIX 1\s*$", re.MULTILINE)
_C_NOISE = (
    re.compile(r"^\s*\d{4} Formula 1 Sporting Regulations \d+/\d+ .*$", re.MULTILINE),
    re.compile(r"^\s*©\d{4} Fédération Internationale de l.Automobile.*$", re.MULTILINE),
)


def _chunk_classic(text: str, source: RegSource) -> list[RegChunk]:
    for pattern in _C_NOISE:
        text = pattern.sub("", text)
    first = _C_ARTICLE.search(text)
    if not first:
        raise ValueError("No article headings found; not a classic-format document.")
    body = text[first.start() :]  # drops the table of contents
    appendix = _C_APPENDIX.search(body)
    body = body[: appendix.start()] if appendix else body

    headings = list(_C_ARTICLE.finditer(body))
    spans = []
    for i, heading in enumerate(headings):
        number, title = heading.group(1), heading.group(2).strip()
        section = _slice(body, headings, i, len(body))[heading.end() - heading.start() :]
        clauses = _sequential(
            (
                m
                for m in _C_CLAUSE.finditer(section)
                if m.group(1) == number and not _is_wrapped_reference(section, m)
            ),
            key=lambda m: (0, int(m.group(2))),
        )
        if not clauses:
            spans.append((number, title, section))
        for j, m in enumerate(clauses):
            spans.append(
                (f"{number}.{m.group(2)}", title, _slice(section, clauses, j, len(section)))
            )
    return _make_chunks(spans, source)


# --- section_b (2026+) ---------------------------------------------------------------------

# Table-of-contents headings end in a page number; body headings don't.
_B_ARTICLE = re.compile(r"^\s*ARTICLE B(\d{1,2}):\s*(.+?)\s*$", re.MULTILINE)
_B_SECTION = re.compile(r"^\s*B(\d{1,2})\.(\d{1,2})\s+([A-Z].*?)\s*$", re.MULTILINE)
_B_APPENDIX = re.compile(r"^\s*APPENDIX B\d+:.*[^\d\s]\s*$", re.MULTILINE)  # body, not TOC
_B_CLAUSE = re.compile(r"^\s*B(\d{1,2})\.(\d{1,2})\.(\d{1,2})\s", re.MULTILINE)
_B_NOISE = (
    re.compile(r"^\s*SECTION B: SPORTING REGULATIONS\s*$", re.MULTILINE),
    re.compile(r"^\s*B\d+ \d{4} Formula 1: Sporting Regulations\s*$", re.MULTILINE),
    re.compile(r"^\s*©\d{4} Fédération Internationale de l.Automobile.*$", re.MULTILINE),
    re.compile(r"^\s*\d{2} [A-Z][a-z]+ \d{4}\s*$", re.MULTILINE),  # issue date line
    re.compile(r"^\s*Issue \d+\s*$", re.MULTILINE),
    re.compile(r"^\s*[0B]\s*$", re.MULTILINE),  # stray page-furniture glyphs
)


def _fix_ligatures(text: str) -> str:
    # The 2026 PDF maps the "ff"/"ffi" ligature glyph to "‘" (body) or "W" (headings):
    # "o‘icials" -> "officials", "e‘ort" -> "effort", "OWicials" -> "Officials".
    text = unicodedata.normalize("NFKC", text)  # "ﬁ" -> "fi", "ﬂ" -> "fl"
    text = re.sub(r"(?<=[A-Za-z])‘", "ff", text)
    return re.sub(r"\bOW(?=ic)", "Off", text)


def _chunk_section_b(text: str, source: RegSource) -> list[RegChunk]:
    text = _fix_ligatures(text)
    for pattern in _B_NOISE:
        text = pattern.sub("", text)

    body_articles = [m for m in _B_ARTICLE.finditer(text) if not re.search(r"\s\d+$", m.group(2))]
    # Appendices repeat article headings; stop at the first non-increasing number.
    articles = []
    for m in body_articles:
        if articles and int(m.group(1)) <= int(articles[-1].group(1)):
            break
        articles.append(m)
    if not articles:
        raise ValueError("No ARTICLE Bn headings found; not a section_b-format document.")
    appendix = _B_APPENDIX.search(text, articles[0].start())
    body_end = appendix.start() if appendix else len(text)

    spans = []
    for i, article in enumerate(articles):
        number = article.group(1)
        article_text = _slice(text, articles, i, body_end)
        section_titles: dict[str, str] = {}
        for m in _B_SECTION.finditer(article_text):
            if m.group(1) == number:
                section_titles.setdefault(m.group(2), m.group(3))
        # Section heading lines would otherwise trail the previous clause's text.
        article_text = _B_SECTION.sub("", article_text)
        clauses = _sequential(
            (
                m
                for m in _B_CLAUSE.finditer(article_text)
                if m.group(1) == number and not _is_wrapped_reference(article_text, m)
            ),
            key=lambda m: (int(m.group(2)), int(m.group(3))),
        )
        for j, m in enumerate(clauses):
            title = section_titles.get(m.group(2), article.group(2))
            spans.append(
                (
                    f"B{number}.{m.group(2)}.{m.group(3)}",
                    title,
                    _slice(article_text, clauses, j, len(article_text)),
                )
            )
    return _make_chunks(spans, source)


def chunk_regulations(raw_text: str, source: RegSource) -> list[RegChunk]:
    if source.fmt == "section_b":
        return _chunk_section_b(raw_text, source)
    return _chunk_classic(raw_text, source)
