"""Map paddock vocabulary to the words the Sporting Regulations actually use.

Retrieval misses happen when a question uses team jargon the regulations never contain: the
2026 text says "suspension", never "red flag" in its clauses on work permitted during a stoppage,
and never mentions an "undercut" at all. Expansion is deterministic so it doesn't depend on the
router LLM remembering to rephrase.
"""

import re

GLOSSARY: dict[str, str] = {
    r"red[\s-]?flag(?:ged)?(?: period)?": "suspension",
    r"under[\s-]?cut|over[\s-]?cut": "pit stop tyre change",
    r"\bvsc\b": "virtual safety car",
    r"\bsc\b": "safety car",
    r"compounds?": "specifications of dry-weather tyres",
    r"degradation|\bdeg\b": "use of tyres",
    r"restart": "resumption",
}


def expand_query(query: str) -> str | None:
    """`query` with paddock jargon replaced by regulation vocabulary, or None if it has none.

    Substitution beats appending: extra words dilute the dense embedding of the original."""
    expanded = query
    for pattern, replacement in GLOSSARY.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded if expanded != query else None


# The regulations use British spelling; BM25 matches exact tokens, so "tire" never hits "tyre".
US_TO_UK: dict[str, str] = {
    r"\btire(s?)\b": r"tyre\1",
    r"\blicense(s?)\b": r"licence\1",
    r"\bcenter\b": "centre",
    r"\bcolor(s?)\b": r"colour\1",
}


def normalize_spelling(query: str) -> str:
    for pattern, replacement in US_TO_UK.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    return query
