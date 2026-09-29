from datetime import date

import pytest

from src.rag.chunking import MAX_CHUNK_CHARS, MIN_TAIL_CHARS, chunk_regulations
from src.rag.sources import RegSource, source_for_race

CLASSIC = RegSource(2024, 6, date(2024, 4, 30), "https://example/2024.pdf")
SECTION_B = RegSource(2026, 8, date(2026, 8, 5), "https://example/2026.pdf", "section_b")

CLASSIC_TEXT = """\
CONTENTS
30) SUPPLY OF TYRES 32
2024 Formula 1 Sporting Regulations 33/106 30 April 2024
©2024 Fédération Internationale de l’Automobile  Issue 6
7) DEAD HEAT
Should two or more drivers finish with the same time, the prizes will be shared.
30) SUPPLY OF TYRES
30.1 Tyre supply
a) The tyre supplier must provide tyres as set out in Article
30.5 which remains unchanged.
30.2 Use of Tyres
Each driver must use at least two different specifications of dry-weather tyres.
41) VOID
APPENDIX 1
Entry form ...
"""

SECTION_B_TEXT = """\
ARTICLE B6: TYRE LIMITATIONS 55
B6.3 Use & Return of Tyres 57
ARTICLE B5: TOTAL TIME CLASSIFIED SESSIONS (TTCS)
B5.14 Suspension Procedure(s)
B5.14.1 The o‘icials may suspend the Race; a wrapped reference to Article
B5.15.1 must not start a new clause.
B5.14.2 Whilst suspended, teams may change tyres.
SECTION B: SPORTING REGULATIONS
B7 2026 Formula 1: Sporting Regulations
05 August 2026
Issue 08
ARTICLE B6: TYRE LIMITATIONS
B6.3 Use & Return of Tyres
B6.3.1 The OWicial allocation applies.
B6.3.2 Each driver must use at least two (2) di‘erent speciﬁcations of dry-weather tyres.
APPENDIX B1: DEFINITIONS
ARTICLE B2: FORMAT OF A COMPETITION
B2.1.1 Appendix text that must be ignored.
"""


def _by_article(chunks):
    return {c.article: c for c in chunks}


def test_classic_splits_clauses_and_skips_toc_noise_void_and_appendix():
    chunks = _by_article(chunk_regulations(CLASSIC_TEXT, CLASSIC))
    assert list(chunks) == ["7", "30.1", "30.2"]
    assert "30.5 which remains unchanged" in chunks["30.1"].text  # wrapped ref, not a clause
    assert "Entry form" not in chunks["30.2"].text
    assert chunks["30.2"].article_title == "SUPPLY OF TYRES"
    assert chunks["30.2"].citation == "2024 Sporting Regulations (Issue 6), Article 30.2"


def test_section_b_splits_clauses_fixes_ligatures_and_strips_page_furniture():
    chunks = _by_article(chunk_regulations(SECTION_B_TEXT, SECTION_B))
    assert list(chunks) == ["B5.14.1", "B5.14.2", "B6.3.1", "B6.3.2"]
    assert "officials" in chunks["B5.14.1"].text
    assert "B5.15.1 must not start" in chunks["B5.14.1"].text
    assert "The Official allocation" in chunks["B6.3.1"].text
    assert "different specifications" in chunks["B6.3.2"].text
    assert "Issue 08" not in chunks["B5.14.2"].text
    assert "Use & Return of Tyres" not in chunks["B5.14.2"].text  # section heading stripped
    assert chunks["B6.3.2"].article_title == "Use & Return of Tyres"
    assert "Appendix text" not in chunks["B6.3.2"].text


def test_long_clause_is_split_into_parts():
    long_clause = "\n".join(f"line {i} " + "x" * 80 for i in range(60))
    text = f"30) SUPPLY OF TYRES\n30.1 Heading\n{long_clause}\n"
    chunks = chunk_regulations(text, CLASSIC)
    assert len(chunks) > 1
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(len(c.text) <= MAX_CHUNK_CHARS + MIN_TAIL_CHARS for c in chunks)
    assert all(len(c.text) >= MIN_TAIL_CHARS for c in chunks)


@pytest.mark.parametrize(
    ("season", "race_date", "issue"),
    [
        (2024, date(2024, 3, 2), 5),  # Bahrain: before Issue 6
        (2024, date(2024, 5, 26), 6),  # Monaco
        (2024, date(2024, 8, 25), 7),  # Zandvoort
        (2026, date(2026, 3, 8), 5),  # Australia
        (2026, date(2026, 9, 26), 8),  # Azerbaijan
    ],
)
def test_source_for_race_picks_issue_in_force(season, race_date, issue):
    assert source_for_race(season, race_date).issue == issue


def test_source_for_race_unknown_season():
    with pytest.raises(ValueError):
        source_for_race(2019, date(2019, 5, 26))
