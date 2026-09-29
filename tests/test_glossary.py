import pytest

from src.rag.glossary import expand_query, normalize_spelling


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What work is allowed during a red flag?", "What work is allowed during a suspension?"),
        ("Red-flagged races", "suspension races"),
        ("Was an undercut possible?", "Was an pit stop tyre change possible?"),
        ("VSC rules", "virtual safety car rules"),
    ],
)
def test_jargon_is_replaced(query, expected):
    assert expand_query(query) == expected


def test_no_jargon_returns_none():
    assert expand_query("What is the pit lane speed limit?") is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How many Tires?", "How many tyres?"),
        ("tire compound", "tyre compound"),
        ("super license points", "super licence points"),
        ("tired drivers", "tired drivers"),  # whole words only
    ],
)
def test_us_spelling_is_normalized(query, expected):
    assert normalize_spelling(query) == expected
