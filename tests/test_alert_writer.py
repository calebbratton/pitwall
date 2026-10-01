from src.agents.alert_writer import _faithful, phrase


class Reply:
    def __init__(self, content):
        self.content = content


class Model:
    def __init__(self, text):
        self.text = text

    def invoke(self, prompt):
        assert "Facts:" in prompt
        return Reply(self.text)


ALERT = {
    "headline": "LEC's undercut on HAM is on",
    "detail": "LEC is 1.4 s behind HAM, 62% to come out ahead.",
    "p": 0.62,
}


def test_phrase_keeps_faithful_lines():
    assert (
        phrase(ALERT, Model("Leclerc can jump Hamilton by pitting now - 62% it works.")) is not None
    )


def test_phrase_rejects_invented_numbers_and_failures():
    assert phrase(ALERT, Model("Leclerc gains 3 places if he pits on lap 30.")) is None
    assert not _faithful("75% chance", ALERT)

    class Broken:
        def invoke(self, prompt):
            raise RuntimeError("rate limited")

    assert phrase(ALERT, Broken()) is None


def test_predictions_must_keep_their_probability():
    assert phrase(ALERT, Model("Leclerc pits now and comes out ahead of Hamilton.")) is None
