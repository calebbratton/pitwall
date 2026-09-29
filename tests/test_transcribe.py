from src.llm.transcribe import clean_transcript, radio_prompt


def _seg(text, logprob=-0.3, compression=1.2, no_speech=0.0):
    return {
        "text": text,
        "avg_logprob": logprob,
        "compression_ratio": compression,
        "no_speech_prob": no_speech,
    }


def test_keeps_confident_segments_and_drops_bad_ones():
    segments = [
        _seg(" Box this lap, box box."),
        _seg(" Bortoleto, Albon, Alonso, Sainz.", logprob=-1.4),  # low confidence: prompt echo
        _seg(" dpnpnpnpnp hbhbhbhb", compression=3.1),  # repetition loop
        _seg(" Thank you.", no_speech=0.9),  # silence
        _seg(" Copy."),
    ]
    assert clean_transcript(segments, "ignored") == "Box this lap, box box. Copy."


def test_collapses_repeated_words_and_strips_garbage_runs():
    segments = [_seg(" Sf. Sf. Sf. Sf. Sf. Tyres are gone. dpnpnpnpnpnpnpnpnpnpnpnpnpnpnpnpnpnp")]
    assert clean_transcript(segments, "") == "Sf. Tyres are gone."


def test_hallucinated_filler_becomes_empty():
    assert clean_transcript([_seg(" Thank you.")], "") == ""
    assert clean_transcript([], " . ") == ""


def test_repeated_sentences_are_collapsed_but_jargon_survives():
    assert clean_transcript([_seg(" Box box. Box box. Stay out.")], "") == "Box box. Stay out."
    real = " Safety car, safety car, box this lap for hards."
    assert clean_transcript([_seg(real)], "") == "Safety car, safety car, box this lap for hards."


def test_radio_prompt_is_a_natural_sentence():
    assert (
        radio_prompt("Isack Hadjar")
        == "Formula 1 team radio with Isack Hadjar and the race engineer."
    )
    assert radio_prompt(None) == "Formula 1 team radio and the race engineer."
