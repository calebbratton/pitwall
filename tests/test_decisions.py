import json

from src.rag.decisions import (
    STEWARDING,
    DocumentRef,
    ajax_html,
    parse_decision,
    parse_rows,
    parse_season_page,
)

PLAIN_ROW = """<li class="document-row key-7">  <a href="/system/files/decision-document/x_-_infringement_-_car_1.pdf" download>
<div class="title">
  Doc 7 - Infringement - Car 1 - Impeding of Car 2
</div><div class="published">Published on <span  class="date-display-single">02.10.26 14:22</span> CET</div></a></li>"""

DRUPAL_ROW = """<li class="document-row key-9">
<div class="node"><a href="/system/files/decision-document/y_-_decision_-_car_3.pdf" download>
<div class="title"><div class="field-items"><div class="field-item even">Doc 9 - Decision - Car 3 - Turn 1 incident</div></div></div>
<div class="published"><div><span class="date-display-single">03.05.26 23:55</span></div></div></a></div></li>"""


def test_parse_rows_reads_both_markups():
    rows = parse_rows(PLAIN_ROW + DRUPAL_ROW, 2026, "Some Grand Prix")
    assert [(r.number, r.title, r.published) for r in rows] == [
        (7, "Infringement - Car 1 - Impeding of Car 2", "02.10.26 14:22"),
        (9, "Decision - Car 3 - Turn 1 incident", "03.05.26 23:55"),
    ]
    assert (
        rows[0].url
        == "https://www.fia.com/system/files/decision-document/x_-_infringement_-_car_1.pdf"
    )


def test_season_page_lists_open_event_docs_and_other_events():
    html = (
        '<div class="event-title active">Open Grand Prix</div>'
        + PLAIN_ROW
        + '<a href="/decision-document-list/nojs/123" class="event-title data-id-123  use-ajax">\n  Earlier Grand Prix  </a>'
    )
    docs, events = parse_season_page(html, 2026)
    assert docs[0].event == "Open Grand Prix"
    assert events == [("123", "Earlier Grand Prix")]
    assert (
        ajax_html(json.dumps([{"command": "settings"}, {"command": "insert", "data": "<p>x</p>"}]))
        == "<p>x</p>"
    )


DECISION_TEXT = """2026 TEST GRAND PRIX
From The Stewards
No / Driver 1 - Test Driver
Competitor Test Team
Time 16:24
Session Free Practice 2
Fact Car 1 unnecessarily impeded Car 2
on the entry to Turn 9.
Infringement Breach of Article B4.1.1 of the FIA F1 Regulations.
Decision Driver: Warning.
The competitor is fined 10,000.
Reason The driver was slow on the racing line.

The Stewards
The other car took evasive action.
Competitors are reminded that they have the right to appeal certain decisions.
Decisions of the Stewards are taken independently of the FIA."""


def test_parse_decision_fields_and_citation():
    ref = DocumentRef(2026, "Test Grand Prix", 7, "Infringement - Car 1 - Impeding", "u", "p")
    d = parse_decision(DECISION_TEXT.replace("No / Driver", "No /\xa0Driver"), ref)
    assert d.competition == "2026 Test Grand Prix"
    assert d.driver == "Test Driver"
    assert d.session == "Free Practice 2"
    assert d.fact == "Car 1 unnecessarily impeded Car 2 on the entry to Turn 9."
    assert d.decision == "Driver: Warning. The competitor is fined 10,000."
    assert d.reason == "The driver was slow on the racing line. The other car took evasive action."
    assert d.key == "Doc 7, 2026 Test Grand Prix"
    assert d.citation.startswith("Stewards' decision: 2026 Test Grand Prix, Doc 7")


def test_stewarding_trigger():
    assert STEWARDING.search("Should Hamilton get a penalty for impeding Bortoleto?")
    assert STEWARDING.search("what did the stewards decide")
    assert not STEWARDING.search("Could McLaren have undercut on lap 30?")


def test_grid_penalties_from_this_weekends_rulings():
    from datetime import UTC, datetime

    from src.rag.decisions import StewardsDecision, grid_penalties

    def ruling(number, title, decision, published):
        return StewardsDecision(
            2026,
            "2026 Test Grand Prix",
            number,
            title,
            "u",
            published,
            "",
            "",
            "",
            "",
            "",
            decision,
            "",
        )

    records = [
        ruling(
            17,
            "Infringement - Car 6 - Changes to PU elements",
            "Drop of 5 grid positions for the next Race in which the driver participates.",
            "02.10.26 10:00",
        ),
        ruling(
            30,
            "Infringement - Car 6 - Impeding of Car 5",
            "Drop of 3 grid positions for the next Race in which the driver participates.",
            "03.10.26 11:00",
        ),
        ruling(
            19,
            "Infringement - Car 41 - Changes to PU elements",
            "The driver is required to start the Race from the pit lane.",
            "02.10.26 10:00",
        ),
        ruling(
            26, "Infringement - Car 44 - Impeding of Car 5", "Driver: Warning.", "02.10.26 14:22"
        ),
        ruling(
            60,
            "Infringement - Car 10 - Impeding",
            "Drop of 3 grid positions for the next Race.",
            "20.09.26 15:00",
        ),
    ]
    drops, pit_lane, notes = grid_penalties(records, datetime(2026, 10, 3, 12, 0, tzinfo=UTC))
    assert drops == {6: 8}  # PU + impeding add up; last event's ruling is out of the window
    assert pit_lane == {41}
    assert len(notes) == 3 and notes[0].startswith("Doc 17: Infringement - Car 6")
