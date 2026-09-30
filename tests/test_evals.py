from src.evals.run import is_relevant, load_scenarios, retrieval_scores, summarise


def test_article_prefix_matching():
    assert is_relevant("B5.12.3", ("B5.12",))
    assert is_relevant("B5.12", ("B5.12",))
    assert not is_relevant("B5.120.1", ("B5.12",))
    assert not is_relevant("B5.1.2", ("B5.12",))


def test_retrieval_scores():
    s = retrieval_scores(["B1.1.1", "B5.12.3", "B5.12.2", "B6.3.6"], ("B5.12",))
    assert s == {"precision": 0.5, "hit": True, "first_relevant_rank": 2}
    assert retrieval_scores(["B1.1.1"], ()) is None
    assert retrieval_scores([], ("B5.12",))["hit"] is False


def test_scenarios_are_well_formed():
    scenarios = load_scenarios()
    assert len(scenarios) == 20
    assert len({s.id for s in scenarios}) == 20
    assert {s.mode for s in scenarios} == {"rules", "race"}
    assert all(s.relevant for s in scenarios if s.mode == "rules")


def test_summary_skips_unscored():
    rows = [
        {
            "route_ok": True,
            "error": None,
            "retrieval": {"precision": 0.4, "hit": True},
            "faithfulness": 1.0,
        },
        {"route_ok": False, "error": "x", "retrieval": None, "faithfulness": None},
    ]
    s = summarise(rows)
    assert s["route_accuracy"] == 0.5 and s["errors"] == 1
    assert s["retrieval_precision"] == 0.4 and s["faithfulness"] == 1.0
