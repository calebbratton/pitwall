from src.models.compounds import c_number, nominations


def test_labels_map_to_the_weekends_compounds():
    assert c_number(2026, "Baku", "MEDIUM") == "C4"
    assert c_number(2026, "Suzuka", "SOFT") == "C3"  # harder than Monaco's MEDIUM
    assert c_number(2026, "Monte Carlo", "MEDIUM") == "C4"
    assert c_number(2026, "Kuala Lumpur", "HARD") == "C2"


def test_unknowns_and_wets_are_none():
    assert c_number(2026, "Baku", "INTERMEDIATE") is None
    assert c_number(2026, "Atlantis", "SOFT") is None
    assert c_number(2019, "Baku", "SOFT") is None


def test_every_2026_race_has_three_distinct_increasing_compounds():
    for location, labels in nominations(2026).items():
        numbers = [int(labels[k][1]) for k in ("HARD", "MEDIUM", "SOFT")]
        assert numbers == sorted(numbers) and len(set(numbers)) == 3, location


def test_c_number_matches_aliases_and_accents():
    from src.models.compounds import c_number

    assert c_number(2026, "Sepang", "HARD") == c_number(2026, "Kuala Lumpur", "HARD") == "C2"
    assert c_number(2026, "montreal", "SOFT") == c_number(2026, "Montréal", "SOFT")
