"""The data refresh job, with its ingest / build / decisions steps faked: no network."""

from datetime import timedelta

from src.tools.openf1 import OpenF1Error
from src.warehouse.refresh import ALL_SESSIONS, refresh


class Calls:
    def __init__(self, new=(), fail=None, decisions_fail=False):
        self.new, self.fail, self.decisions_fail = list(new), fail, decisions_fail
        self.ingested, self.built, self.decisions = [], 0, []

    def ingest(self, years, sessions, settle):
        self.ingested.append((years, tuple(sessions), settle))
        if self.fail:
            raise self.fail
        return self.new

    def build(self):
        self.built += 1

    def fetch_decisions(self, seasons):
        self.decisions.append(seasons)
        if self.decisions_fail:
            raise RuntimeError("fia.com down")
        return ["doc"] * 3


def _run(calls, **kw):
    return refresh(
        [2026],
        ingest_fn=calls.ingest,
        build_fn=calls.build,
        decisions_fn=calls.fetch_decisions,
        **kw,
    )


def test_new_sessions_are_ingested_built_once_and_decisions_fetched():
    calls = Calls(new=[11731, 11735])
    report = _run(calls, settle=timedelta(minutes=45))
    assert calls.ingested == [([2026], ALL_SESSIONS, timedelta(minutes=45))]
    assert "Race" in ALL_SESSIONS and "Qualifying" in ALL_SESSIONS
    assert calls.built == 1 and report.rebuilt and report.sessions == [11731, 11735]
    assert report.decisions == 3


def test_nothing_new_means_no_rebuild():
    calls = Calls(new=[])
    report = _run(calls)
    assert calls.built == 0 and not report.rebuilt


def test_openf1_closed_during_a_live_session_is_skipped_not_raised():
    calls = Calls(fail=OpenF1Error("401: session live"))
    report = _run(calls)
    assert report.sessions == [] and "session live" in report.skipped
    assert calls.built == 0
    assert calls.decisions  # the stewards' documents don't depend on OpenF1


def test_a_failing_decisions_fetch_doesnt_lose_the_warehouse_update():
    calls = Calls(new=[1], decisions_fail=True)
    report = _run(calls)
    assert calls.built == 1 and report.decisions is None and "fia.com down" in report.errors[0]
