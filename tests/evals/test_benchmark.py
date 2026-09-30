"""Benchmark suite: calls the real LLMs (Groq). Run explicitly:

    PITWALL_EVALS=1 pytest tests/evals

Thresholds are floors under the measured baseline (docs/STATUS.md), to catch regressions.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PITWALL_EVALS") != "1", reason="calls real LLMs; set PITWALL_EVALS=1"
)


@pytest.fixture(scope="module")
def summary():
    from src.evals.run import evaluate

    return evaluate()["summary"]


def test_no_errors(summary):
    assert summary["errors"] == 0


def test_routing(summary):
    assert summary["route_accuracy"] >= 0.9


def test_retrieval(summary):
    assert summary["retrieval_hit_rate"] >= 0.8


def test_faithfulness(summary):
    assert summary["faithfulness"] >= 0.8
