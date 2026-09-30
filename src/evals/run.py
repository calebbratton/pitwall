"""Benchmark the post-race chat graph on tests/evals/scenarios.yaml.

Per scenario:
  route      did the router pick the expected mode (rules / race)?
  retrieval  precision@k: share of retrieved regulation clauses that are relevant (hand-labelled
             article prefixes); hit: at least one relevant clause retrieved
  faithful   DeepEval FaithfulnessMetric: share of the answer's claims supported by what the
             analyst was shown (retrieved clauses + telemetry tool results). Judged by the
             "judge" role (Qwen on Groq), a different model family from the analyst.

Graph runs and judge verdicts are cached in data/evals/ (keyed by scenario, and by judge model
for verdicts), so re-scoring doesn't re-run the graph: --rerun forces fresh answers,
--rejudge fresh verdicts. Calls the real LLMs; paced for Groq's free tier.

Usage: python -m src.evals.run [--only vsc-pit madrid-vsc] [--rerun] [--rejudge] [--no-judge]
"""

import argparse
import json
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage

from src.agents.graph import MAX_ANALYST_TELEMETRY_CHARS, build_graph
from src.evals.judge import call_with_backoff
from src.rag.index import DEFAULT_INDEX_PATH, RegulationIndex

SCENARIOS = Path("tests/evals/scenarios.yaml")
OUT = Path("data/evals")
PAUSE_S = 20  # between graph runs: each uses ~3-6K tokens of several models' per-minute budget


@dataclass(frozen=True)
class Scenario:
    id: str
    mode: str
    question: str
    relevant: tuple[str, ...] = ()


def load_scenarios(path: Path = SCENARIOS) -> list[Scenario]:
    return [
        Scenario(s["id"], s["mode"], s["question"], tuple(s.get("relevant", ())))
        for s in yaml.safe_load(path.read_text())
    ]


def is_relevant(article: str, relevant: tuple[str, ...]) -> bool:
    """ "B5.12" covers "B5.12.3" but not "B5.1.2" or "B5.120"."""
    return any(article == r or article.startswith(r + ".") for r in relevant)


def retrieval_scores(articles: list[str], relevant: tuple[str, ...]) -> dict | None:
    if not relevant:
        return None
    hits = [is_relevant(a, relevant) for a in articles]
    return {
        "precision": sum(hits) / len(hits) if hits else 0.0,
        "hit": any(hits),
        "first_relevant_rank": hits.index(True) + 1 if any(hits) else None,
    }


def run_graph(graph, scenario: Scenario) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = call_with_backoff(graph.invoke, {"messages": [HumanMessage(scenario.question)]}, config)
    results = state.get("fetched_telemetry_json", [])
    per_result = MAX_ANALYST_TELEMETRY_CHARS // max(len(results), 1)
    return {
        "question": scenario.question,
        "answer": state["messages"][-1].content,
        "mode": state["route"].mode if state.get("route") else None,
        "error": state.get("error"),
        "rules": [
            {"article": r["article"], "text": r["text"]} for r in state["retrieved_rules_text"]
        ],
        # Exactly what the analyst saw (same clipping as the analyze node).
        "telemetry": [
            {"tool": r["tool"], "args": r["args"], "result": r["result"][:per_result]}
            for r in results
        ],
        "steps": state.get("evaluation_steps", []),
    }


def judge_faithfulness(run: dict, judge) -> dict:
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    context = [f"[{r['article']}] {r['text']}" for r in run["rules"]] + [
        f"{t['tool']}({json.dumps(t['args'])}): {t['result']}" for t in run["telemetry"]
    ]
    metric = FaithfulnessMetric(
        model=judge, async_mode=False, include_reason=True, truths_extraction_limit=40
    )
    metric.measure(
        LLMTestCase(input=run["question"], actual_output=run["answer"], retrieval_context=context)
    )
    return {
        "score": metric.score,
        "reason": metric.reason,
        "unsupported": [v.reason for v in metric.verdicts if v.verdict.strip().lower() == "no"],
    }


def _index_copy() -> Path:
    """Embedded Qdrant allows one process per folder, and the API server may hold it: search
    a throwaway copy instead."""
    tmp = Path(tempfile.mkdtemp(prefix="pitwall-evals-qdrant-"))
    shutil.copytree(DEFAULT_INDEX_PATH, tmp, dirs_exist_ok=True)
    (tmp / ".lock").unlink(missing_ok=True)
    return tmp


def evaluate(
    only: list[str] | None = None,
    rerun: bool = False,
    rejudge: bool = False,
    use_judge: bool = True,
) -> dict:
    from src.seasons import supported_seasons
    from src.tools.openf1 import HttpOpenF1Client

    scenarios = [s for s in load_scenarios() if not only or s.id in only]
    OUT.mkdir(parents=True, exist_ok=True)
    graph = index = judge = None
    index_path = None
    rows = []
    try:
        for n, scenario in enumerate(scenarios, 1):
            path = OUT / f"{scenario.id}.json"
            cached = json.loads(path.read_text()) if path.exists() else {}
            if rerun or cached.get("question") != scenario.question:
                if graph is None:
                    index_path = _index_copy()
                    index = RegulationIndex(index_path)
                    graph = build_graph(HttpOpenF1Client(), index, seasons=supported_seasons())
                print(f"[{n}/{len(scenarios)}] {scenario.id}: running graph", flush=True)
                cached = run_graph(graph, scenario)
                time.sleep(PAUSE_S)
            run = cached
            if use_judge and run.get("error") is None:
                if judge is None:
                    from src.evals.judge import GroqJudge

                    judge = GroqJudge()
                verdicts = run.setdefault("faithfulness", {})
                if rejudge or judge.get_model_name() not in verdicts:
                    print(f"[{n}/{len(scenarios)}] {scenario.id}: judging", flush=True)
                    try:
                        verdicts[judge.get_model_name()] = judge_faithfulness(run, judge)
                    except Exception as e:  # noqa: BLE001 — one failed verdict shouldn't end the run
                        print(f"   judge failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            path.write_text(json.dumps(run, indent=1))
            faith = run.get("faithfulness", {})
            rows.append(
                {
                    "id": scenario.id,
                    "route_ok": run["mode"] == scenario.mode,
                    "error": run.get("error"),
                    "retrieval": retrieval_scores(
                        [r["article"] for r in run["rules"]], scenario.relevant
                    ),
                    "faithfulness": (
                        faith[judge.get_model_name()]["score"]
                        if judge and judge.get_model_name() in faith
                        else None
                    ),
                }
            )
    finally:
        if index is not None:
            index.close()
        if index_path is not None:
            shutil.rmtree(index_path, ignore_errors=True)
    report = {"scenarios": rows, "summary": summarise(rows)}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    return report


def summarise(rows: list[dict]) -> dict:
    scored = [r["retrieval"] for r in rows if r["retrieval"]]
    faith = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "scenarios": len(rows),
        "errors": sum(r["error"] is not None for r in rows),
        "route_accuracy": mean([r["route_ok"] for r in rows]),
        "retrieval_precision": mean([s["precision"] for s in scored]),
        "retrieval_hit_rate": mean([s["hit"] for s in scored]),
        "faithfulness": mean(faith),
        "faithfulness_judged": len(faith),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scenario ids")
    ap.add_argument("--rerun", action="store_true", help="re-run the graph (fresh answers)")
    ap.add_argument("--rejudge", action="store_true", help="re-run the judge")
    ap.add_argument("--no-judge", action="store_true", help="deterministic metrics only")
    args = ap.parse_args()
    report = evaluate(args.only, args.rerun, args.rejudge, not args.no_judge)
    print(f"\n{'scenario':24} {'route':>5} {'prec':>5} {'hit':>4} {'faith':>6}")
    for r in report["scenarios"]:
        ret = r["retrieval"]
        print(
            f"{r['id']:24} {'ok' if r['route_ok'] else 'MISS':>5} "
            f"{ret['precision'] if ret else float('nan'):5.2f} "
            f"{('y' if ret['hit'] else 'n') if ret else '-':>4} "
            f"{r['faithfulness'] if r['faithfulness'] is not None else float('nan'):6.2f}"
            + (f"  error: {r['error']}" if r["error"] else "")
        )
    print("\n" + json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
