"""Terminal chat with Pit Wall AI.

Usage: python -m src.chat [--trace]
Ask about races ("Could McLaren have undercut on lap 30 at Monaco 2024?") or the regulations
("What happens if a race is red-flagged and can't be restarted?"). Follow-ups keep context.
"""

import argparse
import logging
import uuid

from langchain_core.messages import HumanMessage

from src.agents.graph import build_graph, memory_checkpointer
from src.rag.index import RegulationIndex
from src.tools.openf1 import HttpOpenF1Client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true", help="print the graph's evaluation steps")
    args = ap.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    index = RegulationIndex()
    graph = build_graph(HttpOpenF1Client(), index, checkpointer=memory_checkpointer())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("Pit Wall AI. Ask about a race or the regulations. Ctrl-D to quit.\n")
    try:
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                break
            if not question:
                continue
            state = graph.invoke({"messages": [HumanMessage(question)]}, config)
            if args.trace:
                print("\n".join(f"  · {s}" for s in state.get("evaluation_steps", [])))
            print(f"\n{state['messages'][-1].content}\n")
    finally:
        index.close()


if __name__ == "__main__":
    main()
