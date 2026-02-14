#!/usr/bin/env python3
"""Evaluation script for NBA Analytics Copilot.

Runs predefined questions through the graph and checks whether
expected player names appear in the final answer.

This is NOT a unit test — it requires Ollama running on localhost:11434
and a populated DuckDB database (run `python main.py --setup` first).

Usage:
    python eval.py
    python eval.py --model llama3.3
"""

import argparse
import sys
import time

from src.graph import build_graph
from src.config import MAX_ITERATIONS


# ── Evaluation cases ────────────────────────────────────────────
# Each case: (question, list of player names that MUST appear in the answer)

EVAL_CASES = [
    (
        "Who were the best defenders in 2016?",
        ["Anthony Davis", "Rudy Gobert", "Draymond Green"],
    ),
    (
        "Who were the top scorers?",
        ["Westbrook", "Isaiah Thomas", "Harden"],
    ),
    (
        "Who were the best rebounders?",
        ["Andre Drummond", "DeAndre Jordan", "Hassan Whiteside"],
    ),
    (
        "Who were the best playmakers?",
        ["Ricky Rubio", "Chris Paul", "Jeff Teague"],
    ),
    (
        "Which players had the most blocks per game?",
        ["Anthony Davis", "Gobert", "Whiteside"],
    ),
]


def run_eval(model_name: str | None = None):
    graph = build_graph(model_name=model_name)

    passed = 0
    failed = 0
    results = []

    for question, expected_names in EVAL_CASES:
        print(f"\n{'─' * 60}")
        print(f"Q: {question}")
        print(f"   Expected: {expected_names}")

        start = time.time()
        try:
            result = graph.invoke(
                {"messages": [("user", question)]},
                config={"recursion_limit": MAX_ITERATIONS * 10},
            )
            answer = result["messages"][-1].content
            elapsed = time.time() - start
        except Exception as e:
            answer = f"[ERROR] {e}"
            elapsed = time.time() - start

        # Check: how many expected names appear in the answer?
        found = [name for name in expected_names if name.lower() in answer.lower()]
        missing = [
            name for name in expected_names if name.lower() not in answer.lower()
        ]
        hit_rate = len(found) / len(expected_names)

        # Pass if at least half the expected names appear
        is_pass = hit_rate >= 0.5
        status = "PASS" if is_pass else "FAIL"

        if is_pass:
            passed += 1
        else:
            failed += 1

        results.append(
            {
                "question": question,
                "status": status,
                "hit_rate": hit_rate,
                "found": found,
                "missing": missing,
            }
        )

        print(f"   Answer: {answer[:200]}...")
        print(f"   Found: {found}")
        print(f"   Missing: {missing}")
        print(f"   Result: {status} ({hit_rate:.0%} hit rate, {elapsed:.1f}s)")

    # ── Summary ─────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY: {passed}/{total} passed ({passed / total:.0%})")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  [{r['status']}] {r['question']}")
        if r["missing"]:
            print(f"        missing: {r['missing']}")

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NBA Analytics Copilot")
    parser.add_argument("--model", type=str, help="Ollama model name")
    args = parser.parse_args()

    success = run_eval(model_name=args.model)
    sys.exit(0 if success else 1)
