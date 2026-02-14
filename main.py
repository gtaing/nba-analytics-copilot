#!/usr/bin/env python3
"""
NBA Analytics Copilot - CLI Entry Point

Usage:
    python main.py "Who was the best defender in 2016?"
    python main.py --setup                      # Run ETL pipeline
    python main.py --model llama3.2 "..."       # Specify Ollama model
    python main.py -v "Compare LeBron and Curry" # Show agent trace
"""

import argparse
import sys

from src.pipeline import (
    ingest_raw_data,
    build_player_season_features,
    generate_player_summaries,
    build_embeddings,
)
from src.graph import build_graph
from src.config import MAX_ITERATIONS


def run_pipeline():
    """Run the full ETL pipeline to prepare data."""
    print("Running ETL pipeline...")
    print("-" * 40)
    ingest_raw_data()
    build_player_season_features()
    generate_player_summaries()
    build_embeddings()
    print("-" * 40)
    print("Pipeline complete! You can now ask questions.\n")


def main():
    parser = argparse.ArgumentParser(
        description="NBA Analytics Copilot - Ask questions about NBA statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python main.py "Who were the best defenders in 2016?"
            python main.py "Which players averaged more than 2 blocks per game?"
            python main.py --setup  # First time setup
            python main.py -v "Compare LeBron and Curry"
        """,
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask the NBA analytics agent",
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run the ETL pipeline to prepare data (run this first)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Ollama model name (default: llama3.2)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show the multi-agent trace (routing, tool results, iterations)",
    )

    args = parser.parse_args()

    # Run setup if requested
    if args.setup:
        run_pipeline()
        if not args.question:
            return

    # Require a question if not just running setup
    if not args.question:
        parser.print_help()
        print("\nError: Please provide a question or use --setup")
        sys.exit(1)

    # Build and invoke the multi-agent graph
    graph = build_graph(model_name=args.model)
    result = graph.invoke(
        {"messages": [("user", args.question)]},
        config={"recursion_limit": MAX_ITERATIONS * 10},
    )

    # Verbose: show the multi-agent trace
    if args.verbose:
        route = result.get("route", "?")
        iteration = result.get("iteration", 1)
        sql = result.get("sql_result", "")
        rag = result.get("rag_result", "")

        print(f"\n{'=' * 60}")
        print("AGENT TRACE")
        print(f"{'=' * 60}")
        print(f"Route: {route} | Iterations: {iteration}")

        if sql:
            preview = sql[:300] + "..." if len(sql) > 300 else sql
            print(f"\n[SQL Agent]\n{preview}")
        if rag:
            preview = rag[:300] + "..." if len(rag) > 300 else rag
            print(f"\n[RAG Agent]\n{preview}")
        print(f"{'=' * 60}\n")

    answer = result["messages"][-1].content
    print("\n" + answer)


if __name__ == "__main__":
    main()
