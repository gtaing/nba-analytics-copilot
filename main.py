#!/usr/bin/env python3
"""
NBA Analytics Copilot - CLI Entry Point

Usage:
    python main.py "Who was the best defender in 2016?"
    python main.py --setup                    # Run ETL pipeline
    python main.py --backend ollama "..."     # Use Ollama LLM
    python main.py --backend huggingface "..."  # Use HuggingFace model
    python main.py --verbose "..."            # Show debug info
"""

import argparse
import sys

from src.pipeline import (
    ingest_raw_data,
    build_player_season_features,
    generate_player_summaries,
    build_embeddings,
)
from src.agent import NBAAgent


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
  python main.py "Which players had the highest assist-to-turnover ratio?"
  python main.py --setup  # First time setup
  python main.py --backend ollama "Who was the most efficient scorer?"
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
        "--backend",
        choices=["ollama", "huggingface", "none"],
        default="none",
        help="LLM backend to use (default: none - shows raw data)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model name (e.g., 'llama3.2' for Ollama)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including tool selection",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show the raw context used to generate the answer",
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

    # Initialize agent
    agent = NBAAgent(llm_backend=args.backend, model_name=args.model)

    # Ask the question
    if args.show_context:
        answer, context = agent.ask_with_context(args.question)
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)
        print("\n" + "=" * 60)
        print("CONTEXT USED")
        print("=" * 60)
        print(context)
    else:
        answer = agent.ask(args.question, verbose=args.verbose)
        print("\n" + answer)


if __name__ == "__main__":
    main()
