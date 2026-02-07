# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Analytics Copilot — an AI-powered agent that answers natural language questions about NBA statistics using RAG (Retrieval-Augmented Generation) with optional local LLMs. Python 3.13+, managed with `uv`.

## Commands

```bash
# Install dependencies
uv sync

# Install with optional HuggingFace support
uv sync --extra huggingface

# Run the ETL pipeline (required before first query)
python main.py --setup

# Ask a question (no LLM — returns structured data)
python main.py "Who were the best defenders in 2016?"

# Ask with Ollama backend (requires local Ollama running)
python main.py --backend ollama "Who was the most efficient scorer?"

# Verbose mode (shows tool selection and context)
python main.py -v "Who were the best playmakers?"

# Show raw retrieved context alongside answer
python main.py --show-context "Who scored the most?"

# Specify a particular Ollama model
python main.py --backend ollama --model llama3.2 "..."
```

There is no test suite configured.

## Architecture

The project follows a three-stage **RAG pipeline**:

1. **ETL Pipeline** (`src/pipeline/`) — Transforms `data/player_stats_2016.csv` into DuckDB tables:
   - `ingestion.py`: CSV → `raw_player_stats` table (Polars for loading)
   - `features.py`: Computes aggregated stats → `player_season_features` table (PPG, RPG, APG, TS%, AST/TOV ratio, stocks)
   - `summaries.py`: Generates human-readable text per player → `player_summaries` table
   - `embeddings.py`: Encodes summaries via `all-MiniLM-L6-v2` (384-dim) → `player_embeddings` table

2. **Retrieval** (`src/retrieval/semantic.py`) — `SemanticRetriever` class loads embeddings into memory, encodes user questions into the same vector space, ranks players by cosine similarity, and optionally joins structured stats.

3. **Agent** (`src/agent/`) — Two files:
   - `tools.py`: Six tool functions (semantic search, top scorers, top defenders, top playmakers, most efficient, player comparison) that query DuckDB directly.
   - `nba_agent.py`: `NBAAgent` class that does rule-based tool selection (keyword matching on the question), gathers context from selected tools, then optionally generates a natural language answer via Ollama HTTP API (`localhost:11434`) or HuggingFace transformers.

**Entry point**: `main.py` — argparse CLI that either runs `--setup` (ETL) or processes a question through the agent.

## Key Configuration

All constants live in `src/config.py`:
- `DB_PATH = "db/nba.duckdb"` — DuckDB database location
- `RAW_DATA_PATH = "data/player_stats_2016.csv"` — source CSV
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` — sentence-transformers model
- `MIN_GAMES = 50` — minimum games threshold for stat queries
- `HIGH_USAGE_THRESHOLD = 25.0`

Database connection helper: `src/db.py:get_connection()`.

## Data

Single dataset: 2016 NBA season, 487 players, 27 columns. The DuckDB file (`db/nba.duckdb`) is gitignored and regenerated via `--setup`.
