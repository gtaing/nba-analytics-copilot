# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Analytics Copilot — a multi-agent AI system that answers natural language questions about NBA statistics using LangGraph, RAG, dynamic SQL, and local LLMs via Ollama. Python 3.13+, managed with `uv`.

## Commands

```bash
# Install dependencies
uv sync

# Run the ETL pipeline (required before first query)
python main.py --setup

# Ask a question (requires Ollama running on localhost:11434)
python main.py "Who were the best defenders in 2016?"

# Verbose mode (shows routing, agent outputs, iteration count)
python main.py -v "Who were the best playmakers?"

# Specify a particular Ollama model
python main.py --model llama3.3 "Who scored the most?"
```

There is no test suite configured.

## Architecture

The project has three layers:

### 1. ETL Pipeline (`src/pipeline/`)

Transforms `data/player_stats_2016.csv` into DuckDB tables:
- `ingestion.py`: CSV → `raw_player_stats` table (Polars for loading)
- `features.py`: Aggregated stats → `player_season_features` table (PPG, RPG, APG, TS%, stocks)
- `summaries.py`: Text descriptions → `player_summaries` table
- `embeddings.py`: `all-MiniLM-L6-v2` (384-dim) → `player_embeddings` table

### 2. Tool Layer (`src/agent/tools.py` + `src/retrieval/semantic.py`)

Two raw tool functions:
- `search_players(query, top_k)`: Semantic vector search with retry/rephrase on low similarity scores (threshold in config).
- `execute_sql(sql_query)`: Validated SQL execution — SELECT-only, forbidden keyword scan, auto LIMIT 50.

LangChain `@tool` wrappers in `src/graph/tools.py` expose these to the LLM with full schema documentation in docstrings.

### 3. Multi-Agent Graph (`src/graph/`)

LangGraph `StateGraph` with supervisor pattern:

```
START → supervisor → route_question → sql_agent ──┐
                                    → rag_agent ──┤→ synthesizer → check_confidence → END
                                    → both (Send) ┘                                → supervisor (retry)
```

- **State** (`state.py`): `NBAState(MessagesState)` with fields: `route`, `sql_result`, `rag_result`, `iteration`.
- **Nodes** (`nodes.py`): `create_supervisor`, `create_sql_agent`, `create_rag_agent`, `create_synthesizer` — each returns a closure capturing the model name.
- **Builder** (`builder.py`): `build_graph(model_name)` constructs and compiles the graph. Uses `Send` for parallel dispatch, conditional edges for routing and feedback.

**Entry point**: `main.py` — argparse CLI that either runs `--setup` (ETL) or invokes `build_graph().invoke()`.

## Key Configuration

All constants live in `src/config.py`:
- `DB_PATH = "db/nba.duckdb"` — DuckDB database location
- `RAW_DATA_PATH = "data/player_stats_2016.csv"` — source CSV
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` — sentence-transformers model
- `DEFAULT_OLLAMA_MODEL = "llama3.2"` — default LLM
- `OLLAMA_BASE_URL = "http://localhost:11434"` — Ollama endpoint
- `MAX_ITERATIONS = 5` — feedback loop safety cap
- `SIMILARITY_THRESHOLD = 0.15` — RAG retry trigger threshold

Database connection helper: `src/db.py:get_connection()`.

## Data

Single dataset: 2016 NBA season, 487 players, 27 columns. The DuckDB file (`db/nba.duckdb`) is gitignored and regenerated via `--setup`.
