# NBA Analytics Copilot

An AI-powered agent that answers natural language questions about NBA statistics using **RAG (Retrieval-Augmented Generation)** and optional local LLMs.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the ETL pipeline (first time only)
python main.py --setup

# Ask a question
python main.py "Who were the best defenders in 2016?"
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                            │
│              "Who was the best defender in 2016?"               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          AGENT                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. TOOL SELECTION                                       │   │
│  │     Question contains "defender" → select defense tools  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  2. RETRIEVAL (the "R" in RAG)                          │   │
│  │     • Semantic search: find similar player descriptions  │   │
│  │     • Structured query: get top defenders by stats       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3. GENERATION (the "G" in RAG)                         │   │
│  │     • LLM synthesizes answer from retrieved context      │   │
│  │     • Or: structured display of raw data (no LLM mode)   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ANSWER                                  │
│  "Based on 2016 data, the top defenders by stocks (STL+BLK)    │
│   were: 1. Hassan Whiteside (4.2), 2. Anthony Davis (3.9)..."  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts (for GenAI learners)

### 1. RAG (Retrieval-Augmented Generation)

Instead of asking an LLM to answer from memory (which may be outdated or wrong), RAG:
1. **Retrieves** relevant data from a knowledge base
2. **Augments** the LLM prompt with this data
3. **Generates** an answer grounded in facts

This solves the "hallucination" problem - the LLM can only use data you provide.

### 2. Vector Embeddings

Text is converted to numerical vectors (384 dimensions in our case) that capture semantic meaning:
- "LeBron is a great scorer" → [0.12, -0.45, 0.89, ...]
- "James averaged 25 points" → [0.14, -0.42, 0.91, ...] (similar!)

We use `sentence-transformers` to create these embeddings.

### 3. Semantic Search

Find relevant content by comparing vector similarity:
```
Question: "best defenders"  →  embedding  →  compare with all player embeddings
                                                     │
                                          ┌──────────┴──────────┐
                                          │ Most similar players │
                                          └─────────────────────┘
```

### 4. Tool-Based Agents

Instead of hardcoding logic, we give the agent **tools**:
- `search_players`: Semantic search
- `get_top_defenders`: SQL query for defense stats
- `get_top_scorers`: SQL query for scoring stats

The agent decides which tools to use based on the question.

## Project Structure

```
nba-analytics-copilot/
├── main.py                    # CLI entry point
├── pyproject.toml             # Dependencies
├── README.md
├── data/
│   └── player_stats_2016.csv  # Source data
├── db/
│   └── nba.duckdb             # DuckDB database
└── src/
    ├── config.py              # Configuration
    ├── db.py                  # Database connection
    ├── pipeline/              # ETL pipeline
    │   ├── ingestion.py       # Load CSV → DuckDB
    │   ├── features.py        # Compute stats
    │   ├── summaries.py       # Generate text descriptions
    │   └── embeddings.py      # Create vector embeddings
    ├── retrieval/
    │   └── semantic.py        # Semantic search implementation
    └── agent/
        ├── tools.py           # Agent tools (search, query)
        └── nba_agent.py       # Main agent with LLM integration
```

## Usage

### Basic Usage (No LLM)
```bash
python main.py "Who had the best assist-to-turnover ratio?"
```

### With Ollama (Local LLM)
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2

python main.py --backend ollama "Who was the most efficient scorer?"
```

### Verbose Mode
```bash
python main.py -v "Who were the best playmakers?"
```

## Data Pipeline

The ETL pipeline transforms raw CSV data into a queryable knowledge base:

```
CSV Data → DuckDB Tables → Text Summaries → Vector Embeddings
           (structured)    (for humans)     (for semantic search)
```

Run it with:
```bash
python main.py --setup
```

## Limitations & Future Improvements

**Current limitations:**
- Only 2016 NBA season data
- Limited defensive metrics (STL, BLK only - no DRTG, DWS)
- Simple rule-based tool selection

**Potential improvements:**
- Add more seasons via `nba_api` package
- Advanced defensive stats
- LLM-powered tool selection
- Conversation memory
- Player comparison visualizations

## Tech Stack

| Component | Technology |
|-----------|------------|
| Database | DuckDB (embedded OLAP) |
| ETL | Polars, Pandas |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM (optional) | Ollama  |
| CLI | argparse |
