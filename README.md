# NBA Analytics Copilot

An AI-powered **multi-agent system** that answers natural language questions about NBA statistics using RAG (Retrieval-Augmented Generation), dynamic SQL generation, and local LLMs — powered by **LangGraph**.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the ETL pipeline (first time only)
python main.py --setup

# Ask a question (requires Ollama running locally)
python main.py "Who were the best defenders in 2016?"

# Verbose mode — see the full agent trace
python main.py -v "Compare LeBron and Curry"
```

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai) running locally with a model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```

## Architecture

The system uses a **supervisor pattern** with specialist agents orchestrated by LangGraph:

```
                          ┌─────────────┐
                     ┌───▶│  SQL Agent  │───┐
                     │    └─────────────┘   │
┌──────────┐    ┌────┴─────┐                │    ┌─────────────┐
│  User Q  │───▶│ Supervisor│───────────────┼───▶│ Synthesizer │───▶ Answer
└──────────┘    └────┬─────┘                │    └──────┬──────┘
                     │    ┌─────────────┐   │           │
                     └───▶│  RAG Agent  │───┘           │ (low confidence)
                          └─────────────┘               ▼
                                                Retry → Supervisor
```

### How it works

1. **Supervisor** classifies the question via a lightweight LLM call and routes to specialist agent(s).
2. **SQL Agent** generates and executes SQL against DuckDB — handles rankings, filters, comparisons. The system prompt includes the full schema, few-shot examples, and LLM-friendly column names (no special characters). Has an internal retry loop for SQL error correction.
3. **RAG Agent** uses hybrid retrieval: keyword-based stat lookup for ranking queries (e.g. "best defenders" → `ORDER BY stocks_per_game`), with semantic vector search as fallback. Player summaries include qualitative labels ("elite defender", "rim protector") to improve embedding quality.
4. **Parallel dispatch** — for complex questions, both agents run simultaneously via LangGraph's `Send` API.
5. **Synthesizer** merges results from all agents into a coherent answer. If both agents returned errors or empty data, it triggers a feedback loop back to the supervisor for retry.

### Key Concepts

| Concept | Implementation |
|---------|---------------|
| **RAG** | Retrieve data from DuckDB + vector search, augment the LLM prompt, generate grounded answers |
| **Vector Embeddings** | Player summaries encoded via `all-MiniLM-L6-v2` (384-dim) for semantic search |
| **Hybrid Retrieval** | Keyword stat lookup from DB for ranking queries, semantic fallback for open-ended questions |
| **Dynamic SQL** | LLM writes SQL with full schema in prompt + few-shot examples (SELECT-only, validated) |
| **Multi-Agent** | Supervisor routes to specialist agents, each with focused tools and prompts |
| **Parallel Dispatch** | LangGraph `Send` API runs SQL and RAG agents simultaneously |
| **Feedback Loop** | Synthesizer checks data quality; retries via supervisor if insufficient |

## Project Structure

```
nba-analytics-copilot/
├── main.py                    # CLI entry point
├── eval.py                    # Evaluation script (5 Q&A cases, checks expected players)
├── pyproject.toml             # Dependencies (uv)
├── data/
│   └── player_stats_2016.csv  # Source data (2016 NBA season, 487 players)
├── db/
│   └── nba.duckdb             # Embedded database (gitignored, regenerated via --setup)
└── src/
    ├── config.py              # All configuration constants
    ├── db.py                  # DuckDB connection helper
    ├── pipeline/              # ETL pipeline (4 stages)
    │   ├── ingestion.py       # CSV → raw_player_stats table
    │   ├── features.py        # Aggregated stats → player_season_features table
    │   ├── summaries.py       # Text descriptions → player_summaries table
    │   └── embeddings.py      # Vector embeddings → player_embeddings table
    ├── retrieval/
    │   └── semantic.py        # SemanticRetriever (cosine similarity search)
    ├── agent/
    │   └── tools.py           # Raw tool functions (hybrid search_players, execute_sql)
    ├── graph/                 # LangGraph multi-agent system
        ├── state.py           # NBAState TypedDict (shared state contract)
        ├── tools.py           # LangChain @tool wrappers with schema docs
        ├── nodes.py           # Node functions (supervisor, sql_agent, rag_agent, synthesizer)
    │   └── builder.py         # StateGraph construction with Send and conditional edges
    └── tests/
        ├── test_tools.py      # SQL validation, LIMIT enforcement, formatting helpers
        └── test_graph.py      # Routing logic, confidence checks, supervisor parsing
```

## Usage

```bash
# Ask any question about 2016 NBA stats
python main.py "Who averaged a double-double?"
python main.py "Which players had more than 2 blocks per game?"
python main.py "Who was the best 3P shooter in 2016?"

# See the multi-agent trace
python main.py -v "Tell me about the top playmakers"

# Use a different Ollama model
python main.py --model llama3.3 "Best defenders?"
```

## Testing

```bash
# Unit tests (no Ollama required)
python -m pytest tests/ -v

# Evaluation against live LLM (requires Ollama running)
python eval.py
python eval.py --model llama3.3
```

## Data Pipeline

The ETL pipeline transforms raw CSV data into a queryable knowledge base:

```
CSV (27 columns, 487 players)
    ↓ ingestion.py
DuckDB: raw_player_stats
    ↓ features.py
DuckDB: player_season_features (PPG, RPG, APG, TS%, stocks, 3P, FG%, FT%)
    ↓ summaries.py
DuckDB: player_summaries (text with qualitative labels per player)
    ↓ embeddings.py
DuckDB: player_embeddings (384-dim vectors for semantic search)
```

Run with: `python main.py --setup`

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph (StateGraph, Send, conditional edges) |
| LLM Integration | langchain-ollama (ChatOllama) |
| Database | DuckDB (embedded OLAP) |
| ETL | Polars, Pandas |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Testing | pytest (unit) + custom eval script (live LLM) |
| CLI | argparse |

## Known Limitations

- **Single season**: Only 2016 NBA data (expandable via ETL pipeline).
- **Model quality**: Smaller models (3B) may truncate answers or miss players from the data. 7B+ models perform significantly better.
- **3P% precision**: Source data has low precision for shooting percentages (rounded to 1 decimal). Rankings use three-pointers made per game as primary metric instead.
- **Advanced metrics**: No DRTG, DWS, PER, or other advanced stats beyond what's in the source CSV.
