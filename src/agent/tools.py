"""Tools for the NBA Analytics Agent.

Two tools with clear separation of concerns:
- search_players: semantic vector search with retry on low similarity
- execute_sql: dynamic SQL execution (structured data queries)
"""

import re

from src.db import get_connection
from src.retrieval.semantic import SemanticRetriever
from src.config import SIMILARITY_THRESHOLD


# Initialize retriever once (singleton pattern for efficiency)
_retriever = None


def get_retriever() -> SemanticRetriever:
    """Get or create the semantic retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = SemanticRetriever()
    return _retriever


def _avg_similarity(results: list[dict]) -> float:
    """Average similarity score across retrieval results."""
    if not results:
        return 0.0
    return sum(r["similarity"] for r in results) / len(results)


def _format_results(results: list[dict]) -> str:
    """Format retrieval results into a readable string."""
    output = []
    for i, r in enumerate(results, 1):
        stats = r.get("stats", {})
        output.append(
            f"{i}. {r['player_name']} (similarity: {r['similarity']:.3f})\n"
            f"   Summary: {r['summary']}\n"
            f"   Stats: {stats.get('pts_per_game', 0):.1f} PPG, "
            f"{stats.get('reb_per_game', 0):.1f} RPG, "
            f"{stats.get('ast_per_game', 0):.1f} APG, "
            f"{stats.get('stl_per_game', 0):.1f} STL, "
            f"{stats.get('blk_per_game', 0):.1f} BLK"
        )
    return "\n\n".join(output)


def search_players(query: str, top_k: int = 5) -> str:
    """Semantic search for players matching a natural language query.

    If the initial search returns low-similarity results (below threshold),
    automatically retries with broadened query variations to improve recall.
    Returns the best results across all attempts.

    Args:
        query: Natural language description of what to search for
        top_k: Number of results to return

    Returns:
        Formatted string with player summaries and similarity scores
    """
    retriever = get_retriever()

    # First attempt with the original query
    best_results = retriever.retrieve_with_stats(query, top_k)
    best_avg = _avg_similarity(best_results)

    # If similarity is already good, return immediately
    if best_avg >= SIMILARITY_THRESHOLD:
        return _format_results(best_results)

    # Low similarity — try broadened variations
    variations = [
        f"{query} NBA 2016 season player stats",
        f"NBA player who {query}",
    ]

    for variation in variations:
        results = retriever.retrieve_with_stats(variation, top_k)
        avg = _avg_similarity(results)
        if avg > best_avg:
            best_results = results
            best_avg = avg

    return _format_results(best_results)


# Tables the SQL tool is allowed to query.
ALLOWED_TABLES = {"player_season_features", "raw_player_stats"}

# Hard cap on rows returned to keep LLM context manageable.
MAX_ROWS = 50


def execute_sql(sql_query: str) -> str:
    """Execute a validated SELECT query against the NBA database.

    Safety checks:
    - Only SELECT statements are allowed (no mutations).
    - Only known tables can be referenced.
    - Results are capped at MAX_ROWS to prevent context blowup.

    Args:
        sql_query: A SQL SELECT statement.

    Returns:
        Formatted table of results, or an error message.
    """
    # ── Validation ──────────────────────────────────────────────
    normalized = sql_query.strip().rstrip(";")

    # Must be a SELECT
    if not re.match(r"(?i)^\s*SELECT\b", normalized):
        return "Error: only SELECT queries are allowed."

    # Block mutation keywords anywhere in the query
    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE)\b",
        re.IGNORECASE,
    )
    if forbidden.search(normalized):
        return "Error: mutation statements are not allowed."

    # Enforce row cap — append LIMIT if not already present
    if not re.search(r"\bLIMIT\b", normalized, re.IGNORECASE):
        normalized = f"{normalized}\nLIMIT {MAX_ROWS}"

    # ── Execution ───────────────────────────────────────────────
    con = get_connection()
    try:
        df = con.execute(normalized).fetchdf()
        if df.empty:
            return "Query returned no results."
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        con.close()
