"""Tools for the NBA Analytics Agent.

Two tools with clear separation of concerns:
- search_players: hybrid retrieval (keyword stat lookup + semantic search)
- execute_sql: dynamic SQL execution (structured data queries)
"""

import re

from src.db import get_connection
from src.retrieval.semantic import SemanticRetriever
from src.config import SIMILARITY_THRESHOLD, MIN_GAMES


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


# ── Hybrid retrieval ────────────────────────────────────────────
# Maps query keywords to the stat column to sort by.
# Order matters: first match wins.

STAT_KEYWORDS: list[tuple[str, str]] = [
    ("defend",   "stocks_per_game"),
    ("block",    "blk_per_game"),
    ("steal",    "stl_per_game"),
    ("rebound",  "reb_per_game"),
    ("assist",   "ast_per_game"),
    ("playmak",  "ast_per_game"),
    ("3p",       "three_pt_made_per_game"),
    ("three p",  "three_pt_made_per_game"),
    ("three-p",  "three_pt_made_per_game"),
    ("scor",     "pts_per_game"),
    ("point",    "pts_per_game"),
    ("shoot",    "true_shooting_pct"),
    ("efficien", "true_shooting_pct"),
]


def _match_stat_column(query: str) -> str | None:
    """Return the stat column matching keywords in the query, or None."""
    query_lower = query.lower()
    for keyword, column in STAT_KEYWORDS:
        if keyword in query_lower:
            return column
    return None


def _get_stat_leaders(query: str, top_k: int = 5) -> list[dict]:
    """Fetch top players by the stat column matching the query keywords.

    Returns results in the same format as SemanticRetriever.retrieve_with_stats
    so they can be merged with semantic results and passed to _format_results.
    """
    column = _match_stat_column(query)
    if not column:
        return []

    con = get_connection()
    try:
        rows = con.execute(f"""
            SELECT
                f.player_name,
                s.summary,
                f.pts_per_game,
                f.reb_per_game,
                f.ast_per_game,
                f.stl_per_game,
                f.blk_per_game,
                f.true_shooting_pct,
                f.ast_to_tov_ratio,
                f.stocks_per_game
            FROM player_season_features f
            JOIN player_summaries s USING (player_name)
            WHERE f.games_played >= {MIN_GAMES}
            ORDER BY f.{column} DESC
            LIMIT {top_k}
        """).fetchall()
    finally:
        con.close()

    return [
        {
            "player_name": r[0],
            "summary": r[1],
            "similarity": 1.0,  # stat-matched = perfect relevance
            "stats": {
                "pts_per_game": r[2],
                "reb_per_game": r[3],
                "ast_per_game": r[4],
                "stl_per_game": r[5],
                "blk_per_game": r[6],
                "true_shooting_pct": r[7],
                "ast_to_tov_ratio": r[8],
                "stocks_per_game": r[9],
            },
        }
        for r in rows
    ]


def search_players(query: str, top_k: int = 5) -> str:
    """Hybrid search: keyword-based stat lookup + semantic vector search.

    1. If the query contains a recognizable stat keyword (e.g. "defender",
       "scorer"), fetch the actual stat leaders from the database.
    2. Fill remaining slots with semantic search results.
    3. If no keyword matches, fall back to pure semantic search with retry.

    Args:
        query: Natural language description of what to search for
        top_k: Number of results to return

    Returns:
        Formatted string with player summaries and stats
    """
    # ── Hybrid path: keyword stat leaders + semantic fill ───────
    stat_results = _get_stat_leaders(query, top_k)
    if stat_results:
        # Fill remaining slots with semantic results not already included
        retriever = get_retriever()
        semantic_results = retriever.retrieve_with_stats(query, top_k)
        seen = {r["player_name"] for r in stat_results}
        for r in semantic_results:
            if r["player_name"] not in seen and len(stat_results) < top_k:
                stat_results.append(r)
                seen.add(r["player_name"])
        return _format_results(stat_results[:top_k])

    # ── Pure semantic path (no keyword matched) ─────────────────
    retriever = get_retriever()
    best_results = retriever.retrieve_with_stats(query, top_k)
    best_avg = _avg_similarity(best_results)

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
