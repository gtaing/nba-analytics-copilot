"""
Tools for the NBA Analytics Agent.

In Agentic AI, tools are functions that the agent can call to interact
with external systems (databases, APIs, etc.). The agent decides WHICH
tool to use based on the user's question.

This is a key concept: instead of hardcoding logic, we give the agent
capabilities and let it reason about how to use them.
"""

from src.db import get_connection
from src.retrieval.semantic import SemanticRetriever
from src.config import MIN_GAMES


# Initialize retriever once (singleton pattern for efficiency)
_retriever = None


def get_retriever() -> SemanticRetriever:
    """Get or create the semantic retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = SemanticRetriever()
    return _retriever


def search_players(query: str, top_k: int = 5) -> str:
    """
    Semantic search for players matching a natural language query.

    This tool uses embeddings to find players whose descriptions
    are semantically similar to the query. Great for open-ended
    questions like "best defenders" or "efficient scorers".

    Args:
        query: Natural language description of what to search for
        top_k: Number of results to return

    Returns:
        Formatted string with player summaries and similarity scores
    """
    retriever = get_retriever()
    results = retriever.retrieve_with_stats(query, top_k)

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


def query_stats(sql_filter: str = "", order_by: str = "pts_per_game DESC", limit: int = 10) -> str:
    """
    Query player statistics with custom filters and ordering.

    This tool allows structured queries on the stats database.
    Use this when you need specific statistical rankings.

    Args:
        sql_filter: WHERE clause filter (e.g., "games_played >= 50")
        order_by: ORDER BY clause (e.g., "stl_per_game DESC")
        limit: Max number of results

    Returns:
        Formatted table of player statistics
    """
    con = get_connection()

    where_clause = f"WHERE {sql_filter}" if sql_filter else ""

    query = f"""
        SELECT
            player_name,
            games_played,
            pts_per_game,
            reb_per_game,
            ast_per_game,
            stl_per_game,
            blk_per_game,
            true_shooting_pct,
            stocks_per_game
        FROM player_season_features
        {where_clause}
        ORDER BY {order_by}
        LIMIT {limit}
    """

    try:
        df = con.execute(query).fetchdf()
        con.close()
        return df.to_string(index=False)
    except Exception as e:
        con.close()
        return f"Query error: {e}"


def get_top_scorers(min_games: int = MIN_GAMES, limit: int = 10) -> str:
    """Get top scorers by points per game."""
    return query_stats(
        sql_filter=f"games_played >= {min_games}",
        order_by="pts_per_game DESC",
        limit=limit
    )


def get_top_defenders(min_games: int = MIN_GAMES, limit: int = 10) -> str:
    """
    Get top defenders by stocks (steals + blocks) per game.

    Note: This is a simplified defensive metric. Real defensive
    analysis would use DRTG, DWS, and other advanced stats.
    """
    return query_stats(
        sql_filter=f"games_played >= {min_games}",
        order_by="stocks_per_game DESC",
        limit=limit
    )


def get_top_playmakers(min_games: int = MIN_GAMES, limit: int = 10) -> str:
    """Get top playmakers by assists per game."""
    return query_stats(
        sql_filter=f"games_played >= {min_games}",
        order_by="ast_per_game DESC",
        limit=limit
    )


def get_most_efficient(min_games: int = MIN_GAMES, limit: int = 10) -> str:
    """Get most efficient scorers by true shooting percentage."""
    return query_stats(
        sql_filter=f"games_played >= {min_games}",
        order_by="true_shooting_pct DESC",
        limit=limit
    )


def compare_players(player_names: list[str]) -> str:
    """
    Compare specific players side by side.

    Args:
        player_names: List of player names to compare

    Returns:
        Side-by-side comparison of stats
    """
    con = get_connection()
    placeholders = ",".join(f"'{name}'" for name in player_names)

    df = con.execute(f"""
        SELECT
            player_name,
            games_played,
            pts_per_game,
            reb_per_game,
            ast_per_game,
            stl_per_game,
            blk_per_game,
            true_shooting_pct,
            ast_to_tov_ratio
        FROM player_season_features
        WHERE player_name IN ({placeholders})
    """).fetchdf()
    con.close()

    if df.empty:
        return f"No players found matching: {player_names}"

    return df.to_string(index=False)


# Tool registry for the agent
TOOLS = {
    "search_players": {
        "function": search_players,
        "description": "Semantic search for players matching a natural language query. "
                       "Best for open-ended questions about player types or characteristics.",
    },
    "get_top_scorers": {
        "function": get_top_scorers,
        "description": "Get the top scorers by points per game.",
    },
    "get_top_defenders": {
        "function": get_top_defenders,
        "description": "Get the best defenders by steals + blocks (stocks) per game.",
    },
    "get_top_playmakers": {
        "function": get_top_playmakers,
        "description": "Get the best playmakers by assists per game.",
    },
    "get_most_efficient": {
        "function": get_most_efficient,
        "description": "Get the most efficient scorers by true shooting percentage.",
    },
    "compare_players": {
        "function": compare_players,
        "description": "Compare specific players side by side.",
    },
}
