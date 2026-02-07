"""LangChain tool wrappers over existing NBA analytics functions.

Two tools:
- search_players: semantic vector search (meaning-based retrieval)
- query_db: dynamic SQL against DuckDB (structured data queries)

The @tool decorator generates a JSON schema from each function's signature
and docstring, which the LLM reads to decide when and how to call each tool.
"""

from langchain_core.tools import tool

from src.agent.tools import (
    search_players as _search_players,
    execute_sql as _execute_sql,
)


@tool
def search_players(query: str, top_k: int = 5) -> str:
    """Search for NBA players using semantic similarity.

    Uses vector embeddings to find players whose profile matches the query.
    Best for open-ended questions about player types or characteristics.

    Args:
        query: Natural language description of what to search for.
        top_k: Number of results to return.
    """
    return _search_players(query, top_k)


@tool
def query_db(sql_query: str) -> str:
    """Execute a SQL SELECT query against the NBA stats database (DuckDB).

    Use this tool to answer any statistical question about NBA players.
    Write a SQL query that retrieves the data needed to answer the question.
    Only SELECT queries are allowed — no mutations.

    Available tables and their columns:

    Table: player_season_features (487 rows — one per player, aggregated)
        player_name         VARCHAR   — full name, e.g. 'LeBron James'
        games_played        BIGINT    — number of games in the season
        pts_per_game        DOUBLE    — points per game
        reb_per_game        DOUBLE    — rebounds per game
        ast_per_game        DOUBLE    — assists per game
        stl_per_game        DOUBLE    — steals per game
        blk_per_game        DOUBLE    — blocks per game
        true_shooting_pct   DOUBLE    — true shooting % (0-1 scale)
        ast_to_tov_ratio    DOUBLE    — assist-to-turnover ratio
        tov_per_game        DOUBLE    - turnover per game
        stocks_per_game     DOUBLE    — steals + blocks per game
        three_pt_pct        DOUBLE    — three-point shooting percentage (0-1)
        three_pt_made_per_game   DOUBLE — three-pointers made per game
        three_pt_attempted_per_game DOUBLE — three-point attempts per game
        fg_pct              DOUBLE    — field goal percentage (0-1)
        ft_pct              DOUBLE    — free throw percentage (0-1)

    Tips:
    - Use player_season_features for all queries (clean column names, no quoting needed).
    - Use single quotes for string values: WHERE player_name = 'LeBron James'
    - Filter with games_played >= 50 for meaningful per-game averages.
    - Results are capped at 50 rows.

    Args:
        sql_query: A valid SQL SELECT statement.
    """
    return _execute_sql(sql_query)


# Per-agent tool lists — each specialist gets only its own tools.
# This focuses the LLM: the SQL agent only sees query_db, so it can't
# accidentally fall back to semantic search (and vice versa).
sql_tools = [query_db]
rag_tools = [search_players]

# Full list for any node that needs all tools.
nba_tools = [search_players, query_db]
