"""Feature engineering for player statistics."""

from src.db import get_connection


def build_player_season_features():
    """Build aggregated player features table."""
    con = get_connection()
    con.execute("""
        CREATE OR REPLACE TABLE player_season_features AS
        SELECT
            concat(firstName, ' ', lastName) AS player_name,
            GP AS games_played,
            AVG(PTS) AS pts_per_game,
            AVG(REB) AS reb_per_game,
            AVG(AST) AS ast_per_game,
            AVG(STL) AS stl_per_game,
            AVG(BLK) AS blk_per_game,
            AVG(ts) AS true_shooting_pct,
            AVG(AST) / NULLIF(AVG("TO"), 0) AS ast_to_tov_ratio,
            AVG(STL) + AVG(BLK) AS stocks_per_game,
            AVG("3P%") AS three_pt_pct,
            AVG("3PM") AS three_pt_made_per_game,
            AVG("3PA") AS three_pt_attempted_per_game,
            AVG("FG%") AS fg_pct,
            AVG("FT%") AS ft_pct
        FROM raw_player_stats
        GROUP BY player_name, games_played
    """)
    con.close()
    print("✓ Built player season features")
