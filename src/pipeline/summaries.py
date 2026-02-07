"""Generate text summaries for each player."""

from src.db import get_connection


def generate_player_summaries():
    """Create human-readable summaries for semantic embedding."""
    con = get_connection()

    rows = con.execute("""
        SELECT
            player_name,
            games_played,
            pts_per_game,
            reb_per_game,
            ast_per_game,
            stl_per_game,
            blk_per_game,
            true_shooting_pct,
            COALESCE(ast_to_tov_ratio, 0) AS ast_to_tov_ratio,
            stocks_per_game
        FROM player_season_features
    """).fetchall()

    def _build_labels(pts, reb, ast, stl, blk, ts, stocks):
        """Assign qualitative labels based on stat thresholds."""
        labels = []
        if pts >= 20:
            labels.append("elite scorer")
        if reb >= 10:
            labels.append("dominant rebounder")
        if ast >= 7:
            labels.append("elite playmaker")
        if stocks >= 2.5:
            labels.append("elite defender")
        if blk >= 1.5:
            labels.append("rim protector")
        if stl >= 1.5:
            labels.append("ball hawk")
        if ts >= 0.60:
            labels.append("efficient shooter")
        return labels

    def make_summary(r):
        name, gp, pts, reb, ast, stl, blk, ts, ast_tov, stocks = r

        labels = _build_labels(pts, reb, ast, stl, blk, ts, stocks)
        label_line = ""
        if labels:
            label_line = f" He is known as an {', '.join(labels)}."

        return (
            f"{name} played {gp} games in the 2016 season."
            f"{label_line} "
            f"He averaged {pts:.1f} points, {reb:.1f} rebounds, and {ast:.1f} assists per game. "
            f"His true shooting percentage was {ts*100:.1f}%. "
            f"Defensively, he averaged {stl:.1f} steals and {blk:.1f} blocks per game "
            f"({stocks:.1f} stocks combined). "
            f"His assist-to-turnover ratio was {ast_tov:.2f}."
        )

    summaries = [(r[0], make_summary(r)) for r in rows]

    con.execute("""
        CREATE OR REPLACE TABLE player_summaries (
            player_name TEXT,
            summary TEXT
        )
    """)

    con.executemany("INSERT INTO player_summaries VALUES (?, ?)", summaries)
    con.close()
    print(f"✓ Generated {len(summaries)} player summaries")
