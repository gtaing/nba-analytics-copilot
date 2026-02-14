"""Generate text summaries for each player."""

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple
from src.db import get_connection


@dataclass(frozen=True)
class PlayerSeasonStats:
    player_name: str
    games_played: int
    pts_per_game: float
    reb_per_game: float
    ast_per_game: float
    stl_per_game: float
    blk_per_game: float
    true_shooting_pct: float
    ast_to_tov_ratio: float
    stocks_per_game: float

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "PlayerSeasonStats":
        return cls(
            player_name=row[0],
            games_played=row[1],
            pts_per_game=row[2],
            reb_per_game=row[3],
            ast_per_game=row[4],
            stl_per_game=row[5],
            blk_per_game=row[6],
            true_shooting_pct=row[7],
            ast_to_tov_ratio=row[8],
            stocks_per_game=row[9],
        )


def generate_player_summaries() -> None:
    """Create human-readable summaries for semantic embedding."""
    con: Any = get_connection()

    rows: Sequence[Sequence[Any]] = con.execute("""
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

    players: List[PlayerSeasonStats] = [PlayerSeasonStats.from_row(r) for r in rows]

    def _build_labels(player: PlayerSeasonStats) -> List[str]:
        """Assign qualitative labels based on stat thresholds."""

        rules: List[Tuple[Callable[[PlayerSeasonStats], bool], str]] = [
            (lambda p: p.pts_per_game >= 20, "elite scorer"),
            (lambda p: p.reb_per_game >= 10, "dominant rebounder"),
            (lambda p: p.ast_per_game >= 7, "elite playmaker"),
            (lambda p: p.stocks_per_game >= 2.5, "elite defender"),
            (lambda p: p.blk_per_game >= 1.5, "rim protector"),
            (lambda p: p.stl_per_game >= 1.5, "ball hawk"),
            (lambda p: p.true_shooting_pct >= 0.60, "efficient shooter"),
        ]

        return [label for predicate, label in rules if predicate(player)]

    def make_summary(player: PlayerSeasonStats) -> str:
        labels = _build_labels(player)
        label_line = f" He is known as an {', '.join(labels)}." if labels else ""

        return (
            f"{player.player_name} played {player.games_played} games in the 2016 season."
            f"{label_line} "
            f"He averaged {player.pts_per_game:.1f} points, "
            f"{player.reb_per_game:.1f} rebounds, and "
            f"{player.ast_per_game:.1f} assists per game. "
            f"His true shooting percentage was "
            f"{player.true_shooting_pct * 100:.1f}%. "
            f"Defensively, he averaged {player.stl_per_game:.1f} steals "
            f"and {player.blk_per_game:.1f} blocks per game "
            f"({player.stocks_per_game:.1f} stocks combined). "
            f"His assist-to-turnover ratio was "
            f"{player.ast_to_tov_ratio:.2f}."
        )

    summaries: List[Tuple[str, str]] = [
        (p.player_name, make_summary(p)) for p in players
    ]

    con.execute("""
        CREATE OR REPLACE TABLE player_summaries (
            player_name TEXT,
            summary TEXT
        )
    """)

    con.executemany(
        "INSERT INTO player_summaries VALUES (?, ?)",
        summaries,
    )

    con.close()
    print(f"✓ Generated {len(summaries)} player summaries")
