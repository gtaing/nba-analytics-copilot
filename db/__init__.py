from pathlib import Path

# Expose DB path objects for consumers
DB_DIR = Path(__file__).resolve().parent
DB_FILE = DB_DIR / "nba.duckdb"

__all__ = ["DB_DIR", "DB_FILE"]
