"""Raw data ingestion from CSV to DuckDB."""

import polars as pl
from src.db import get_connection
from src.config import RAW_DATA_PATH


def ingest_raw_data():
    """Load raw player stats CSV into DuckDB."""
    player_stats = (
        pl.read_csv(RAW_DATA_PATH)
        .with_columns([
            pl.col("firstName").str.replace("'", " ").alias("firstName"),
            pl.col("lastName").str.replace("'", " ").alias("lastName")
        ])
    )


    con = get_connection()
    con.execute(
        "CREATE OR REPLACE TABLE raw_player_stats AS SELECT * FROM player_stats"
    )
    con.close()
    print(f"✓ Ingested raw data from {RAW_DATA_PATH}")
