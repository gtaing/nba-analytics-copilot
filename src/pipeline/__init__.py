"""ETL pipeline for NBA data processing."""

from src.pipeline.ingestion import ingest_raw_data
from src.pipeline.features import build_player_season_features
from src.pipeline.summaries import generate_player_summaries
from src.pipeline.embeddings import build_embeddings

__all__ = [
    "ingest_raw_data",
    "build_player_season_features",
    "generate_player_summaries",
    "build_embeddings",
]
