"""Build vector embeddings for player summaries."""

from sentence_transformers import SentenceTransformer
from src.db import get_connection
from src.config import EMBEDDING_MODEL


def build_embeddings():
    """Encode player summaries into vector embeddings."""
    con = get_connection()
    model = SentenceTransformer(EMBEDDING_MODEL)

    rows = con.execute("""
        SELECT player_name, summary
        FROM player_summaries
    """).fetchall()

    print(f"  Encoding {len(rows)} summaries...")
    embeddings = [
        (name, model.encode(summary).tolist()) for name, summary in rows
    ]

    con.execute("""
        CREATE OR REPLACE TABLE player_embeddings (
            player_name TEXT,
            embedding FLOAT[]
        )
    """)

    con.executemany("INSERT INTO player_embeddings VALUES (?, ?)", embeddings)
    con.close()
    print(f"✓ Built embeddings ({EMBEDDING_MODEL})")
