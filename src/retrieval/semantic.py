"""Semantic retrieval using vector embeddings."""

import numpy as np
from sentence_transformers import SentenceTransformer
from src.db import get_connection
from src.config import EMBEDDING_MODEL


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SemanticRetriever:
    """
    Retriever that uses semantic similarity to find relevant players.

    This implements the 'R' in RAG - finding relevant context based on
    the semantic meaning of the question, not just keyword matching.
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._embeddings_cache = None

    def _load_embeddings(self):
        """Load all embeddings from database (cached)."""
        if self._embeddings_cache is not None:
            return self._embeddings_cache

        con = get_connection()
        rows = con.execute("""
            SELECT e.player_name, e.embedding, s.summary
            FROM player_embeddings e
            JOIN player_summaries s USING (player_name)
        """).fetchall()
        con.close()

        self._embeddings_cache = [
            {
                "player_name": row[0],
                "embedding": np.array(row[1]),
                "summary": row[2],
            }
            for row in rows
        ]
        return self._embeddings_cache

    def retrieve_by_question(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Find players most relevant to the question using semantic similarity.

        This is the key fix: we now actually use the question embedding
        to rank ALL players by relevance, not just filter to top scorers.

        Args:
            question: Natural language question about NBA players
            top_k: Number of results to return

        Returns:
            List of dicts with player_name, summary, and similarity score
        """
        # Encode the question into the same vector space
        question_embedding = self.model.encode(question)

        # Load all player embeddings
        all_players = self._load_embeddings()

        # Compute similarity between question and each player summary
        results = []
        for player in all_players:
            similarity = cosine_similarity(question_embedding, player["embedding"])
            results.append({
                "player_name": player["player_name"],
                "summary": player["summary"],
                "similarity": similarity,
            })

        # Sort by similarity (highest first) and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def retrieve_with_stats(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve players with both their summaries and structured stats.

        Combines semantic retrieval with structured data for richer context.
        """
        # Get semantically relevant players
        relevant = self.retrieve_by_question(question, top_k)
        player_names = [r["player_name"] for r in relevant]

        # Fetch their structured stats
        con = get_connection()
        placeholders = ",".join(f"'{name}'" for name in player_names)
        stats = con.execute(f"""
            SELECT
                player_name,
                games_played,
                pts_per_game,
                reb_per_game,
                ast_per_game,
                stl_per_game,
                blk_per_game,
                true_shooting_pct,
                ast_to_tov_ratio,
                stocks_per_game
            FROM player_season_features
            WHERE player_name IN ({placeholders})
        """).fetchdf()
        con.close()

        # Merge stats into results
        stats_dict = stats.set_index("player_name").to_dict("index")
        for r in relevant:
            if r["player_name"] in stats_dict:
                r["stats"] = stats_dict[r["player_name"]]

        return relevant
