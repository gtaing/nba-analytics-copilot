"""Tests for the tool layer: SQL validation guards and formatting helpers.

Strategy:
- execute_sql validation (SELECT-only, forbidden keywords, auto LIMIT) is tested
  WITHOUT a database — the function returns error strings before ever touching the DB.
- For the execution path, we mock get_connection() to control what the DB returns.
- _avg_similarity and _format_results are pure functions — no mocking needed.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.agent.tools import execute_sql, _avg_similarity, _format_results


# ── execute_sql: validation guards ──────────────────────────────
# These tests verify the regex checks BEFORE any DB call happens.
# No database or LLM is involved.


class TestSQLValidation:
    """Test that execute_sql rejects dangerous queries."""

    def test_rejects_insert(self):
        result = execute_sql("INSERT INTO player_season_features VALUES ('x')")
        assert "Error" in result

    def test_rejects_delete(self):
        result = execute_sql("DELETE FROM player_season_features")
        assert "Error" in result

    def test_rejects_drop(self):
        result = execute_sql("DROP TABLE player_season_features")
        assert "Error" in result

    def test_rejects_update(self):
        result = execute_sql("UPDATE player_season_features SET pts_per_game = 0")
        assert "Error" in result

    def test_rejects_create(self):
        result = execute_sql("CREATE TABLE hacked (id INT)")
        assert "Error" in result

    def test_rejects_truncate(self):
        result = execute_sql("TRUNCATE TABLE player_season_features")
        assert "Error" in result

    def test_rejects_plain_text(self):
        """A non-SQL string should be rejected (not a SELECT)."""
        result = execute_sql("hello world")
        assert result == "Error: only SELECT queries are allowed."

    def test_rejects_mutation_hidden_in_select(self):
        """Mutation keyword inside a SELECT should still be blocked."""
        result = execute_sql("SELECT * FROM player_season_features; DROP TABLE x")
        assert "Error" in result


# ── execute_sql: LIMIT enforcement ──────────────────────────────
# We mock the DB to verify that LIMIT is auto-appended.


class TestSQLLimitEnforcement:
    """Test that execute_sql auto-appends LIMIT when missing."""

    @patch("src.agent.tools.get_connection")
    def test_auto_appends_limit(self, mock_get_conn):
        """Query without LIMIT should get LIMIT 50 appended."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
            {"player_name": ["Test Player"]}
        )
        mock_get_conn.return_value = mock_conn

        execute_sql("SELECT player_name FROM player_season_features")

        executed_sql = mock_conn.execute.call_args[0][0]
        assert "LIMIT 50" in executed_sql

    @patch("src.agent.tools.get_connection")
    def test_preserves_existing_limit(self, mock_get_conn):
        """Query that already has LIMIT should NOT get a second one."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
            {"player_name": ["Test Player"]}
        )
        mock_get_conn.return_value = mock_conn

        execute_sql("SELECT player_name FROM player_season_features LIMIT 5")

        executed_sql = mock_conn.execute.call_args[0][0]
        assert executed_sql.count("LIMIT") == 1

    @patch("src.agent.tools.get_connection")
    def test_empty_result_message(self, mock_get_conn):
        """Empty DataFrame should return a clear message, not crash."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame()
        mock_get_conn.return_value = mock_conn

        result = execute_sql(
            "SELECT * FROM player_season_features WHERE player_name = 'Nobody'"
        )
        assert result == "Query returned no results."

    @patch("src.agent.tools.get_connection")
    def test_sql_error_is_caught(self, mock_get_conn):
        """DB exceptions should be returned as error strings, not raised."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("no such column: fake_col")
        mock_get_conn.return_value = mock_conn

        result = execute_sql("SELECT fake_col FROM player_season_features")
        assert result.startswith("SQL error:")
        assert "fake_col" in result


# ── _avg_similarity ─────────────────────────────────────────────
# Pure function — no mocking needed.


class TestAvgSimilarity:
    def test_empty_list(self):
        assert _avg_similarity([]) == 0.0

    def test_single_result(self):
        assert _avg_similarity([{"similarity": 0.8}]) == 0.8

    def test_multiple_results(self):
        results = [{"similarity": 0.6}, {"similarity": 0.4}]
        assert _avg_similarity(results) == pytest.approx(0.5)


# ── _format_results ─────────────────────────────────────────────
# Pure function — no mocking needed.


class TestFormatResults:
    def test_includes_player_name(self):
        results = [
            {
                "player_name": "LeBron James",
                "similarity": 0.95,
                "summary": "Great player",
                "stats": {
                    "pts_per_game": 25.3,
                    "reb_per_game": 7.4,
                    "ast_per_game": 8.7,
                    "stl_per_game": 1.2,
                    "blk_per_game": 0.6,
                },
            }
        ]
        formatted = _format_results(results)
        assert "LeBron James" in formatted
        assert "25.3 PPG" in formatted
        assert "0.95" in formatted

    def test_missing_stats_default_to_zero(self):
        """Results without a stats dict should show 0.0 for all stats."""
        results = [
            {
                "player_name": "Bench Player",
                "similarity": 0.10,
                "summary": "Rarely plays",
            }
        ]
        formatted = _format_results(results)
        assert "0.0 PPG" in formatted

    def test_numbering(self):
        """Multiple results should be numbered 1., 2., etc."""
        results = [
            {"player_name": "A", "similarity": 0.9, "summary": "a"},
            {"player_name": "B", "similarity": 0.8, "summary": "b"},
        ]
        formatted = _format_results(results)
        assert formatted.startswith("1. A")
        assert "2. B" in formatted
