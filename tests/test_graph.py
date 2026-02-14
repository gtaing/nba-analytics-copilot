"""Tests for graph routing logic, confidence checks, and supervisor parsing.

Strategy:
- route_question and check_confidence are pure functions (state dict → decision).
  No mocking needed — we just pass in hand-crafted state dicts.
- The supervisor contains LLM-dependent logic, so we mock ChatOllama to return
  controlled responses and verify the parsing produces the correct route.
"""

from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END

from src.graph.builder import route_question, check_confidence
from src.graph.nodes import create_supervisor
from src.config import MAX_ITERATIONS


# ── Helpers ─────────────────────────────────────────────────────


def _make_state(**overrides) -> dict:
    """Build a minimal NBAState dict for testing."""
    state = {
        "messages": [HumanMessage(content="test question")],
        "route": "both",
        "sql_result": "",
        "rag_result": "",
        "iteration": 0,
    }
    state.update(overrides)
    return state


# ── route_question ──────────────────────────────────────────────
# Pure function: reads state["route"], returns Send objects.


class TestRouteQuestion:
    def test_sql_dispatches_sql_agent_only(self):
        sends = route_question(_make_state(route="sql"))
        assert len(sends) == 1
        assert sends[0].node == "sql_agent"

    def test_semantic_dispatches_rag_agent_only(self):
        sends = route_question(_make_state(route="semantic"))
        assert len(sends) == 1
        assert sends[0].node == "rag_agent"

    def test_both_dispatches_two_agents(self):
        sends = route_question(_make_state(route="both"))
        assert len(sends) == 2
        nodes = {s.node for s in sends}
        assert nodes == {"sql_agent", "rag_agent"}

    def test_unknown_route_defaults_to_rag(self):
        """Any unrecognized route falls through to the else branch (rag_agent)."""
        sends = route_question(_make_state(route="garbage"))
        assert len(sends) == 1
        assert sends[0].node == "rag_agent"


# ── check_confidence ────────────────────────────────────────────
# Pure function: reads sql_result, rag_result, iteration → END or "supervisor".


class TestCheckConfidence:
    def test_ends_when_sql_has_data(self):
        state = _make_state(sql_result="player_name\nLeBron James")
        assert check_confidence(state) == END

    def test_ends_when_rag_has_data(self):
        state = _make_state(rag_result="1. LeBron James (similarity: 0.9)")
        assert check_confidence(state) == END

    def test_retries_when_both_empty(self):
        state = _make_state(sql_result="", rag_result="", iteration=0)
        assert check_confidence(state) == "supervisor"

    def test_retries_on_sql_error(self):
        state = _make_state(sql_result="SQL error: no such column", rag_result="")
        assert check_confidence(state) == "supervisor"

    def test_retries_on_no_results(self):
        state = _make_state(sql_result="Query returned no results.", rag_result="")
        assert check_confidence(state) == "supervisor"

    def test_ends_at_max_iterations_even_with_no_data(self):
        """Safety cap: stop retrying after MAX_ITERATIONS regardless of data."""
        state = _make_state(sql_result="", rag_result="", iteration=MAX_ITERATIONS)
        assert check_confidence(state) == END

    def test_ends_when_only_rag_whitespace(self):
        """Whitespace-only RAG result is treated as empty → retry."""
        state = _make_state(sql_result="", rag_result="   ", iteration=0)
        assert check_confidence(state) == "supervisor"


# ── Supervisor response parsing ─────────────────────────────────
# We mock the LLM to return exact strings and verify the parsing
# logic maps them to the correct route. This tests the regex/string
# matching, not the LLM's ability to classify.


class TestSupervisorParsing:
    """Test that the supervisor correctly parses LLM classification responses."""

    def _run_supervisor(self, llm_response_text: str, iteration: int = 0) -> str:
        """Helper: create supervisor, mock LLM to return given text, return route."""
        with patch("src.graph.nodes._make_llm") as mock_make_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content=llm_response_text)
            mock_make_llm.return_value = mock_llm

            supervisor_fn = create_supervisor("fake-model")
            state = _make_state(iteration=iteration)
            result = supervisor_fn(state)
            return result["route"]

    def test_parses_sql(self):
        assert self._run_supervisor("sql") == "sql"

    def test_parses_semantic(self):
        assert self._run_supervisor("semantic") == "semantic"

    def test_parses_both(self):
        assert self._run_supervisor("both") == "both"

    def test_case_insensitive(self):
        assert self._run_supervisor("SQL") == "sql"
        assert self._run_supervisor("SEMANTIC") == "semantic"

    def test_extra_whitespace(self):
        assert self._run_supervisor("  sql  \n") == "sql"

    def test_ambiguous_defaults_to_both(self):
        """If the LLM returns something unexpected, default to 'both'."""
        assert self._run_supervisor("I think sql and semantic") == "both"
        assert self._run_supervisor("not sure") == "both"

    def test_retry_always_returns_both(self):
        """On retry (iteration > 0), supervisor skips LLM and returns 'both'."""
        assert self._run_supervisor("sql", iteration=1) == "both"
        assert self._run_supervisor("semantic", iteration=2) == "both"
