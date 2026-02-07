"""Graph construction for the NBA analytics multi-agent system.

The graph implements a supervisor pattern with parallel dispatch:

    START ──▶ supervisor ──▶ route_question
                               ├─ "sql"      ──▶ sql_agent ──────────┐
                               ├─ "semantic"  ──▶ rag_agent ──────────┤
                               └─ "both"     ──▶ sql_agent ─┐        │
                                                 rag_agent ─┘(Send)  │
                                                                      ▼
                                                              synthesizer
                                                                   │
                                               ┌───────────────────┤
                                               ▼                   ▼
                                          (sufficient)        (insufficient
                                              END           & iteration < max)
                                                                   │
                                                                   ▼
                                                              supervisor
                                                              (retry)

Send dispatches both agents in parallel. Each writes to its own state
field (sql_result / rag_result). After both complete, their state updates
merge and the synthesizer sees both results.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src.graph.state import NBAState
from src.graph.nodes import (
    create_supervisor,
    create_sql_agent,
    create_rag_agent,
    create_synthesizer,
)
from src.config import MAX_ITERATIONS


def route_question(state: NBAState):
    """Conditional edge after supervisor: dispatch to specialist(s).

    Returns either a node name (single dispatch) or a list of Send objects
    (parallel dispatch). LangGraph handles both:
    - String → single node execution
    - [Send, Send] → parallel execution, states merge after completion
    """
    route = state.get("route", "both")

    if route == "both":
        return [Send("sql_agent", state), Send("rag_agent", state)]
    elif route == "sql":
        return [Send("sql_agent", state)]
    else:
        return [Send("rag_agent", state)]


def check_confidence(state: NBAState) -> str:
    """Conditional edge after synthesizer: finish or retry.

    Checks whether the specialist agents returned meaningful data.
    If not (and we haven't hit the iteration cap), routes back to the
    supervisor for another attempt with broader coverage.
    """
    sql = state.get("sql_result", "")
    rag = state.get("rag_result", "")
    iteration = state.get("iteration", 0)

    has_sql = sql and not sql.startswith(("SQL error:", "Error:", "Query returned no"))
    has_rag = bool(rag and rag.strip())

    if (has_sql or has_rag) or iteration >= MAX_ITERATIONS:
        return END
    return "supervisor"


def build_graph(model_name: str | None = None):
    """Build and compile the NBA analytics multi-agent graph.

    Args:
        model_name: Ollama model to use (defaults to config.DEFAULT_OLLAMA_MODEL).

    Returns:
        A compiled LangGraph ready to .invoke() or .stream().
    """
    graph = StateGraph(NBAState)

    # Register nodes
    graph.add_node("supervisor", create_supervisor(model_name))
    graph.add_node("sql_agent", create_sql_agent(model_name))
    graph.add_node("rag_agent", create_rag_agent())
    graph.add_node("synthesizer", create_synthesizer(model_name))

    # Wire edges
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_question)
    graph.add_edge("sql_agent", "synthesizer")
    graph.add_edge("rag_agent", "synthesizer")
    graph.add_conditional_edges("synthesizer", check_confidence, {END: END, "supervisor": "supervisor"})

    return graph.compile()
