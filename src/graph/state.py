"""Shared state definition for the NBA analytics graph.

The State is a TypedDict that flows through every node. Each node reads the
fields it needs and writes its outputs back — the same model as XCom in
Airflow or asset materialization in Dagster.

Fields use two merge strategies:
- `messages` has the add_messages reducer (appends, never overwrites).
- All other fields use last-write-wins (the default for TypedDict fields).
  This works because each specialist agent writes to its own field —
  sql_agent writes sql_result, rag_agent writes rag_result — no conflicts
  even during parallel execution via Send.
"""

from langgraph.graph import MessagesState


class NBAState(MessagesState):
    """Graph state for the NBA analytics multi-agent system.

    messages:    conversation history (inherited, with add_messages reducer)
    route:       supervisor's routing decision ("sql" | "semantic" | "both")
    sql_result:  output from the SQL agent
    rag_result:  output from the RAG agent
    iteration:   feedback loop counter (incremented by synthesizer)
    """

    route: str
    sql_result: str
    rag_result: str
    iteration: int
