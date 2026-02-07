"""Node functions for the NBA analytics multi-agent graph.

Each node is a function: State → partial State update.

Nodes in the multi-agent system:
- supervisor:   classifies the question, sets the routing decision
- sql_agent:    generates and executes SQL via an internal ReAct loop
- rag_agent:    runs semantic search with retry (Phase 3 enhancement)
- synthesizer:  merges results into a final answer, checks data quality
"""

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from src.graph.tools import query_db, sql_tools
from src.graph.state import NBAState
from src.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL
from src.agent.tools import search_players as raw_search_players


# ── Helpers ─────────────────────────────────────────────────────

def _get_question(state: NBAState) -> str:
    """Extract the original user question (always the first message)."""
    return state["messages"][0].content


def _make_llm(model_name: str | None = None):
    """Create a base ChatOllama instance."""
    return ChatOllama(
        model=model_name or DEFAULT_OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# ── Supervisor ──────────────────────────────────────────────────

SUPERVISOR_PROMPT = """\
Classify this NBA analytics question into ONE category.
Respond with EXACTLY one word — no explanation.

- sql: for questions that need numerical ranking or filtering
  (e.g. "best scorers", "best defenders", "who had the most blocks",
   "top rebounders", "compare LeBron and Curry", "highest shooting percentage")
  Any question with "best", "top", "most", "highest", "lowest" is sql.
- semantic: for purely descriptive questions with no ranking
  (e.g. "describe LeBron's play style", "what kind of player is Curry")
- both: when unsure, or when the question needs stats AND descriptions

Respond with one word: sql, semantic, or both"""


def create_supervisor(model_name: str | None = None):
    """Create the supervisor node.

    The supervisor makes a single lightweight LLM call to classify the
    question. This replaces the old keyword-matching _select_tools().

    On retry (iteration > 0), it always picks "both" for maximum coverage.
    """
    llm = _make_llm(model_name)

    def supervisor(state: NBAState) -> dict:
        iteration = state.get("iteration", 0)

        # On retry, widen the net — use both agents
        if iteration > 0:
            return {"route": "both"}

        question = _get_question(state)
        response = llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=question),
        ])

        # Parse the classification loosely — default to "both" on ambiguity
        text = response.content.strip().lower()
        if "sql" in text and "semantic" not in text and "both" not in text:
            route = "sql"
        elif "semantic" in text and "sql" not in text and "both" not in text:
            route = "semantic"
        else:
            route = "both"

        return {"route": route}

    return supervisor


# ── SQL Agent ───────────────────────────────────────────────────

SQL_AGENT_PROMPT = """\
You are an NBA SQL analyst. Given a question, write a SQL query to answer it.
You MUST call the query_db tool with your SQL query. Do NOT answer in text.

Available table: player_season_features (2016 season only — do NOT filter by year)
Columns:
  player_name         VARCHAR  — full name, e.g. 'LeBron James'
  games_played        BIGINT   — number of games in the season
  pts_per_game        DOUBLE   — points per game
  reb_per_game        DOUBLE   — rebounds per game
  ast_per_game        DOUBLE   — assists per game
  stl_per_game        DOUBLE   — steals per game (defensive stat)
  blk_per_game        DOUBLE   — blocks per game (defensive stat)
  true_shooting_pct   DOUBLE   — true shooting percentage (0-1)
  ast_to_tov_ratio    DOUBLE   — assist-to-turnover ratio
  stocks_per_game     DOUBLE   — steals + blocks combined (defensive stat)

Tips:
- Use stocks_per_game to rank defenders (steals + blocks).
- Filter games_played >= 50 for meaningful per-game averages.
- Use single quotes for strings: WHERE player_name = 'LeBron James'
- Always SELECT the relevant stat columns alongside player_name so the data is useful.
- For "best" or "top" questions, use ORDER BY ... DESC LIMIT 10.

If a query errors, read the error and try a corrected query."""


def create_sql_agent(model_name: str | None = None):
    """Create the SQL specialist agent.

    Runs an internal ReAct loop (up to 3 attempts): the LLM generates SQL
    via tool calling, the tool executes it, and on error the LLM sees the
    error message and can retry with corrected SQL.

    Writes results to state["sql_result"] — not to messages.
    """
    llm = _make_llm(model_name).bind_tools(sql_tools)

    def sql_agent(state: NBAState) -> dict:
        question = _get_question(state)
        internal_msgs = [
            SystemMessage(content=SQL_AGENT_PROMPT),
            HumanMessage(content=question),
        ]

        last_result = ""
        ERROR_PREFIXES = ("SQL error:", "Error:", "Query returned no")

        for _ in range(3):
            response = llm.invoke(internal_msgs)
            internal_msgs.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                result = query_db.invoke(tc["args"])
                last_result = result
                internal_msgs.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

                # Stop retrying on success
                if not result.startswith(ERROR_PREFIXES):
                    return {"sql_result": result}

        return {"sql_result": last_result}

    return sql_agent


# ── RAG Agent ───────────────────────────────────────────────────

def create_rag_agent():
    """Create the RAG (semantic search) specialist agent.

    No LLM call needed — this agent calls the search_players function
    directly. The Phase 3 retry/rephrase logic is built into that function.
    Interpretation happens later in the synthesizer.

    Writes results to state["rag_result"] — not to messages.
    """

    def rag_agent(state: NBAState) -> dict:
        question = _get_question(state)
        result = raw_search_players(question)
        return {"rag_result": result}

    return rag_agent


# ── Synthesizer ─────────────────────────────────────────────────

SYNTHESIZER_PROMPT = """\
You are an NBA analytics expert. Synthesize a clear, concise answer from the
data below. Rules:
1. ONLY use facts present in the provided data. Never invent statistics.
2. If data is empty or contains errors, say so honestly.
3. Lead with the direct answer, then cite supporting numbers.
4. Be concise — no filler."""


def create_synthesizer(model_name: str | None = None):
    """Create the synthesizer node.

    Takes the structured outputs from the specialist agents (sql_result,
    rag_result) and uses the LLM to produce a coherent final answer.

    Also increments the iteration counter for the feedback loop.
    """
    llm = _make_llm(model_name)

    def synthesizer(state: NBAState) -> dict:
        question = _get_question(state)
        sql_result = state.get("sql_result", "")
        rag_result = state.get("rag_result", "")
        iteration = state.get("iteration", 0)

        # Build context from available results
        context_parts = []
        if sql_result:
            context_parts.append(f"=== SQL Query Results ===\n{sql_result}")
        if rag_result:
            context_parts.append(f"=== Semantic Search Results ===\n{rag_result}")

        if not context_parts:
            from langchain_core.messages import AIMessage
            return {
                "messages": [AIMessage(
                    content="I could not retrieve relevant data for this question."
                )],
                "iteration": iteration + 1,
            }

        context = "\n\n".join(context_parts)

        response = llm.invoke([
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(
                content=f"Question: {question}\n\nAvailable Data:\n{context}"
            ),
        ])

        return {
            "messages": [response],
            "iteration": iteration + 1,
        }

    return synthesizer
