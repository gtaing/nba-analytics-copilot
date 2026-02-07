"""
NBA Analytics Agent with local LLM support.

This agent uses local LLMs via Ollama or HuggingFace - no API costs required.

Supported backends:
- Ollama: Run models like llama3, mistral, phi3 locally
- HuggingFace: Use models like Flan-T5 directly from transformers

Key Agentic AI concepts demonstrated:
- Tool selection: Agent decides which tool to use based on the question
- Context injection: Retrieved data is passed to the LLM as context
- RAG: Retrieval-Augmented Generation pattern
"""

import re
from typing import Optional

from src.agent.tools import TOOLS, search_players


class NBAAgent:
    """
    An AI agent specialized in NBA analytics.

    The agent uses a two-step process:
    1. RETRIEVE: Use tools to gather relevant data from the database
    2. GENERATE: Use an LLM to synthesize a natural language answer
    """

    def __init__(self, llm_backend: str = "ollama", model_name: Optional[str] = None):
        """
        Initialize the NBA Agent.

        Args:
            llm_backend: "ollama", "huggingface", or "none" (no LLM, raw data only)
            model_name: Model to use (defaults: ollama="llama3.2", hf="google/flan-t5-base")
        """
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.generator = None
        self._setup_llm()

    def _setup_llm(self):
        """Initialize the LLM based on backend choice."""
        if self.llm_backend == "ollama":
            self._setup_ollama()
        # "none" = no LLM setup needed

    def _setup_ollama(self):
        """
        Setup Ollama client.

        Ollama runs LLMs locally. Install it from https://ollama.ai
        Then run: ollama pull llama3.2
        """
        try:
            import httpx
            # Test if Ollama is running
            response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if response.status_code == 200:
                self.model_name = self.model_name or "llama3.2"
                print(f"✓ Connected to Ollama (model: {self.model_name})")
            else:
                print("Warning: Ollama not responding. Falling back to no-LLM mode.")
                self.llm_backend = "none"
        except Exception:
            print("Warning: Ollama not available. Falling back to no-LLM mode.")
            print("  To use Ollama: install from https://ollama.ai, then run 'ollama pull llama3.2'")
            self.llm_backend = "none"

    def _select_tools(self, question: str) -> list[str]:
        """
        Decide which tools to use based on the question.

        This is a simple rule-based tool selector. In a more advanced
        agent, you might use an LLM to decide which tools to call.
        """
        question_lower = question.lower()
        selected = []

        # Always use semantic search for context
        selected.append("search_players")

        # Add specialized tools based on question content
        if any(word in question_lower for word in ["scor", "point", "ppg"]):
            selected.append("get_top_scorers")

        if any(word in question_lower for word in ["defend", "defense", "defensive", "block", "steal"]):
            selected.append("get_top_defenders")

        if any(word in question_lower for word in ["assist", "playmaker", "passing", "pass"]):
            selected.append("get_top_playmakers")

        if any(word in question_lower for word in ["efficient", "shooting", "ts%"]):
            selected.append("get_most_efficient")

        return selected if selected else ["search_players"]

    def _gather_context(self, question: str) -> str:
        """
        Use tools to gather relevant context for the question.
        This is the RETRIEVAL step of RAG.
        """
        selected_tools = self._select_tools(question)
        context_parts = []

        for tool_name in selected_tools:
            if tool_name not in TOOLS:
                continue

            tool = TOOLS[tool_name]
            func = tool["function"]

            if tool_name == "search_players":
                result = func(question, top_k=5)
            else:
                result = func()

            context_parts.append(f"## {tool_name.replace('_', ' ').title()}\n{result}")

        return "\n\n".join(context_parts)

    def _generate_with_ollama(self, question: str, context: str) -> str:
        """Generate answer using Ollama."""
        import httpx

        prompt = f"""You are an NBA analytics expert. Answer this question using ONLY the data provided.

Question: {question}

Data:
{context}

Give a clear, specific answer with statistics. Be concise."""

        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60.0,
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Ollama error: {response.text}"

    def ask(self, question: str, verbose: bool = False) -> str:
        """
        Answer an NBA analytics question.

        Args:
            question: Natural language question about NBA
            verbose: If True, print intermediate steps

        Returns:
            Natural language answer based on the data
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"Selected tools: {self._select_tools(question)}")
            print(f"{'='*60}\n")

        # Step 1: Retrieve context
        context = self._gather_context(question)

        # Step 2: Generate answer
        if self.llm_backend == "ollama":
            return self._generate_with_ollama(question, context)
        elif self.llm_backend == "huggingface":
            return self._generate_with_huggingface(question, context)
        else:
            return self._generate_simple(question, context)

    def ask_with_context(self, question: str) -> tuple[str, str]:
        """Return both the answer and the raw context used."""
        context = self._gather_context(question)

        if self.llm_backend == "ollama":
            answer = self._generate_with_ollama(question, context)
        elif self.llm_backend == "huggingface":
            answer = self._generate_with_huggingface(question, context)
        else:
            answer = self._generate_simple(question, context)

        return answer, context
