"""
backend/agents/graphiti_agent.py

LangGraph RAG agent — same state machine as agentic-graph-rag's graph_agent.py
with two changes:

  1. retrieve_service.retrieve_hybrid() is now async (Graphiti.search() is async).
     The _retrieve node awaits it via asyncio.run() so the sync LangGraph
     interface stays unchanged.

  2. graph_service.structured_retriever() is gone — Graphiti does that
     internally. The agent just calls retrieve_hybrid() and gets back a
     combined list of text strings (graph facts + FAISS chunks).

Everything else — router, HyDE, grade_documents, transform_query, generate,
calculator tool, all state fields — is identical to the original.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from backend.core.config import settings
from backend.tools.calculator import calculate
from backend.tools.query_expander import generate_hyde_document

logger = logging.getLogger("graphiti-rag.agent")


class AgentState(TypedDict):
    question:          str
    original_question: str
    chat_history:      str
    documents:         List[str]
    decision:          str
    generation:        str
    steps:             List[str]
    retry_count:       int
    mode:              str
    temperature:       float
    max_tokens:        int


class GraphitiRAGAgent:
    """
    LangGraph agent backed by Graphiti (temporal graph) + FAISS hybrid retrieval.
    Graph compiled once at construction — same FIX 1 as the original.
    """

    def __init__(self, retrieve_service, model_name: str = settings.OLLAMA_MODEL):
        self.retrieve_service = retrieve_service
        self.model_name       = model_name
        self.max_retries      = settings.MAX_ITERATIONS

        self._json_llm = ChatOllama(model=model_name, temperature=0, format="json")
        self._llm      = ChatOllama(model=model_name, temperature=0)
        self._app      = self._build_graph()
        logger.info("GraphitiRAGAgent ready (model=%s)", model_name)

    # ------------------------------------------------------------------
    # Internal helpers — identical to original
    # ------------------------------------------------------------------

    def _invoke_json(self, prompt: str, fallback: Dict) -> Dict:
        """Strip markdown fences before JSON parsing — same FIX 3 as original."""
        try:
            res     = self._json_llm.invoke([HumanMessage(content=prompt)])
            content = res.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            logger.warning("JSON parse failed: %s", e)
            return fallback

    def _writer(self, temperature: float, max_tokens: int) -> ChatOllama:
        return ChatOllama(model=self.model_name, temperature=temperature, num_predict=max_tokens)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _router(self, state: AgentState) -> Dict:
        logger.info("--- ROUTER ---")
        question = state.get("original_question", state["question"])
        prompt = (
            f"You are a router.\n"
            f"1. If user asks for info/facts/summary output 'vectorstore'.\n"
            f"2. If user says hi/hello/thanks output 'chitchat'.\n"
            f"Question: {question}\n"
            f'Return JSON: {{"datasource": "vectorstore" | "chitchat"}}'
        )
        result   = self._invoke_json(prompt, {"datasource": "vectorstore"})
        decision = result.get("datasource", "vectorstore")
        return {"decision": decision, "steps": ["router"]}

    def _chitchat(self, state: AgentState) -> Dict:
        logger.info("--- CHITCHAT ---")
        prompt = (
            f"Previous chat:\n{state.get('chat_history', '')}\n\n"
            f"User: {state['original_question']}\n"
            f"Reply politely and conversationally."
        )
        reply = self._writer(state["temperature"], state["max_tokens"]).invoke(
            [HumanMessage(content=prompt)]
        ).content
        return {"generation": reply, "steps": ["chitchat"]}

    def _retrieve(self, state: AgentState) -> Dict:
        """
        Calls retrieve_hybrid() — now async because Graphiti.search() is async.
        Uses asyncio.run() to call from the sync LangGraph context.
        """
        logger.info("--- RETRIEVE (mode=%s) ---", state["mode"])
        top_k = settings.TOP_K_RETRIEVAL * 2 if state["mode"] == "detailed" else settings.TOP_K_RETRIEVAL
        search_query = state["question"]

        if settings.USE_HYDE:
            try:
                search_query = generate_hyde_document(state["question"])
            except Exception as e:
                logger.warning("HyDE failed: %s — using raw query", e)

        try:
            # retrieve_hybrid is async; we're inside a sync LangGraph node
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're inside an existing event loop (e.g. FastAPI) — use a thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run,
                            self.retrieve_service.retrieve_hybrid(search_query, top_k=top_k)
                        )
                        docs = future.result()
                else:
                    docs = loop.run_until_complete(
                        self.retrieve_service.retrieve_hybrid(search_query, top_k=top_k)
                    )
            except RuntimeError:
                docs = asyncio.run(
                    self.retrieve_service.retrieve_hybrid(search_query, top_k=top_k)
                )
        except Exception as e:
            logger.error("Retrieval error: %s", e)
            docs = []

        return {"documents": docs, "steps": ["retrieve"]}

    def _grade_documents(self, state: AgentState) -> Dict:
        logger.info("--- GRADE DOCUMENTS ---")
        if not state["documents"]:
            return {"documents": []}
        doc_txt = "\n\n".join(
            [f"[{i}] {d[:300]}..." for i, d in enumerate(state["documents"])]
        )
        prompt = (
            f"Identify relevant docs for: {state['question']}\n"
            f"Docs:\n{doc_txt}\n"
            f'Return JSON {{"indices": [0, 2, ...]}} of relevant docs. '
            f"If unsure, include the document."
        )
        result  = self._invoke_json(prompt, {"indices": list(range(len(state["documents"])))})
        indices = result.get("indices", [])
        try:
            filtered = [state["documents"][i] for i in indices if i < len(state["documents"])]
        except Exception:
            filtered = state["documents"]
        return {"documents": filtered, "steps": ["grade_documents"]}

    def _transform_query(self, state: AgentState) -> Dict:
        logger.info("--- TRANSFORM QUERY ---")
        prompt = (
            f"Context: {state.get('chat_history', '')}\n"
            f"User Question: {state['question']}\n\n"
            f"Rewrite to be standalone and search-friendly. "
            f"Replace pronouns with specific names from context if possible.\n"
            f"Output ONLY the rewritten question string."
        )
        new_q = self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"question": new_q, "retry_count": state["retry_count"] + 1}

    def _generate(self, state: AgentState) -> Dict:
        logger.info("--- GENERATE (mode=%s) ---", state["mode"])
        context  = "\n\n".join(state["documents"])
        question = state["original_question"]
        history  = state.get("chat_history", "")

        if state["mode"] == "detailed":
            system_prompt = (
                f"You are a comprehensive analyst. Provide a detailed answer "
                f"using up to {state['max_tokens']} tokens. Cover all aspects. "
                f"Do NOT output raw JSON or mention tool names."
            )
        else:
            system_prompt = "You are a concise assistant. Answer directly and briefly. Do not output JSON."

        prompt = f"""{system_prompt}

Relevant Context:
{context}

Chat History:
{history}

Question: {question}
Answer:"""

        writer         = self._writer(state["temperature"], state["max_tokens"])
        writer_w_tools = writer.bind_tools([calculate])
        messages       = [HumanMessage(content=prompt)]
        response       = writer_w_tools.invoke(messages)

        if response.tool_calls:
            messages.append(response)
            for tc in response.tool_calls:
                if tc["name"] == "calculate":
                    try:
                        result = calculate.invoke(tc["args"])
                        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                    except Exception as e:
                        messages.append(ToolMessage(content=f"Calculation failed: {e}", tool_call_id=tc["id"]))
            response = writer_w_tools.invoke(messages)

        return {"generation": response.content, "steps": state.get("steps", []) + ["generate"]}

    # ------------------------------------------------------------------
    # Graph wiring — identical to original
    # ------------------------------------------------------------------

    def _route_decision(self, state: AgentState) -> str:
        return state["decision"]

    def _decide_to_generate(self, state: AgentState) -> str:
        if not state["documents"]:
            if state["retry_count"] >= self.max_retries:
                return "generate"
            return "transform_query"
        return "generate"

    def _build_graph(self):
        wf = StateGraph(AgentState)
        wf.add_node("router",          self._router)
        wf.add_node("chitchat",        self._chitchat)
        wf.add_node("retrieve",        self._retrieve)
        wf.add_node("grade_documents", self._grade_documents)
        wf.add_node("transform_query", self._transform_query)
        wf.add_node("generate",        self._generate)

        wf.set_entry_point("router")
        wf.add_conditional_edges(
            "router", self._route_decision,
            {"chitchat": "chitchat", "vectorstore": "retrieve"},
        )
        wf.add_edge("chitchat",       END)
        wf.add_edge("retrieve",       "grade_documents")
        wf.add_conditional_edges(
            "grade_documents", self._decide_to_generate,
            {"transform_query": "transform_query", "generate": "generate"},
        )
        wf.add_edge("transform_query", "retrieve")
        wf.add_edge("generate",        END)
        return wf.compile()

    def query(
        self,
        query:        str,
        mode:         str   = "concise",
        temperature:  float = 0.0,
        max_tokens:   int   = settings.MAX_TOKENS,
        chat_history: str   = "",
    ) -> Dict:
        initial: AgentState = {
            "question":          query,
            "original_question": query,
            "chat_history":      chat_history,
            "documents":         [],
            "decision":          "vectorstore",
            "generation":        "",
            "steps":             [],
            "retry_count":       0,
            "mode":              mode,
            "temperature":       temperature,
            "max_tokens":        max_tokens,
        }
        result = self._app.invoke(initial)
        return {
            "answer":   result.get("generation", ""),
            "sources":  result.get("documents",  []),
            "metadata": {"steps": result.get("steps", [])},
        }
