"""
backend/services/retrieve_service.py

Hybrid retrieval: Graphiti (primary) + FAISS (fallback) + reranker.

Retrieval priority:
  1. Graphiti.search() — temporal knowledge graph, entity-aware, BFS + vector
  2. FAISS similarity search — offline fallback, same embed cache pipeline
  3. Reranker (BAAI/bge-reranker-v2-m3) — scores and re-orders combined candidates

Why keep FAISS when we have Graphiti?
  - Graphiti requires Neo4j running. FAISS works when Neo4j is down.
  - During development on your local machine, you don't always want Neo4j up.
  - FAISS covers documents that weren't ingested with --skip-faiss.
  - The reranker blends both pools so quality improves with more candidates.

Key difference from agentic-graph-rag:
  - graph_service.structured_retriever() (Cypher entity query) is replaced
    by graphiti_svc.search() which does the same thing internally plus
    temporal filtering and cross-encoder reranking on the graph side.
  - Everything else (embed cache, FAISS search, SQLite metadata, reranker)
    is identical to the original retrieve_service.py.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from backend.core.config import settings
from backend.services.embed_cache_service import EmbedCacheService
from backend.services.graphiti_service import GraphitiService
from backend.tools.embedder import Embedder

logger = logging.getLogger("graphiti-rag.retrieve")

DEFAULT_INDEX    = settings.FAISS_INDEX_PATH
DEFAULT_META     = settings.FAISS_META_PATH
DEFAULT_DB       = settings.META_DB_PATH
DEFAULT_MODEL    = settings.EMBEDDING_MODEL


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    meta = []
    if not os.path.exists(path):
        return meta
    with open(path, "r", encoding="utf8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                meta.append(json.loads(ln))
            except Exception:
                meta.append({"text": ln})
    return meta


class RetrieveService:
    def __init__(
        self,
        index_path:       str                          = DEFAULT_INDEX,
        meta_path:        str                          = DEFAULT_META,
        db_path:          str                          = DEFAULT_DB,
        embed_cache:      Optional[EmbedCacheService]  = None,
        embedder:         Optional[Embedder]           = None,
        reranker_obj:     Optional[Any]                = None,
        reranker_enabled: bool                         = False,
        graphiti_svc:     Optional[GraphitiService]    = None,
    ):
        self.index_path      = index_path
        self.meta_path       = meta_path
        self.db_path         = db_path
        self.reranker        = reranker_obj
        self.reranker_enabled = bool(reranker_enabled)
        self.graphiti_svc    = graphiti_svc

        # Metadata store: SQLite preferred, JSONL fallback (identical to original)
        self.db_conn: Optional[sqlite3.Connection] = None
        self.meta: List[Dict[str, Any]] = []
        if os.path.exists(self.db_path):
            try:
                self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.db_conn.row_factory = sqlite3.Row
                logger.info("RetrieveService connected to SQLite: %s", self.db_path)
            except Exception as e:
                logger.error("SQLite connection failed: %s", e)
        if self.db_conn is None:
            self.meta = _load_jsonl(self.meta_path)
            logger.info("JSONL metadata loaded: %d entries", len(self.meta))

        # FAISS index (lazy load)
        self._index: Optional[faiss.Index] = None
        self._ensure_index()

        # Embedder + cache (identical to original)
        self.embedder    = embedder or Embedder()
        self.embed_model = getattr(self.embedder, "model", DEFAULT_MODEL)
        self.embed_cache = embed_cache or EmbedCacheService()

    # ------------------------------------------------------------------
    # Primary hybrid retrieval — called by the agent
    # ------------------------------------------------------------------

    async def retrieve_hybrid(self, query: str, top_k: int = settings.TOP_K_RETRIEVAL) -> List[str]:
        """
        Combines:
          1. Graphiti knowledge graph search (temporal, entity-aware)
          2. FAISS vector search (offline fallback)
          3. Reranker over the combined pool

        Returns a list of plain text strings — same interface as the original.
        """
        docs: List[str] = []
        fetch_k = settings.RERANKER_INITIAL_K if (self.reranker_enabled and self.reranker) else top_k

        # ── 1. Graphiti search ────────────────────────────────────────────────
        graphiti_hits: List[str] = []
        if self.graphiti_svc:
            try:
                results = await self.graphiti_svc.search(query, limit=fetch_k)
                for r in results:
                    fact = r.get("fact", "")
                    if not fact:
                        continue
                    # Annotate with temporal context when available — the LLM
                    # can use this to give time-aware answers
                    if r.get("valid_at") and not r.get("invalid_at"):
                        fact = f"[Valid from {r['valid_at']}] {fact}"
                    elif r.get("invalid_at"):
                        fact = f"[Valid {r['valid_at']} → {r['invalid_at']}] {fact}"
                    graphiti_hits.append(fact)
                if graphiti_hits:
                    logger.debug("Graphiti returned %d hits", len(graphiti_hits))
            except Exception as e:
                logger.error("Graphiti search failed: %s", e)

        # ── 2. FAISS fallback ─────────────────────────────────────────────────
        faiss_hits: List[str] = []
        if self._index is not None:
            try:
                qvec = self._embed_text(query).astype("float32")
                D, I = self._search_faiss(qvec, fetch_k)
                if I.size > 0:
                    for idx in I[0]:
                        if idx < 0:
                            continue
                        meta = self._get_meta(int(idx))
                        text = meta.get("text", "")
                        if text:
                            faiss_hits.append(text)
                logger.debug("FAISS returned %d hits", len(faiss_hits))
            except Exception as e:
                logger.error("FAISS search failed: %s", e)

        # ── 3. Combine + rerank ───────────────────────────────────────────────
        # Graphiti hits first — they're richer (entities + relationships + temporal).
        # FAISS hits fill in anything Graphiti missed.
        all_candidates = graphiti_hits + [h for h in faiss_hits if h not in graphiti_hits]

        if not all_candidates:
            return docs

        if self.reranker_enabled and self.reranker and all_candidates:
            try:
                candidate_dicts = [{"meta": {"text": t}} for t in all_candidates]
                reranked = self.reranker.rerank(query, candidate_dicts, top_k=top_k)
                docs = [d["meta"]["text"] for d in reranked]
                logger.debug("Reranker top score: %.4f", reranked[0].get("_rerank_score", 0))
            except Exception as e:
                logger.error("Reranker failed: %s — using raw candidates", e)
                docs = all_candidates[:top_k]
        else:
            docs = all_candidates[:top_k]

        return docs

    # ------------------------------------------------------------------
    # FAISS helpers (identical to original retrieve_service.py)
    # ------------------------------------------------------------------

    def _get_meta(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            return {}
        if self.db_conn:
            try:
                cur = self.db_conn.cursor()
                cur.execute(
                    "SELECT chunk_id, doc_name, text, start_token, end_token FROM chunks WHERE chunk_id = ?",
                    (idx,),
                )
                row = cur.fetchone()
                return dict(row) if row else {}
            except Exception as e:
                logger.error("SQLite fetch error idx=%d: %s", idx, e)
                return {}
        if 0 <= idx < len(self.meta):
            return self.meta[idx]
        return {}

    def _ensure_index(self) -> None:
        if self._index is None and os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
                logger.info("FAISS index loaded — ntotal=%d", self._index.ntotal)
            except Exception as e:
                logger.error("FAISS load failed: %s", e)

    def _search_faiss(self, vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_index()
        if self._index is None:
            return np.array([[]]), np.array([[]])
        q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        return self._index.search(q, top_k)

    def _embed_text(self, text: str) -> np.ndarray:
        vec = self.embed_cache.get_vector(text, self.embed_model)
        if vec is not None:
            return vec.astype("float32")
        vec = self.embedder.embed_batch([text])[0]
        try:
            self.embed_cache.set_vector(text, self.embed_model, vec)
        except Exception:
            logger.exception("Embed cache write failed")
        return np.asarray(vec, dtype=np.float32)

    def close(self) -> None:
        if self.db_conn:
            try:
                self.db_conn.close()
            except Exception:
                pass
        try:
            self.embed_cache.close()
        except Exception:
            pass
