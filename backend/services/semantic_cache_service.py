# backend/services/semantic_cache_service.py

import sqlite3
import faiss
import os
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer

# Import your centralized settings
from backend.core.config import settings

logger = logging.getLogger("agentic-rag.semantic_cache")

class SemanticCacheService:
    def __init__(
        self, 
        db_path: str = settings.MEMORY_DB_PATH, 
        model_name: str = settings.SEMANTIC_CACHE_MODEL,
        threshold: float = settings.SEMANTIC_CACHE_THRESHOLD
    ):
        self.db_path = db_path
        self.threshold = threshold
        
        logger.info(f"Loading Semantic Cache Encoder ({model_name})... This might take a moment.")
        self.encoder = SentenceTransformer(model_name)
        self.dim = self.encoder.get_embedding_dimension()
        
        # IndexFlatIP uses Inner Product (Cosine Similarity if vectors are normalized)
        self.index = faiss.IndexFlatIP(self.dim)
        
        self.answers = []
        self.past_queries = []

        # Boot up the cache using past memory
        self._load_from_sqlite_memory()

    def _load_from_sqlite_memory(self):
        """Reads past conversations from your SQLite memory and loads them into FAISS."""
        if not os.path.exists(self.db_path):
            logger.info("No existing SQLite memory found. Starting with empty semantic cache.")
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        try:
            # Order by timestamp so we can pair up User -> Assistant turns sequentially
            cur.execute("SELECT conv_id, role, content FROM memory ORDER BY ts ASC")
            rows = cur.fetchall()
            
            # Group by conversation ID
            convs = {}
            for r in rows:
                convs.setdefault(r["conv_id"], []).append((r["role"], r["content"]))

            # Pair up "user" questions with the following "assistant" answers
            pair_count = 0
            for cid, turns in convs.items():
                for i in range(len(turns) - 1):
                    if turns[i][0] == "user" and turns[i+1][0] == "assistant":
                        user_query = turns[i][1]
                        ai_answer = turns[i+1][1]
                        self._add_to_faiss(user_query, ai_answer)
                        pair_count += 1
                        
            logger.info(f"Loaded {pair_count} past Q&A pairs from memory into Semantic Cache.")
        except Exception as e:
            logger.error(f"Failed to load from memory into Semantic Cache: {e}")
        finally:
            conn.close()

    def _add_to_faiss(self, query: str, answer: str):
        """Helper to embed, normalize, and add to FAISS."""
        vec = self.encoder.encode([query])
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.answers.append(answer)
        self.past_queries.append(query)

    def check_cache(self, query: str) -> Optional[str]:
        """Checks if a highly similar query has been asked before."""
        if self.index.ntotal == 0:
            return None

        # Embed the new question
        vec = self.encoder.encode([query])
        faiss.normalize_L2(vec)

        # Search for the top 1 closest past question
        scores, indices = self.index.search(vec, k=1)
        
        # If the similarity score is higher than our threshold, it's a hit!
        best_score = scores[0][0]
        if best_score >= self.threshold:
            best_idx = indices[0][0]
            logger.info(f"⚡ SEMANTIC CACHE HIT! Score: {best_score:.4f} (Matched: '{self.past_queries[best_idx]}')")
            return self.answers[best_idx]
            
        return None

    def add_new_turn(self, query: str, answer: str):
        """Adds a brand new conversation turn to the active memory cache."""
        self._add_to_faiss(query, answer)