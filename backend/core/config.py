"""
backend/core/config.py — extended from agentic-graph-rag with Graphiti additions.
New fields marked with # NEW
"""

from __future__ import annotations
from pydantic_settings import BaseSettings
from typing import List, Optional
import os, json


def _parse_cors(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x and str(x).strip()]
    except Exception:
        pass
    return [p.strip() for p in raw.split(",") if p.strip()]


class Settings(BaseSettings):
    # --- API ---
    API_TITLE:   str  = "Graphiti RAG API"
    API_VERSION: str  = "1.0.0"
    DEBUG:       bool = False
    CORS_ORIGINS: Optional[str] = None

    # --- Ollama ---
    OLLAMA_BASE_URL:  str = "http://localhost:11434"
    OLLAMA_MODEL:     str = "qwen2.5:7b"
    EMBEDDING_MODEL:  str = "qwen3-embedding:8b"
    EMBEDDING_DIM:    int = 4096 

    # --- RAG ---
    MAX_TOKENS:     int = 1024
    MAX_ITERATIONS: int = 7
    TOP_K_RETRIEVAL: int = 5

    # --- Vector chunking (FAISS ingest) ---
    CHUNK_TOKENS:         int = 512
    CHUNK_OVERLAP:        int = 100
    EMBEDDING_BATCH_SIZE: int = 16

    # --- Semantic chunking (Graphiti ingest) ---
    SEMANTIC_CHUNK_THRESHOLD_TYPE: str = "percentile"
    SEMANTIC_CHUNK_BREAKPOINT:     int = 85

    # --- Coref ---
    USE_COREF: bool = True

    # --- Neo4j ---
    NEO4J_URI:      str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # --- Watcher ---
    WATCH_DIR: str = "knowledge"

    # --- File paths ---
    FAISS_INDEX_PATH: str = "backend/db/vector_data/knowledge_faiss.index"
    FAISS_META_PATH:  str = "backend/db/vector_data/knowledge_meta.jsonl"
    META_DB_PATH:     str = "backend/db/vector_data/metadata_store.db"
    MEMORY_DB_PATH:   str = "backend/db/memory/memory_store.sqlite"
    EMBEDDING_CACHE_DB: str = "backend/db/embedding_cache/embed_cache.sqlite"
    MEMORY_MAX_TURNS: int = 20

    # --- Reranker ---
    RERANKER_ENABLED:    bool  = True
    RERANKER_MODEL:      str   = "BAAI/bge-reranker-v2-m3"
    RERANKER_INITIAL_K:  int   = 15
    RERANKER_BACKEND:    str   = "cross-encoder"
    RERANKER_NORMALIZE:  str   = "sigmoid"
    RERANKER_BATCH_SIZE: int   = 8

    # --- Semantic cache ---
    SEMANTIC_CACHE_MODEL:     str   = "BAAI/bge-large-en-v1.5"
    SEMANTIC_CACHE_THRESHOLD: float = 0.85

    # --- Feature flags ---
    USE_HYDE:         bool = True
    CHAINLIT_ENABLED: bool = True

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }

    @property
    def CORS(self) -> List[str]:
        raw = os.getenv("CORS_ORIGINS", None) or self.CORS_ORIGINS
        return _parse_cors(raw)


settings = Settings()
