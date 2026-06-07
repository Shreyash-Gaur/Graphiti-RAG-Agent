"""
backend/services/graphiti_service.py

Thin wrapper around the Graphiti client.

Responsibilities:
  - Owns the single Graphiti instance (one per process, reused across requests)
  - Configures Graphiti to use your local Ollama LLM + embedder — same models
    already running for the rest of the stack (OLLAMA_MODEL, EMBEDDING_MODEL)
  - Exposes search() so retrieve_service.py can call Graphiti without importing
    graphiti_core directly
  - Exposes close() for FastAPI lifespan cleanup

What this does NOT do:
  - Ingestion — that belongs in scripts/ingest_watch.py
  - FAISS / reranking — that belongs in retrieve_service.py
  - Graph traversal Cypher — Graphiti handles that internally

Why a wrapper instead of using Graphiti directly?
  retrieve_service.py needs a consistent interface. If you later swap Graphiti
  for a different temporal graph library, you change this file only.
"""

from __future__ import annotations

import logging
from typing import Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from backend.core.config import settings

logger = logging.getLogger("graphiti-rag.graphiti_service")


class GraphitiService:
    """
    Wraps the Graphiti client with Ollama-backed LLM, embedder, and cross-encoder.

    Ollama exposes an OpenAI-compatible API at /v1 — Graphiti's OpenAIClient
    works against it without modification. The only differences vs the cloud
    config are:
      - api_key="ollama"  (Ollama ignores this but the field is required)
      - base_url points to your local Ollama instance
      - embedding_dim must match the model (1024 for mxbai-embed-large,
        768 for nomic-embed-text)

    Note on embedding_dim:
      qwen3-embedding:8b → 4096
      mxbai-embed-large:latest → 1024
      nomic-embed-text         → 768
      If you change EMBEDDING_MODEL, update EMBEDDING_DIM in .env too.
    """

    def __init__(self):
        ollama_base = settings.OLLAMA_BASE_URL.rstrip("/") + "/v1"

        llm_config = LLMConfig(
            api_key="ollama",
            model=settings.OLLAMA_MODEL,
            small_model=settings.OLLAMA_MODEL,
            base_url=ollama_base,
        )
        llm_client = OpenAIClient(config=llm_config)

        embedder_config = OpenAIEmbedderConfig(
            api_key="ollama",
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_dim=settings.EMBEDDING_DIM,
            base_url=ollama_base,
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        cross_encoder = OpenAIRerankerClient(
            client=llm_client,
            config=llm_config,
        )

        self.client = Graphiti(
            settings.NEO4J_URI,
            settings.NEO4J_USERNAME,
            settings.NEO4J_PASSWORD,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )
        logger.info(
            "GraphitiService ready — model=%s embedding=%s dim=%d",
            settings.OLLAMA_MODEL, settings.EMBEDDING_MODEL, settings.EMBEDDING_DIM,
        )

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """
        Searches the Graphiti knowledge graph.
        Returns a list of dicts with 'fact', 'valid_at', 'invalid_at' keys.
        retrieve_service.py calls this as the primary (Neo4j) retrieval path.
        """
        try:
            results = await self.client.search(query)
            formatted = []
            for r in results[:limit]:
                formatted.append({
                    "fact":       r.fact,
                    "valid_at":   str(r.valid_at)   if getattr(r, "valid_at",   None) else None,
                    "invalid_at": str(r.invalid_at) if getattr(r, "invalid_at", None) else None,
                    "uuid":       r.uuid,
                })
            logger.debug("Graphiti search returned %d results for query: %.60s", len(formatted), query)
            return formatted
        except Exception as e:
            logger.error("Graphiti search failed: %s", e)
            return []

    async def close(self) -> None:
        try:
            await self.client.close()
            logger.info("Graphiti connection closed.")
        except Exception as e:
            logger.warning("Error closing Graphiti: %s", e)
