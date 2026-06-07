"""
backend/main.py — identical structure to agentic-graph-rag with GraphitiService replacing GraphService.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config     import settings
from backend.core.logger     import setup_logging
from backend.core.exceptions import AgenticRAGException

from backend.services.retrieve_service       import RetrieveService
from backend.services.embed_cache_service    import EmbedCacheService
from backend.services.memory_service         import MemoryService
from backend.services.semantic_cache_service import SemanticCacheService
from backend.services.graphiti_service       import GraphitiService
from backend.agents.graphiti_agent           import GraphitiRAGAgent
from backend.models.request_models           import QueryRequest, RetrieveRequest
from backend.models.response_models          import QueryResponse, RetrieveResponse, DocumentResult

setup_logging()
logger = logging.getLogger("graphiti-rag.api")

retrieve_service: Optional[RetrieveService]     = None
embed_cache:      Optional[EmbedCacheService]   = None
memory_service:   Optional[MemoryService]       = None
semantic_cache:   Optional[SemanticCacheService] = None
graphiti_svc:     Optional[GraphitiService]     = None
rag_agent:        Optional[GraphitiRAGAgent]    = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_service, semantic_cache, embed_cache, retrieve_service, graphiti_svc, rag_agent

    logger.info("Starting Graphiti-RAG service...")

    for name, factory, target in [
        ("MemoryService",       lambda: MemoryService(max_history=settings.MEMORY_MAX_TURNS, use_sqlite=True, db_path=settings.MEMORY_DB_PATH, preload=False), "memory_service"),
        ("SemanticCacheService",lambda: SemanticCacheService(), "semantic_cache"),
        ("EmbedCacheService",   lambda: EmbedCacheService(db_path=settings.EMBEDDING_CACHE_DB), "embed_cache"),
    ]:
        try:
            globals()[target] = factory()
            logger.info("%s ready.", name)
        except Exception as e:
            logger.exception("%s init failed: %s", name, e)

    # Reranker
    reranker_obj = None
    if settings.RERANKER_ENABLED:
        try:
            from backend.tools.reranker import Reranker
            reranker_obj = Reranker()
            logger.info("Reranker loaded.")
        except Exception as e:
            logger.exception("Reranker init failed: %s", e)

    # GraphitiService (replaces GraphService)
    try:
        graphiti_svc = GraphitiService()
        await graphiti_svc.client.build_indices_and_constraints()
        logger.info("GraphitiService ready.")
    except Exception as e:
        logger.error("GraphitiService init failed: %s", e)
        graphiti_svc = None

    # RetrieveService
    try:
        retrieve_service = RetrieveService(
            embed_cache=globals().get("embed_cache"),
            reranker_obj=reranker_obj,
            reranker_enabled=bool(reranker_obj),
            graphiti_svc=graphiti_svc,
        )
        logger.info("RetrieveService ready.")
    except Exception as e:
        logger.exception("RetrieveService init failed: %s", e)

    # Agent
    if retrieve_service:
        try:
            rag_agent = GraphitiRAGAgent(retrieve_service=retrieve_service)
            logger.info("GraphitiRAGAgent ready.")
        except Exception as e:
            logger.exception("Agent init failed: %s", e)

    yield

    logger.info("Shutting down...")
    if graphiti_svc:
        await graphiti_svc.close()
    for svc in [retrieve_service, globals().get("embed_cache"), globals().get("memory_service")]:
        if svc and hasattr(svc, "close"):
            try:
                svc.close()
            except Exception:
                pass


APP = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION, lifespan=lifespan)
APP.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


@APP.get("/health")
async def health():
    return {
        "status":        "ok",
        "retriever":     bool(retrieve_service),
        "rag_agent":     bool(rag_agent),
        "graphiti":      bool(graphiti_svc),
        "semantic_cache": bool(semantic_cache),
    }


@APP.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")
    try:
        session_id = req.conversation_id or "default"
        use_cache = (
            semantic_cache
            and req.mode == "concise"
            and req.temperature <= 0.1
            and req.max_tokens <= settings.MAX_TOKENS
            and not req.bypass_cache
        )
        if use_cache:
            cached = semantic_cache.check_cache(req.query)
            if cached:
                if memory_service:
                    memory_service.add_turn(session_id, req.query, cached)
                return QueryResponse(query=req.query, answer=cached, sources=[], num_sources=0, prompt="semantic_cache", metadata={"cached": True})

        chat_history = memory_service.get_context(session_id, last_n=10) if memory_service else ""

        output    = rag_agent.query(query=req.query, mode=req.mode, temperature=req.temperature, max_tokens=req.max_tokens, chat_history=chat_history)
        ai_answer = output.get("answer", "No answer generated.")

        if memory_service:
            memory_service.add_turn(session_id, req.query, ai_answer)
        if semantic_cache and req.mode == "concise" and not req.bypass_cache:
            semantic_cache.add_new_turn(req.query, ai_answer)

        doc_results = [
            DocumentResult(text=str(t)[:500], score=1.0, metadata={}, source="graphiti/faiss", chunk_id=i)
            for i, t in enumerate(output.get("sources", [])) if t
        ]
        return QueryResponse(query=req.query, answer=ai_answer, sources=doc_results, num_sources=len(doc_results), prompt="", metadata=output.get("metadata", {}))
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@APP.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(req: RetrieveRequest):
    if not retrieve_service:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    try:
        texts = await retrieve_service.retrieve_hybrid(req.query, top_k=req.top_k)
        results = [DocumentResult(text=t, score=1.0, metadata={}, source="graphiti/faiss", chunk_id=i) for i, t in enumerate(texts)]
        return RetrieveResponse(query=req.query, results=results, num_results=len(results))
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(e))


INGEST_DIR = Path(settings.WATCH_DIR)
INGEST_DIR.mkdir(parents=True, exist_ok=True)

@APP.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    filename = Path(file.filename).name
    out_path = INGEST_DIR / filename
    try:
        with out_path.open("wb") as fh:
            fh.write(await file.read())
        return {"status": "accepted", "filename": filename, "note": "Watcher will detect and ingest shortly."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:APP", host="0.0.0.0", port=8000, reload=settings.DEBUG)
