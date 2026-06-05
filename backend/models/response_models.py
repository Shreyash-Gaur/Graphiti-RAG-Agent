# backend/models/response_models.py

from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# ------------------------------
# Document / Chunk returned during retrieval
# ------------------------------
class DocumentResult(BaseModel):
    """
    Represents a retrieved vector chunk and its metadata.
    """
    text: str
    score: float
    metadata: Dict[str, Any] = {}
    source: Optional[str] = None
    chunk_id: Optional[int] = None


# ------------------------------
# Retrieve Endpoint Response
# ------------------------------
class RetrieveResponse(BaseModel):
    query: str
    results: List[DocumentResult]
    num_results: int


# ------------------------------
# Final RAG Answer
# ------------------------------
class QueryResponse(BaseModel):
    """
    Response schema for /query, /agent_query.
    """
    query: str
    answer: str
    sources: List[DocumentResult]
    num_sources: int
    prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ------------------------------
# Agentic Multi-step Reasoning Output
# ------------------------------
class IterationStep(BaseModel):
    step: int
    reasoning: str
    retrieved: List[DocumentResult]


class IterativeQueryResponse(BaseModel):
    """
    Multi-step agent response.
    """
    query: str
    answer: str
    iterations: List[IterationStep]
    final_sources: List[DocumentResult]


# ------------------------------
# Ingestion API Response
# ------------------------------
class IngestResponse(BaseModel):
    document: str
    num_chunks: int
    dim: int
    status: str


class BatchIngestResponse(BaseModel):
    results: List[IngestResponse]


# ------------------------------
# Health Check Model
# ------------------------------
class HealthResponse(BaseModel):
    status: str
    retriever_loaded: bool
    reranker_loaded: bool
    ollama_configured: bool
    num_documents: Optional[int] = None
    embedding_model: Optional[str] = None

