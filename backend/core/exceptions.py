# backend/core/exceptions.py

"""
Custom exceptions used across the Agentic RAG backend.
"""


class AgenticRAGException(Exception):
    """Base exception for Agentic RAG system."""


class RetrievalError(AgenticRAGException):
    """Error during document retrieval."""


class EmbeddingError(AgenticRAGException):
    """Error during embedding generation."""


class LLMError(AgenticRAGException):
    """Error during LLM interaction."""


class AgentError(AgenticRAGException):
    """Error during agent execution."""


class IngestionError(AgenticRAGException):
    """Error during document ingestion."""


class ConfigurationError(AgenticRAGException):
    """Invalid or missing configuration."""
