# backend/tools/query_expander.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from backend.core.config import settings
import logging

logger = logging.getLogger("agentic-rag.hyde")

def generate_hyde_document(query: str) -> str:
    """
    Implements HyDE (Hypothetical Document Embeddings).
    Takes a short user query and generates a fake, hypothetical answer.
    This generated text is mathematically much closer to the target documents in the vector database.
    """
    logger.info(f"Generating HyDE document for query: '{query}'")
    try:
        # Use a low temperature so the LLM acts factual, not creative
        llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.1)
        
        prompt = (
            f"Please write a short, factual paragraph answering the following question or explaining the topic. "
            f"Do not include introductory filler, just provide the expected facts.\n\n"
            f"Topic: {query}\n"
            f"Factual Explanation:"
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        hypothetical_doc = response.content.strip()
        
        logger.info(f"HyDE generated {len(hypothetical_doc)} characters of context.")
        return hypothetical_doc
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return query  # Fallback to the original query if generation fails