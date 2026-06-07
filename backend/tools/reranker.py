import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from backend.core.config import settings


logger = logging.getLogger("agentic-rag.reranker")


class Reranker:
    """
    Production-grade hybrid reranker with score normalization.

    Supports:
    - CrossEncoder (sentence-transformers)
    - FlagEmbedding (BGE)
    - GPU auto-detection
    - FP16
    - Feature toggle
    - Score normalization (minmax / softmax / none)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        enabled: Optional[bool] = None,
        backend: Optional[str] = None,  # "cross-encoder" or "flag"
        batch_size: int = 16,
        device: Optional[str] = None,
        normalize: Optional[str] = None,  # "minmax", "softmax", "none"
    ):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.enabled = enabled if enabled is not None else settings.RERANKER_ENABLED
        self.backend = backend or settings.RERANKER_BACKEND
        self.batch_size = batch_size or settings.RERANKER_BATCH_SIZE
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize or settings.RERANKER_NORMALIZE


        self.model = None

        if not self.enabled:
            logger.info("Reranker disabled via configuration.")
            return

        try:
            if self.backend == "flag":
                from FlagEmbedding import FlagReranker

                logger.info(f"[RERANKER] Loading FlagEmbedding model: {self.model_name}")
                self.model = FlagReranker(
                    self.model_name,
                    use_fp16=(self.device == "cuda")
                )
            else:
                from sentence_transformers import CrossEncoder

                logger.info(f"[RERANKER] Loading CrossEncoder model: {self.model_name} on {self.device}")
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device
                )

        except ImportError as e:
            logger.error(f"Reranker backend not installed: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.enabled = False

    # ---------------------------
    # Score Normalization
    # ---------------------------

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return scores

        if len(scores) == 0:
            return scores

        if self.normalize == "sigmoid":
            # Best for CrossEncoders/BGE to convert logits to probabilities (0 to 1)
            return 1 / (1 + np.exp(-scores))

        if self.normalize == "minmax":
            min_s = np.min(scores)
            max_s = np.max(scores)
            if max_s - min_s < 1e-8:
                return np.ones_like(scores)
            return (scores - min_s) / (max_s - min_s)

        if self.normalize == "softmax":
            shifted = scores - np.max(scores)  # numerical stability
            exp_scores = np.exp(shifted)
            return exp_scores / np.sum(exp_scores)

        return scores

    # ---------------------------
    # Scoring
    # ---------------------------

    def _score(self, query: str, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)

        pairs = [[query, text] for text in texts]

        if self.backend == "flag":
            scores = self.model.compute_score(pairs)
        else:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )

        scores = np.asarray(scores, dtype=np.float32)
        return self._normalize_scores(scores)

    # ---------------------------
    # Public API
    # ---------------------------

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        if not self.enabled or not self.model or not documents:
            return documents[:top_k]

        texts = [doc.get("meta", {}).get("text", "") for doc in documents]

        try:
            scores = self._score(query, texts)

            for doc, score in zip(documents, scores):
                doc["_rerank_score"] = float(score)

            documents.sort(key=lambda x: x["_rerank_score"], reverse=True)
            return documents[:top_k]

        except Exception as e:
            logger.error(f"Reranking computation failed: {e}")
            return documents[:top_k]
