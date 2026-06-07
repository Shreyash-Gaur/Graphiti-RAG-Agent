# backend/tools/embedder.py
import os
import json
import logging
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from typing import List

logger = logging.getLogger("graphiti-rag.embedder")


class Embedder:
    """
    Ollama-backed embedder using the /api/embed endpoint (Ollama 0.2.0+).

    /api/embed replaces the old /api/embeddings endpoint. Key differences:
      - Field is "input" (string or list), not "prompt"
      - Response is always {"embeddings": [[...]]}  — list of lists
      - Supports native batching: pass a list, get back a list of vectors
      - Returns L2-normalized vectors by default
    """

    def __init__(self, base_url=None, model=None, timeout=150):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model    = model    or os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
        self.timeout  = timeout

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def _embed_one(self, text: str) -> np.ndarray:
        """Single text embed — delegates to embed_batch for consistency."""
        return self.embed_batch([text])[0]

    def embed(self, text: str) -> np.ndarray:
        """Public single-text API."""
        return self._embed_one(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch embed via /api/embed.
        Single HTTP call regardless of batch size — Ollama handles it natively.
        Falls back to per-item embedding if the batch call fails.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        url     = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}

        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            if not isinstance(data, dict) or "embeddings" not in data:
                # Dump debug info and raise — do not silently return zeros
                debug_path = "backend/db/embedding_cache/embed_debug.json"
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                with open(debug_path, "w") as f:
                    json.dump({"response": data, "texts_sample": texts[:2]}, f, indent=2)
                raise RuntimeError(
                    f"Unexpected /api/embed response shape. Saved to {debug_path}."
                )

            vecs = [np.asarray(v, dtype=np.float32) for v in data["embeddings"]]

            if len(vecs) != len(texts):
                raise RuntimeError(
                    f"Ollama returned {len(vecs)} embeddings for {len(texts)} inputs."
                )

            dim = len(vecs[0])
            out = np.zeros((len(vecs), dim), dtype=np.float32)
            for i, v in enumerate(vecs):
                out[i] = v
            return out

        except RuntimeError:
            raise  # already descriptive, don't wrap again

        except Exception as e:
            # Network error, timeout, JSON parse failure, etc.
            # Fall back to per-item calls so a single bad text doesn't kill the batch
            logger.warning(
                "Batch embed failed (%s) — falling back to per-item embedding", e
            )
            results = []
            for text in texts:
                try:
                    single_payload = {"model": self.model, "input": text}
                    r = self.session.post(url, json=single_payload, timeout=self.timeout)
                    r.raise_for_status()
                    d = r.json()
                    results.append(np.asarray(d["embeddings"][0], dtype=np.float32))
                except Exception as inner_e:
                    raise RuntimeError(
                        f"Per-item fallback also failed for text '{text[:50]}...': {inner_e}"
                    ) from e

            dim = len(results[0])
            out = np.zeros((len(results), dim), dtype=np.float32)
            for i, v in enumerate(results):
                out[i] = v
            return out