# backend/services/embed_cache_service.py
"""
Embedding cache service (SQLite-backed).

Usage:
    from services.embed_cache_service import EmbedCacheService
    cache = EmbedCacheService(db_path="backend/db/embedding_cache/embed_cache.sqlite")
    vec = cache.get_vector("hello world", model="mxbai-embed-large:latest")
    if vec is None:
        vec = embedder.embed_batch(["hello world"])[0]
        cache.set_vector("hello world", model="mxbai-embed-large:latest", vector=vec)
    cache.close()
"""

from __future__ import annotations
import sqlite3
import os
import time
import hashlib
import logging
from typing import Optional, List, Dict, Tuple
import numpy as np
from backend.core.config import settings

logger = logging.getLogger("agentic-rag.embedcache")

def _text_model_key(text: str, model: str) -> str:
    h = hashlib.sha256()
    # normalize text and model deterministically
    b = (text or "").encode("utf-8") + b"\x1f" + (model or "").encode("utf-8")
    h.update(b)
    return h.hexdigest()


class EmbedCacheService:
    def __init__(self, db_path: str = settings.EMBEDDING_CACHE_DB, enable_wal: bool = True):
        self.db_path = db_path
        # ensure directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if enable_wal:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
            except Exception:
                pass
        self._init_db()

    def _init_db(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embed_cache (
                key TEXT PRIMARY KEY,
                text TEXT,
                model TEXT,
                dim INTEGER,
                vec BLOB,
                ts REAL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model ON embed_cache(model);")
        self._conn.commit()

    # -------------------------
    # Helpers for packing/unpacking vectors
    # -------------------------
    @staticmethod
    def _pack_vector(vec: np.ndarray) -> bytes:
        # Ensure float32 contiguous
        a = np.asarray(vec, dtype=np.float32)
        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        return a.tobytes()

    @staticmethod
    def _unpack_vector(buf: bytes, dim: int) -> np.ndarray:
        if buf is None:
            return None
        # interpret bytes as float32 and reshape
        arr = np.frombuffer(buf, dtype=np.float32)
        if dim is not None and arr.size != dim:
            # best-effort reshape if mismatch; otherwise return as-is
            try:
                arr = arr.reshape(dim)
            except Exception:
                logger.warning("EmbedCache: stored dim mismatch (%s != %s), returning flat array", arr.size, dim)
        return arr

    # -------------------------
    # Single operations
    # -------------------------
    def get_vector(self, text: str, model: str) -> Optional[np.ndarray]:
        """
        Return numpy float32 vector if present, else None.
        """
        key = _text_model_key(text, model)
        cur = self._conn.cursor()
        cur.execute("SELECT dim, vec FROM embed_cache WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        dim = row["dim"]
        buf = row["vec"]
        return self._unpack_vector(buf, dim)

    def set_vector(self, text: str, model: str, vector: np.ndarray) -> None:
        """
        Insert or update a single embedding.
        """
        key = _text_model_key(text, model)
        dim = int(np.asarray(vector).size)
        buf = self._pack_vector(vector)
        ts = float(time.time())
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO embed_cache (key, text, model, dim, vec, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (key, text, model, dim, buf, ts),
        )
        self._conn.commit()

    # -------------------------
    # Batch operations
    # -------------------------
    def get_batch(self, texts: List[str], model: str) -> Tuple[List[Optional[np.ndarray]], List[str]]:
        """
        Return two lists aligned to input `texts`:
          - vectors: np.ndarray or None for each text
          - keys: cache keys for each text (useful for later upserts)
        This performs a single SQL query for speed.
        """
        if not texts:
            return [], []

        keys = [_text_model_key(t, model) for t in texts]
        placeholders = ",".join("?" for _ in keys)
        cur = self._conn.cursor()
        # Query only rows for those keys
        cur.execute(f"SELECT key, dim, vec FROM embed_cache WHERE key IN ({placeholders})", tuple(keys))
        rows = cur.fetchall()
        row_map = {r["key"]: (r["dim"], r["vec"]) for r in rows}

        vectors = []
        for k in keys:
            if k in row_map:
                dim, buf = row_map[k]
                vectors.append(self._unpack_vector(buf, dim))
            else:
                vectors.append(None)
        return vectors, keys

    def set_batch(self, texts: List[str], model: str, vectors: List[np.ndarray]) -> None:
        """
        Bulk upsert embeddings; `texts` and `vectors` must be same length.
        """
        if not texts:
            return
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors length mismatch")

        now = float(time.time())
        params = []
        for t, v in zip(texts, vectors):
            key = _text_model_key(t, model)
            dim = int(np.asarray(v).size)
            buf = self._pack_vector(v)
            params.append((key, t, model, dim, buf, now))

        cur = self._conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO embed_cache (key, text, model, dim, vec, ts) VALUES (?, ?, ?, ?, ?, ?)",
            params,
        )
        self._conn.commit()

    # -------------------------
    # Utility
    # -------------------------
    def has(self, text: str, model: str) -> bool:
        key = _text_model_key(text, model)
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM embed_cache WHERE key = ? LIMIT 1", (key,))
        return cur.fetchone() is not None

    def count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) as c FROM embed_cache")
        return int(cur.fetchone()["c"] or 0)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = None
