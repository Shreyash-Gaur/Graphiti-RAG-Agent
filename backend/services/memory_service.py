# backend/services/memory_service.py
"""
MemoryService (SQLite-backed, thread-safe)
Features:
 - Keeps per-conversation history in memory for fast read.
 - Persists all turns to SQLite for durability.
 - Bounded history (max_history) kept in-memory per conversation.
 - Safe for multi-threaded FastAPI usage (uses RLock).
 - Full API: add_turn, get_context, get_history, export/import.
"""

from __future__ import annotations
import sqlite3
import threading
import json
import time
import os
import logging
from typing import List, Dict, Optional, Any
from backend.core.config import settings

logger = logging.getLogger("agentic-rag.memory")

class MemoryService:
    def __init__(
        self,
        max_history: int = settings.MEMORY_MAX_TURNS,
        use_sqlite: bool = True,
        db_path: str = settings.MEMORY_DB_PATH,
        preload: bool = True,
    ):
        self.max_history = int(max_history)
        self.use_sqlite = bool(use_sqlite)
        self.db_path = db_path
        self._lock = threading.RLock()
        self._store: Dict[str, List[Dict[str, Any]]] = {}

        # Ensure DB directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        if self.use_sqlite:
            self._init_db()
            if preload:
                try:
                    self._load_all_to_memory()
                except Exception as e:
                    logger.warning("Preload memory failed: %s", e)

    # ---------------------------
    # Database bootstrap
    # ---------------------------
    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                ts REAL NOT NULL,
                role TEXT,
                content TEXT,
                meta TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_id ON memory(conv_id);")
        conn.commit()
        self._conn = conn

    def _load_all_to_memory(self) -> None:
        if not self._conn:
            return
        cur = self._conn.cursor()
        cur.execute("SELECT conv_id, ts, role, content, meta FROM memory ORDER BY id ASC")
        rows = cur.fetchall()
        with self._lock:
            for r in rows:
                conv_id = r["conv_id"]
                turn = {
                    "ts": float(r["ts"]),
                    "role": r["role"],
                    "content": r["content"],
                    "meta": json.loads(r["meta"]) if r["meta"] else {},
                }
                self._store.setdefault(conv_id, []).append(turn)
            for cid, turns in list(self._store.items()):
                if len(turns) > self.max_history:
                    self._store[cid] = turns[-self.max_history :]

    # ---------------------------
    # Core Internal Helper
    # ---------------------------
    def _add_single_turn(
        self,
        conv_id: str,
        role: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> None:
        ts = float(ts or time.time())
        meta = meta or {}
        turn = {"ts": ts, "role": role, "content": content, "meta": meta}

        with self._lock:
            buf = self._store.setdefault(conv_id, [])
            buf.append(turn)
            if self.use_sqlite and self._conn:
                try:
                    cur = self._conn.cursor()
                    cur.execute(
                        "INSERT INTO memory (conv_id, ts, role, content, meta) VALUES (?, ?, ?, ?, ?)",
                        (conv_id, ts, role, content, json.dumps(meta, ensure_ascii=False)),
                    )
                    self._conn.commit()
                except Exception as e:
                    logger.exception("Failed to persist memory to sqlite: %s", e)
            if len(buf) > self.max_history:
                self._store[conv_id] = buf[-self.max_history :]

    # ---------------------------
    # Public API
    # ---------------------------
    def add_turn(self, session_id: str, user_input: str, ai_output: str) -> None:
        """Saves a conversation pair (User + AI)."""
        try:
            self._add_single_turn(conv_id=session_id, role="user", content=user_input)
            self._add_single_turn(conv_id=session_id, role="assistant", content=ai_output)
            logger.info(f"Saved turn to memory for session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to add turn pair: {e}")
            raise e

    def get_context(self, session_id: str, last_n: int = 5) -> str:
        """Retrieves history as a formatted string for the LLM."""
        with self._lock:
            history = self._store.get(session_id, [])
            relevant_history = history[-last_n:]
            formatted_context = []
            for item in relevant_history:
                role = item.get("role", "unknown").capitalize()
                content = item.get("content", "")
                formatted_context.append(f"{role}: {content}")
            return "\n".join(formatted_context)

    # ---------------------------
    # Restored Utilities
    # ---------------------------
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Return raw list of dicts (Useful for Frontend)."""
        with self._lock:
            return list(self._store.get(session_id, []))

    def export_history(self, session_id: str) -> str:
        """Return JSON string of history."""
        hist = self.get_history(session_id)
        return json.dumps(hist, ensure_ascii=False, indent=2)

    def import_history(self, session_id: str, turns: List[Dict[str, Any]]) -> None:
        """Bulk import turns."""
        if not isinstance(turns, list):
            raise ValueError("turns must be a list")
        with self._lock:
            self._store[session_id] = turns[-self.max_history :]
            if self.use_sqlite and self._conn:
                try:
                    cur = self._conn.cursor()
                    for t in self._store[session_id]:
                        cur.execute(
                            "INSERT INTO memory (conv_id, ts, role, content, meta) VALUES (?, ?, ?, ?, ?)",
                            (
                                session_id,
                                float(t.get("ts", time.time())),
                                t.get("role", ""),
                                t.get("content", ""),
                                json.dumps(t.get("meta", {}), ensure_ascii=False),
                            ),
                        )
                    self._conn.commit()
                except Exception as e:
                    logger.exception("Failed to import history: %s", e)

    def clear_history(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)
            if self.use_sqlite and self._conn:
                try:
                    self._conn.execute("DELETE FROM memory WHERE conv_id = ?", (session_id,))
                    self._conn.commit()
                except Exception as e:
                    logger.exception("Failed to clear memory: %s", e)

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None