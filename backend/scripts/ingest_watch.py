"""
backend/scripts/ingest_watch.py

Unified ingestion watcher for Graphiti-RAG-Agent.

What this does that quickstart.py / llm_evolution.py never did:
  A. Watches a folder for new files (PDF, TXT, MD) — same pattern as ingest_graph_watch.py
  B. Semantic chunking per file — SemanticChunker splits on meaning boundaries,
     not fixed token counts, so each episode fed to Graphiti is a complete thought.
     This directly improves entity extraction quality inside Graphiti.
  C. Coreference resolution per chunk (optional, controlled by USE_COREF in .env)
     Rewrites pronouns and partial names to full canonical forms before Graphiti
     sees the text — same Tier 3 logic from agentic-graph-rag.
  D. Calls graphiti.add_episode() per chunk — this replaces the entire
     LLMGraphTransformer + add_graph_documents() + APOC merge pipeline.
     Graphiti handles entity extraction, deduplication, temporal edges,
     and vector indexing in one call.
  E. ALSO ingests into FAISS for hybrid retrieval fallback — same as
     ingest_multi_docs.py. Graphiti's search is the primary path; FAISS
     is the fallback when Graphiti returns nothing.
  F. Syncs FAISS metadata to SQLite after each file.

Why both Graphiti AND FAISS?
  Graphiti's vector search requires a running Neo4j instance. FAISS works
  offline. retrieve_service.py tries Graphiti first, falls back to FAISS.
  During development (Neo4j down), queries still work.

Usage:
  python -m backend.scripts.ingest_watch
  python -m backend.scripts.ingest_watch --watch knowledge --interval 15
  python -m backend.scripts.ingest_watch --skip-coref    # faster, less accurate
  python -m backend.scripts.ingest_watch --skip-faiss    # Graphiti only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress LangChain deprecation warnings — nothing to migrate to yet
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage
from graphiti_core.nodes import EpisodeType

from backend.core.config import settings
from backend.services.graphiti_service import GraphitiService

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Do NOT call setup_logging() here — that writes to app.log which is the
# FastAPI server log. The ingest watcher is a separate process and should
# log to its own file AND the terminal independently.

def _setup_ingest_logging() -> logging.Logger:
    log_dir = Path("backend/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)

        # Terminal
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        root.addHandler(ch)

        # Ingest-specific log file (separate from app.log)
        fh = logging.FileHandler(log_dir / "ingest.log", encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Silence noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    return logging.getLogger("graphiti-rag.ingest")


logger = _setup_ingest_logging()


# ---------------------------------------------------------------------------
# Coref resolution — identical to agentic-graph-rag Tier 3
# ---------------------------------------------------------------------------

def resolve_coreferences(text: str, llm: ChatOllama) -> str:
    """
    Rewrites text replacing pronouns and partial names with full canonical forms.
    One LLM call per chunk. Skip with --skip-coref for faster dev ingestion.
    """
    prompt = (
        "Rewrite the following text replacing all partial names, pronouns, and aliases "
        "with their full canonical name as established in the text.\n\n"
        "Rules:\n"
        "- Replace pronouns (he, she, they, his, her) with the actual person's full name\n"
        "- Replace partial names with full names only if the full form is in this same text\n"
        "- If ambiguous or full form not present, leave as-is\n"
        "- Do NOT change any facts, dates, numbers, or relationships\n"
        "- Return ONLY the rewritten text, no explanation\n\n"
        f"Text:\n{text}\n\nRewritten text:"
    )
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        rewritten = result.content.strip()
        if len(rewritten) < len(text) * 0.5:
            logger.warning(
                "Coref result suspiciously short (%d vs %d chars) — using original",
                len(rewritten), len(text),
            )
            return text
        return rewritten
    except Exception as e:
        logger.error("Coref resolution failed: %s — using original", e)
        return text


# ---------------------------------------------------------------------------
# FAISS ingestion — delegates to existing ingest_multi_docs.py
# ---------------------------------------------------------------------------

def ingest_to_faiss(file_path: Path) -> bool:
    ingest_script = REPO_ROOT / "backend" / "scripts" / "ingest_multi_docs.py"
    sync_script   = REPO_ROOT / "backend" / "scripts" / "convert_meta_to_sqlite.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    cmd = [
        sys.executable, str(ingest_script),
        "--input",        str(file_path),
        "--out-index",    settings.FAISS_INDEX_PATH,
        "--out-meta",     settings.FAISS_META_PATH,
        "--cache-db",     settings.EMBEDDING_CACHE_DB,
        "--chunk-tokens", str(settings.CHUNK_TOKENS),
        "--overlap",      str(settings.CHUNK_OVERLAP),
        "--batch",        str(settings.EMBEDDING_BATCH_SIZE),
        "--append",
    ]
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        logger.error("FAISS ingest failed for %s", file_path.name)
        return False

    sync_cmd = [
        sys.executable, str(sync_script),
        "--meta", settings.FAISS_META_PATH,
        "--out",  settings.META_DB_PATH,
    ]
    sync_result = subprocess.run(sync_cmd, env=env)
    if sync_result.returncode != 0:
        logger.error("SQLite sync failed for %s", file_path.name)
        return False

    logger.info("FAISS + SQLite ingest complete: %s", file_path.name)
    return True


# ---------------------------------------------------------------------------
# Main per-file ingestion pipeline
# ---------------------------------------------------------------------------

async def ingest_file(
    file_path:    Path,
    graphiti_svc: GraphitiService,
    skip_coref:   bool = False,
    skip_faiss:   bool = False,
) -> None:
    logger.info("=== Ingesting: %s ===", file_path.name)

    # ── A. Load ──────────────────────────────────────────────────────────────
    try:
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path))
        docs = loader.load()
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path.name, e)
        return

    # ── B. Semantic chunking ──────────────────────────────────────────────────
    logger.info("Semantic chunking %s...", file_path.name)
    try:
        embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=settings.SEMANTIC_CHUNK_THRESHOLD_TYPE,
            breakpoint_threshold_amount=settings.SEMANTIC_CHUNK_BREAKPOINT,
        )
        chunks = splitter.split_documents(docs)
        logger.info("Created %d semantic chunks from %s", len(chunks), file_path.name)
    except Exception as e:
        logger.error("Semantic chunking failed: %s — falling back to full doc", e)
        chunks = docs

    # ── C. Coreference resolution ─────────────────────────────────────────────
    effective_coref = not skip_coref and settings.USE_COREF
    if effective_coref:
        llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)
        logger.info("Running coref resolution on %d chunks...", len(chunks))
        resolved = 0
        for i, chunk in enumerate(chunks):
            # FIX: per-chunk progress log so terminal is never silent
            logger.info("  Coref %d/%d...", i + 1, len(chunks))
            original = chunk.page_content
            chunk.page_content = resolve_coreferences(chunk.page_content, llm)
            if chunk.page_content != original:
                resolved += 1
        logger.info("Coref complete — changed %d/%d chunks", resolved, len(chunks))
    else:
        logger.info("Coref resolution skipped.")

    # ── D. Graphiti ingestion ─────────────────────────────────────────────────
    logger.info("Adding %d episodes to Graphiti...", len(chunks))
    episode_errors = 0
    for i, chunk in enumerate(chunks):
        episode_name = f"{file_path.stem}_chunk_{i:04d}"
        try:
            await graphiti_svc.client.add_episode(
                name=episode_name,
                episode_body=chunk.page_content,
                source=EpisodeType.text,
                source_description=f"File: {file_path.name}, chunk {i}/{len(chunks)}",
                reference_time=datetime.now(timezone.utc),
            )
            # FIX: changed from DEBUG to INFO so episode progress is visible
            logger.info("  Episode %d/%d added: %s", i + 1, len(chunks), episode_name)
        except Exception as e:
            episode_errors += 1
            logger.error("Failed to add episode %s: %s", episode_name, e)
            if episode_errors > len(chunks) // 2:
                logger.error(
                    "Too many episode failures — stopping Graphiti ingest for %s",
                    file_path.name,
                )
                break

    logger.info(
        "Graphiti ingest complete for %s — %d/%d episodes succeeded",
        file_path.name, len(chunks) - episode_errors, len(chunks),
    )

    # ── E. FAISS fallback ingest ──────────────────────────────────────────────
    if not skip_faiss:
        logger.info("Running FAISS fallback ingest for %s...", file_path.name)
        ingest_to_faiss(file_path)
    else:
        logger.info("FAISS ingest skipped (--skip-faiss).")


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

def find_files(dirpath: Path) -> list[Path]:
    exts = {".pdf", ".txt", ".md"}
    if not dirpath.exists():
        return []
    return [p for p in dirpath.glob("*") if p.suffix.lower() in exts and p.is_file()]


async def watch_loop(
    watch_dir:  Path,
    interval:   int,
    skip_coref: bool,
    skip_faiss: bool,
) -> None:
    graphiti_svc = GraphitiService()
    try:
        await graphiti_svc.client.build_indices_and_constraints()
        logger.info("Graphiti indices ready.")
    except Exception as e:
        logger.warning("Graphiti index build note (may already exist): %s", e)

    seen = {p.name for p in find_files(watch_dir)}
    logger.info(
        "Watcher started — watching: %s | skipping %d existing files | interval: %ds",
        watch_dir.resolve(), len(seen), interval,
    )
    logger.info(
        "Config — coref: %s | FAISS fallback: %s | chunking: semantic(%s, %d)",
        "enabled" if (not skip_coref and settings.USE_COREF) else "disabled",
        "enabled" if not skip_faiss else "disabled",
        settings.SEMANTIC_CHUNK_THRESHOLD_TYPE,
        settings.SEMANTIC_CHUNK_BREAKPOINT,
    )

    try:
        while True:
            await asyncio.sleep(interval)
            current = {p.name: p for p in find_files(watch_dir)}
            new_names = set(current.keys()) - seen

            for name in sorted(new_names):
                logger.info("New file detected: %s", name)
                try:
                    await ingest_file(
                        current[name],
                        graphiti_svc,
                        skip_coref=skip_coref,
                        skip_faiss=skip_faiss,
                    )
                except Exception:
                    logger.exception("Unhandled error ingesting %s", name)
                finally:
                    seen.add(name)

            deleted = seen - set(current.keys())
            for name in deleted:
                seen.discard(name)

    except asyncio.CancelledError:
        pass
    finally:
        await graphiti_svc.close()
        logger.info("Watcher stopped — Graphiti connection closed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch a folder and ingest new files into Graphiti + FAISS"
    )
    parser.add_argument("--watch",      default=settings.WATCH_DIR)
    parser.add_argument("--interval",   type=int, default=10)
    parser.add_argument("--skip-coref", action="store_true",
                        help="Skip coreference resolution (faster, less accurate)")
    parser.add_argument("--skip-faiss", action="store_true",
                        help="Skip FAISS fallback ingest (Graphiti only)")
    args = parser.parse_args()

    watch_dir = Path(args.watch)
    watch_dir.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(watch_loop(
            watch_dir=watch_dir,
            interval=args.interval,
            skip_coref=args.skip_coref,
            skip_faiss=args.skip_faiss,
        ))
    except KeyboardInterrupt:
        logger.info("Watcher stopped by user.")


if __name__ == "__main__":
    main()