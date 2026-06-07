#!/usr/bin/env python3
"""
Ingest multiple documents (PDF or .txt) into a FAISS index + metadata JSONL.

Features:
- Walks an input directory (or accepts a single file).
- Extracts text from PDFs (PyPDF2) or reads .txt files.
- Splits into paragraph blocks, then token-chunks using tiktoken (if installed) or whitespace.
- Uses backend.services.embed_cache_service.EmbedCacheService to avoid re-embedding identical text.
- Uses backend.tools.embedder.Embedder to compute embeddings for cache misses.
- Supports indexing metric: 'l2' (IndexFlatL2) or 'cosine' (IndexFlatIP with normalized vectors).
- Can append to an existing FAISS index (incremental) or create a fresh one.
- Emits metadata JSONL file with one object per chunk: {chunk_id, pid, doc_name, block_id, start_token, end_token, text}
- Atomic write strategy for index + metadata.

Example:
  python backend/scripts/ingest_multi_docs.py \
    --input knowledge \
    --out-index backend/db/vector_data/knowledge_faiss.index \
    --out-meta backend/db/vector_data/knowledge_meta.jsonl \
    --batch 64 --chunk-tokens 512 --overlap 128 --metric cosine

"""
from __future__ import annotations
import sys
from pathlib import Path
# Ensure repo root is on sys.path BEFORE any package imports that expect `backend` package.
REPO_ROOT = Path(__file__).resolve().parents[2]  # script is backend/scripts -> parents[2] => repo root
sys.path.insert(0, str(REPO_ROOT))


import argparse
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import numpy as np
import faiss
import time
import tempfile
import shutil
import logging

# ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.embed_cache_service import EmbedCacheService
from backend.tools.embedder import Embedder

# PDF reader
from PyPDF2 import PdfReader

# Tokenizer (tiktoken preferred)
try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
    def tokenize_ids(text: str) -> List[int]:
        return ENCODER.encode(text)
    def detokenize_ids(ids: List[int]) -> str:
        return ENCODER.decode(ids)
    TOKENIZER_NAME = "tiktoken"
except Exception:
    TOKENIZER_NAME = "whitespace"
    def tokenize_ids(text: str):
        return text.split()
    def detokenize_ids(tokens):
        if not tokens:
            return ""
        if isinstance(tokens[0], int):
            return " ".join(str(t) for t in tokens)
        return " ".join(tokens)

LOG = logging.getLogger("ingest_multi_docs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf8", errors="ignore")


def chunk_tokens_by_stride(text: str, chunk_tokens: int = 512, overlap: int = 128) -> List[Tuple[int,int,str]]:
    toks = tokenize_ids(text)
    stride = max(1, chunk_tokens - overlap)
    chunks = []
    # detect numeric token ids or whitespace tokens
    if toks and isinstance(toks[0], int):
        for start in range(0, len(toks), stride):
            end = min(start + chunk_tokens, len(toks))
            chunk_ids = toks[start:end]
            chunk_text = detokenize_ids(chunk_ids)
            chunks.append((start, end, chunk_text))
            if end == len(toks):
                break
    else:
        for start in range(0, len(toks), stride):
            end = min(start + chunk_tokens, len(toks))
            tks = toks[start:end]
            chunk_text = detokenize_ids(tks)
            chunks.append((start, end, chunk_text))
            if end == len(toks):
                break
    return chunks


def find_documents(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    docs = []
    for ext in ("*.pdf","*.txt"):
        docs.extend(sorted(p.glob("**/"+ext)))
    return docs


def load_existing_index(index_path: Path) -> Optional[faiss.Index]:
    if not index_path.exists():
        return None
    return faiss.read_index(str(index_path))


def save_faiss_atomic(index: faiss.Index, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(out_path)


def append_to_index(orig_index: Optional[faiss.Index], vectors: np.ndarray, metric: str) -> faiss.Index:
    if orig_index is None:
        dim = vectors.shape[1]
        if metric == "l2":
            idx = faiss.IndexFlatL2(dim)
        else:
            # cosine -> use inner product index; ensure vectors are normalized before adding
            idx = faiss.IndexFlatIP(dim)
        idx.add(vectors)
        return idx
    else:
        # If existing index dim != new dim, raise
        if orig_index.d != vectors.shape[1]:
            raise RuntimeError(f"Index dim mismatch: index.d={orig_index.d} vs vectors.shape[1]={vectors.shape[1]}")
        orig_index.add(vectors)
        return orig_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input file or directory (PDFs / .txt)")
    parser.add_argument("--out-index", default="backend/db/vector_data/knowledge_faiss.index", help="Output FAISS index path")
    parser.add_argument("--out-meta", default="backend/db/vector_data/knowledge_meta.jsonl", help="Output metadata jsonl path")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--metric", choices=["l2","cosine"], default="l2", help="Distance metric for FAISS; cosine uses IndexFlatIP with normalized vectors")
    parser.add_argument("--embed-model", default=None, help="Embedder model override (env default used if not set)")
    parser.add_argument("--cache-db", default="backend/db/embedding_cache/embed_cache.sqlite", help="Embed cache sqlite path")
    parser.add_argument("--append", action="store_true", help="Append to existing index/meta instead of overwriting")
    args = parser.parse_args()

    input_path = args.input
    out_index = Path(args.out_index)
    out_meta = Path(args.out_meta)
    embed_model = args.embed_model

    LOG.info("Ingest: input=%s metric=%s tokenizer=%s append=%s", input_path, args.metric, TOKENIZER_NAME, args.append)
    docs = find_documents(input_path)
    LOG.info("Found %d documents", len(docs))
    if not docs:
        LOG.error("No documents found in %s", input_path)
        return

    # init services
    embed_cache = EmbedCacheService(db_path=args.cache_db)
    embedder = Embedder(model=embed_model) if embed_model else Embedder()
    LOG.info("Embedder model: %s", getattr(embedder, "model", "(unknown)"))

    # optionally load existing meta and index if append
    existing_meta = []
    next_chunk_id = 0
    if args.append and out_meta.exists():
        with out_meta.open("r", encoding="utf8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    obj = json.loads(ln)
                    existing_meta.append(obj)
                except Exception:
                    continue
        if existing_meta:
            next_chunk_id = max(m.get("chunk_id", -1) for m in existing_meta) + 1
    LOG.info("Starting chunk id at %d", next_chunk_id)

    existing_index = None
    if args.append and out_index.exists():
        try:
            existing_index = load_existing_index(out_index)
            LOG.info("Loaded existing index ntotal=%s dim=%s", existing_index.ntotal, existing_index.d)
        except Exception as e:
            LOG.exception("Failed to read existing index: %s", e)
            existing_index = None

    all_meta_to_write = []
    all_vectors = []
    # Process each document
    for doc_id, path in enumerate(docs):
        LOG.info("Processing document (%d/%d): %s", doc_id+1, len(docs), path)
        try:
            if path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(path)
            else:
                text = read_text_file(path)
        except Exception as e:
            LOG.exception("Failed to read %s: %s", path, e)
            continue

        # produce paragraph blocks
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        blocks = []
        tmp = []
        tmp_chars = 0
        for p in paras:
            tmp.append(p)
            tmp_chars += len(p)
            if tmp_chars > 4000:
                blocks.append(" ".join(tmp))
                tmp = []
                tmp_chars = 0
        if tmp:
            blocks.append(" ".join(tmp))

        LOG.info("Created %d blocks", len(blocks))

        # chunk each block and collect chunk_texts + meta
        chunk_texts = []
        chunk_meta = []
        for blk_id, blk in enumerate(blocks):
            chunks = chunk_tokens_by_stride(blk, chunk_tokens=args.chunk_tokens, overlap=args.overlap)
            for (s,e,ctext) in chunks:
                meta = {
                    "chunk_id": next_chunk_id,
                    "pid": doc_id,
                    "doc_name": str(path.name),
                    "block_id": blk_id,
                    "start_token": int(s),
                    "end_token": int(e),
                }
                chunk_meta.append(meta)
                chunk_texts.append(ctext)
                next_chunk_id += 1

        LOG.info("Document produced %d chunks", len(chunk_texts))
        if not chunk_texts:
            continue

        # Batch embed with cache assistance
        n_total = len(chunk_texts)
        dim = None
        vectors = np.zeros((n_total, 1), dtype=np.float32)  # placeholder, will be resized
        idx = 0
        for i in range(0, n_total, args.batch):
            batch_texts = chunk_texts[i:i+args.batch]
            # ask cache first and compute missing
            cached_vecs, _keys = embed_cache.get_batch(batch_texts, embedder.model)
            miss_idx = [j for j,v in enumerate(cached_vecs) if v is None]
            # results list aligned to batch_texts
            results = [np.asarray(v, dtype=np.float32) if v is not None else None for v in cached_vecs]
            if miss_idx:
                to_compute = [batch_texts[j] for j in miss_idx]
                try:
                    computed = embedder.embed_batch(to_compute)
                except Exception:
                    computed = [embedder.embed(t) for t in to_compute]
                computed = [np.asarray(c, dtype=np.float32) for c in computed]
                # fill in results and write back to cache
                for local_i, vec in enumerate(computed):
                    global_pos = miss_idx[local_i]
                    results[global_pos] = vec
                try:
                    embed_cache.set_batch(to_compute, embedder.model, computed)
                except Exception:
                    LOG.exception("Failed to set embed cache batch")

            # determine dim if not set
            for r in results:
                if dim is None and r is not None:
                    dim = int(r.shape[0])

            # stack results
            batch_arr = np.vstack([np.asarray(r, dtype=np.float32) for r in results])
            if vectors.shape[1] == 1 and vectors.shape[0] == n_total and dim is not None:
                vectors = np.zeros((n_total, dim), dtype=np.float32)
            vectors[idx: idx + batch_arr.shape[0], :] = batch_arr
            idx += batch_arr.shape[0]

        # optional normalization for cosine
        if args.metric == "cosine":
            # normalize each vector to unit L2
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            vectors = vectors / norms

        # add to global arrays
        all_vectors.append(vectors)
        all_meta_to_write.extend([{**m, "text": chunk_texts[i]} for i,m in enumerate(chunk_meta)])

    if not all_meta_to_write:
        LOG.error("No chunks produced overall; exiting.")
        return

    # combine vectors
    vectors_cat = np.vstack(all_vectors)
    LOG.info("Total vectors to add: %d dim=%d", vectors_cat.shape[0], vectors_cat.shape[1])

    # build/append FAISS index
    if existing_index is not None:
        final_index = append_to_index(existing_index, vectors_cat, args.metric)
    else:
        final_index = append_to_index(None, vectors_cat, args.metric)

    # save index atomically
    LOG.info("Saving FAISS index to %s", out_index)
    save_faiss_atomic(final_index, out_index)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    # write metadata: if append, append lines; otherwise write new file
    mode = "a" if (args.append and out_meta.exists()) else "w"
    LOG.info("Writing metadata to %s (mode=%s)", out_meta, mode)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf8", dir=str(out_meta.parent)) as tf:
        for m in (existing_meta if (args.append and existing_meta) else []) + all_meta_to_write:
            tf.write(json.dumps(m, ensure_ascii=False) + "\n")
        tf.flush()
        tmpname = tf.name
    # atomic replace
    shutil.move(tmpname, out_meta)

    LOG.info("Done. Wrote %d metadata entries. Index ntotal=%d", len(all_meta_to_write) + (len(existing_meta) if existing_meta else 0), final_index.ntotal)


if __name__ == "__main__":
    main()
