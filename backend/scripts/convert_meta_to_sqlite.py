#!/usr/bin/env python3
"""
Convert metadata JSONL -> SQLite metadata_store.db

Usage:
  python backend/scripts/convert_meta_to_sqlite.py --meta backend/db/vector_data/knowledge_meta.jsonl --out backend/db/vector_data/metadata_store.db
"""
from __future__ import annotations
import argparse, json, sqlite3
from pathlib import Path

def create_db(out_path: Path):
    conn = sqlite3.connect(str(out_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY,
            pid INTEGER,
            doc_name TEXT,
            block_id INTEGER,
            start_token INTEGER,
            end_token INTEGER,
            text TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_name ON chunks(doc_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pid ON chunks(pid)")
    conn.commit()
    return conn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", default="backend/db/vector_data/metadata_store.db")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise SystemExit("meta not found: " + str(meta_path))

    out_path = Path(args.out)
    conn = create_db(out_path)
    cur = conn.cursor()
    inserted = 0
    with meta_path.open("r", encoding="utf8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                m = json.loads(ln)
            except Exception:
                continue
            chunk_id = m.get("chunk_id")
            pid = m.get("pid", m.get("pdf", None))
            doc_name = m.get("doc_name", m.get("pdf", m.get("pid", "")))
            block_id = m.get("block_id")
            start_token = m.get("start_token")
            end_token = m.get("end_token")
            text = m.get("text", "")
            try:
                cur.execute(
                    "INSERT OR REPLACE INTO chunks (chunk_id,pid,doc_name,block_id,start_token,end_token,text) VALUES (?,?,?,?,?,?,?)",
                    (chunk_id, pid, doc_name, block_id, start_token, end_token, text)
                )
                inserted += 1
            except Exception as e:
                print("Insert error:", e)
    conn.commit()
    conn.close()
    print(f"Wrote {inserted} rows to {out_path}")

if __name__ == "__main__":
    main()
