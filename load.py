"""
Load chunks.jsonl into Postgres with locally-computed embeddings.

Embeddings run on-device via fastembed (bge-small-en-v1.5, 384 dims). No
document content leaves the machine — which matters when the corpus is
client-confidential exhibitor manuals, and removes per-query embedding
cost from the unit economics.

Usage:
    uv run python load.py chunks.jsonl
    uv run python load.py chunks.jsonl --truncate      # rebuild from scratch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from fastembed import TextEmbedding

DSN = os.getenv("PG_DSN", "postgresql:///exhibitor")
MODEL = "BAAI/bge-small-en-v1.5"
BATCH = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--truncate", action="store_true", help="clear the table first")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"{args.jsonl} not found — run ingest.py first")

    rows = [json.loads(line) for line in args.jsonl.open()]
    print(f"{len(rows)} chunks from {args.jsonl}")

    print(f"loading {MODEL} (first run downloads ~130MB)...")
    embedder = TextEmbedding(model_name=MODEL)

    with psycopg.connect(DSN, autocommit=False) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if args.truncate:
                cur.execute("TRUNCATE chunks RESTART IDENTITY")
                print("table truncated")

            for start in range(0, len(rows), BATCH):
                batch = rows[start:start + BATCH]
                vectors = list(embedder.embed([r["embed_text"] for r in batch]))

                cur.executemany(
                    """
                    INSERT INTO chunks
                      (show_id, doc_title, clause_id, section_path, page,
                       kind, part, content, embed_text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (r["show_id"], r["doc_title"], r["clause_id"], r["section_path"],
                         r["page"], r["kind"], r["part"], r["content"], r["embed_text"], v)
                        for r, v in zip(batch, vectors)
                    ],
                )
                print(f"  {min(start + BATCH, len(rows))}/{len(rows)}", end="\r")

        conn.commit()

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT show_id, kind, count(*), count(embedding) AS embedded
            FROM chunks GROUP BY show_id, kind ORDER BY show_id, kind
        """)
        print("\n")
        for show_id, kind, n, embedded in cur.fetchall():
            flag = "" if n == embedded else f"  ⚠ {n - embedded} missing embeddings"
            print(f"  {show_id:24} {kind:12} {n:4}{flag}")


if __name__ == "__main__":
    main()