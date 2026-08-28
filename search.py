"""
Hybrid retrieval over exhibitor manuals.

Four stages, each of which you can turn off to see what it buys:

  1. vector search   — semantic, finds paraphrases, blind to identifiers
  2. keyword search  — exact, finds "Form E-4" and "4,00 m", blind to synonyms
  3. RRF fusion      — combines the two rankings without normalising scores
  4. cross-encoder   — rereads query+passage together, reorders the top 20

The show_id filter is applied inside both searches. It is not a performance
detail: returning one show's rules for another show is a correctness failure,
and in a multi-tenant platform it is the isolation boundary.

Usage:
    uv run python search.py "maximum stand height"
    uv run python search.py "escape route width" --show fachdental-frankfurt
    uv run python search.py "Form E-4 deadline" --compare
"""

from __future__ import annotations

import argparse
import os
import textwrap

import psycopg
from pgvector.psycopg import register_vector
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

DSN = os.getenv("PG_DSN", "postgresql:///exhibitor")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 50          # per retrieval arm
FUSED = 20               # what goes into the reranker
RRF_K = 60               # standard smoothing constant

_embedder: TextEmbedding | None = None
_reranker: TextCrossEncoder | None = None


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


# ---------------------------------------------------------------- SQL

HYBRID_SQL = """
WITH vec AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(qvec)s) AS rank
    FROM chunks
    WHERE (%(show_id)s::text IS NULL OR show_id = %(show_id)s::text)
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(k)s
),
kw AS (
    SELECT id, ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %(q)s)) DESC) AS rank
    FROM chunks
    WHERE (%(show_id)s::text IS NULL OR show_id = %(show_id)s::text)
      AND tsv @@ plainto_tsquery('english', %(q)s)
    LIMIT %(k)s
)
SELECT c.id, c.show_id, c.clause_id, c.section_path, c.page, c.kind, c.content,
       vec.rank AS vec_rank,
       kw.rank  AS kw_rank,
       COALESCE(1.0 / (%(rrf_k)s + vec.rank), 0)
     + COALESCE(1.0 / (%(rrf_k)s + kw.rank),  0) AS rrf
FROM vec
FULL OUTER JOIN kw USING (id)
JOIN chunks c ON c.id = COALESCE(vec.id, kw.id)
ORDER BY rrf DESC
LIMIT %(limit)s
"""

SINGLE_ARM_SQL = {
    "vector": """
        SELECT id, show_id, clause_id, section_path, page, kind, content,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %(qvec)s) AS vec_rank,
               NULL::bigint AS kw_rank, 0.0 AS rrf
        FROM chunks
        WHERE (%(show_id)s::text IS NULL OR show_id = %(show_id)s::text)
        ORDER BY embedding <=> %(qvec)s
        LIMIT %(limit)s
    """,
    "keyword": """
        SELECT id, show_id, clause_id, section_path, page, kind, content,
               NULL::bigint AS vec_rank,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %(q)s)) DESC) AS kw_rank,
               0.0 AS rrf
        FROM chunks
        WHERE (%(show_id)s::text IS NULL OR show_id = %(show_id)s::text)
          AND tsv @@ plainto_tsquery('english', %(q)s)
        LIMIT %(limit)s
    """,
}


# ---------------------------------------------------------------- retrieval

def _rows_to_dicts(cur) -> list[dict]:
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def search(
    query: str,
    show_id: str | None = None,
    mode: str = "hybrid",          # hybrid | vector | keyword
    rerank: bool = True,
    limit: int = 5,
) -> list[dict]:
    qvec = list(embedder().query_embed(query))[0]

    params = {
        "q": query,
        "qvec": qvec,
        "show_id": show_id,
        "k": CANDIDATES,
        "rrf_k": RRF_K,
        "limit": FUSED if rerank else limit,
    }

    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(HYBRID_SQL if mode == "hybrid" else SINGLE_ARM_SQL[mode], params)
            rows = _rows_to_dicts(cur)

    if not rows:
        return []

    if rerank:
        scores = list(reranker().rerank(query, [r["content"] for r in rows]))
        for r, s in zip(rows, scores):
            r["rerank_score"] = s
        rows.sort(key=lambda r: r["rerank_score"], reverse=True)
        for i, r in enumerate(rows, 1):
            r["final_rank"] = i
        rows = rows[:limit]

    return rows


# ---------------------------------------------------------------- display

def show(rows: list[dict], title: str) -> None:
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")
    if not rows:
        print("  (nothing)")
        return
    for i, r in enumerate(rows, 1):
        ranks = f"vec={r['vec_rank'] or '–'} kw={r['kw_rank'] or '–'}"
        if r.get("rerank_score") is not None:
            ranks += f" rerank={r['rerank_score']:.2f}"
        print(f"\n{i}. [{r['clause_id'] or r['kind']}] {r['show_id']} p.{r['page']}   {ranks}")
        print(f"   {r['section_path'][:88]}")
        body = textwrap.shorten(r["content"], width=220, placeholder=" …")
        print(textwrap.fill(body, width=76, initial_indent="   ", subsequent_indent="   "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--show", dest="show_id", default=None, help="filter to one show_id")
    ap.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("-n", "--limit", type=int, default=5)
    ap.add_argument("--compare", action="store_true",
                    help="run all four configurations side by side")
    args = ap.parse_args()

    if args.compare:
        show(search(args.query, args.show_id, "vector", False, args.limit),
             "VECTOR ONLY — semantic, blind to identifiers")
        show(search(args.query, args.show_id, "keyword", False, args.limit),
             "KEYWORD ONLY — exact, blind to synonyms")
        show(search(args.query, args.show_id, "hybrid", False, args.limit),
             "HYBRID (RRF) — fused, not reranked")
        show(search(args.query, args.show_id, "hybrid", True, args.limit),
             "HYBRID + CROSS-ENCODER RERANK")
    else:
        show(search(args.query, args.show_id, args.mode, not args.no_rerank, args.limit),
             f"{args.mode}{'' if args.no_rerank else ' + rerank'}"
             + (f"  ·  show_id={args.show_id}" if args.show_id else ""))


if __name__ == "__main__":
    main()