"""
Draft golden eval questions from chunks that are actually in the database.

Why not write them by hand? You should still review every one — but the
questions must be grounded in real corpus content, and sampling the database
guarantees that. The model drafts; you verify. A golden set you did not check
is not a golden set.

Includes deliberate negatives: questions the corpus cannot answer, where the
correct behaviour is refusal. Those are the most valuable rows in the file,
because hallucination under absence is the failure that costs an exhibitor money.

Usage:
    uv run python make_golden.py --per-show 12 -o evals/golden.jsonl
    uv run python make_golden.py --review evals/golden.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import psycopg
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("PG_DSN", "postgresql:///exhibitor")
MODEL = "claude-sonnet-4-5"
client = Anthropic()

DRAFT_PROMPT = """Here is a clause from a trade fair exhibitor manual.

show_id: {show_id}
clause_id: {clause_id}
section: {section_path}
---
{content}
---

Write ONE question a real exhibitor would ask, whose answer is contained in this
clause. Use the exhibitor's vocabulary, not the manual's — someone asks "can I
hang a banner?", not "what are the suspended rigging provisions?".

Then list the specific facts the answer must contain (numbers, deadlines,
conditions, exceptions). Be exact about figures and units as written.

Return ONLY JSON, no markdown fence:
{{"q": "...", "expected_facts": ["...", "..."], "difficulty": "easy|medium|hard"}}

Mark it "hard" if the clause contains an exception, a threshold with a
condition attached, or a rule that depends on stand size or hall."""

# Questions no exhibitor manual answers. Correct behaviour is explicit refusal.
NEGATIVES = [
    "What is the wifi password for the exhibitor lounge?",
    "How many visitors attended last year's edition?",
    "Can you recommend a good hotel near the venue?",
    "What is the phone number of my account manager?",
    "How much does a 6x6 stand cost to rent?",
    "Which competitors are exhibiting in my hall?",
    "What is the catering menu for the VIP lounge?",
    "Can I get a list of attendee email addresses?",
]


def sample_chunks(per_show: int) -> list[dict]:
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT show_id FROM chunks ORDER BY 1")
        shows = [r[0] for r in cur.fetchall()]

        out = []
        for show in shows:
            cur.execute(
                """
                SELECT show_id, clause_id, section_path, page, kind, content
                FROM chunks
                WHERE show_id = %s
                  AND length(content) BETWEEN 200 AND 2000
                  AND (clause_id IS NOT NULL OR kind = 'table_row')
                ORDER BY random()
                LIMIT %s
                """,
                (show, per_show),
            )
            cols = [d.name for d in cur.description]
            out.extend(dict(zip(cols, r)) for r in cur.fetchall())
    return out


def draft(chunk: dict) -> dict | None:
    resp = client.messages.create(
        model=MODEL, max_tokens=600,
        messages=[{"role": "user", "content": DRAFT_PROMPT.format(**chunk)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        item = json.loads(text)
    except json.JSONDecodeError:
        return None

    return {
        "q": item["q"],
        "show_id": chunk["show_id"],
        "expected_clause": chunk["clause_id"],
        "expected_facts": item.get("expected_facts", []),
        "difficulty": item.get("difficulty", "medium"),
        "answerable": True,
        "source_page": chunk["page"],
        "reviewed": False,
    }


def build(per_show: int, out_path: Path) -> None:
    chunks = sample_chunks(per_show)
    print(f"sampled {len(chunks)} chunks")

    rows = []
    for i, c in enumerate(chunks, 1):
        item = draft(c)
        if item:
            rows.append(item)
        print(f"  {i}/{len(chunks)}", end="\r")

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT show_id FROM chunks ORDER BY 1")
        shows = [r[0] for r in cur.fetchall()]

    for q in NEGATIVES:
        rows.append({
            "q": q,
            "show_id": random.choice(shows),
            "expected_clause": None,
            "expected_facts": [],
            "difficulty": "hard",
            "answerable": False,
            "source_page": None,
            "reviewed": False,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_neg = sum(1 for r in rows if not r["answerable"])
    print(f"\nwrote {len(rows)} rows ({n_neg} unanswerable) to {out_path}")
    print(f"\nNow review them:  uv run python make_golden.py --review {out_path}")


def review(path: Path) -> None:
    """Walk the file. Keep, edit, or drop each row. Nothing counts until reviewed."""
    rows = [json.loads(l) for l in path.open()]
    pending = [r for r in rows if not r["reviewed"]]
    print(f"{len(pending)} unreviewed of {len(rows)}\n")

    for r in rows:
        if r["reviewed"]:
            continue
        print("─" * 74)
        print(f"[{r['difficulty']}] {r['show_id']}  clause={r['expected_clause']}  p.{r['source_page']}")
        print(f"Q: {r['q']}")
        for f in r["expected_facts"]:
            print(f"   • {f}")
        ans = input("\n[k]eep  [e]dit question  [d]rop  [q]uit > ").strip().lower()

        if ans == "q":
            break
        if ans == "d":
            r["_drop"] = True
        elif ans == "e":
            new_q = input("new question: ").strip()
            if new_q:
                r["q"] = new_q
                print("⚠ expected_facts no longer match — [d]rop this row or edit the file by hand")
                r["_drop"] = input("drop? [y/N] > ").strip().lower() == "y"
            r["reviewed"] = True
        else:
            r["reviewed"] = True

    kept = [r for r in rows if not r.get("_drop")]
    with path.open("w") as f:
        for r in kept:
            r.pop("_drop", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_reviewed = sum(1 for r in kept if r["reviewed"])
    print(f"\n{n_reviewed}/{len(kept)} reviewed, {len(rows) - len(kept)} dropped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-show", type=int, default=12)
    ap.add_argument("-o", "--out", type=Path, default=Path("evals/golden.jsonl"))
    ap.add_argument("--review", type=Path, help="review an existing file")
    args = ap.parse_args()

    if args.review:
        review(args.review)
    else:
        build(args.per_show, args.out)