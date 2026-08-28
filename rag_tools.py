"""
Retrieval-backed tools, with programmatic citation verification.

The verification step is the thing that turns "trust me, it's grounded" into a
mechanical guarantee: parse the clause IDs out of the model's answer, check each
one appeared in the retrieved set, reject the answer if any did not. In a domain
where a wrong clause means a stand is rejected on build-up day, that check is
the difference between a product and a liability.

Usage:
    uv run python rag_tools.py "maximum stand height" --show buchmesse-technical
"""

from __future__ import annotations

import json
import os
import re

import psycopg
from pydantic import BaseModel

from search import search
from agent import Tool, run_loop, Budget, Trace

DSN = os.getenv("PG_DSN", "postgresql:///exhibitor")
RELEVANCE_FLOOR = 0.0        # cross-encoder score below this = not really relevant
TOP_K = 5

CLAUSE_CITE_RE = re.compile(r"\b(\d{1,2}(?:\.\d{1,2}){1,3})\b")


# ---------------------------------------------------------------- tools

class RegulationArgs(BaseModel):
    show_id: str
    query: str


class ShowArgs(BaseModel):
    pass


def normalise_show_id(show_id: str) -> str:
    """Models write 'Ambiente 2027'; the database holds 'ambiente-2027'.
    Without this, every real query silently returns zero rows."""
    return show_id.strip().lower().replace(" ", "-").replace("_", "-")


def list_shows() -> str:
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT show_id, count(*) FROM chunks GROUP BY 1 ORDER BY 1")
        return json.dumps([{"show_id": s, "clauses": n} for s, n in cur.fetchall()])


def search_regulations(show_id: str, query: str) -> str:
    show = normalise_show_id(show_id)
    rows = search(query, show_id=show, mode="hybrid", rerank=True, limit=TOP_K)

    kept = [r for r in rows if r.get("rerank_score", 1.0) > RELEVANCE_FLOOR]
    if not kept:
        return json.dumps({
            "show_id": show,
            "results": [],
            "note": ("Nothing in this show's manual matches that question. "
                     "Say so explicitly — do not answer from general knowledge."),
        })

    return json.dumps({
        "show_id": show,
        "results": [
            {
                "clause_id": r["clause_id"],
                "section": r["section_path"],
                "page": r["page"],
                "text": r["content"][:1200],
                "relevance": round(float(r.get("rerank_score", 0)), 2),
            }
            for r in kept
        ],
    }, ensure_ascii=False)


SEARCH_TOOL = Tool(
    name="search_regulations",
    description=(
        "Search a show's technical guidelines and exhibitor manual. Returns passages "
        "with clause IDs and page numbers. Use for any question about booth height, "
        "rigging, electrical, fire safety, escape routes, materials, or stand approval. "
        "Call once per distinct topic; call again with different wording if the first "
        "search returns nothing useful."
    ),
    args=RegulationArgs,
    fn=search_regulations,
)

LIST_SHOWS_TOOL = Tool(
    name="list_shows",
    description="List the shows whose manuals are available, with clause counts.",
    args=ShowArgs,
    fn=lambda: list_shows(),
)


# ---------------------------------------------------------------- verification
HALL_RE = re.compile(r"\bhalls?\b[^:\n]*", re.IGNORECASE)

def cited_clauses(answer: str) -> set[str]:
    masked = HALL_RE.sub(" ", answer)
    return set(CLAUSE_CITE_RE.findall(masked))

def retrieved_clauses(trace_payloads: list[str]) -> set[str]:
    out: set[str] = set()
    for payload in trace_payloads:
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
        for r in data.get("results", []):
            if r.get("clause_id"):
                out.add(r["clause_id"])
    return out


UNITS_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\s*(?:m|mm|cm|kg|kn|%|€|a\.m\.|p\.m\.)", re.IGNORECASE)

def verify_citations(answer, retrieved):
    measurements = {m.split()[0] for m in UNITS_RE.findall(answer)}
    claimed = cited_clauses(answer) - measurements
    invented = {c for c in claimed
                if c not in retrieved
                and not any(c.startswith(r + ".") for r in retrieved)}
    return (not invented), invented


SYSTEM = (
    "You answer exhibitor questions strictly from the show's technical guidelines.\n"
    "Rules:\n"
    "- Cite the clause ID in brackets after every factual claim, e.g. [3.6.4].\n"
    "- Never state a rule that did not appear in a search result. If the manual "
    "does not cover it, say so and tell the exhibitor to contact technical services.\n"
    "- Do not compare numbers yourself in prose without quoting both values from "
    "the source. State the limit and the exhibitor's value side by side.\n"
    "- If a search returns nothing, try different wording once, then stop."
)


def answer_question(question: str, show_id: str | None = None, verbose: bool = True) -> dict:
    budget, trace = Budget(limit_usd=0.50), Trace()
    prompt = question if not show_id else f"Show: {show_id}\n\n{question}"

    answer = run_loop(
        prompt,
        tools=[SEARCH_TOOL, LIST_SHOWS_TOOL],
        system=SYSTEM,
        max_iterations=6,
        budget=budget,
        trace=trace,
        label="rag",
        verbose=verbose,
    )

    # Re-run the searches the agent made to recover what was retrieved.
    payloads = []
    for line in trace.lines:
        m = re.search(r"search_regulations\((\{.*\})\)", line)
        if m:
            try:
                args = eval(m.group(1))          # trace lines are our own repr
                payloads.append(search_regulations(**args))
            except Exception:
                pass

    retrieved = retrieved_clauses(payloads)
    ok, invented = verify_citations(answer, retrieved)

    return {
        "answer": answer,
        "verified": ok,
        "invented_citations": sorted(invented),
        "retrieved_clauses": sorted(retrieved),
        "cost_usd": round(budget.spent, 4),
        "tool_calls": trace.tool_calls,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--show", dest="show_id", default=None)
    args = ap.parse_args()

    result = answer_question(args.question, args.show_id)

    print("\n" + "=" * 78)
    print(result["answer"])
    print("=" * 78)
    if result["verified"]:
        print(f"✓ all citations verified against {len(result['retrieved_clauses'])} retrieved clauses")
    else:
        print(f"✗ INVENTED CITATIONS: {result['invented_citations']}")
        print(f"  retrieved was: {result['retrieved_clauses']}")
    print(f"  ${result['cost_usd']} · {result['tool_calls']} tool calls")