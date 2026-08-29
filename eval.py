"""
Score the system against the golden set.

Two configurations, same questions, same corpus:
  single  — one agent, retrieval + list_shows
  multi   — supervisor delegating to compliance / deadline / drafting specialists

Four metrics, deliberately separate so a failure is diagnosable:
  retrieval    did the expected clause come back at all?      (retrieval stage)
  facts        does the answer contain the required figures?  (generation stage)
  refusal      on unanswerable questions, did it refuse?      (hallucination)
  citations    were all cited clauses actually retrieved?     (grounding)

Retrieval is scored deterministically. Facts and refusal use an LLM judge,
because "4,00 m" / "4.00 m" / "4 m" are the same fact and substring matching
would score a correct answer wrong.

Usage:
    uv run python eval.py --config single
    uv run python eval.py --compare
    uv run python eval.py --compare --limit 20        # cheaper smoke run
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

from agent import Tool, run_loop, Budget, Trace
from rag_tools import (
    SEARCH_TOOL, LIST_SHOWS_TOOL, RegulationArgs, SYSTEM,
    search_regulations, verify_citations,
)

client = Anthropic()
JUDGE_MODEL = "claude-sonnet-4-5"
GOLDEN = Path("evals/golden-all.jsonl")
WORKERS = 6


# ---------------------------------------------------------------- instrumentation

class Recorder:
    """Wraps a tool so we capture what retrieval actually returned, instead of
    reconstructing it from log lines afterwards. (FAILURES.md: observability
    designed after the fact.)"""

    def __init__(self):
        self.payloads: list[str] = []

    def wrap(self, tool: Tool) -> Tool:
        def recorded(**kw):
            out = tool.fn(**kw)
            self.payloads.append(out)
            return out
        return Tool(tool.name, tool.description, tool.args, recorded)

    @property
    def clauses(self) -> set[str]:
        found: set[str] = set()
        for p in self.payloads:
            try:
                for r in json.loads(p).get("results", []):
                    if r.get("clause_id"):
                        found.add(r["clause_id"])
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        return found


# ---------------------------------------------------------------- configurations

def run_single(question: str, show_id: str | None) -> dict:
    rec = Recorder()
    budget, trace = Budget(limit_usd=0.50), Trace()
    prompt = question if not show_id else f"Show: {show_id}\n\n{question}"

    answer = run_loop(
        prompt,
        tools=[rec.wrap(SEARCH_TOOL), LIST_SHOWS_TOOL],
        system=SYSTEM,
        max_iterations=6, budget=budget, trace=trace,
        label="single", verbose=False,
    )
    return {"answer": answer, "retrieved": rec.clauses,
            "cost": budget.spent, "tool_calls": trace.tool_calls}


COMPLIANCE_SYSTEM = (
    "You retrieve and report show regulations. Search the manual, then report "
    "findings verbatim with clause IDs. Do not draft prose for the exhibitor and "
    "do not infer rules that were not returned. If nothing relevant is found, say so."
)

DRAFT_SYSTEM = (
    "You write the reply to the exhibitor from findings you are given. Preserve every "
    "clause citation exactly. Add nothing that is not in the findings. If the findings "
    "say the manual does not cover it, say that plainly."
)

SUPERVISOR_SYSTEM = (
    "You coordinate specialists to answer an exhibitor's question. First call "
    "find_regulations with the exhibitor's question. Then call draft_reply with the "
    "findings, passing them through COMPLETE and UNABRIDGED — do not summarise or "
    "select. Return the drafted reply as your final answer."
)


class FindArgs(RegulationArgs):
    pass


class DraftArgs(RegulationArgs.__base__):          # pydantic BaseModel
    findings: str


def run_multi(question: str, show_id: str | None) -> dict:
    rec = Recorder()
    budget, trace = Budget(limit_usd=1.00), Trace()

    def find_regulations(show_id: str, query: str) -> str:
        return run_loop(
            f"Show: {show_id}\nQuestion: {query}",
            tools=[rec.wrap(SEARCH_TOOL)],
            system=COMPLIANCE_SYSTEM,
            max_iterations=5, budget=budget, trace=trace,
            depth=1, label="compliance", verbose=False,
        )

    def draft_reply(findings: str) -> str:
        return run_loop(
            findings, tools=[], system=DRAFT_SYSTEM,
            max_iterations=2, budget=budget, trace=trace,
            depth=1, label="drafting", verbose=False,
        )

    specialists = [
        Tool("find_regulations",
             "Search the show manual and report findings with clause IDs.",
             FindArgs, find_regulations),
        Tool("draft_reply",
             "Write the exhibitor-facing reply from findings. Call last.",
             DraftArgs, draft_reply),
    ]

    prompt = question if not show_id else f"Show: {show_id}\n\n{question}"
    answer = run_loop(prompt, tools=specialists, system=SUPERVISOR_SYSTEM,
                      max_iterations=6, budget=budget, trace=trace,
                      label="supervisor", verbose=False)

    return {"answer": answer, "retrieved": rec.clauses,
            "cost": budget.spent, "tool_calls": trace.tool_calls}


CONFIGS = {"single": run_single, "multi": run_multi}


# ---------------------------------------------------------------- scoring

JUDGE_PROMPT = """Question asked: {q}

Answer given:
---
{answer}
---

Facts the answer must contain:
{facts}

For each required fact, decide whether the answer states it. Numbers written
differently but meaning the same thing count as present ("4,00 m" = "4.00 m" = "4 m").
A fact hedged as uncertain still counts as present.

Also decide: does the answer REFUSE — i.e. state that the manual does not cover
this and the exhibitor should ask someone else? An answer that supplies substantive
regulatory content is not a refusal, even if it adds a caveat.

Return ONLY JSON: {{"present": <int>, "total": <int>, "refused": <bool>}}"""


def judge(q: str, answer: str, facts: list[str]) -> dict:
    facts_txt = "\n".join(f"- {f}" for f in facts) or "(none — this question is unanswerable)"
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=300,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(q=q, answer=answer, facts=facts_txt)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"present": 0, "total": len(facts), "refused": False}


def clause_hit(expected: str | None, retrieved: set[str]) -> bool | None:
    """None = not applicable (table rows carry no clause_id).
    A parent counts: retrieving 3.5.6 when 3.5.6.2 was expected is a partial win."""
    if not expected:
        return None
    if expected in retrieved:
        return True
    return any(expected.startswith(r + ".") or r.startswith(expected + ".")
               for r in retrieved)


@dataclass
class Scores:
    rows: list[dict] = field(default_factory=list)

    def add(self, **kw):
        self.rows.append(kw)

    def summary(self) -> dict:
        ans = [r for r in self.rows if r["answerable"]]
        neg = [r for r in self.rows if not r["answerable"]]
        applicable = [r for r in ans if r["clause_hit"] is not None]

        return {
            "n": len(self.rows),
            "retrieval": sum(r["clause_hit"] for r in applicable) / max(len(applicable), 1),
            "facts": sum(r["fact_ratio"] for r in ans) / max(len(ans), 1),
            "refusal": sum(r["refused"] for r in neg) / max(len(neg), 1),
            "hallucinated": sum(not r["refused"] for r in neg),
            "citations": sum(r["citations_ok"] for r in self.rows) / max(len(self.rows), 1),
            "cost": sum(r["cost"] for r in self.rows),
            "latency": sum(r["latency"] for r in self.rows) / max(len(self.rows), 1),
        }


def score_one(row: dict, config: str) -> dict:
    fn = CONFIGS[config]
    t0 = time.monotonic()
    try:
        out = fn(row["q"], row.get("show_id"))
    except Exception as e:
        return {"q": row["q"], "answerable": row["answerable"], "clause_hit": False,
                "fact_ratio": 0.0, "refused": False, "citations_ok": False,
                "cost": 0.0, "latency": time.monotonic() - t0, "error": str(e)[:120]}

    latency = time.monotonic() - t0
    verdict = judge(row["q"], out["answer"], row["expected_facts"])
    ok, invented = verify_citations(out["answer"], out["retrieved"])

    return {
        "q": row["q"],
        "answerable": row["answerable"],
        "clause_hit": clause_hit(row["expected_clause"], out["retrieved"]),
        "fact_ratio": verdict["present"] / max(verdict["total"], 1),
        "refused": verdict["refused"],
        "citations_ok": ok,
        "invented": sorted(invented),
        "cost": out["cost"],
        "latency": latency,
        "answer": out["answer"][:400],
    }


def run(config: str, rows: list[dict]) -> Scores:
    scores = Scores()
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(score_one, r, config): r for r in rows}
        for fut in as_completed(futures):
            scores.rows.append(fut.result())
            done += 1
            print(f"  {config}: {done}/{len(rows)}", end="\r")
    print()
    return scores


def report(name: str, s: Scores) -> None:
    m = s.summary()
    print(f"\n{name}")
    print(f"  retrieval recall     {m['retrieval']:.0%}")
    print(f"  fact coverage        {m['facts']:.0%}")
    print(f"  correct refusal      {m['refusal']:.0%}   ({m['hallucinated']} hallucinated)")
    print(f"  citations verified   {m['citations']:.0%}")
    print(f"  cost                 ${m['cost']:.3f}   (${m['cost']/max(m['n'],1):.4f}/question)")
    print(f"  mean latency         {m['latency']:.1f}s")

    worst = sorted((r for r in s.rows if r["answerable"]), key=lambda r: r["fact_ratio"])[:3]
    if worst:
        print("  worst answers:")
        for r in worst:
            print(f"    {r['fact_ratio']:.0%}  {r['q'][:64]}")
    liars = [r for r in s.rows if not r["answerable"] and not r["refused"]]
    for r in liars:
        print(f"  ⚠ answered an unanswerable question: {r['q'][:64]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="single")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--limit", type=int)
    ap.add_argument("-o", "--out", type=Path, default=Path("evals/results.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.golden.open()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} questions "
          f"({sum(1 for r in rows if not r['answerable'])} unanswerable)")

    configs = list(CONFIGS) if args.compare else [args.config]
    results = {}
    for c in configs:
        s = run(c, rows)
        report(c.upper(), s)
        results[c] = {"summary": s.summary(), "rows": s.rows}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nfull results → {args.out}")


if __name__ == "__main__":
    main()