# Exhibitor Manual Copilot — Architecture

A grounded question-answering system over trade fair exhibitor manuals. Built
in six days as a working prototype; every claim below is measured against a
69-question evaluation set, not asserted.

---

## The problem

Trade fair organisers publish technical guidelines — stand height limits,
rigging rules, fire regulations, escape route widths, submission deadlines —
as 40–60 page PDFs, different for every show and often every hall. Exhibitor
services teams answer the same questions by hand before every show.

The failure that matters is not an unhelpful answer. It is a **confident wrong
answer**: an exhibitor builds to a rule that does not apply, and the stand is
rejected at build-up, in another country, days before opening. Every design
decision below follows from that asymmetry.

---

## Pipeline

```
PDF → structural chunking → embed → Postgres (pgvector + tsvector)
                                        ↓
question → hybrid search (vector + BM25, RRF) → rerank → relevance floor
                                        ↓
                        agent loop with guards → answer
                                        ↓
                          citation verification → accept / reject
```

---

## Decisions and why

### Structural chunking, not fixed windows

Chunk boundaries follow the document's own numbering. A clause and its
exception must land in the same chunk: "island stands may not exceed 4.0 m"
separated from "structures above 2.5 m require written approval" produces an
answer that is technically true and operationally wrong.

Chunks carry `show_id`, `clause_id`, `section_path`, `page`, `kind`. The
section path is **prepended before embedding** — "Maximum 4,00 m" embeds
almost meaninglessly; "Stand Construction > Height > Island Stands: Maximum
4,00 m" embeds well. Higher leverage than the choice of embedding model.

Deadline and dimension tables are chunked **one row per chunk**. In testing,
table rows outranked prose clauses on retrieval for questions about specific
figures — the numbers exhibitors actually need live in tables.

*Measured:* 463 chunks from 3 manuals (Buchmesse, Fachdental Frankfurt,
Mesago Formnext). A 4th document yielded zero chunks — see Known limitations.

### Postgres, not a dedicated vector database

pgvector for embeddings, a `GENERATED ALWAYS` tsvector column for keyword
search. Keyword indexing stays in sync with content with zero application
code. At this scale a separate vector store adds an operational surface and
buys nothing; the threshold to revisit is roughly 10M chunks or a need for
distributed sharding.

### Local embeddings

`bge-small-en-v1.5` (384d) via fastembed, on-device. Exhibitor manuals are
client-confidential documents, and "no document content leaves our
infrastructure" is a procurement answer as much as a technical one. It also
removes per-query embedding cost from the unit economics.

### Hybrid search with reciprocal rank fusion

Vector search alone fails on exactly what this domain is full of: clause
numbers, form names, hall identifiers, measurements. "Form E-4" carries almost
no semantic signal. Keyword search alone fails on the vocabulary gap between
how exhibitors ask ("can I hang a banner?") and how manuals write ("suspended
rigging provisions").

RRF (k=60) fuses the two rankings without needing to normalise between two
incomparable scoring systems.

### Cross-encoder reranking

Top 20 from RRF → `ms-marco-MiniLM-L-6-v2` → top 5. A cross-encoder reads
query and passage together rather than comparing independently computed
vectors.

*Observed:* on one query, the top keyword hit scored **-0.43** and was demoted
to 4th; a passage arriving at vector rank 4 with no keyword match was promoted
to 2nd. The reranker corrects both arms.

### Relevance floor → honest refusal

Any result scoring at or below 0.0 after reranking is dropped. When nothing
survives, the tool returns an explicit instruction to tell the user the manual
does not cover it.

*Measured:* **100% correct refusal on 16 unanswerable questions.** This is the
single most important number in the system.

### Citation verification

Clause IDs are parsed out of the answer and checked against what retrieval
actually returned. A citation not in the retrieved set was invented.

*Caught in practice:* the model retrieved clause `3.5.6` and cited `3.5.6.4` —
plausible, specific, and unverifiable by reading. Descendants of a retrieved
clause are now accepted, which is a compromise; the correct fix is leaf-level
chunking so citations resolve exactly.

### Comparison in code, not in prose

`compare_to_limit(value, limit)` returns a computed verdict.

*Why it exists:* in testing the system stated that a 3.5 m sign **"exceeds the
4.0 m maximum height"** and recommended the exhibitor lower it — advice that
would have caused an unnecessary redesign. Models reason over "3.5" and "4.0"
as text. Any threshold an exhibitor depends on belongs in code.

---

## Single agent vs. multi-agent — measured

69 golden questions (16 unanswerable), same corpus, same tools.

| | single | supervisor + specialists |
|---|---|---|
| retrieval recall | 76% | **86%** |
| fact coverage | **65%** | 46% |
| correct refusal | **100%** | 94% (1 hallucinated) |
| citations verified | 93% | **96%** |
| cost / question | **$0.0144** | $0.0872 |
| mean latency | **12.1s** | 53.5s |

**The multi-agent system retrieves better and answers worse.** It finds the
right clause more often, then loses facts at the handoff, where the supervisor
summarises specialist findings before passing them on. An explicit instruction
to pass findings "complete and unabridged" did not prevent this.

It also answered one unanswerable question, where the single agent refused all
16.

**Decision: single agent with well-scoped tools.** Revisit when a specialist
genuinely needs different tools, and structure the handoff as typed data
(`{clause_id, rule, condition}`) rather than prose. Prose between components
gets rewritten; structure does not.

This conclusion is worth more than the architecture choice. Three earlier
single-run comparisons produced three *contradictory* verdicts — one of which
turned out to be an artifact of a test stub more permissive than reality. At
n=69 the result is stable across repeat runs (76%/65% then 76%/64%).
**n=1 cannot distinguish architecture from noise.**

---

## Guards in the agent loop

Every tool call passes: unknown-tool check → schema validation → repetition
detection → timeout → exception capture → output truncation. Every failure
path returns an error *message to the model* rather than raising, because a
model told what went wrong usually recovers.

The repetition guard hashes `(name, sorted(args))` and short-circuits after
three identical calls. Repetition loops are the most common agent failure in
production and are rarely guarded for.

On hitting the iteration limit, one final call is made with tools removed,
asking for a best-effort answer. Failing loudly with nothing is worse than
degrading.

A shared `Budget` object is threaded through the whole call tree so nested
agents cannot spend without limit. This is also where per-tenant cost
attribution would attach.

---

## Known limitations

**Multi-column PDF extraction.** Naive geometric sorting interleaves columns
word by word; document order is better but produces occasional out-of-order
section paths. Correct fix is block clustering by x-coordinate.

**Silent zero-yield ingest.** One manual produced 0 chunks — both parsers
failed on a malformed PDF. Fault isolation means the run completes, but a
document that yields nothing looks like success. Needs a minimum-yield
assertion per document.

**Citation verification by regex over prose** is a stopgap. Distinguishing a
clause number from a hall number from a measurement is genuinely ambiguous in
free text. The right design has the model emit citations in a structured field.

**The judge is the same model family being judged** and may be generous to its
own phrasing. Needs a different judge model and a human-scored subset to
calibrate against.

**`trust` auth on localhost Postgres.** Fine on a laptop, wrong everywhere
else. Needs a real role and `scram-sha-256` before anything is deployed.

---

## What would change in Rust

- **Structured concurrency.** A tool timeout in Python leaks its thread —
  `cancel_futures` cannot stop work already running, and Python has no way to
  kill a thread. Measured directly: a 10s timeout returned after 30s until the
  executor lifecycle was managed manually. Tokio's `CancellationToken` and
  `JoinSet` give real cooperative cancellation with structured cleanup, which
  matters for a service holding thousands of long-lived agent sessions against
  flaky external tools.
- **Type-safe tool schemas** via derive macros — one definition serving both
  the model-facing schema and runtime validation. The prototype achieves this
  with Pydantic after an early bug where two tools shared one validation model.
- **Parallel tool execution** with proper cancellation semantics rather than a
  thread pool.

---

## MCP boundary

Tools are exposed over MCP (`exhibitor-ops`): `list_shows`,
`search_regulations`, `get_clause`, `compare_to_limit`.

The runtime does not know about trade fairs. It knows about tools. Exhibitor
regulations are one server; a floor-plan service would be another; the next
vertical is a third. Same loop, same guards, same eval harness — **verticals
plug in rather than forking the product.**

Note that safety constraints must live in the *tool descriptions*, not the
system prompt. A consuming agent never sees your prompt; the description is
the only thing that travels with the tool.

**Dependency risk:** the MCP Python SDK went 1.x → 2.0 with breaking renames
on 2026-07-28 and broke every package that declared an unbounded `mcp>=1.x`.
A tool layer built on this needs explicit upper bounds and a vendoring policy.

---

## Cost

$0.0144 per question at current pricing, 12.1s mean latency. Prompt caching is
not yet enabled; tool schemas are re-sent on every call in the loop and are
identical across turns, so caching them is the obvious first optimisation.