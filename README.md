# Exhibitor Manual Copilot

Grounded question-answering over trade fair exhibitor manuals. An agent that
answers exhibitor questions from a show's technical guidelines, cites the
clause it used, and refuses when the manual doesn't cover it.

Built as a six-day prototype. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
design decisions and what was measured, and [FAILURES.md](FAILURES.md) for what
broke along the way.

## What it does

```
$ uv run python rag_tools.py "What's the tallest I can build my stand?" \
    --show buchmesse-technical

Maximum stand height depends on the hall [3.5.6]:
  - Halls 6.1, 6.2: max 3.7 m
  - Halls 4.1, 4.2: max 4 m
  - Halls 3.0, 3.1, 4.0, 5.0, 5.1, 6.0: max 5 m
Structures higher than 2.5 m require official authorisation [3.5.6.2].
Stands exceeding 4 m are subject to a surcharge graded by stand size,
except light fittings without promotional content [3.5.6.4].

✓ all citations verified against 5 retrieved clauses
  $0.0152 · 1 tool call
```

Ask it something the manual doesn't cover and it says so, rather than
inventing a rule.

## Results

69 golden questions, 16 of them unanswerable:

| | single agent | supervisor + specialists |
|---|---|---|
| retrieval recall | 76% | **86%** |
| fact coverage | **65%** | 46% |
| correct refusal | **100%** | 94% |
| citations verified | 93% | **96%** |
| cost / question | **$0.0144** | $0.0872 |
| mean latency | **12.1s** | 53.5s |

The multi-agent version retrieves better and answers worse — it loses facts
when the supervisor summarises specialist findings. Single agent wins.

## Setup

Requires Python 3.12+, PostgreSQL 17 with pgvector, and an Anthropic API key.

```bash
# database
brew install postgresql@17 pgvector && brew services start postgresql@17
createdb exhibitor
psql exhibitor -f schema.sql

# project
uv sync
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

## Corpus

Exhibitor manuals are not included — they're the publishers' copyright. Put
public PDFs in `corpus/` and rebuild. These were used for development:

- Frankfurter Buchmesse — Terms and Conditions of Participation, Technical Regulations
- Infotage Fachdental Frankfurt — Technical Guidelines (Messe Frankfurt)
- Formnext 2024 — Stand Construction Guidelines (Mesago)

```bash
uv run python ingest.py corpus/ --show-id mixed --inspect   # tune, then:
uv run python ingest.py corpus/ --show-id mixed -o chunks.jsonl
uv run python load.py chunks.jsonl --truncate
psql exhibitor -c "UPDATE chunks SET show_id = doc_title;"
```

`--inspect` prints chunk statistics and flags missing clause IDs and
mid-clause splits. Expect to tune the heading regex for your documents —
every publisher numbers differently.

## Use

```bash
# retrieval only, compare the stages
uv run python search.py "escape route width" --compare
uv run python search.py "4,00 m height" --show fachdental-frankfurt

# full agent with citation verification
uv run python rag_tools.py "How many exits does my 80 m² stand need?" \
    --show fachdental-frankfurt

# evaluation
uv run python eval.py --config single --limit 20     # smoke test
uv run python eval.py --compare                      # full run, ~$4

# MCP server
uv run mcp dev mcp_server.py
```

For Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "exhibitor-ops": {
      "command": "uv",
      "args": ["--directory", "/path/to/exhibitor-agent",
               "run", "python", "mcp_server.py"]
    }
  }
}
```

## Layout

```
agent.py          agent loop, guards, tool registry, budget
ingest.py         PDF → structural chunks with clause metadata
load.py           chunks → Postgres with local embeddings
search.py         hybrid retrieval: vector + BM25, RRF, cross-encoder rerank
rag_tools.py      retrieval-backed tools + citation verification
mcp_server.py     MCP server exposing the tools
eval.py           scoring against the golden set
make_golden.py    draft and review golden questions
evals/            golden-all.jsonl (69 questions), results.json
tests/            failure-mode suite for the guards
```

## Tests

```bash
uv run pytest -v                    # 13 unit tests, no API calls
uv run pytest -m integration -v     # hits the real API
```

The unit tests use stub clients to exercise loop behaviour — repetition
guards, iteration limits, graceful degradation — deterministically and for
free. The integration tests check that the agent admits failure rather than
hallucinating when a tool is broken.

## License

Code is MIT. The corpus is not distributed.