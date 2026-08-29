"""
MCP server exposing exhibitor operations.

The architectural point: the agent runtime does not know about trade fairs. It
knows about tools. Exhibitor regulations are one MCP server; a floor-plan
service would be another; the next vertical is a third. Same loop, same guards,
same eval harness — verticals plug in rather than forking the product.

Install (Claude Desktop) — add to claude_desktop_config.json:

    {
      "mcpServers": {
        "exhibitor-ops": {
          "command": "uv",
          "args": ["--directory", "/Users/thanhhc/exhibitor-agent",
                   "run", "python", "mcp_server.py"]
        }
      }
    }

Deps:
    uv add "mcp[cli]"
"""

from __future__ import annotations

import json
import os

import psycopg
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

load_dotenv()

from search import search
from rag_tools import normalise_show_id, RELEVANCE_FLOOR

DSN = os.getenv("PG_DSN", "postgresql:///exhibitor")
mcp = MCPServer("exhibitor-ops")


@mcp.tool()
def list_shows() -> str:
    """List the trade fairs whose technical guidelines are indexed, with clause counts.

    Call this first if the user names a show informally — the show_id needed by
    the other tools is the slug returned here.
    """
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT show_id, count(*) FILTER (WHERE kind = 'clause'),
                   count(*) FILTER (WHERE kind = 'table_row')
            FROM chunks GROUP BY 1 ORDER BY 1
        """)
        return json.dumps([
            {"show_id": s, "clauses": c, "table_rows": t}
            for s, c, t in cur.fetchall()
        ])


@mcp.tool()
def search_regulations(show_id: str, query: str, limit: int = 5) -> str:
    """Search one show's technical guidelines and exhibitor manual.

    Returns passages with clause IDs, section paths and page numbers. Use for any
    question about stand height, rigging, electrical, fire safety, escape routes,
    materials, permits or approvals.

    Results are scoped to a single show — a rule from another fair does not apply
    and is never returned. If nothing relevant is found, say so rather than
    answering from general knowledge: exhibitor regulations differ by venue and a
    plausible-sounding wrong answer causes stands to be rejected on site.
    """
    show = normalise_show_id(show_id)
    rows = search(query, show_id=show, mode="hybrid", rerank=True, limit=limit)
    kept = [r for r in rows if r.get("rerank_score", 1.0) > RELEVANCE_FLOOR]

    if not kept:
        return json.dumps({
            "show_id": show, "results": [],
            "note": "No relevant passage found. Tell the user this is not covered "
                    "in the manual and to contact the show's technical services.",
        })

    return json.dumps({
        "show_id": show,
        "results": [{
            "clause_id": r["clause_id"],
            "section": r["section_path"],
            "page": r["page"],
            "kind": r["kind"],
            "text": r["content"][:1500],
            "relevance": round(float(r.get("rerank_score", 0)), 2),
        } for r in kept],
    }, ensure_ascii=False)


@mcp.tool()
def get_clause(show_id: str, clause_id: str) -> str:
    """Retrieve one specific clause verbatim by its number, e.g. '3.5.6.2'.

    Use when a clause has been cited and the exact wording matters, or to verify
    that a clause number actually exists before repeating it to an exhibitor.
    """
    show = normalise_show_id(show_id)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT clause_id, section_path, page, content
            FROM chunks
            WHERE show_id = %s AND clause_id = %s
            ORDER BY page LIMIT 5
        """, (show, clause_id))
        rows = cur.fetchall()

    if not rows:
        return json.dumps({
            "found": False,
            "note": f"No clause {clause_id} in {show}. Do not cite it.",
        })
    return json.dumps({
        "found": True,
        "clauses": [{"clause_id": c, "section": s, "page": p, "text": t}
                    for c, s, p, t in rows],
    }, ensure_ascii=False)


@mcp.tool()
def compare_to_limit(value: float, limit: float, unit: str = "m") -> str:
    """Compare a measurement against a regulatory limit. Use this instead of
    reasoning about numbers in prose.

    Numeric comparison in natural language is unreliable — models have stated
    that 3.5 m "exceeds" a 4.0 m limit. Any threshold check an exhibitor depends
    on should go through this tool.
    """
    return json.dumps({
        "value": value, "limit": limit, "unit": unit,
        "exceeds_limit": value > limit,
        "within_limit": value <= limit,
        "margin": round(limit - value, 3),
        "statement": (f"{value}{unit} {'exceeds' if value > limit else 'is within'} "
                      f"the {limit}{unit} limit"),
    })


if __name__ == "__main__":
    mcp.run()