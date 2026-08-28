"""
Structural chunking for exhibitor manuals.

The core idea: exhibitor manuals are heavily numbered documents. Use the
document's own hierarchy as the chunk boundary, never a character count.
A clause and its exception must land in the same chunk, because splitting
them produces a confidently wrong answer.

Usage:
    uv run python ingest.py corpus/ambiente-2027.pdf --show-id ambiente-2027
    uv run python ingest.py corpus/ --show-id ambiente-2027 --inspect
    uv run python ingest.py corpus/ --show-id ambiente-2027 -o chunks.jsonl

Deps:
    uv add pymupdf pdfplumber
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pymupdf as fitz
import pdfplumber


# Target sizes in characters. ~4 chars per token, so 1200–2400 chars ≈ 300–600 tokens.
TARGET_CHARS = 2400
HARD_MAX_CHARS = 4000      # above this we must split, even mid-clause
MIN_CHARS = 120            # below this, merge upward rather than emit a fragment


# ---------------------------------------------------------------- data model

@dataclass
class Chunk:
    show_id: str
    doc_title: str
    clause_id: str | None
    section_path: str
    page: int
    content: str
    kind: str = "clause"            # clause | table_row | preamble
    part: str | None = None         # "2/3" when an oversized clause had to be split

    @property
    def embed_text(self) -> str:
        """What actually gets embedded. The section path carries the meaning that
        the clause body assumes. 'Maximum 4.0 m' embeds almost meaninglessly;
        'Stand Construction > Height > Island Stands: Maximum 4.0 m' embeds well."""
        prefix = self.section_path
        if self.clause_id:
            prefix = f"{prefix} [{self.clause_id}]"
        return f"{prefix}\n{self.content}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["embed_text"] = self.embed_text
        return d


# ---------------------------------------------------------------- heading detection

# Matches: "3", "3.2", "3.2.1", "3.2.1.4" followed by a title.
# Requires the title to start with a capital or quote to avoid matching
# "...as set out in 3.2.1 below" mid-sentence.
CLAUSE_RE = re.compile(
    r"^\s*(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z“\"'][^\n]{2,60}?)\s*$"
)

# Some manuals use "Section 4 —" or "Appendix B:" instead of pure numbering.
ALT_HEADING_RE = re.compile(
    r"^\s*(?P<num>(?:Section|Appendix|Annex|Part)\s+[A-Z0-9]+)\s*[—–\-:.]\s*(?P<title>[^\n]{2,80})\s*$",
    re.IGNORECASE,
)
MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DATEY_RE = re.compile(rf"^\s*\d{{1,2}}\.?\s+{MONTHS}\b", re.IGNORECASE)

def is_plausible_heading(clause_id: str, last_at_depth: dict[int, str]) -> bool:
    if not re.match(r"^\d", clause_id):
        return True
    if int(clause_id.split(".")[0]) > 12:
        return False
    d = depth_of(clause_id)
    prev = last_at_depth.get(d)
    if not prev:
        return True
    if d > 1 and clause_id.rsplit(".", 1)[0] != prev.rsplit(".", 1)[0]:
        return True                      # different parent: not a sibling, no ordering claim
    try:
        return int(clause_id.split(".")[-1]) > int(prev.split(".")[-1])
    except ValueError:
        return True

def match_heading(line: str) -> tuple[str, str] | None:
    for rx in (CLAUSE_RE, ALT_HEADING_RE):
        m = rx.match(line)
        if m:
            return m.group("num").rstrip("."), m.group("title").strip()
    return None


def depth_of(clause_id: str) -> int:
    """'3.2.1' -> 3. Non-numeric headings ('Appendix B') sit at depth 1."""
    return clause_id.count(".") + 1 if re.match(r"^\d", clause_id) else 1


# ---------------------------------------------------------------- boilerplate removal

def find_boilerplate(pages: list[list[str]], threshold: float = 0.6) -> set[str]:
    """Lines appearing on most pages are headers/footers, not content.
    Left in, they pollute every chunk and drag down embedding quality."""
    if len(pages) < 4:
        return set()
    counts: dict[str, int] = {}
    for lines in pages:
        for line in set(l.strip() for l in lines[:3] + lines[-3:]):
            if 3 < len(line) < 120:
                counts[line] = counts.get(line, 0) + 1
    cutoff = len(pages) * threshold
    return {line for line, n in counts.items() if n >= cutoff}


PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?\d{1,3}\s*(?:/\s*\d{1,3})?\s*$", re.IGNORECASE)


# ---------------------------------------------------------------- extraction

def extract_pages(pdf_path: Path) -> tuple[str, list[list[str]]]:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        text = text.replace("\xad", "")                    # soft hyphens
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # words split across lines
        pages.append(text.splitlines())
    doc.close()
    return pdf_path.stem, pages


def extract_table_rows(pdf_path: Path, show_id: str, doc_title: str) -> list[Chunk]:
    """Deadline tables are the highest-value content in an exhibitor manual and the
    worst thing to chunk as a block: one row per chunk so 'Form E-4' matches sharply."""
    chunks: list[Chunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables() or []:
                if len(table) < 2:
                    continue
                header = [(c or "").strip() for c in table[0]]
                if not any(header):
                    continue
                for row in table[1:]:
                    cells = [(c or "").strip() for c in row]
                    if not any(cells):
                        continue
                    pairs = [f"{h}: {c}" for h, c in zip(header, cells) if c]
                    if len(pairs) < 2:
                        continue
                    chunks.append(Chunk(
                        show_id=show_id,
                        doc_title=doc_title,
                        clause_id=None,
                        section_path=f"{doc_title} > Table (p.{pno})",
                        page=pno,
                        content=" | ".join(pairs),
                        kind="table_row",
                    ))
    return chunks


# ---------------------------------------------------------------- chunking
TRAILING_JUNK = re.compile(
    r"\b(is|are|was|were|be|of|for|to|in|on|with|and|or|the|a|an|that|which|如)\s*$",
    re.IGNORECASE,
)

@dataclass
class _Section:
    clause_id: str
    title: str
    page: int
    path: list[str]
    lines: list[str] = field(default_factory=list)


def split_oversized(text: str, limit: int = HARD_MAX_CHARS) -> list[str]:
    """Last resort for a clause longer than the hard cap. Split on sentence
    boundaries, never mid-sentence. Both halves keep the same clause metadata."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.;:])\s+", text)
    out, buf = [], ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > limit:
            out.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf.strip())
    return out


def chunk_pdf(pdf_path: Path, show_id: str) -> list[Chunk]:
    doc_title, pages = extract_pages(pdf_path)
    boilerplate = find_boilerplate(pages)

    sections: list[_Section] = []
    stack: list[tuple[int, str, str]] = []      # (depth, clause_id, title)
    current: _Section | None = None
    last_at_depth: dict[int, str] = {}

    for pno, lines in enumerate(pages, start=1):
        for raw in lines:
            line = raw.strip()
            if not line or line in boilerplate or PAGE_NUM_RE.match(line):
                continue
            hit = match_heading(line)
            if hit  and not DATEY_RE.match(line) and not TRAILING_JUNK.search(hit[1]) and is_plausible_heading(hit[0], last_at_depth):
                clause_id, title = hit
                d = depth_of(clause_id)
                while stack and stack[-1][0] >= d:
                    stack.pop()
                for deeper in [k for k in last_at_depth if k > d]:
                    del last_at_depth[deeper]
                last_at_depth[d] = clause_id
                stack.append((d, clause_id, title))

                path = [f"{cid} {t}" for _, cid, t in stack]
                current = _Section(clause_id=clause_id, title=title, page=pno, path=path)
                sections.append(current)
            elif current is not None:
                current.lines.append(line)
            else:
                # Text before the first heading — front matter.
                current = _Section(clause_id="", title="Preamble", page=pno,
                                   path=[f"{doc_title} > Preamble"])
                current.lines.append(line)
                sections.append(current)

    chunks: list[Chunk] = []
    for sec in sections:
        body = " ".join(sec.lines).strip()
        # A heading with no body is a container (e.g. "3 Stand Construction").
        # Its title already lives in the section_path of its children — drop it.
        if len(body) < MIN_CHARS:
            continue

        header_line = f"{sec.clause_id} {sec.title}".strip()
        full = f"{header_line}. {body}" if header_line else body
        parts = split_oversized(full)
        section_path = " > ".join([doc_title] + sec.path)

        for i, part_text in enumerate(parts, start=1):
            chunks.append(Chunk(
                show_id=show_id,
                doc_title=doc_title,
                clause_id=sec.clause_id or None,
                section_path=section_path,
                page=sec.page,
                content=part_text,
                kind="preamble" if not sec.clause_id else "clause",
                part=f"{i}/{len(parts)}" if len(parts) > 1 else None,
            ))

    try:
        chunks.extend(extract_table_rows(pdf_path, show_id, doc_title))
    except Exception as e:
        print(f"  ({pdf_path.name}: table extraction failed, keeping {len(chunks)} text chunks — {type(e).__name__})")
    return chunks

# ---------------------------------------------------------------- inspection

def inspect(chunks: list[Chunk], sample: int = 8) -> None:
    """Look at the output before you embed anything. Regexes never survive
    contact with a real manual unchanged — budget time to tune them."""
    sizes = sorted(len(c.content) for c in chunks)
    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1

    print(f"\n{len(chunks)} chunks   {by_kind}")
    if sizes:
        p = lambda q: sizes[min(int(len(sizes) * q), len(sizes) - 1)]
        print(f"chars  min {sizes[0]}  p50 {p(.5)}  p90 {p(.9)}  max {sizes[-1]}")
        print(f"≈tokens       p50 {p(.5)//4}  p90 {p(.9)//4}")

    orphans = [c for c in chunks if c.part]
    if orphans:
        print(f"\n⚠  {len(orphans)} chunks were split mid-clause — check these first")

    no_id = sum(1 for c in chunks if not c.clause_id and c.kind == "clause")
    if no_id:
        print(f"⚠  {no_id} clause chunks have no clause_id — heading regex is missing them")

    print("\n" + "=" * 70)
    step = max(1, len(chunks) // sample)
    for c in chunks[::step][:sample]:
        print(f"\n[{c.kind}] {c.section_path}")
        print(f"  clause={c.clause_id} page={c.page} chars={len(c.content)}"
              + (f" part={c.part}" if c.part else ""))
        print(f"  {c.content[:280]}{'…' if len(c.content) > 280 else ''}")


# ---------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="PDF file or directory of PDFs")
    ap.add_argument("--show-id", required=True)
    ap.add_argument("-o", "--out", type=Path, help="write JSONL here")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    if not args.path.exists():
        sys.exit(f"{args.path} does not exist — create it and add exhibitor manual PDFs")
    pdfs = sorted(args.path.glob("*.pdf")) if args.path.is_dir() else [args.path]
    if not pdfs:
        sys.exit(f"no PDFs found in {args.path}")

    all_chunks: list[Chunk] = []
    failed: list[tuple[str, str]] = []
    for pdf in pdfs:
        try:
            chunks = chunk_pdf(pdf, args.show_id)
            print(f"{pdf.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            failed.append((pdf.name, f"{type(e).__name__}: {e}"))
            print(f"{pdf.name}: FAILED — {type(e).__name__}")

    if failed:
        print(f"\n⚠ {len(failed)} of {len(pdfs)} documents failed:")
        for name, err in failed:
            print(f"   {name}: {err[:120]}")

    if args.inspect:
        inspect(all_chunks)

    if args.out:
        with args.out.open("w") as f:
            for c in all_chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        print(f"\nwrote {len(all_chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()