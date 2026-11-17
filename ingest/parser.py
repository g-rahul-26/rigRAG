"""PDF -> clean page texts (+ optional tables as markdown)."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf

PAGE_NO_RE = re.compile(r"^\s*(page\s*)?\d{1,3}(\s*(of|/)\s*\d{1,3})?\s*$", re.I)


def read_pdf_pages(path: Path) -> list[str]:
    with pymupdf.open(path) as doc:
        return [page.get_text("text") for page in doc]


def remove_headers_footers(pages: list[str]) -> list[str]:
    """Drop lines repeated on many pages (letterheads, footers) and bare page numbers."""
    if len(pages) < 3:
        return [_strip_page_numbers(p) for p in pages]
    counts: Counter[str] = Counter()
    for p in pages:
        lines = {l.strip() for l in p.splitlines() if l.strip()}
        counts.update(lines)
    threshold = max(3, int(len(pages) * 0.5))
    repeated = {l for l, c in counts.items() if c >= threshold and len(l) < 120}
    cleaned = []
    for p in pages:
        keep = [l for l in p.splitlines() if l.strip() not in repeated]
        cleaned.append(_strip_page_numbers("\n".join(keep)))
    return cleaned


def _strip_page_numbers(text: str) -> str:
    lines = [l for l in text.splitlines() if not PAGE_NO_RE.match(l)]
    text = "\n".join(lines)
    text = text.replace(" ", " ").replace("", "•")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # join hyphenated line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def table_to_markdown(table: list[list]) -> str:
    rows = [[re.sub(r"\s+", " ", str(c or "")).strip() for c in row] for row in table if row and any(row)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)


def read_tables(path: Path, max_pages: int = 200) -> list[tuple[int, str]]:
    """Return [(page_number, markdown_table)]. Slow, so it is optional (--tables)."""
    import pdfplumber

    out = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages], start=1):
            try:
                for t in page.extract_tables():
                    md = table_to_markdown(t)
                    if md and len(md) > 60:
                        out.append((i, md))
            except Exception:  # broken tables shouldn't stop ingestion
                continue
    return out


def parse_pdf(path: Path, with_tables: bool = False) -> dict:
    pages = remove_headers_footers(read_pdf_pages(path))
    tables = read_tables(path) if with_tables else []
    return {"pages": pages, "tables": tables, "n_pages": len(pages)}
