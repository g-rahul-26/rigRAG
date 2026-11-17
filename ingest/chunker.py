"""Structure-aware chunking: one chunk ~= one numbered paragraph of the regulation.

Why: regulations are written as numbered rules ("38. Periodic updation ...").
Cutting every N characters splits a rule in half and mixes two rules in one chunk,
which hurts retrieval AND makes citations ("Para 38") impossible.
"""
from __future__ import annotations

import re

MONTH_WORDS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b"
PARA_START = re.compile(
    r"^\s*(?:(?:Para(?:graph)?|Section|Clause)\s+)?"
    r"(\d{1,3}(?:\.\d{1,3}){0,3})\s*[.)]?\s+"
    rf"(?!{MONTH_WORDS})(?=[A-Z(“\"'])"
)
CHAPTER_RE = re.compile(r"^\s*(CHAPTER|Chapter|PART|Part|SECTION|Section)\s+([IVXLC]+|\d+)\b[\s\-–:]*(.*)$")


def _words(s: str) -> int:
    return len(s.split())


NUMBER_ONLY = re.compile(r"^\s*(?:(?:Para(?:graph)?|Section|Clause)\s+)?\d{1,3}(?:\.\d{1,3}){0,3}\s*[.)]?\s*$")


def _lines_with_pages(pages: list[str]) -> list[tuple[str, int]]:
    """All non-empty lines with their page number. PDFs often put the paragraph number on its
    own line ("42." newline "Periodic updation ...") - join those so the paragraph is detected."""
    raw = [(l.strip(), n) for n, page in enumerate(pages, start=1) for l in page.splitlines() if l.strip()]
    out: list[tuple[str, int]] = []
    i = 0
    while i < len(raw):
        line, page = raw[i]
        if NUMBER_ONLY.match(line) and i + 1 < len(raw):
            nxt = raw[i + 1][0]
            if re.match(r"[A-Z(“\"']", nxt):
                out.append((f"{line} {nxt}", page))
                i += 2
                continue
        out.append((line, page))
        i += 1
    return out


def split_into_units(pages: list[str]) -> list[dict]:
    """Walk through all lines and start a new unit whenever a numbered paragraph begins."""
    units: list[dict] = []
    current = {"para": "Preamble", "chapter": "", "lines": [], "page_start": 1, "page_end": 1}
    chapter = ""
    for line, page_no in _lines_with_pages(pages):
        ch = CHAPTER_RE.match(line)
        if ch and len(line) < 150:
            chapter = re.sub(r"\s+", " ", line)
        m = PARA_START.match(line)
        if m:
            if current["lines"]:
                units.append(current)
            current = {"para": m.group(1), "chapter": chapter, "lines": [], "page_start": page_no, "page_end": page_no}
        current["lines"].append((line, page_no))
        current["page_end"] = page_no
    if current["lines"]:
        units.append(current)
    for u in units:
        lines = u.pop("lines")
        u["word_pages"] = [pg for line, pg in lines for _ in line.split()]
        u["text"] = " ".join(line for line, _ in lines)
        u["text"] = re.sub(r"\s+", " ", u["text"]).strip()
    return units


# Sub-clause starts: (i) "Aadhaar number" ...   (a) Where ...   (12) ...   ii. ...
CLAUSE_START = re.compile(r"(?=\s\((?:[ivxlc]{1,6}|[a-z]{1,2}|\d{1,2})\)\s+[“\"'A-Z])")


def _pack(pieces: list[str], max_words: int) -> list[str]:
    """Greedily pack small pieces into chunks of <= max_words without cutting a piece."""
    out, cur = [], ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if cur and _words(cur) + _words(piece) > max_words:
            out.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        out.append(cur)
    return out


def _split_long(text: str, max_words: int, overlap: int = 40) -> list[str]:
    """Split a long paragraph WITHOUT cutting a definition or sentence in half:
    1) at sub-clause boundaries ((i), (a), (12)...), 2) inside a too-long clause at sentence ends,
    3) as a last resort by words."""
    clauses = CLAUSE_START.split(text)
    pieces: list[str] = []
    for clause in clauses:
        if _words(clause) <= max_words:
            pieces.append(clause)
            continue
        sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z(\d“\"])", clause)
        for sent in sentences:
            w = sent.split()
            if len(w) <= max_words:
                pieces.append(sent)
            else:
                step = max_words - overlap
                pieces.extend(" ".join(w[i: i + max_words]) for i in range(0, len(w), step))
    return _pack(pieces, max_words)


def merge_small(units: list[dict], min_words: int) -> list[dict]:
    merged: list[dict] = []
    buf: dict | None = None
    for u in units:
        if buf is None:
            buf = dict(u)
        else:
            buf["text"] += " " + u["text"]
            buf["word_pages"] = buf.get("word_pages", []) + u.get("word_pages", [])
            buf["page_end"] = u["page_end"]
            first = buf["para"].split("–")[0]
            buf["para"] = f"{first}–{u['para']}" if first != u["para"] else first
        if _words(buf["text"]) >= min_words:
            merged.append(buf)
            buf = None
    if buf:
        if merged and _words(buf["text"]) < min_words:
            merged[-1]["text"] += " " + buf["text"]
            merged[-1]["word_pages"] = merged[-1].get("word_pages", []) + buf.get("word_pages", [])
            merged[-1]["page_end"] = buf["page_end"]
        else:
            merged.append(buf)
    return merged


def _piece_pages(unit: dict, pieces: list[str]) -> list[tuple[int, int]]:
    """Find which page each split piece starts/ends on (so 'part 7' doesn't show the page of part 1)."""
    wp = unit.get("word_pages") or []
    words = unit["text"].split()
    if len(wp) != len(words):
        return [(unit["page_start"], unit["page_end"])] * len(pieces)
    spans, cursor = [], 0
    for piece in pieces:
        pw = piece.split()
        start = cursor
        head = pw[:6]
        for i in range(max(0, cursor - 60), len(words) - len(head) + 1):
            if words[i: i + len(head)] == head:
                start = i
                break
        end = min(start + len(pw) - 1, len(wp) - 1)
        spans.append((wp[start], wp[end]))
        cursor = start + 1
    return spans


def make_chunks(doc_id: str, meta: dict, pages: list[str], tables: list[tuple[int, str]],
                max_words: int = 280, min_words: int = 40) -> list[dict]:
    units = merge_small(split_into_units(pages), min_words)
    chunks: list[dict] = []
    for u in units:
        pieces = _split_long(u["text"], max_words) if _words(u["text"]) > max_words else [u["text"]]
        spans = _piece_pages(u, pieces)
        for j, piece in enumerate(pieces):
            chunks.append({
                "type": "text",
                "para": u["para"] + (f" (part {j + 1})" if len(pieces) > 1 else ""),
                "chapter": u["chapter"],
                "page_start": spans[j][0],
                "page_end": spans[j][1],
                "text": piece,
            })
    for page_no, md in tables:
        for j, piece in enumerate(_split_long(md, max_words, overlap=0)):
            chunks.append({"type": "table", "para": f"Table on p.{page_no}" + (f" (part {j+1})" if j else ""),
                           "chapter": "", "page_start": page_no, "page_end": page_no, "text": piece})

    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{doc_id}::{i:04d}"
        c["doc_id"] = doc_id
        for k in ("regulator", "title", "circular_no", "date", "year", "url"):
            c[k] = meta.get(k, "")
        # "Contextual chunk header": the embedding also sees WHICH document / chapter this is.
        header = f"{meta.get('regulator','').upper()} | {meta.get('title','')}"
        if c["chapter"]:
            header += f" | {c['chapter']}"
        header += f" | Para {c['para']}"
        c["embed_text"] = f"{header}\n{c['text']}"
    return chunks
