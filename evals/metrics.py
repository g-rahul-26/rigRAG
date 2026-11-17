"""Retrieval metrics. A retrieved chunk is 'correct' if it comes from the expected document
(and, if given, the expected paragraph)."""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def doc_matches(chunk: dict, expected_doc: str) -> bool:
    if not expected_doc:
        return False
    e = _norm(expected_doc)
    for field in ("doc_id", "circular_no", "file", "title"):
        v = _norm(str(chunk.get(field, "")))
        if v and (v == e or (field in ("title", "file") and len(e) > 8 and e in v)):
            return True
    return False


def para_matches(chunk_para: str, expected_para: str) -> bool:
    if not expected_para:
        return True
    p = (chunk_para or "").split(" (part")[0]
    exp = str(expected_para).strip()
    if "–" in p:  # merged range like "12–14"
        lo, hi = p.split("–", 1)
        try:
            return float(lo.split(".")[0]) <= float(exp.split(".")[0]) <= float(hi.split(".")[0])
        except ValueError:
            return exp in (lo, hi)
    return p == exp or p.startswith(exp + ".") or exp.startswith(p + ".")


def is_relevant(chunk: dict, item: dict) -> bool:
    exp = item.get("expected_doc", "")
    if doc_matches(chunk, exp) and para_matches(chunk.get("para", ""), item.get("expected_para", "")):
        return True
    # identical paragraph found in another Direction (deduplicated) also counts as a hit
    return bool(exp) and _norm(exp) in {_norm(d) for d in chunk.get("also_in_ids", [])}


def hit_at_k(chunks: list[dict], item: dict, k: int) -> float:
    return float(any(is_relevant(c, item) for c in chunks[:k]))


def reciprocal_rank(chunks: list[dict], item: dict) -> float:
    for i, c in enumerate(chunks, start=1):
        if is_relevant(c, item):
            return 1.0 / i
    return 0.0
