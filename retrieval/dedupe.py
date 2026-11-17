"""Many RBI Directions repeat the SAME paragraph (Commercial Banks, NBFCs, Payments Banks ... KYC Directions).
Showing 5 copies of one rule wastes all the context slots. We keep the best-ranked copy and remember
where else it appears ("Same rule also in: ...")."""
from __future__ import annotations

import re


def _shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i: i + n]) for i in range(max(1, len(words) - n + 1))}


def similarity(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def dedupe(chunks: list[dict], threshold: float = 0.75) -> list[dict]:
    kept: list[dict] = []
    for c in chunks:
        dup_of = next((k for k in kept if similarity(k["text"], c["text"]) >= threshold), None)
        if dup_of is None:
            kept.append({**c, "also_in": [], "also_in_ids": []})
        elif c["doc_id"] != dup_of["doc_id"] and c["title"] not in dup_of["also_in"]:
            dup_of["also_in"].append(c["title"])
            dup_of["also_in_ids"].append(c["doc_id"])
    return kept
