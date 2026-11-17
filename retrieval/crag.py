"""Corrective RAG (CRAG): grade retrieved chunks; drop irrelevant ones.
If nothing is relevant, the pipeline retries once with a rewritten query; if still nothing,
the answer model gets the top chunks and decides itself (it may reply NOT_FOUND).

All passages are graded in ONE LLM call (not one call per chunk) to save free-tier quota."""
from __future__ import annotations

import re

from generation import prompts

MAX_CHARS_PER_PASSAGE = 2200  # chunks are ~280 words (~1700 chars): show the WHOLE chunk to the grader


def _parse_ids(value) -> set[int]:
    """Accept [1, 3], ["1", "[3]"], "1,3", [{"passage": 1}] ... -> {1, 3}"""
    ids: set[int] = set()
    items = value if isinstance(value, list) else [value]
    for x in items:
        if isinstance(x, dict):
            x = next((v for k, v in x.items() if k.lower() in ("id", "passage", "number", "index", "n")), "")
        for m in re.findall(r"\d+", str(x)):
            ids.add(int(m))
    return ids


def grade(llm, question: str, chunks: list[dict], fast: bool = False, debug: dict | None = None) -> list[dict]:
    if not chunks:
        return []
    passages = "\n\n".join(f"[{i}] {c['text'][:MAX_CHARS_PER_PASSAGE]}" for i, c in enumerate(chunks, start=1))
    try:
        data = llm.chat_json([{"role": "user", "content": prompts.GRADER.format(question=question, passages=passages)}],
                             fast=fast)
    except Exception as e:
        if debug is not None:
            debug.setdefault("errors", []).append(f"grader: {e}")
        return chunks  # grading failed -> don't throw results away
    if debug is not None:
        debug.setdefault("grader_raw", []).append(data)
    key = next((k for k in data if "relevant" in k.lower()), None)
    if key is None:
        return chunks  # unexpected format -> keep everything
    keep = {i for i in _parse_ids(data[key]) if 1 <= i <= len(chunks)}
    return [c for i, c in enumerate(chunks, start=1) if i in keep]
