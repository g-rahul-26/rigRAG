"""Build the context, call the LLM, and turn [S#] markers into real citations."""
from __future__ import annotations

import re

from config import NOT_FOUND_MESSAGE
from generation import prompts


def _approx_tokens(text: str) -> int:
    return len(text) // 4 + 1


def source_label(c: dict) -> str:
    parts = [c.get("regulator", "").upper(), c.get("title", "")[:90]]
    if c.get("circular_no"):
        parts.append(c["circular_no"])
    parts.append(f"Para {c.get('para', '?')}")
    parts.append(f"p.{c.get('page_start', '?')}")
    if c.get("date"):
        parts.append(c["date"])
    return " | ".join(p for p in parts if p)


def build_context(chunks: list[dict], max_tokens: int) -> tuple[str, list[dict]]:
    blocks, used, total = [], [], 0
    for c in chunks:
        block = f"[S{len(used) + 1}] ({source_label(c)})\n{c['text']}"
        t = _approx_tokens(block)
        if used and total + t > max_tokens:
            break
        blocks.append(block)
        used.append(c)
        total += t
    return "\n\n".join(blocks), used


def answer_language(question: str) -> str:
    from retrieval.query_transform import is_hinglish

    letters = [ch for ch in question if ch.isalpha()]
    if letters and sum(1 for ch in letters if 0x0900 <= ord(ch) <= 0x097F) / len(letters) > 0.3:
        return "Hindi in Devanagari script"
    if is_hinglish(question):
        return "Hinglish (Hindi words written in English/Roman letters ONLY - do NOT use Devanagari script)"
    if letters and sum(1 for ch in letters if ord(ch) > 0x024F) / len(letters) > 0.3:
        return "the same language and script as the question"
    return "English"


def generate_answer(llm, question: str, chunks: list[dict], max_tokens_context: int) -> tuple[str, list[dict]]:
    context, used = build_context(chunks, max_tokens_context)
    messages = [
        {"role": "system", "content": prompts.ANSWER_SYSTEM},
        {"role": "user", "content": prompts.ANSWER_USER.format(context=context, question=question,
                                                               language=answer_language(question))},
    ]
    return llm.chat(messages, fast=False, max_tokens=900), used


_CITE_GROUP = re.compile(r"[\[【［〔(]\s*(?:[^\]\d\[\n]{0,15}?\s)?(S\s*\d+(?:\s*[,;，、–—\-]\s*S?\s*\d+)*)\s*[\]】］〕)]")


def normalize_citations(text: str) -> str:
    """Turn every citation style LLMs produce into [S1][S2]:
    【S1】  [ S1 ]  [S1, S3]  [S1–S3]  (S2)  ->  [S1] [S1][S3] [S1][S2][S3] [S2]"""
    def repl(m: re.Match) -> str:
        nums: list[int] = []
        parts = re.split(r"\s*([,;，、–—\-])\s*", m.group(1))
        prev_sep = ","
        for part in parts:
            if part in (",", ";", "，", "、", "–", "—", "-"):
                prev_sep = part
                continue
            n = int(re.sub(r"\D", "", part))
            if prev_sep in ("–", "—", "-") and nums and n > nums[-1] and n - nums[-1] < 10:
                nums.extend(range(nums[-1] + 1, n + 1))
            else:
                nums.append(n)
        return "".join(f"[S{n}]" for n in dict.fromkeys(nums))
    return _CITE_GROUP.sub(repl, text)


def postprocess(raw: str, used: list[dict], docs: dict) -> dict:
    """-> {answer, not_found, citations, warnings}"""
    text = normalize_citations(raw.strip())
    text = re.sub(r"\s+(\[S\d+\])", r"\1", text)  # "sentence [S1]" -> "sentence[S1]" spacing tidy
    if not text or text.upper().startswith("NOT_FOUND") or (len(text) < 40 and "NOT_FOUND" in text.upper()):
        return {"answer": NOT_FOUND_MESSAGE, "not_found": True, "citations": [], "warnings": []}

    cited = []
    for m in re.finditer(r"\[S(\d+)\]", text):
        n = int(m.group(1))
        if 1 <= n <= len(used) and n not in cited:
            cited.append(n)
    # remove citations that point to sources that don't exist (hallucinated [S9])
    text = re.sub(r"\[S(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(used) else "", text)

    citations, warnings = [], []
    for n in cited:
        c = used[n - 1]
        citations.append({
            "ref": f"S{n}", "label": source_label(c), "title": c.get("title", ""), "regulator": c.get("regulator", ""),
            "circular_no": c.get("circular_no", ""), "para": c.get("para", ""), "page": c.get("page_start"),
            "date": c.get("date", ""), "url": c.get("url", ""), "doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
            "text": c["text"], "also_in": c.get("also_in", []), "also_in_ids": c.get("also_in_ids", []),
        })
        doc = docs.get(c["doc_id"], {})
        for newer_id in doc.get("superseded_by", []):
            newer = docs.get(newer_id, {})
            warnings.append(f"⚠️ [S{n}] '{doc.get('title','')[:80]}' appears to be superseded by "
                            f"'{newer.get('title','')[:80]}' ({newer.get('date','')}). Verify the latest instruction.")
    if not citations:
        warnings.append("⚠️ The answer has no citations - treat it with caution.")
    return {"answer": text, "not_found": False, "citations": citations, "warnings": warnings}
