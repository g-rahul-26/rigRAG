"""LLM-as-judge with simple BINARY pass/fail labels (as recommended in Hamel Husain's evals post).
Binary labels are easier to agree on and to compare with your own human labels later."""
from __future__ import annotations

FAITHFULNESS = """You are a strict reviewer. Decide if EVERY factual claim in the ANSWER is supported \
by the SOURCES. Ignore citation formatting. If the answer adds facts not in the sources, it FAILS.
Return JSON: {{"pass": true|false, "reason": "one short sentence"}}

SOURCES:
{sources}

ANSWER:
{answer}"""

CORRECTNESS = """Compare the ANSWER to the REFERENCE answer for the QUESTION. It PASSES if it states the same \
key facts (numbers, periods, conditions) as the reference, even if worded differently or longer. \
It FAILS if a key fact is missing, different, or if it refuses to answer.
Return JSON: {{"pass": true|false, "reason": "one short sentence"}}

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}"""


def _ask(llm, prompt: str) -> dict:
    data = llm.chat_json([{"role": "user", "content": prompt}], fast=False, max_tokens=300)
    val = data.get("pass")
    if isinstance(val, str):
        val = val.strip().lower() in ("true", "yes", "pass")
    return {"pass": bool(val), "reason": str(data.get("reason", ""))[:300]}


def judge_faithfulness(llm, answer: str, sources: list[dict]) -> dict:
    src = "\n\n".join(f"[S{i}] {c['text'][:1200]}" for i, c in enumerate(sources, start=1))
    return _ask(llm, FAITHFULNESS.format(sources=src or "(none)", answer=answer))


def judge_correctness(llm, question: str, answer: str, reference: str) -> dict:
    return _ask(llm, CORRECTNESS.format(question=question, reference=reference, answer=answer))
