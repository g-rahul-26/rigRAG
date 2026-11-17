"""Create a DRAFT test set by asking the LLM to write questions from random chunks.

    python -m evals.generate_testset --n 60

Then OPEN evals/testset.draft.jsonl, read every line, fix or delete bad questions,
add your own hand-written ones (incl. a few questions the circulars do NOT answer,
with "type": "not_in_corpus"), and save it as evals/testset.jsonl.
Reviewing the questions yourself is the most important part - don't skip it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from generation.llm import LLM  # noqa: E402
from retrieval.store import read_jsonl  # noqa: E402

PROMPT = """You are creating an evaluation set for a Q&A system on Indian financial regulations.
Read the passage and write ONE realistic question a compliance officer or a customer might ask, \
that this passage answers. Use everyday wording - do NOT copy phrases from the passage. \
Also write the short correct answer using ONLY the passage.
If the passage has no concrete rule/number/condition worth asking about, return {{"skip": true}}.
Return JSON: {{"question": "...", "answer": "..."}}

Document: {title}
Passage (Para {para}):
{text}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=2.0, help="pause between calls (free-tier friendly)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "testset.draft.jsonl"))
    args = ap.parse_args()

    chunks = [c for c in read_jsonl(settings.processed_dir / "chunks.jsonl")
              if c["type"] == "text" and len(c["text"].split()) > 60 and c["para"] != "Preamble"]
    if not chunks:
        print("No chunks found - build the index first.")
        return
    random.seed(args.seed)
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    picked, docs = [], list(by_doc)
    while len(picked) < min(args.n * 2, len(chunks)):  # round-robin across documents
        for d in docs:
            if by_doc[d]:
                picked.append(by_doc[d].pop(random.randrange(len(by_doc[d]))))
        if not any(by_doc.values()):
            break

    llm = LLM()
    out, i = [], 0
    for c in picked:
        if len(out) >= args.n:
            break
        data = llm.chat_json([{"role": "user", "content": PROMPT.format(title=c["title"], para=c["para"], text=c["text"])}],
                             fast=False)
        time.sleep(args.sleep)
        if data.get("skip") or not data.get("question"):
            continue
        i += 1
        out.append({"id": f"q{i:03d}", "type": "factual", "question": data["question"], "expected_answer": data.get("answer", ""),
                    "expected_doc": c["doc_id"], "expected_para": c["para"].split(" (part")[0].split("–")[0],
                    "source_title": c["title"]})
        print(f"[{len(out)}/{args.n}] {data['question']}")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(out)} draft questions to {args.out}\nREVIEW them, then save as evals/testset.jsonl")


if __name__ == "__main__":
    main()
