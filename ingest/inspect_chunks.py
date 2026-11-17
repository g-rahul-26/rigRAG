"""Check chunking quality.

    python -m ingest.inspect_chunks                       # health report for every document
    python -m ingest.inspect_chunks --doc "Payments Banks" --page 43   # raw PDF lines of one page

Healthy document: many distinct paragraphs, few chunks in 'Preamble', few giant multi-part paragraphs.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from retrieval.store import read_jsonl  # noqa: E402


def report() -> None:
    docs = {d["doc_id"]: d for d in read_jsonl(settings.processed_dir / "docs.jsonl")}
    by_doc = defaultdict(list)
    for c in read_jsonl(settings.processed_dir / "chunks.jsonl"):
        by_doc[c["doc_id"]].append(c)
    print(f"{'chunks':>6} {'paras':>5} {'preamble%':>9} {'biggest para (parts)':>22}  title")
    for doc_id, chunks in by_doc.items():
        base = [c["para"].split(" (part")[0] for c in chunks if c["type"] == "text"]
        counts = Counter(base)
        pre = sum(1 for b in base if b.startswith("Preamble"))
        big, n = counts.most_common(1)[0] if counts else ("-", 0)
        flag = "  <-- check" if (pre / max(len(base), 1) > 0.2 or n > 25) else ""
        print(f"{len(chunks):6} {len(counts):5} {pre / max(len(base), 1):8.0%} {big[:14]:>16} ({n:3})  "
              f"{docs.get(doc_id, {}).get('title', '')[:60]}{flag}")


def show_page(doc_query: str, page: int) -> None:
    import pymupdf

    docs = read_jsonl(settings.processed_dir / "docs.jsonl")
    match = [d for d in docs if doc_query.lower() in d["title"].lower() or doc_query.lower() in d["file"].lower()]
    if not match:
        print("No document title/file contains", doc_query)
        return
    d = match[0]
    path = next(settings.raw_dir.rglob(d["file"]))
    print(f"{d['title']}\n{path}\n--- page {page}: raw lines (repr shows hidden characters) ---")
    with pymupdf.open(path) as pdf:
        for line in pdf[page - 1].get_text("text").splitlines()[:60]:
            print(repr(line))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="")
    ap.add_argument("--page", type=int, default=0)
    args = ap.parse_args()
    if args.doc and args.page:
        show_page(args.doc, args.page)
    else:
        report()


if __name__ == "__main__":
    main()
