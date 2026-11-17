"""Run the test set against one or more pipeline versions and print a comparison table.

    python -m evals.run_evals --versions v0 v1 v2 --retrieval-only        # no LLM needed for v0-v2
    python -m evals.run_evals --versions v0 v2 v6 --limit 30               # full: answers + LLM judge
    python -m evals.run_evals --versions v6 --no-judge                     # answers, skip judging

Free-tier tip: start with --limit 20, and use --sleep to stay under rate limits.
Results (every trace + summary table) are saved in evals/results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from evals.judge import judge_correctness, judge_faithfulness  # noqa: E402
from evals.metrics import doc_matches, hit_at_k, reciprocal_rank  # noqa: E402
from pipeline import VERSION_NAMES, RegRAG  # noqa: E402
from retrieval.embeddings import get_embedder  # noqa: E402
from retrieval.store import Store, read_jsonl  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, pct=True):
    if x is None:
        return "–"
    return f"{x * 100:.0f}%" if pct else f"{x:.2f}"


def evaluate_version(version: str, items: list[dict], store: Store, args, judge_llm) -> tuple[dict, list[dict]]:
    rag = RegRAG(version=version, store=store)
    traces = []
    for i, item in enumerate(items, 1):
        t0 = time.time()
        rec = {"id": item["id"], "version": version, "question": item["question"], "type": item.get("type", "factual")}
        try:
            if args.retrieval_only:
                chunks, debug = rag.retrieve(item["question"])
                res = {"answer": None, "not_found": not chunks, "citations": [], "sources": chunks, "debug": debug}
            else:
                res = rag.ask(item["question"])
        except Exception as e:
            print(f"  [{i}] ERROR {e}")
            rec["error"] = str(e)[:300]
            traces.append(rec)
            continue

        sources = res["sources"]
        cand_ids = res["debug"].get("candidates", [])
        candidates = [store.chunks[c] for c in cand_ids if c in store.chunks]
        rec.update({"answer": res["answer"], "not_found": res["not_found"],
                    "retrieved": [{"chunk_id": c["chunk_id"], "para": c["para"], "title": c["title"][:80]} for c in sources],
                    "latency": round(time.time() - t0, 2), "llm_calls": res["debug"].get("llm_calls")})

        if rec["type"] == "not_in_corpus":
            rec["abstained"] = float(res["not_found"])
        else:
            rec["hit@1"] = hit_at_k(sources, item, 1)
            rec["hit@k"] = hit_at_k(sources, item, settings.final_k)
            rec["mrr"] = reciprocal_rank(sources, item)
            rec["candidate_recall"] = hit_at_k(candidates, item, len(candidates)) if candidates else None
            if not args.retrieval_only:
                rec["citation_hit"] = float(any(doc_matches(c, item) or item.get("expected_doc", "") in c.get("also_in_ids", []) for c in res["citations"]))
                rec["wrongly_abstained"] = float(res["not_found"])
                if judge_llm is not None and not res["not_found"]:
                    f = judge_faithfulness(judge_llm, res["answer"], sources)
                    c = judge_correctness(judge_llm, item["question"], res["answer"], item.get("expected_answer", ""))
                    rec.update({"faithful": float(f["pass"]), "faithful_reason": f["reason"],
                                "correct": float(c["pass"]), "correct_reason": c["reason"]})
                elif judge_llm is not None:
                    rec.update({"faithful": 1.0, "correct": 0.0, "correct_reason": "abstained"})
        traces.append(rec)
        status = f"hit@k={rec.get('hit@k', rec.get('abstained'))}"
        if "correct" in rec:
            status += f" correct={rec['correct']:.0f} faithful={rec['faithful']:.0f}"
        print(f"  [{i}/{len(items)}] {status}  {item['question'][:70]}")
        time.sleep(args.sleep)

    ok = [t for t in traces if "error" not in t]
    fact = [t for t in ok if t["type"] != "not_in_corpus"]
    neg = [t for t in ok if t["type"] == "not_in_corpus"]
    summary = {
        "version": version, "name": VERSION_NAMES[version], "n": len(ok), "errors": len(traces) - len(ok),
        "hit@1": mean([t.get("hit@1") for t in fact]), "hit@k": mean([t.get("hit@k") for t in fact]),
        "mrr": mean([t.get("mrr") for t in fact]), "candidate_recall": mean([t.get("candidate_recall") for t in fact]),
        "citation_hit": mean([t.get("citation_hit") for t in fact]), "correct": mean([t.get("correct") for t in fact]),
        "faithful": mean([t.get("faithful") for t in fact]),
        "wrongly_abstained": mean([t.get("wrongly_abstained") for t in fact]),
        "abstain_when_unanswerable": mean([t.get("abstained") for t in neg]),
        "avg_latency_s": mean([t.get("latency") for t in ok]), "avg_llm_calls": mean([t.get("llm_calls") for t in ok]),
    }
    return summary, traces


def to_markdown(rows: list[dict]) -> str:
    cols = [("version", "Version", None), ("name", "Technique", None), ("hit@1", "Hit@1", True), ("hit@k", f"Hit@{settings.final_k}", True),
            ("mrr", "MRR", False), ("candidate_recall", f"Recall@{settings.candidates_k}", True),
            ("citation_hit", "Citation hit", True), ("correct", "Correct", True), ("faithful", "Faithful", True),
            ("abstain_when_unanswerable", "Refuses unanswerable", True), ("avg_latency_s", "Latency (s)", False)]
    head = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols)
    lines = [head]
    for r in rows:
        vals = [str(r[k]) if pct is None else fmt(r[k], pct) for k, _, pct in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default=str(Path(__file__).parent / "testset.jsonl"))
    ap.add_argument("--versions", nargs="+", default=["v0", "v2", "v6"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retrieval-only", action="store_true", help="skip answer generation (cheapest)")
    ap.add_argument("--no-judge", action="store_true", help="generate answers but skip LLM-as-judge")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between questions")
    args = ap.parse_args()

    items = read_jsonl(Path(args.testset))
    if not items:
        print(f"No test set at {args.testset}. Create one: python -m evals.generate_testset  (then review it!)")
        return
    if args.limit:
        items = items[: args.limit]

    store = Store(settings.processed_dir, settings.index_dir, get_embedder(settings.embed_model))
    judge_llm = None
    if not args.retrieval_only and not args.no_judge:
        from generation.llm import LLM
        judge_llm = LLM()

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for v in args.versions:
        print(f"\n=== {v}: {VERSION_NAMES[v]} ({len(items)} questions) ===")
        summary, traces = evaluate_version(v, items, store, args, judge_llm)
        rows.append(summary)
        with open(RESULTS / f"{stamp}_{v}.jsonl", "w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    md = to_markdown(rows)
    print("\n" + md)
    (RESULTS / f"{stamp}_summary.md").write_text(md + "\n", encoding="utf-8")
    with open(RESULTS / f"{stamp}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved traces and summary to {RESULTS}")


if __name__ == "__main__":
    main()
