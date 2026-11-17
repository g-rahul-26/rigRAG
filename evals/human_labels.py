"""Level-2 evals: label answers yourself, then check how often the LLM judge agrees with you.

    python -m evals.human_labels label     --run evals/results/<stamp>_v6.jsonl
    python -m evals.human_labels agreement --run evals/results/<stamp>_v6.jsonl

If the judge disagrees with you a lot, fix the judge prompt (evals/judge.py) before trusting its scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.store import read_jsonl  # noqa: E402

LABELS = Path(__file__).parent / "results" / "human_labels.jsonl"


def load_labels() -> dict:
    return {(r["run"], r["id"]): r["good"] for r in read_jsonl(LABELS)}


def label(run: Path) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    done = load_labels()
    traces = [t for t in read_jsonl(run) if t.get("answer") and "error" not in t]
    LABELS.parent.mkdir(exist_ok=True)
    with open(LABELS, "a", encoding="utf-8") as f:
        for i, t in enumerate(traces, 1):
            if (run.name, t["id"]) in done:
                continue
            print("\n" + "=" * 80)
            print(f"[{i}/{len(traces)}]  Q: {t['question']}\n")
            print(t["answer"])
            print("\nRetrieved:", "; ".join(f"{r['title'][:40]} (Para {r['para']})" for r in t.get("retrieved", [])))
            ans = input("\nIs this a GOOD answer (correct + grounded)? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()
            if ans == "q":
                break
            if ans in ("y", "n"):
                note = input("Optional note (why?): ").strip()
                f.write(json.dumps({"run": run.name, "id": t["id"], "good": ans == "y", "note": note}, ensure_ascii=False) + "\n")
                f.flush()
    print(f"\nLabels saved to {LABELS}")


def agreement(run: Path) -> None:
    labels = load_labels()
    pairs = []
    for t in read_jsonl(run):
        key = (run.name, t["id"])
        if key in labels and "correct" in t and "faithful" in t:
            judge_good = bool(t["correct"]) and bool(t["faithful"])
            pairs.append((labels[key], judge_good, t))
    if not pairs:
        print("No overlapping labels. Run 'label' first on a run that was judged (no --no-judge).")
        return
    tp = sum(1 for h, j, _ in pairs if h and j)
    tn = sum(1 for h, j, _ in pairs if not h and not j)
    fp = sum(1 for h, j, _ in pairs if not h and j)   # judge says good, you said bad (dangerous!)
    fn = sum(1 for h, j, _ in pairs if h and not j)
    n = len(pairs)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    # For catching BAD answers (usually the minority class) precision/recall on the "bad" class matters most
    bad_prec = tn / (tn + fn) if tn + fn else 0
    bad_rec = tn / (tn + fp) if tn + fp else 0
    print(f"Labelled pairs: {n}")
    print(f"Raw agreement:            {(tp + tn) / n:.0%}")
    print(f"Judge precision ('good'): {prec:.0%}   recall: {rec:.0%}")
    print(f"Judge precision ('bad'):  {bad_prec:.0%}   recall: {bad_rec:.0%}   <- how well it catches bad answers")
    print(f"Confusion: TP={tp} TN={tn} FP={fp} (judge too lenient) FN={fn} (judge too strict)")
    print("\nDisagreements (read these to improve the judge prompt):")
    for h, j, t in pairs:
        if h != j:
            print(f"- {t['id']}: you={'good' if h else 'bad'} judge={'good' if j else 'bad'} | {t['question'][:70]}")
            print(f"    judge said: {t.get('correct_reason','')} / {t.get('faithful_reason','')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["label", "agreement"])
    ap.add_argument("--run", required=True, help="a trace file from evals/results/")
    args = ap.parse_args()
    (label if args.command == "label" else agreement)(Path(args.run))


if __name__ == "__main__":
    main()
