"""Ask a question from the terminal.

    python ask.py "What is the periodicity of KYC updation for high risk customers?"
    python ask.py "..." --version v2 --regulator rbi --debug
"""
from __future__ import annotations

import argparse
import json
import sys

from pipeline import VERSIONS, RegRAG


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Hindi / ₹ on Windows terminals
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--version", default=None, choices=list(VERSIONS))
    ap.add_argument("--regulator", choices=["rbi", "sebi"], default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rag = RegRAG(version=args.version)
    res = rag.ask(args.question, regulators=[args.regulator] if args.regulator else None)
    print("\n" + res["answer"] + "\n")
    for c in res["citations"]:
        print(f"  [{c['ref']}] {c['label']}")
        if c["url"]:
            print(f"        {c['url']}")
    for w in res["warnings"]:
        print(w)
    if args.debug:
        dbg = {k: v for k, v in res["debug"].items() if k != "raw_answer"}
        print("\nDEBUG:", json.dumps(dbg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
