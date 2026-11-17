"""Parse every PDF in data/raw, chunk it, link superseded circulars, and build the search index.

    python -m ingest.build_index            # text only (fast)
    python -m ingest.build_index --tables   # also extract tables (slower, better for limits/fees)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from ingest.chunker import make_chunks  # noqa: E402
from ingest.metadata import extract_metadata, normalise_circular  # noqa: E402
from ingest.parser import parse_pdf  # noqa: E402
from retrieval.embeddings import get_embedder  # noqa: E402
from retrieval.store import build_store, write_jsonl  # noqa: E402


def find_pdfs(raw_dir: Path) -> list[tuple[Path, str]]:
    out = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() == ".pdf":
            hint = p.parent.name.lower() if p.parent.name.lower() in ("rbi", "sebi") else ""
            out.append((p, hint))
    return out


def link_supersession(docs: list[dict]) -> int:
    """If doc B says it supersedes circular X and doc A *is* circular X -> A.superseded_by = B."""
    by_no = {d["circular_no"]: d for d in docs if d["circular_no"]}
    links = 0
    for b in docs:
        for no in b["supersedes"]:
            a = by_no.get(normalise_circular(no))
            if a and a is not b and (not a["date"] or not b["date"] or a["date"] <= b["date"]):
                a.setdefault("superseded_by", [])
                if b["doc_id"] not in a["superseded_by"]:
                    a["superseded_by"].append(b["doc_id"])
                    links += 1
    for d in docs:
        d.setdefault("superseded_by", [])
    return links


def build(raw_dir: Path, processed_dir: Path, index_dir: Path, embed_model: str, with_tables: bool = False) -> tuple[list, list]:
    pdfs = find_pdfs(raw_dir)
    if not pdfs:
        print(f"No PDFs in {raw_dir}. Run the scraper or copy PDFs into data/raw/rbi and data/raw/sebi.")
        return [], []
    docs, all_chunks = [], []
    for i, (path, hint) in enumerate(pdfs, 1):
        try:
            parsed = parse_pdf(path, with_tables=with_tables)
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] x could not read {path.name}: {e}")
            continue
        if sum(len(p) for p in parsed["pages"]) < 200:
            print(f"[{i}/{len(pdfs)}] x {path.name}: almost no text (scanned PDF?) - skipped")
            continue
        sidecar_path = path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
        meta = extract_metadata(parsed["pages"], sidecar, regulator_hint=hint)
        doc_id = f"{meta['regulator']}_{hashlib.md5(path.name.encode()).hexdigest()[:10]}"
        meta.update({"doc_id": doc_id, "file": path.name, "n_pages": parsed["n_pages"]})
        chunks = make_chunks(doc_id, meta, parsed["pages"], parsed["tables"],
                             settings.chunk_max_words, settings.chunk_min_words)
        meta["n_chunks"] = len(chunks)
        docs.append(meta)
        all_chunks.extend(chunks)
        print(f"[{i}/{len(pdfs)}] {meta['regulator'].upper():4} {meta['date'] or '----------'} "
              f"{len(chunks):4} chunks  {meta['title'][:70]}")
    if not docs:
        return [], []
    links = link_supersession(docs)
    print(f"\nLinked {links} superseded documents.")
    write_jsonl(processed_dir / "docs.jsonl", docs)
    write_jsonl(processed_dir / "chunks.jsonl", all_chunks)
    print(f"Embedding {len(all_chunks)} chunks with {embed_model} ...")
    build_store(all_chunks, get_embedder(embed_model), index_dir)
    return docs, all_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true", help="extract tables with pdfplumber (slower)")
    args = ap.parse_args()
    settings.ensure_dirs()
    t0 = time.time()
    docs, chunks = build(settings.raw_dir, settings.processed_dir, settings.index_dir, settings.embed_model, args.tables)
    if docs:
        print(f"\nDone in {time.time() - t0:.0f}s: {len(docs)} documents, {len(chunks)} chunks.")
        print('Next: streamlit run app/streamlit_app.py   (or)   python ask.py "your question"')


if __name__ == "__main__":
    main()
