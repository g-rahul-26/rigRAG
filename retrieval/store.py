"""Index storage: ChromaDB for vectors + BM25 for keywords + the chunk texts themselves."""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = set("""a an the of to in for on and or is are be by with as at from that this which shall
under any such its it their may will not all into per vide has have was were been""".split())

COLLECTION = "regrag"


def tokenize(text: str) -> list[str]:
    """Keeps reference numbers like 'dor.aml.rec.no.27/14.01.001/2024-25' as one token AND its parts."""
    text = text.lower()
    toks = []
    for t in re.findall(r"[a-z0-9]+(?:[./\-()][a-z0-9]+)*", text):
        if t in STOPWORDS:
            continue
        toks.append(t)
        if re.search(r"[./\-()]", t):
            toks.extend(p for p in re.split(r"[./\-()]", t) if p and p not in STOPWORDS)
    return toks


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _chroma_client(index_dir: Path):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(path=str(index_dir / "chroma"), settings=ChromaSettings(anonymized_telemetry=False))


def build_store(chunks: list[dict], embedder, index_dir: Path, batch: int = 256) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    client = _chroma_client(index_dir)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "embed_model": embedder.model_name})
    for i in range(0, len(chunks), batch):
        part = chunks[i: i + batch]
        vecs = embedder.embed_passages([c["embed_text"] for c in part])
        col.add(
            ids=[c["chunk_id"] for c in part],
            embeddings=[v.tolist() for v in vecs],
            documents=[c["text"] for c in part],
            metadatas=[{"doc_id": c["doc_id"], "regulator": c["regulator"], "year": int(c["year"] or 0),
                        "para": c["para"], "type": c["type"]} for c in part],
        )
        print(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)} chunks")
    bm25_tokens = [tokenize(c["embed_text"]) for c in chunks]
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"ids": [c["chunk_id"] for c in chunks], "tokens": bm25_tokens}, f)


class Store:
    """Loads the index once and answers dense / keyword searches with optional filters."""

    def __init__(self, processed_dir: Path, index_dir: Path, embedder):
        self.embedder = embedder
        self.chunks = {c["chunk_id"]: c for c in read_jsonl(processed_dir / "chunks.jsonl")}
        self.docs = {d["doc_id"]: d for d in read_jsonl(processed_dir / "docs.jsonl")}
        if not self.chunks:
            raise FileNotFoundError("No chunks found. Run:  python -m ingest.build_index")
        col = _chroma_client(index_dir).get_collection(COLLECTION)
        built_with = (col.metadata or {}).get("embed_model")
        if built_with and built_with != embedder.model_name:
            raise RuntimeError(f"Index was built with '{built_with}' but EMBED_MODEL is '{embedder.model_name}'. "
                               "Rebuild the index: python -m ingest.build_index")
        self.col = col
        with open(index_dir / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25_ids = data["ids"]
        self.bm25 = BM25Okapi(data["tokens"])

    # ---- filters -------------------------------------------------------
    @staticmethod
    def _chroma_where(regulators: list[str] | None, years: tuple[int, int] | None):
        conds = []
        if regulators:
            conds.append({"regulator": {"$in": list(regulators)}})
        if years:
            conds += [{"year": {"$gte": years[0]}}, {"year": {"$lte": years[1]}}]
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def _passes(self, c: dict, regulators, years) -> bool:
        if regulators and c["regulator"] not in regulators:
            return False
        if years and not (years[0] <= int(c["year"] or 0) <= years[1]):
            return False
        return True

    # ---- searches ------------------------------------------------------
    def dense(self, query: str, k: int, regulators=None, years=None, query_vec: np.ndarray | None = None) -> list[tuple[str, float]]:
        vec = query_vec if query_vec is not None else self.embedder.embed_query(query)
        res = self.col.query(query_embeddings=[vec.tolist()], n_results=min(k, len(self.chunks)),
                             where=self._chroma_where(regulators, years))
        ids, dists = res["ids"][0], res["distances"][0]
        return [(i, 1 - d) for i, d in zip(ids, dists)]

    def keyword(self, query: str, k: int, regulators=None, years=None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1]
        out = []
        for idx in order:
            if scores[idx] <= 0:
                break
            cid = self.bm25_ids[idx]
            if self._passes(self.chunks[cid], regulators, years):
                out.append((cid, float(scores[idx])))
            if len(out) >= k:
                break
        return out

    def doc_for(self, chunk_id: str) -> dict:
        return self.docs.get(self.chunks[chunk_id]["doc_id"], {})
