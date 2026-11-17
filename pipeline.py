"""The RegRAG pipeline. Each VERSION switches on one more technique, so you can measure
exactly how much each one helps (see evals/run_evals.py).

    v0  dense vector search only                 (baseline)
    v1  + hybrid search (BM25 keywords + vectors)
    v2  + cross-encoder reranking
    v3  + multi-query / RAG-fusion
    v4  + HyDE (hypothetical document embedding)
    v5  + routing (RBI vs SEBI)
    v6  + CRAG (grade chunks, retry, refuse if nothing relevant)
"""
from __future__ import annotations

import time

from config import NOT_FOUND_MESSAGE, settings
from generation.answer import generate_answer, postprocess
from retrieval import crag, query_transform
from retrieval.dedupe import dedupe
from retrieval.embeddings import get_embedder
from retrieval.hybrid import rrf
from retrieval.rerank import rerank
from retrieval.router import route
from retrieval.store import Store

_FEATURES = ["hybrid", "rerank", "multi_query", "hyde", "route", "crag"]
VERSIONS = {f"v{i}": {f: (j < i) for j, f in enumerate(_FEATURES)} for i in range(len(_FEATURES) + 1)}
VERSION_NAMES = {
    "v0": "Dense only (baseline)", "v1": "+ Hybrid BM25", "v2": "+ Reranker", "v3": "+ Multi-query",
    "v4": "+ HyDE", "v5": "+ Routing", "v6": "+ CRAG (full)",
}


class RegRAG:
    def __init__(self, version: str | None = None, store: Store | None = None, llm=None, rerank_model: str | None = None):
        self.version = version or settings.default_version
        if self.version not in VERSIONS:
            raise ValueError(f"Unknown version {self.version}. Choose from {list(VERSIONS)}")
        self.cfg = VERSIONS[self.version]
        self.store = store or Store(settings.processed_dir, settings.index_dir, get_embedder(settings.embed_model))
        self._llm = llm
        self.rerank_model = rerank_model or settings.rerank_model

    @property
    def llm(self):
        if self._llm is None:
            from generation.llm import LLM

            self._llm = LLM()
        return self._llm

    # ------------------------------------------------------------------ retrieval
    def _search(self, query: str, k: int, regulators, years):
        if self.cfg["hybrid"]:
            return [self.store.dense(query, k, regulators, years), self.store.keyword(query, k, regulators, years)]
        return [self.store.dense(query, k, regulators, years)]

    def _retrieve_once(self, q: str, regulators, years, debug: dict) -> list[dict]:
        k, n = settings.candidates_k, settings.final_k
        queries = [q]
        if self.cfg["multi_query"]:
            try:
                queries += query_transform.multi_query(self.llm, q)
            except Exception as e:
                debug["errors"].append(f"multi_query: {e}")
        debug["queries"] = queries

        lists = []
        for qq in queries:
            lists += self._search(qq, k, regulators, years)
        if self.cfg["hyde"]:
            try:
                passage = query_transform.hyde(self.llm, q)
                debug["hyde"] = passage
                vec = self.store.embedder.embed_passages([passage])[0]
                lists.append(self.store.dense(passage, k, regulators, years, query_vec=vec))
            except Exception as e:
                debug["errors"].append(f"hyde: {e}")

        fused = rrf(lists, top_n=k) if len(lists) > 1 else lists[0][:k]
        debug["candidates"] = [cid for cid, _ in fused]

        if self.cfg["rerank"]:
            ranked = rerank(q, [(cid, self.store.chunks[cid]["embed_text"]) for cid, _ in fused], self.rerank_model, len(fused))
        else:
            ranked = fused
        chunks = [{**self.store.chunks[cid], "score": round(float(s), 4)} for cid, s in ranked]
        unique = dedupe(chunks)
        debug["duplicates_removed"] = len(chunks) - len(unique)
        return unique[:n]

    def retrieve(self, question: str, regulators: list[str] | None = None, years: tuple[int, int] | None = None) -> tuple[list[dict], dict]:
        t0 = time.time()
        debug: dict = {"version": self.version, "errors": []}
        q = question
        if query_transform.needs_translation(question):
            try:
                q = query_transform.translate_to_english(self.llm, question)
                debug["translated"] = q
            except Exception as e:
                debug["errors"].append(f"translate: {e}")

        expanded = query_transform.expand_acronyms(q)
        if expanded != q:
            debug["expanded"] = expanded
            q = expanded

        if self.cfg["route"] and not regulators:
            regulators = route(q, self.llm)
            debug["route"] = regulators
        if regulators and not any(c["regulator"] in regulators for c in self.store.chunks.values()):
            regulators = None  # routed to a regulator we have no documents for

        chunks = self._retrieve_once(q, regulators, years, debug)
        if not chunks and regulators:
            chunks = self._retrieve_once(q, None, years, debug)

        if self.cfg["crag"]:
            fast = settings.grader_use_fast_model
            first = chunks
            kept = crag.grade(self.llm, q, chunks, fast=fast, debug=debug)
            debug["graded_out"] = [c["chunk_id"] for c in chunks if c not in kept]
            if not kept:
                retry_q = query_transform.rewrite_for_retry(self.llm, q)
                debug["retry_query"] = retry_q
                retry_chunks = self._retrieve_once(retry_q, regulators, years, debug)
                kept = crag.grade(self.llm, q, retry_chunks, fast=fast, debug=debug)
            if not kept and first:
                # Safety net: the grader can be wrong. Give the answer model the top 3 original chunks;
                # it still replies NOT_FOUND if they are really irrelevant (rule 6 of the answer prompt).
                kept = first[:3]
                debug["crag_fallback"] = True
            chunks = kept
        debug["final"] = [c["chunk_id"] for c in chunks]
        debug["retrieval_seconds"] = round(time.time() - t0, 2)
        return chunks, debug

    # ------------------------------------------------------------------ full answer
    def ask(self, question: str, regulators: list[str] | None = None, years: tuple[int, int] | None = None) -> dict:
        t0 = time.time()
        calls_before = self.llm.calls if self._llm else 0
        chunks, debug = self.retrieve(question, regulators, years)
        if not chunks:
            out = {"answer": NOT_FOUND_MESSAGE, "not_found": True, "citations": [], "warnings": []}
            used = []
        else:
            raw, used = generate_answer(self.llm, question, chunks, settings.max_context_tokens)
            debug["raw_answer"] = raw
            out = postprocess(raw, used, self.store.docs)
        debug["total_seconds"] = round(time.time() - t0, 2)
        debug["llm_calls"] = (self.llm.calls if self._llm else 0) - calls_before
        return {"question": question, **out, "sources": used, "debug": debug}
