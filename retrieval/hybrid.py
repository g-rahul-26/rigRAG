"""Reciprocal Rank Fusion (RRF): merge several ranked lists into one.

score(chunk) = sum over lists of 1 / (k + rank)
Simple, needs no score normalisation, and works very well in practice.
"""
from __future__ import annotations


def rrf(ranked_lists: list[list[tuple[str, float]]], k: int = 60, top_n: int | None = None) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (cid, _) in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_n] if top_n else fused


def hybrid_search(store, query: str, k: int, regulators=None, years=None, query_vec=None) -> list[tuple[str, float]]:
    dense = store.dense(query, k, regulators, years, query_vec=query_vec)
    sparse = store.keyword(query, k, regulators, years)
    return rrf([dense, sparse], top_n=k)
