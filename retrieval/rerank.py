"""Cross-encoder reranker: reads (question, passage) TOGETHER, so it is much more accurate
than embeddings, but slower - so we only rerank the top ~20 candidates."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _load(model_name: str):
    from sentence_transformers import CrossEncoder

    print(f"Loading reranker {model_name} (first time downloads it)...")
    return CrossEncoder(model_name, max_length=512)


def rerank(query: str, candidates: list[tuple[str, str]], model_name: str, top_n: int) -> list[tuple[str, float]]:
    """candidates = [(chunk_id, text)] -> [(chunk_id, score)] best first."""
    if not candidates:
        return []
    model = _load(model_name)
    scores = model.predict([(query, text) for _, text in candidates])
    ranked = sorted(zip([c[0] for c in candidates], [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
