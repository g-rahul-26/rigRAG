"""Central configuration. Everything can be overridden from the .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).parent.resolve()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    # ---------- paths ----------
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"            # PDFs go here: data/raw/rbi, data/raw/sebi
    processed_dir: Path = ROOT / "data" / "processed"  # docs.jsonl, chunks.jsonl
    index_dir: Path = ROOT / "data" / "index"        # chroma db + bm25.pkl
    cache_dir: Path = ROOT / "data" / "cache"        # LLM response cache

    # ---------- models (all free, run locally) ----------
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ---------- LLM providers (free tiers) ----------
    # Order = fallback chain. Remove the ones you don't have keys for.
    llm_providers: list[str] = field(
        default_factory=lambda: [p.strip() for p in _env("LLM_PROVIDERS", "groq,gemini,ollama").split(",") if p.strip()]
    )
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_model: str = _env("GROQ_MODEL", "openai/gpt-oss-120b")          # main answers
    groq_fast_model: str = _env("GROQ_FAST_MODEL", "openai/gpt-oss-20b")  # rewriting / grading
    gemini_api_key: str = _env("GEMINI_API_KEY", "")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-flash-latest")
    gemini_fast_model: str = _env("GEMINI_FAST_MODEL", "gemini-flash-lite-latest")
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = _env("OLLAMA_MODEL", "llama3.2:3b")
    ollama_fast_model: str = _env("OLLAMA_FAST_MODEL", "llama3.2:3b")

    # ---------- retrieval ----------
    candidates_k: int = int(_env("CANDIDATES_K", "20"))   # how many chunks each retriever returns
    final_k: int = int(_env("FINAL_K", "5"))              # how many chunks go to the LLM
    # Groq free tier ~8K tokens/minute -> keep the context small
    max_context_tokens: int = int(_env("MAX_CONTEXT_TOKENS", "2500"))

    # ---------- chunking ----------
    chunk_max_words: int = int(_env("CHUNK_MAX_WORDS", "280"))
    chunk_min_words: int = int(_env("CHUNK_MIN_WORDS", "40"))

    # CRAG relevance grader: the small "fast" model was too strict -> use the main model by default
    grader_use_fast_model: bool = _env("GRADER_USE_FAST_MODEL", "0") == "1"

    # ---------- misc ----------
    default_version: str = _env("DEFAULT_VERSION", "v6")
    use_cache: bool = _env("USE_LLM_CACHE", "1") == "1"

    def ensure_dirs(self) -> None:
        for p in [self.raw_dir / "rbi", self.raw_dir / "sebi", self.processed_dir, self.index_dir, self.cache_dir]:
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()

NOT_FOUND_MESSAGE = (
    "I could not find this in the RBI/SEBI circulars I have indexed. "
    "Please check the official websites (rbi.org.in / sebi.gov.in) or rephrase your question."
)
