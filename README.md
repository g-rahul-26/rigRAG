# 📜 RegRAG — Cited Q&A over RBI & SEBI Regulations

A Retrieval-Augmented Generation (RAG) system that answers questions about Indian financial
regulations (RBI Master Directions, SEBI circulars). **Every answer is cited** to the exact circular
and paragraph. It warns you when a circular has been superseded, and it says *"not found"*
instead of making things up.

**Cost: ₹0.** Free LLM APIs (Groq / Gemini) or a local LLM (Ollama), plus local embedding and reranking models.

```
INGESTION                                       QUERY
PDFs → parse → paragraph-aware chunks           question → (translate) → route RBI/SEBI
     → metadata (circular no, date,                      → multi-query + HyDE
       supersession links)                               → hybrid search (BM25 + vectors, RRF)
     → embeddings → ChromaDB + BM25                      → cross-encoder rerank
                                                         → CRAG relevance check (retry / refuse)
                                                         → LLM answer with [S#] citations
```

| Version | Technique added |
|---|---|
| v0 | Dense vector search (baseline) |
| v1 | + Hybrid search (BM25 + vectors, Reciprocal Rank Fusion) |
| v2 | + Cross-encoder reranker |
| v3 | + Multi-query / RAG-Fusion |
| v4 | + HyDE |
| v5 | + Routing (RBI vs SEBI) |
| v6 | + Corrective RAG (grade → retry → refuse) |

---

## 1. Setup (one time)

**You need:** Python 3.10–3.12, about 3 GB of free disk space, and 8 GB RAM.

```bash
# go into the project folder
cd regrag

# create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# (Linux only, optional) install the small CPU version of PyTorch first to avoid a 2 GB CUDA download:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

**Get a free LLM key** (pick at least one):
- **Groq** (recommended): https://console.groq.com/keys
- **Gemini** (backup): https://aistudio.google.com/apikey
- **Ollama** (offline, no key): install from https://ollama.com, then run `ollama pull llama3.2:3b`

Copy `.env.example` to `.env` and paste your key:
```bash
# Windows:  copy .env.example .env
# Mac/Linux: cp .env.example .env
```

**Check that everything works**. This runs offline tests that need no keys and no downloads:
```bash
pytest -q
```
You should see `20 passed`.

---

## 2. Get the documents

**Option A: scraper**
```bash
python -m ingest.scraper --source rbi --keyword KYC --limit 20
python -m ingest.scraper --source sebi --limit 20
```
**Option B: a list of URLs.** Put one URL per line in a file (see `data/urls_example.txt`):
```bash
python -m ingest.scraper --source sebi --urls-file data/urls_example.txt
```
**Option C: manual (always works).** Download PDFs in your browser and put them in
`data/raw/rbi/` and `data/raw/sebi/`.

> 💡 Start small: about 20–50 documents on one topic, such as KYC/AML. Government sites change their
> layout often. If the scraper finds 0 links, use option B or C.

## 3. Build the index
```bash
python -m ingest.build_index            # add --tables to also extract tables (slower)
```
The first run downloads the embedding model (~130 MB).

## 4. Ask questions
```bash
streamlit run app/streamlit_app.py
```
or from the terminal:
```bash
python ask.py "What is the periodicity of KYC updation for high risk customers?" --debug
python ask.py "..." --version v2
```
The first question also downloads the reranker (~90 MB).

## 5. Evaluate (the most important part!)
```bash
# 1) generate DRAFT questions from your documents
python -m evals.generate_testset --n 50
# 2) open evals/testset.draft.jsonl, fix or delete bad questions, add 5-10 questions your
#    circulars DON'T answer ("type": "not_in_corpus"), then save as evals/testset.jsonl
#    (format: evals/testset.FORMAT_EXAMPLE.jsonl)

# 3) compare versions
python -m evals.run_evals --versions v0 v1 v2 --retrieval-only     # cheap, no LLM for v0-v2
python -m evals.run_evals --versions v0 v2 v4 v6 --limit 30        # full, with LLM judge

# 4) label answers yourself and check whether the LLM judge agrees with you
python -m evals.human_labels label     --run evals/results/<stamp>_v6.jsonl
python -m evals.human_labels agreement --run evals/results/<stamp>_v6.jsonl
```
The summary table (`evals/results/*_summary.md`) goes straight into this README and your resume.

**Metrics:** Hit@1/Hit@5 and MRR (did we retrieve the right paragraph?), Recall@20 (was it among the
candidates?), Citation hit, Correct and Faithful (LLM judge, pass/fail), and Refuses unanswerable.

---

## Project structure
```
config.py              all settings (read from .env)
pipeline.py            RegRAG class, versions v0-v6
ask.py                 command-line Q&A
ingest/   scraper.py   RBI/SEBI downloader
          parser.py    PDF text, headers/footers removed, tables
          chunker.py   paragraph-aware chunking + contextual headers
          metadata.py  circular no, dates, supersession detection
          build_index.py
retrieval/ store.py    ChromaDB + BM25    hybrid.py  RRF fusion
          rerank.py    cross-encoder      query_transform.py  multi-query, HyDE, translation
          router.py    RBI/SEBI routing   crag.py   corrective RAG grading
generation/ llm.py     Groq → Gemini → Ollama fallback, retry, disk cache
          prompts.py   all prompts        answer.py  context building, citations, warnings
evals/    generate_testset.py  run_evals.py  judge.py  metrics.py  human_labels.py
app/streamlit_app.py   web UI
tests/                 offline tests (synthetic PDFs, fake LLM)
```

## Free-tier notes
- Groq's free tier allows only a few thousand tokens per minute. The context is capped by
  `MAX_CONTEXT_TOKENS`, and the cheap tasks (rewriting, grading) use the smaller `*_FAST_MODEL`.
- v6 makes about 4 LLM calls per question. Repeated questions are served from `data/cache/`, so they cost nothing.
- If you hit a rate limit, the client waits and retries, then falls back to the next provider in `LLM_PROVIDERS`.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No LLM provider configured` | Put `GROQ_API_KEY` in `.env` (the file must be named exactly `.env`) |
| `Index was built with X but EMBED_MODEL is Y` | Re-run `python -m ingest.build_index` after changing `EMBED_MODEL` |
| `model_not_found` from Groq | Model names change. Copy a current one from console.groq.com/docs/models into `.env` |
| A PDF is skipped as "scanned" | It is an image-only PDF. OCR isn't included yet, so use a text version |
| Hindi answers are weak | Set `EMBED_MODEL=intfloat/multilingual-e5-small` and rebuild the index |

## Roadmap
- [ ] Compare mode ("how did rule X change between 2016 and 2024?")
- [ ] Daily auto-update with GitHub Actions cron
- [ ] Expose RegRAG as an MCP server
- [ ] Deploy on Hugging Face Spaces

*Disclaimer: this is an educational project, not legal advice. Always verify with the official circular.*
