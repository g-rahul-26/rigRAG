"""Offline test fixtures: SYNTHETIC PDFs (made-up rules, not real regulations),
a hashing embedder (no model download) and a fake LLM (no API key)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FILLER = ("This sentence is filler text added only so that the paragraph is long enough for the "
          "chunker to treat it as a complete unit in this synthetic test document.")

DOC_A = f"""SYNTHETIC TEST DOCUMENT - NOT A REAL REGULATION
RESERVE BANK OF INDIA
RBI/2030-31/11
DEMO.TEST.No.1/01.01.001/2030-31
January 5, 2030
Master Direction - Widget Account Directions, 2030
1. Short title and commencement. These directions shall be called the Widget Account Directions. {FILLER}
2. Periodic review of widget accounts. Regulated entities shall review every widget account once in seven years for low risk customers and once in two years for high risk customers. {FILLER}
3. Purple stamp requirement. Every widget application form shall carry a purple stamp issued by the branch manager before the account is opened. {FILLER}
"""

DOC_B = f"""SYNTHETIC TEST DOCUMENT - NOT A REAL REGULATION
RESERVE BANK OF INDIA
RBI/2031-32/22
March 10, 2031
Circular - Widget Account (Amendment) Directions, 2031
1. Supersession. This circular is issued in supersession of circular RBI/2030-31/11 dated January 5, 2030. {FILLER}
2. Revised review period. Regulated entities shall now review high risk widget accounts once every year. {FILLER}
"""

DOC_C = f"""SYNTHETIC TEST DOCUMENT - NOT A REAL REGULATION
SECURITIES AND EXCHANGE BOARD OF INDIA
SEBI/HO/DEMO/TEST/CIR/2030/5
June 1, 2030
Circular on Gadget Fund Disclosures
1. Applicability. This circular applies to all gadget fund managers registered with the board. {FILLER}
2. Monthly disclosure. Gadget fund managers shall disclose the portfolio of each gadget fund on their website within ten days from the close of each month. {FILLER}
"""


def make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=9)
    doc.save(path)
    doc.close()


class FakeLLM:
    calls = 0

    def chat(self, messages, fast=False, temperature=0.0, max_tokens=900, json_mode=False):
        self.calls += 1
        content = " ".join(m["content"] for m in messages)
        if "search queries" in content:
            return json.dumps({"queries": ["widget account review periodicity", "review of widget accounts"]})
        if "Write a short passage" in content:
            return "Regulated entities shall review widget accounts periodically based on risk."
        if "Which regulator" in content:
            return json.dumps({"route": "both"})
        if "check whether search results" in content:
            return json.dumps({"relevant": [] if "moon" in content.lower() else [1, 2]})
        if "Write ONE better search query" in content:
            return "moon base lending rules"
        if "Translate" in content:
            return "How often must high risk widget accounts be reviewed?"
        if "RegRAG" in content:  # answer prompt
            if "moon" in content.lower():
                return "NOT_FOUND"
            return "High risk widget accounts must be reviewed once in two years [S1]. See also [S9]."
        return "{}"

    def chat_json(self, messages, fast=True, max_tokens=600):
        from generation.llm import parse_json

        return parse_json(self.chat(messages, fast=fast, json_mode=True))


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Build a full index from the synthetic PDFs into a temp folder."""
    from ingest.build_index import build

    base = tmp_path_factory.mktemp("regrag")
    raw = base / "raw"
    (raw / "rbi").mkdir(parents=True)
    (raw / "sebi").mkdir(parents=True)
    make_pdf(raw / "rbi" / "widget_md.pdf", DOC_A)
    make_pdf(raw / "rbi" / "widget_amendment.pdf", DOC_B)
    make_pdf(raw / "sebi" / "gadget_fund.pdf", DOC_C)
    docs, chunks = build(raw, base / "processed", base / "index", "hash")
    return {"base": base, "docs": docs, "chunks": chunks}


@pytest.fixture(scope="session")
def store(built):
    from retrieval.embeddings import get_embedder
    from retrieval.store import Store

    return Store(built["base"] / "processed", built["base"] / "index", get_embedder("hash"))


@pytest.fixture()
def fake_llm():
    return FakeLLM()
