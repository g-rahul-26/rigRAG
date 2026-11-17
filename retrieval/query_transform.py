"""Query translation tricks from the RAG course: multi-query, HyDE, translation, retry-rewrite."""
from __future__ import annotations

import re

from generation import prompts


# Common regulatory abbreviations -> full forms used inside the documents.
# Add your own as you find failures in the evals!
GLOSSARY = {
    "kyc": "Know Your Customer", "v-cip": "Video based Customer Identification Process", "vcip": "Video based Customer Identification Process",
    "cdd": "Customer Due Diligence", "edd": "Enhanced Due Diligence", "pep": "Politically Exposed Person",
    "ovd": "Officially Valid Document", "ucic": "Unique Customer Identification Code", "ckycr": "Central KYC Records Registry",
    "pmla": "Prevention of Money-Laundering Act", "aml": "Anti-Money Laundering", "cft": "Combating Financing of Terrorism",
    "str": "Suspicious Transaction Report", "ctr": "Cash Transaction Report", "fiu-ind": "Financial Intelligence Unit-India",
    "re": "Regulated Entity", "res": "Regulated Entities", "nbfc": "Non-Banking Financial Company",
    "kfs": "Key Fact Statement", "apr": "Annual Percentage Rate", "dla": "Digital Lending App", "lsp": "Lending Service Provider",
    "fldg": "First Loss Default Guarantee", "ppi": "Prepaid Payment Instrument", "npa": "Non-Performing Asset",
    "ecb": "External Commercial Borrowings", "fema": "Foreign Exchange Management Act", "lodr": "Listing Obligations and Disclosure Requirements",
    "aif": "Alternative Investment Fund", "amc": "Asset Management Company", "kra": "KYC Registration Agency",
}


def expand_acronyms(question: str) -> str:
    """'What is V-CIP?' -> 'What is V-CIP? (Video based Customer Identification Process)'"""
    found = []
    for tok in re.findall(r"[A-Za-z][A-Za-z\-]+", question):
        full = GLOSSARY.get(tok.lower())
        if full and tok.isupper() or (full and "-" in tok):
            if full.lower() not in question.lower() and full not in found:
                found.append(full)
    return f"{question} ({'; '.join(found)})" if found else question


HINGLISH_WORDS = {
    "kya", "hai", "hain", "ke", "ki", "ka", "ko", "liye", "kaun", "kaunse", "kaise", "kitni", "kitna", "kitne",
    "kab", "kahan", "kyun", "kyon", "chahiye", "karna", "karne", "karte", "hota", "hoti", "mera", "meri", "mujhe",
    "agar", "nahi", "nahin", "aur", "bhi", "se", "kholne", "khata", "kar", "sakte", "sakta", "padega", "batao",
}


def is_hinglish(text: str) -> bool:
    """Hindi written in English letters, e.g. 'account kholne ke liye kya chahiye'."""
    words = re.findall(r"[a-z]+", text.lower())
    return len(words) >= 3 and sum(w in HINGLISH_WORDS for w in words) >= 2


def needs_translation(text: str) -> bool:
    """True for non-Latin scripts (Hindi, Tamil, Kannada, ...) and for Hinglish."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for ch in letters if ord(ch) > 0x024F)
    return non_latin / len(letters) > 0.3 or is_hinglish(text)


def translate_to_english(llm, question: str) -> str:
    out = llm.chat([{"role": "user", "content": prompts.TRANSLATE.format(question=question)}], fast=True, max_tokens=200)
    return out.strip().strip('"') or question


def multi_query(llm, question: str, n: int = 3) -> list[str]:
    data = llm.chat_json([{"role": "user", "content": prompts.MULTI_QUERY.format(question=question, n=n)}])
    qs = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
    return qs[:n]


def hyde(llm, question: str) -> str:
    return llm.chat([{"role": "user", "content": prompts.HYDE.format(question=question)}], fast=True, max_tokens=300)


def rewrite_for_retry(llm, question: str) -> str:
    out = llm.chat([{"role": "user", "content": prompts.REWRITE_FOR_RETRY.format(question=question)}],
                   fast=True, max_tokens=100)
    return re.sub(r"^(query:)\s*", "", out.strip(), flags=re.I).strip('"') or question
