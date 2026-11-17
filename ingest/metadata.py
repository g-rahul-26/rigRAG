"""Extract metadata (circular number, date, title, supersession) from regulation text."""
from __future__ import annotations

import re
from datetime import datetime

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
MONTHS_SHORT = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"

# Circular / reference numbers seen in RBI and SEBI documents
CIRCULAR_PATTERNS = [
    r"RBI/\d{4}-\d{2,4}/\d+",                                              # RBI/2023-24/12
    r"RBI/[A-Z]{2,}/\d{4}-\d{2,4}/\d+",                                    # RBI/DOR/2023-24/105
    r"\b[A-Z]{2,6}(?:\.[A-Z0-9\-]+){1,6}\.?\s?No\.?\s?[\w\-]+/[\d\.]+/\d{4}-\d{2,4}",  # DOR.AML.REC.No.27/14.01.001/2024-25
    r"SEBI/HO/[A-Z0-9_\-/()]+/\d{4}/\d+",                                  # SEBI/HO/MIRSD/.../CIR/2024/12
    r"\bHO/\d+/\d+/\d+\(\d+\)\d{4}-[A-Z0-9\-_]+/I/\d+/\d{4}",              # new SEBI style HO/49/11/11(123)2026-CFD.../I/8760/2026
    r"CIR/[A-Z]{2,}/[A-Z0-9/\-]+/\d{4}",                                   # CIR/MRD/DP/12/2014
]
CIRCULAR_RE = re.compile("|".join(f"(?:{p})" for p in CIRCULAR_PATTERNS))

DATE_PATTERNS = [
    (re.compile(rf"\b({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})\b"), "%B %d %Y"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS}),?\s+(\d{{4}})\b"), "%d %B %Y"),
    (re.compile(rf"\b({MONTHS_SHORT})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b"), "%b %d %Y"),
    (re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b"), "%d %m %Y"),
]

SUPERSEDE_RE = re.compile(r"supersed|repeal|rescind|stand(?:s)? withdrawn|hereby withdrawn|in supersession", re.I)


def parse_date(text: str) -> str:
    """Return the first date in text as YYYY-MM-DD, or ''."""
    best = None
    for rx, fmt in DATE_PATTERNS:
        for m in rx.finditer(text):
            parts = " ".join(m.groups()).replace("Sept", "Sep")
            try:
                d = datetime.strptime(parts, fmt)
            except ValueError:
                continue
            if 1990 <= d.year <= 2100 and (best is None or m.start() < best[0]):
                best = (m.start(), d)
            break  # first match per pattern is enough
    return best[1].strftime("%Y-%m-%d") if best else ""


def normalise_circular(no: str) -> str:
    return re.sub(r"\s+", "", no).rstrip(".").upper()


def find_circular_numbers(text: str) -> list[str]:
    seen, out = set(), []
    for m in CIRCULAR_RE.finditer(text):
        n = normalise_circular(m.group(0))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def detect_regulator(text: str, fallback: str = "") -> str:
    head = text[:4000].upper()
    if "SECURITIES AND EXCHANGE BOARD" in head or "SEBI/" in head:
        return "sebi"
    if "RESERVE BANK OF INDIA" in head or "RBI/" in head:
        return "rbi"
    return fallback or "unknown"


def guess_title(first_page: str) -> str:
    lines = [l.strip() for l in first_page.splitlines() if l.strip()]
    keywords = ("direction", "circular", "master", "guidelines", "framework", "regulations", "notification", "policy")
    for l in lines[:60]:
        if 15 <= len(l) <= 220 and any(k in l.lower() for k in keywords) and not CIRCULAR_RE.search(l):
            return re.sub(r"\s+", " ", l)
    for l in lines[:20]:
        if len(l) > 20:
            return l[:200]
    return "Untitled document"


def find_superseded(text: str) -> list[str]:
    """Circular numbers mentioned in sentences that talk about superseding / repealing."""
    out: list[str] = []
    sentences = re.split(r"(?<=[.;])\s+", text)
    for i, s in enumerate(sentences):
        if SUPERSEDE_RE.search(s):
            window = " ".join(sentences[i: i + 3])  # the numbers are often listed right after
            out.extend(find_circular_numbers(window))
    # Repeal annexes are usually tables at the end: "List of circulars repealed"
    m = re.search(r"(list of (?:circulars|notifications)[^\n]{0,60}(?:repealed|superseded|consolidated))", text, re.I)
    if m:
        out.extend(find_circular_numbers(text[m.start():]))
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def extract_metadata(pages: list[str], sidecar: dict | None = None, regulator_hint: str = "") -> dict:
    sidecar = sidecar or {}
    first = pages[0] if pages else ""
    first_two = "\n".join(pages[:2])
    full = "\n".join(pages)

    circulars = find_circular_numbers(first_two)
    own_no = normalise_circular(sidecar["circular_no"]) if sidecar.get("circular_no") else (circulars[0] if circulars else "")

    date = ""
    if sidecar.get("date"):
        date = parse_date(sidecar["date"])
    date = date or parse_date(first_two)

    updated = re.search(rf"Updated as on\s+((?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}})", full[:6000], re.I)
    superseded = [n for n in find_superseded(full) if n != own_no]

    return {
        "regulator": sidecar.get("regulator") or detect_regulator(first_two, regulator_hint),
        "title": (sidecar.get("title") or "").strip() or guess_title(first),
        "circular_no": own_no,
        "other_refs": circulars[1:5],
        "date": date,
        "year": int(date[:4]) if date else 0,
        "updated_as_on": parse_date(updated.group(1)) if updated else "",
        "supersedes": superseded,
        "url": sidecar.get("pdf_url") or sidecar.get("source_page") or "",
        "is_master": bool(re.search(r"master (direction|circular)", (sidecar.get("title") or first[:1500]), re.I)),
    }
