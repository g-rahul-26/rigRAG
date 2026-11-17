"""Download RBI / SEBI regulation PDFs into data/raw/<regulator>/.

Three ways to get data (use whichever works for you):

  1. Automatic listing pages
       python -m ingest.scraper --source rbi  --keyword KYC --limit 30
       python -m ingest.scraper --source sebi --limit 30

  2. A text file of URLs (one per line). Works with:
       - direct PDF links   (https://rbidocs.rbi.org.in/rdocs/notification/PDFs/....PDF)
       - RBI detail pages   (https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12345)
       - SEBI circular pages (https://www.sebi.gov.in/legal/circulars/dec-2024/..._90088.html)
       python -m ingest.scraper --source rbi --urls-file my_rbi_urls.txt

  3. Manual: download PDFs in your browser and drop them in data/raw/rbi or data/raw/sebi.
     No scraper needed - the parser reads metadata from the PDF text.

Every downloaded PDF gets a sidecar <name>.json with title / url / date / circular number.
Government websites change their HTML often. If the automatic mode finds 0 links,
use option 2 or 3 (and tell me, we can fix the selectors).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 RegRAG-student-project"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}

LISTING_PAGES = {
    "rbi": [
        "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx",   # all Master Directions
        "https://www.rbi.org.in/Scripts/NotificationUser.aspx",       # latest notifications
    ],
    "sebi": [
        # Legal -> Circulars (first page). Later pages load via JavaScript, use --urls-file for more.
        "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0",
        # Legal -> Master Circulars
        "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=6&smid=0",
    ],
}

# Short forms people type -> words RBI/SEBI actually use in titles
KEYWORD_ALIASES = {
    "kyc": ["kyc", "know your customer"],
    "aml": ["aml", "anti-money laundering", "anti money laundering", "money laundering"],
    "nbfc": ["nbfc", "non-banking financial", "non banking financial"],
    "fema": ["fema", "foreign exchange"],
    "upi": ["upi", "unified payments"],
    "ppi": ["ppi", "prepaid payment"],
    "mf": ["mutual fund"],
    "lodr": ["lodr", "listing obligations"],
    "ipo": ["ipo", "public issue", "issue of capital"],
    "aif": ["aif", "alternative investment"],
}


def keyword_terms(keyword: str) -> list[str]:
    terms = []
    for part in [k.strip().lower() for k in keyword.split(",") if k.strip()]:
        terms.extend(KEYWORD_ALIASES.get(part, [part]))
    return terms


def title_matches(title: str, terms: list[str]) -> bool:
    t = re.sub(r"[\s\-–]+", " ", title.lower())
    return any(term in t for term in terms)


def dedupe_by_title(items: list[dict]) -> list[dict]:
    """The listing often has both a detail-page link and a PDF link for the same title: keep one (prefer PDF)."""
    best: dict[str, dict] = {}
    for it in items:
        key = re.sub(r"[^a-z0-9]", "", it.get("title", "").lower()) or it["url"]
        if key not in best or (is_pdf_link(it["url"]) and not is_pdf_link(best[key]["url"])):
            best[key] = it
    return list(best.values())


session = requests.Session()
session.headers.update(HEADERS)


def get(url: str, retries: int = 3, delay: float = 2.0) -> requests.Response | None:
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r
            print(f"  [{r.status_code}] {url}")
        except requests.RequestException as e:
            print(f"  [error attempt {attempt}] {url}: {e}")
        time.sleep(delay * attempt)
    return None


def safe_name(title: str, url: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", title or "doc").strip("_")[:80] or "doc"
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{base}_{h}"


def is_pdf_link(href: str) -> bool:
    return href.lower().split("?")[0].split("#")[0].endswith(".pdf")


def clean_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^PDF\s*-\s*", "", text, flags=re.I)
    text = re.sub(r"\(\s*\d+\s*kb\s*\)", "", text, flags=re.I).strip()
    return text


# ------------------------------------------------------------------ RBI
def rbi_links_from_listing(html: str, base: str) -> list[dict]:
    """Return [{title, url(pdf or detail page), date}] from an RBI listing page."""
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    current_date = ""
    for el in soup.find_all(["td", "a", "b", "span"]):
        txt = el.get_text(" ", strip=True)
        m = re.fullmatch(r"[A-Z][a-z]{2} \d{1,2}, \d{4}", txt)
        if m:
            current_date = txt
            continue
        if el.name != "a" or not el.get("href"):
            continue
        href = urljoin(base, el["href"])
        title = clean_title(el.get("title") or el.get("aria-label") or txt)
        if is_pdf_link(href) or re.search(r"BS_ViewMasDirections\.aspx\?id=\d+|NotificationUser\.aspx\?Id=\d+", href, re.I):
            if href in seen or len(title) < 8:
                continue
            seen.add(href)
            out.append({"title": title, "url": href, "date": current_date})
    return out


def rbi_pdf_from_detail(url: str) -> str | None:
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        if is_pdf_link(href) and "rbidocs" in href.lower():
            return href
    return None


# ------------------------------------------------------------------ SEBI
def sebi_links_from_listing(html: str, base: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base, a["href"])
        if re.search(r"/legal/(master-)?circulars/[a-z]{3}-\d{4}/.+_\d+\.html", href):
            if href in seen:
                continue
            seen.add(href)
            row = a.find_parent("tr")
            date = ""
            if row:
                m = re.search(r"[A-Z][a-z]{2} \d{1,2}, \d{4}", row.get_text(" ", strip=True))
                date = m.group(0) if m else ""
            out.append({"title": clean_title(a.get("title") or a.get_text(" ", strip=True)), "url": href, "date": date})
    return out


def sebi_resolve_page(url: str) -> dict:
    """Open a SEBI circular HTML page and find the PDF + circular number + date."""
    info = {"pdf": None, "circular_no": "", "date": "", "title": ""}
    r = get(url)
    if not r:
        return info
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    m = re.search(r"Circular No\.?\s*:?\s*([A-Za-z0-9/\-_().]+)", text)
    if m:
        info["circular_no"] = m.group(1)
    m = re.search(r"[A-Z][a-z]{2} \d{1,2}, \d{4}", text)
    if m:
        info["date"] = m.group(0)
    h1 = soup.find(["h1", "title"])
    if h1:
        info["title"] = clean_title(h1.get_text(" ", strip=True))
    # PDF is shown in an iframe like  ../../../web/?file=https://www.sebi.gov.in/sebi_data/attachdocs/...pdf
    candidates = [f.get("src", "") for f in soup.find_all(["iframe", "embed"])] + [a.get("href", "") for a in soup.find_all("a")]
    for c in candidates:
        if not c:
            continue
        full = urljoin(url, c)
        q = parse_qs(urlparse(full).query)
        if "file" in q and is_pdf_link(q["file"][0]):
            info["pdf"] = q["file"][0]
            return info
        if is_pdf_link(full) and "sebi_data" in full:
            info["pdf"] = full
            return info
    m = re.search(r"https?://www\.sebi\.gov\.in/sebi_data/[^\s\"'#]+?\.pdf", r.text, re.I)
    if m:
        info["pdf"] = m.group(0)
    return info


# ------------------------------------------------------------------ download
def download_pdf(pdf_url: str, meta: dict, out_dir: Path) -> bool:
    name = safe_name(meta.get("title", ""), pdf_url)
    pdf_path = out_dir / f"{name}.pdf"
    if pdf_path.exists():
        print(f"  = already have {pdf_path.name}")
        return True
    r = get(pdf_url)
    if not r or not r.content[:5].startswith(b"%PDF"):
        print(f"  x not a PDF: {pdf_url}")
        return False
    pdf_path.write_bytes(r.content)
    meta = {**meta, "pdf_url": pdf_url}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  + saved {pdf_path.name} ({len(r.content)//1024} KB)")
    return True


def resolve_and_download(source: str, item: dict, out_dir: Path) -> bool:
    url = item["url"]
    meta = {"regulator": source, "title": item.get("title", ""), "date": item.get("date", ""), "source_page": url}
    if is_pdf_link(url):
        return download_pdf(url, meta, out_dir)
    if source == "rbi":
        pdf = rbi_pdf_from_detail(url)
    else:
        info = sebi_resolve_page(url)
        pdf = info["pdf"]
        meta["circular_no"] = info["circular_no"]
        meta["date"] = meta["date"] or info["date"]
        meta["title"] = meta["title"] or info["title"]
    if not pdf:
        print(f"  x no PDF found on {url}")
        return False
    return download_pdf(pdf, meta, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["rbi", "sebi"], required=True)
    ap.add_argument("--keyword", default="", help='filter titles, e.g. KYC  or  "KYC,AML,digital lending"')
    ap.add_argument("--list-titles", action="store_true", help="only print the titles found, download nothing")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--urls-file", default="", help="text file with one URL per line")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between downloads (be polite)")
    args = ap.parse_args()

    settings.ensure_dirs()
    out_dir = settings.raw_dir / args.source

    if args.urls_file:
        urls = [u.strip() for u in Path(args.urls_file).read_text(encoding="utf-8").splitlines() if u.strip() and not u.startswith("#")]
        items = [{"url": u, "title": "", "date": ""} for u in urls]
    else:
        items = []
        for page in LISTING_PAGES[args.source]:
            print(f"Reading listing: {page}")
            r = get(page)
            if not r:
                continue
            found = rbi_links_from_listing(r.text, page) if args.source == "rbi" else sebi_links_from_listing(r.text, page)
            print(f"  found {len(found)} links")
            items.extend(found)
        if not items:
            print("\nNo links found. The website layout may have changed or blocked the request.\n"
                  "Use --urls-file or download PDFs manually into", out_dir)
            return

    if not args.urls_file:
        items = dedupe_by_title(items)
    if args.list_titles:
        for it in items:
            print(f"- {it.get('date',''):12} {it.get('title','')}")
        print(f"\n{len(items)} titles. Pick a word from them for --keyword")
        return
    if args.keyword:
        terms = keyword_terms(args.keyword)
        matched = [i for i in items if title_matches(i.get("title", ""), terms)]
        print(f"Keyword {terms}: {len(matched)} of {len(items)} titles match")
        if not matched:
            print("\nNo title matched. Some titles that were found:")
            for it in items[:15]:
                print("  -", it.get("title", "")[:110])
            print("\nTry another --keyword, or see all titles with:  --list-titles")
            return
        items = matched
    items = items[: args.limit]
    print(f"\nDownloading {len(items)} documents into {out_dir}\n")

    ok = 0
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.get('title') or item['url']}")
        ok += resolve_and_download(args.source, item, out_dir)
        time.sleep(args.delay)
    print(f"\nDone: {ok}/{len(items)} PDFs saved. Next: python -m ingest.build_index")


if __name__ == "__main__":
    main()
