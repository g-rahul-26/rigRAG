from ingest.chunker import PARA_START, make_chunks
from ingest.metadata import extract_metadata, find_circular_numbers, parse_date


def test_parse_date_formats():
    assert parse_date("dated February 25, 2016") == "2016-02-25"
    assert parse_date("dated 25th February 2016") == "2016-02-25"
    assert parse_date("on 05.01.2030 the") == "2030-01-05"
    assert parse_date("no date here") == ""


def test_circular_numbers():
    text = "Ref RBI/2023-24/105 and DOR.AML.REC.No.27/14.01.001/2024-25 and SEBI/HO/MIRSD/DOP/CIR/2024/12"
    nos = find_circular_numbers(text)
    assert "RBI/2023-24/105" in nos
    assert any(n.startswith("DOR.AML.REC.NO.27") for n in nos)
    assert "SEBI/HO/MIRSD/DOP/CIR/2024/12" in nos


def test_para_regex_ignores_dates_and_lowercase():
    assert PARA_START.match("38. Periodic updation shall be")
    assert PARA_START.match("2.1 The bank shall")
    assert not PARA_START.match("25 February 2016 was the date")
    assert not PARA_START.match("100 crore rupees or more")


def test_chunker_keeps_paragraphs():
    filler = " word" * 50
    pages = [f"Intro text{filler}\n1. First Rule about apples.{filler}\n2. Second Rule about pears.{filler}"]
    meta = {"regulator": "rbi", "title": "T", "circular_no": "", "date": "", "year": 0, "url": ""}
    chunks = make_chunks("d1", meta, pages, [], max_words=280, min_words=40)
    paras = [c["para"] for c in chunks]
    assert paras == ["Preamble", "1", "2"]
    assert "apples" in chunks[1]["text"] and "pears" not in chunks[1]["text"]
    assert chunks[1]["embed_text"].startswith("RBI | T | Para 1")


def test_long_paragraph_is_split():
    pages = ["1. Long Rule. " + " ".join(f"Sentence number {i} is here." for i in range(200))]
    meta = {"regulator": "rbi", "title": "T"}
    chunks = make_chunks("d1", meta, pages, [], max_words=100, min_words=10)
    assert len(chunks) > 3
    assert all(len(c["text"].split()) <= 160 for c in chunks)
    assert chunks[0]["para"].startswith("1 (part 1)")


def test_metadata_supersedes():
    pages = ["RESERVE BANK OF INDIA\nRBI/2031-32/22\nMarch 10, 2031\nCircular - X\n"
             "1. This circular is issued in supersession of circular RBI/2030-31/11 dated January 5, 2030."]
    m = extract_metadata(pages)
    assert m["regulator"] == "rbi"
    assert m["circular_no"] == "RBI/2031-32/22"
    assert m["date"] == "2031-03-10"
    assert "RBI/2030-31/11" in m["supersedes"]


def test_build_links_supersession(built):
    docs = {d["file"]: d for d in built["docs"]}
    assert len(docs) == 3
    old, new = docs["widget_md.pdf"], docs["widget_amendment.pdf"]
    assert new["doc_id"] in old["superseded_by"]
    assert docs["gadget_fund.pdf"]["regulator"] == "sebi"
    paras = {c["para"] for c in built["chunks"] if c["doc_id"] == old["doc_id"]}
    assert {"2", "3"} <= paras


def test_number_on_its_own_line_is_detected():
    filler = " word" * 50
    pages = [f"Intro{filler}\n42.\nPeriodic Updation of KYC.{filler}\n43.\nOther Rule.{filler}"]
    chunks = make_chunks("d", {"regulator": "rbi", "title": "T"}, pages, [], 280, 40)
    assert [c["para"] for c in chunks] == ["Preamble", "42", "43"]


def test_parts_get_their_own_pages():
    pages = ["1. Long Rule. " + " ".join(f"Sentence {i} here." for i in range(150)),
             " ".join(f"More text {i} here." for i in range(150))]
    chunks = make_chunks("d", {"regulator": "rbi", "title": "T"}, pages, [], 100, 10)
    assert chunks[0]["page_start"] == 1
    assert chunks[-1]["page_start"] == 2


def test_definitions_are_not_cut_in_half():
    defs = " ".join(f'({r}) "Term{i}" means a long definition sentence number {i} with several extra words to make it longer than usual.'
                    for i, r in enumerate(["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]))
    pages = ["3. Definitions. In these Directions, unless the context otherwise requires: " + defs]
    chunks = make_chunks("d", {"regulator": "rbi", "title": "T"}, pages, [], 60, 10)
    assert len(chunks) > 1
    for c in chunks[1:]:
        assert c["text"].lstrip().startswith("(")  # every later part starts at a clause boundary
