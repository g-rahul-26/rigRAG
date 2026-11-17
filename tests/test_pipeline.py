from config import NOT_FOUND_MESSAGE
from evals.metrics import hit_at_k, para_matches, reciprocal_rank
from generation.answer import normalize_citations, postprocess
from pipeline import VERSIONS, RegRAG
from retrieval.hybrid import rrf
from retrieval.query_transform import expand_acronyms, is_hinglish, needs_translation
from retrieval.router import route
from retrieval.store import tokenize


def test_versions_are_cumulative():
    assert not any(VERSIONS["v0"].values())
    assert all(VERSIONS["v6"].values())
    assert sum(VERSIONS["v3"].values()) == 3


def test_tokenize_keeps_reference_numbers():
    toks = tokenize("Circular DOR.AML.REC.No.27/14.01.001/2024-25 dated")
    assert "dor.aml.rec.no.27/14.01.001/2024-25" in toks
    assert "aml" in toks


def test_rrf_prefers_items_in_both_lists():
    fused = rrf([[("a", 1), ("b", 1)], [("b", 1), ("c", 1)]])
    assert fused[0][0] == "b"


def test_router_keywords():
    assert route("Can a mutual fund invest in this?") == ["sebi"]
    assert route("Rules for NBFC gold loan") == ["rbi"]
    assert route("What is the rule?") == ["rbi", "sebi"]


def test_translation_detection():
    assert needs_translation("उच्च जोखिम वाले ग्राहकों के लिए KYC")
    assert not needs_translation("What is KYC?")


def test_keyword_search_finds_exact_rule(store):
    hits = store.keyword("purple stamp branch manager", k=3)
    assert "purple stamp" in store.chunks[hits[0][0]]["text"].lower()


def test_filters(store):
    hits = store.dense("disclosure", k=10, regulators=["sebi"])
    assert hits and all(store.chunks[c]["regulator"] == "sebi" for c, _ in hits)
    hits = store.keyword("widget", k=10, years=(2031, 2031))
    assert all(store.chunks[c]["year"] == 2031 for c, _ in hits)


def test_v1_retrieval_and_metrics(store):
    rag = RegRAG(version="v1", store=store)
    chunks, debug = rag.retrieve("purple stamp on widget application form")
    item = {"expected_doc": chunks[0]["doc_id"], "expected_para": "3"}
    assert hit_at_k(chunks, item, 5) == 1.0
    assert reciprocal_rank(chunks, item) > 0
    assert debug["final"]


def test_v6_full_answer_with_citations_and_supersession_warning(store, fake_llm):
    rag = RegRAG(version="v6", store=store, llm=fake_llm, rerank_model="unused")
    rag.cfg = {**rag.cfg, "rerank": False}  # reranker needs a model download; tested manually
    res = rag.ask("How often should high risk widget accounts be reviewed?")
    assert not res["not_found"]
    assert res["citations"] and res["citations"][0]["ref"] == "S1"
    assert "[S9]" not in res["answer"]  # hallucinated citation removed
    assert res["debug"]["llm_calls"] >= 3


def test_v6_refuses_when_nothing_relevant(store, fake_llm):
    rag = RegRAG(version="v6", store=store, llm=fake_llm)
    rag.cfg = {**rag.cfg, "rerank": False}
    res = rag.ask("What are the rules for lending on the moon?")
    assert res["not_found"] and res["answer"] == NOT_FOUND_MESSAGE
    assert "retry_query" in res["debug"]


def test_hindi_question_is_translated(store, fake_llm):
    rag = RegRAG(version="v1", store=store, llm=fake_llm)
    _, debug = rag.retrieve("उच्च जोखिम वाले विजेट खाते की समीक्षा कितनी बार?")
    assert debug["translated"].startswith("How often")


def test_postprocess_not_found():
    out = postprocess("NOT_FOUND", [], {})
    assert out["not_found"]


def test_para_matching():
    assert para_matches("38", "38")
    assert para_matches("38.1", "38")
    assert para_matches("12–14", "13")
    assert not para_matches("4", "40")
    assert para_matches("5 (part 2)", "5")


def test_citation_styles_are_normalised():
    assert normalize_citations("fact 【S1】【S2】.") == "fact [S1][S2]."
    assert normalize_citations("fact [ S1 ][ S3 ]") == "fact [S1][S3]"
    assert normalize_citations("fact [S1, S4]") == "fact [S1][S4]"
    assert normalize_citations("fact [S1–S3]") == "fact [S1][S2][S3]"
    assert normalize_citations("see (S2) and (section 5)") == "see [S2] and (section 5)"


def test_postprocess_reads_fullwidth_citations():
    used = [{"doc_id": "d", "chunk_id": "d::1", "text": "t", "title": "T", "para": "1"}]
    out = postprocess("High risk: two years 【S1】【S2】.", used, {})
    assert [c["ref"] for c in out["citations"]] == ["S1"]
    assert "【" not in out["answer"] and "[S2]" not in out["answer"]


def test_acronym_expansion():
    assert "Video based Customer Identification Process" in expand_acronyms("What is V-CIP?")
    assert expand_acronyms("what is the rule") == "what is the rule"


def test_hinglish_detection():
    assert is_hinglish("Bank account kholne ke liye kaun se documents chahiye?")
    assert needs_translation("KYC kitni baar update karna hota hai?")
    assert not is_hinglish("What documents are needed to open a bank account?")


def test_crag_fallback_lets_answer_model_decide(store, fake_llm):
    rag = RegRAG(version="v6", store=store, llm=fake_llm)
    rag.cfg = {**rag.cfg, "rerank": False}
    res = rag.ask("What are the rules for lending on the moon?")
    assert res["debug"].get("crag_fallback") is True
    assert res["not_found"]  # answer model still refuses irrelevant chunks


def test_citation_with_word_prefix():
    assert normalize_citations("उत्तर [स्रोत S1] और [Source S2]") == "उत्तर [S1] और [S2]"


def test_grader_parses_odd_formats():
    from retrieval.crag import _parse_ids
    assert _parse_ids([1, "3", "[4]"]) == {1, 3, 4}
    assert _parse_ids("1, 2") == {1, 2}
    assert _parse_ids([{"passage": 5}]) == {5}


def test_dedupe_keeps_one_copy_and_remembers_others():
    from retrieval.dedupe import dedupe
    text = "Regulated entities shall carry out periodic updation of KYC at least once in every two years for high risk customers."
    chunks = [{"doc_id": "a", "title": "Commercial Banks KYC", "text": text},
              {"doc_id": "b", "title": "NBFC KYC", "text": text + " "},
              {"doc_id": "c", "title": "Other", "text": "Completely different rule about V-CIP and video based identification."}]
    out = dedupe(chunks)
    assert [c["doc_id"] for c in out] == ["a", "c"]
    assert out[0]["also_in"] == ["NBFC KYC"]
    assert hit_at_k(out, {"expected_doc": "b"}, 1) == 1.0  # copy in NBFC Direction still counts


def test_answer_language_hint():
    from generation.answer import answer_language
    assert answer_language("What is KYC?") == "English"
    assert answer_language("वीडियो KYC क्या है?").startswith("Hindi")
    assert answer_language("Bank account kholne ke liye kaun se documents chahiye?").startswith("Hinglish")
