"""All prompts in one place, so they are easy to read, tweak and evaluate."""

ANSWER_SYSTEM = """You are RegRAG, an assistant that answers questions about Indian financial regulations \
(RBI and SEBI) using ONLY the numbered sources provided.

Rules:
1. Use only facts from the sources. Never use outside knowledge.
2. After every sentence that states a fact, cite ONLY the 1-2 sources that directly support it.
   Use exactly this format with plain square brackets and no spaces: [S1] or [S2][S4]. Never cite all sources by default.
3. If the sources answer the question only PARTIALLY, give the part they support and clearly say what is not covered.
4. If the question contains a wrong assumption (e.g. a wrong number or period), politely correct it using the sources.
5. If the question asks to compare institutions or rules, compare what each source says, naming the institution.
6. Reply exactly NOT_FOUND only when NONE of the sources contain anything relevant to the question,
   or when the question is not about financial regulation at all.
7. Quote numbers, limits, time periods and dates exactly as written in the sources.
8. If sources disagree, prefer the most recent one and say that an older instruction differs.
9. Be concise: a direct answer first, then short supporting points if needed.
10. Reply in the same language as the question: Hindi in Devanagari -> Hindi in Devanagari; Hinglish (Hindi written
    in English letters) -> Hinglish in English letters only, never Devanagari; English -> English.
    Citations are always exactly [S1] - never translate the word inside the brackets.
"""

ANSWER_USER = """Sources:
{context}

Question: {question}

Write the answer in: {language}
Answer (with [S#] citations):"""

MULTI_QUERY = """You help search a database of RBI and SEBI regulations.
Rewrite the user's question into {n} different search queries that use formal regulatory wording \
(e.g. "customer due diligence" instead of "ID check", "periodic updation" instead of "re-KYC").
Always expand abbreviations into their full form in at least one query (e.g. V-CIP -> Video based Customer Identification Process).
Return JSON: {{"queries": ["...", "..."]}}

Question: {question}"""

HYDE = """Write a short passage (80-120 words) that looks like it was taken from an RBI Master Direction \
or SEBI circular and that would answer the question below. Use formal regulatory language and \
paragraph-style wording. It is fine to guess the details - it is only used for search.

Question: {question}

Passage:"""

TRANSLATE = """Translate this question (it may be Hindi, Hinglish or another Indian language) into clear English. \
Keep regulatory terms precise (e.g. "video KYC" -> "Video based Customer Identification Process (V-CIP)"). \
Return only the translation.

Question: {question}"""

ROUTER = """Which regulator's documents should be searched to answer this question?
- "rbi": banks, NBFCs, payments, KYC/AML for banks, loans, deposits, forex (FEMA), cooperative banks
- "sebi": stock market, listed companies, mutual funds, brokers, depositories, IPOs, investment advisers
- "both": unclear or genuinely needs both
Return JSON: {{"route": "rbi" | "sebi" | "both"}}

Question: {question}"""

GRADER = """You check whether search results are useful for answering a question about regulations.
For each numbered passage, decide if it contains information that helps answer the question \
directly OR partially - definitions, conditions, procedures and related requirements all count as relevant. \
Abbreviations may appear in full form (e.g. "V-CIP" = "Video based Customer Identification Process"). \
When unsure, include the passage. Only exclude passages that are clearly about something else. Return JSON: {{"relevant": [list of passage numbers]}}. \
Return an empty list if none are relevant.

Question: {question}

Passages:
{passages}"""

REWRITE_FOR_RETRY = """The search for this question found nothing relevant in RBI/SEBI regulations.
Write ONE better search query using official regulatory terminology. Return only the query.

Question: {question}"""
