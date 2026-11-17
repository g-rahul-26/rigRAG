"""Route a question to RBI, SEBI or both. Cheap keyword rules first, LLM only if unclear."""
from __future__ import annotations

import re

from generation import prompts

RBI_TERMS = r"\b(bank|banks|banking|nbfc|rbi|deposit|loan|lending|credit card|debit card|upi|neft|rtgs|imps|payment|" \
            r"wallet|ppi|fema|forex|foreign exchange|remittance|ecb|priority sector|npa|provisioning|crr|slr|" \
            r"co-operative|cooperative|microfinance|gold loan|digital lending|ombudsman)\b"
SEBI_TERMS = r"\b(sebi|listed|listing|ipo|share|shares|stock|stock exchange|broker|mutual fund|amc|depositor(y|ies)|demat|" \
             r"investment adviser|research analyst|portfolio manager|aif|reit|invit|lodr|insider trading|takeover|" \
             r"debenture|bond issue|derivative|f&o|clearing corporation|kra|nism)\b"


def route(question: str, llm=None) -> list[str]:
    q = question.lower()
    rbi = len(re.findall(RBI_TERMS, q))
    sebi = len(re.findall(SEBI_TERMS, q))
    if rbi and not sebi:
        return ["rbi"]
    if sebi and not rbi:
        return ["sebi"]
    if llm is not None:
        try:
            r = llm.chat_json([{"role": "user", "content": prompts.ROUTER.format(question=question)}]).get("route", "both")
            if r in ("rbi", "sebi"):
                return [r]
        except Exception:
            pass
    return ["rbi", "sebi"]
