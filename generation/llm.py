"""One client for three FREE providers (all speak the OpenAI API format):

  groq   -> https://console.groq.com     (free key, fast)
  gemini -> https://aistudio.google.com  (free key, backup)
  ollama -> https://ollama.com           (runs on your laptop, no key, no limits)

If one provider hits its rate limit or fails, the next one in LLM_PROVIDERS is tried.
Responses are cached on disk so re-running the same question costs no quota.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass

from config import settings


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    fast_model: str


def available_providers() -> list[Provider]:
    table = {
        "groq": Provider("groq", "https://api.groq.com/openai/v1", settings.groq_api_key,
                         settings.groq_model, settings.groq_fast_model),
        "gemini": Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
                           settings.gemini_api_key, settings.gemini_model, settings.gemini_fast_model),
        "ollama": Provider("ollama", settings.ollama_base_url, "ollama", settings.ollama_model, settings.ollama_fast_model),
    }
    out = []
    for name in settings.llm_providers:
        p = table.get(name)
        if p and (p.api_key or name == "ollama"):
            out.append(p)
    return out


class LLMError(RuntimeError):
    pass


class _Cache:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k):
        row = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def set(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
        self.db.commit()


class LLM:
    def __init__(self, providers: list[Provider] | None = None, use_cache: bool | None = None):
        from openai import OpenAI

        self.providers = providers if providers is not None else available_providers()
        if not self.providers:
            raise LLMError("No LLM provider configured. Put GROQ_API_KEY in .env (free at console.groq.com) "
                           "or install Ollama and set LLM_PROVIDERS=ollama")
        self.clients = {p.name: OpenAI(base_url=p.base_url, api_key=p.api_key, max_retries=0, timeout=90)
                        for p in self.providers}
        use_cache = settings.use_cache if use_cache is None else use_cache
        self.cache = _Cache(settings.cache_dir / "llm_cache.sqlite") if use_cache else None
        self.calls = 0  # real (non-cached) API calls, shown in the UI

    def chat(self, messages: list[dict], fast: bool = False, temperature: float = 0.0,
             max_tokens: int = 900, json_mode: bool = False) -> str:
        key = hashlib.sha256(json.dumps([messages, fast, temperature, json_mode], ensure_ascii=False).encode()).hexdigest()
        if self.cache and (hit := self.cache.get(key)) is not None:
            return hit

        errors = []
        for p in self.providers:
            model = p.fast_model if fast else p.model
            kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if p.name == "groq" and "gpt-oss" in model:
                kwargs["extra_body"] = {"reasoning_effort": "low"}  # saves tokens on the free tier
            for attempt in range(3):
                try:
                    resp = self.clients[p.name].chat.completions.create(**kwargs)
                    text = (resp.choices[0].message.content or "").strip()
                    self.calls += 1
                    if self.cache and text:
                        self.cache.set(key, text)
                    return text
                except Exception as e:  # noqa: BLE001 - we want to fall through to the next provider
                    msg = str(e)
                    status = getattr(e, "status_code", None)
                    if status == 429 or "rate limit" in msg.lower():
                        wait = _retry_after(e) or 5 * (attempt + 1)
                        if wait <= 30 and attempt < 2:
                            print(f"[{p.name}] rate limited, waiting {wait:.0f}s ...")
                            time.sleep(wait)
                            continue
                    if json_mode and status == 400 and "response_format" in msg:
                        kwargs.pop("response_format", None)  # provider doesn't support JSON mode
                        continue
                    errors.append(f"{p.name}/{model}: {msg[:200]}")
                    break
        raise LLMError("All LLM providers failed:\n  " + "\n  ".join(errors))

    def chat_json(self, messages: list[dict], fast: bool = True, max_tokens: int = 600) -> dict:
        text = self.chat(messages, fast=fast, max_tokens=max_tokens, json_mode=True)
        return parse_json(text)


def _retry_after(e) -> float | None:
    try:
        h = e.response.headers.get("retry-after")
        return float(h) if h else None
    except Exception:
        return None


def parse_json(text: str) -> dict:
    """LLMs sometimes wrap JSON in ```json fences or add text around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}
