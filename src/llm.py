"""Thin, provider-agnostic LLM client over plain HTTPS (no vendor SDKs).

Provider is picked from whichever key is set (or LLM_PROVIDER):
  ANTHROPIC_API_KEY -> Anthropic Messages API
  OPENAI_API_KEY    -> OpenAI Chat Completions
  GEMINI_API_KEY    -> Google Gemini generateContent (has a free tier)
LLM_MODEL overrides the default model name for the chosen provider.
Keys are read from environment variables or Streamlit secrets.
"""
from __future__ import annotations

import json
import os
import re

import requests

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-3.5-flash-lite",
}
# Tried in order when the main model is overloaded (503) or rate-limited (429).
# Free-tier limits are per model, so falling back spreads the load.
FALLBACK_MODELS = {
    "gemini": ["gemini-3.6-flash", "gemini-flash-lite-latest"],
}
RETRY_STATUS = {429, 500, 503}
KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def _load_dotenv():
    """Minimal .env reader for local runs (no extra dependency)."""
    from pathlib import Path

    f = Path(__file__).resolve().parents[1] / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip() and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


_load_dotenv()


def _get(name):
    v = os.environ.get(name)
    if v:
        return v
    try:  # Streamlit secrets, only when running inside Streamlit
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self):
        prov = (_get("LLM_PROVIDER") or "").lower().strip()
        if not prov:
            prov = next((p for p, k in KEY_NAMES.items() if _get(k)), "")
        self.provider = prov or None
        self.key = _get(KEY_NAMES[prov]) if prov in KEY_NAMES else None
        if prov == "gemini" and not self.key:
            self.key = _get("GOOGLE_API_KEY")
        self.model = _get("LLM_MODEL") or DEFAULT_MODELS.get(prov)

    @property
    def available(self) -> bool:
        return bool(self.provider and self.key)

    def label(self):
        return f"{self.provider} / {self.model}" if self.available else "not configured"

    def complete(self, system: str, user: str, json_mode=False, max_tokens=600, timeout=25) -> str:
        if not self.available:
            raise LLMError("No LLM API key configured")
        models = [self.model] + [m for m in FALLBACK_MODELS.get(self.provider, []) if m != self.model]
        err = None
        for model in models:
            try:
                out = self._call(model, system, user, json_mode, max_tokens, timeout)
                self.last_model = model
                return out
            except LLMError as e:
                err = e
                if getattr(e, "status", None) not in RETRY_STATUS:
                    raise
        raise err

    def _call(self, model, system, user, json_mode, max_tokens, timeout) -> str:
        try:
            if self.provider == "anthropic":
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    # no temperature: newer Claude models reject non-default sampling params
                    json={"model": model, "max_tokens": max_tokens, "system": system,
                          "messages": [{"role": "user", "content": user}]},
                    timeout=timeout,
                )
                r.raise_for_status()
                return "".join(b.get("text", "") for b in r.json()["content"])
            if self.provider == "openai":
                body = {"model": model, "max_completion_tokens": max_tokens,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
                if json_mode:
                    body["response_format"] = {"type": "json_object"}
                r = requests.post("https://api.openai.com/v1/chat/completions",
                                  headers={"Authorization": f"Bearer {self.key}"}, json=body, timeout=timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            if self.provider == "gemini":
                cfg = {"temperature": 0, "maxOutputTokens": max_tokens + 2048}  # headroom for thinking tokens
                if json_mode:
                    cfg["responseMimeType"] = "application/json"
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": self.key},
                    json={"systemInstruction": {"parts": [{"text": system}]},
                          "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": cfg},
                    timeout=timeout,
                )
                r.raise_for_status()
                return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
        except requests.HTTPError as e:
            err = LLMError(f"{self.provider} API error {e.response.status_code} ({model}): {e.response.text[:200]}")
            err.status = e.response.status_code
            raise err from e
        except requests.RequestException as e:
            raise LLMError(f"{self.provider} request failed: {e}") from e
        raise LLMError(f"Unknown provider {self.provider}")


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise LLMError("Model did not return JSON")
    return json.loads(m.group(0))
