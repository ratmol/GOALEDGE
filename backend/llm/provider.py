"""
LLM provider selection — makes The Gaffer work locally AND when deployed.

Locally you can run Ollama (free). On Vercel there is no Ollama, so we point the
SAME analyst at a free hosted open-source model via any OpenAI-compatible API
(Groq, OpenRouter, Together, ...). One env var switches provider; nothing else
changes.

Env:
  LLM_API_KEY    hosted key (if set, hosted provider is used)
  LLM_BASE_URL   OpenAI-compatible base (default: Groq)
  LLM_MODEL      model name (default: an open Llama on Groq)
  OLLAMA_HOST / OLLAMA_MODEL  local fallback (see ollama_client)

Selection order: hosted key present -> HostedLLM; else local Ollama if running;
else NullLLM (analyst then returns a deterministic, model-only summary).
"""
from __future__ import annotations

import os

import requests

from backend.llm.ollama_client import OllamaClient

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


class HostedLLM:
    """OpenAI-compatible chat-completions client (Groq/OpenRouter/Together/...)."""
    supports_tools = False   # we use single-shot context injection (RAG-friendly)

    def __init__(self, api_key: str, base_url: str = LLM_BASE_URL,
                 model: str = LLM_MODEL):
        self.api_key = api_key
        self.base = base_url.rstrip("/")
        self.model = model

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = requests.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": 0.4},
            timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class NullLLM:
    supports_tools = False

    def available(self) -> bool:
        return False

    def generate(self, prompt: str, system: str | None = None) -> str:
        return ""


def get_llm():
    key = os.getenv("LLM_API_KEY", "").strip()
    if key:
        return HostedLLM(key)
    oc = OllamaClient()
    if oc.available():
        return oc
    return NullLLM()


def describe(client) -> str:
    if isinstance(client, HostedLLM):
        return f"hosted:{client.model}"
    if isinstance(client, OllamaClient):
        return f"ollama:{client.model}"
    return "none"


if __name__ == "__main__":
    import types
    # selection: hosted key -> HostedLLM
    os.environ["LLM_API_KEY"] = "test-key"
    assert isinstance(get_llm(), HostedLLM); print("hosted selected OK")
    # no key + no ollama -> NullLLM
    os.environ["LLM_API_KEY"] = ""
    import backend.llm.provider as P
    P.OllamaClient = lambda *a, **k: types.SimpleNamespace(
        available=lambda: False, model="x")
    assert isinstance(get_llm(), NullLLM); print("null selected OK")
    # HostedLLM.generate request shaping (mocked transport)
    calls = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.update(url=url, headers=headers, body=json)
        return types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": " Spain edge it. "}}]})
    P.requests.post = fake_post
    out = HostedLLM("k", model="llama-3.3-70b-versatile").generate("hi", system="sys")
    assert out == "Spain edge it.", out
    assert calls["url"].endswith("/chat/completions")
    assert calls["headers"]["Authorization"] == "Bearer k"
    assert calls["body"]["messages"][0]["role"] == "system"
    print("hosted generate shaping OK ->", out)
