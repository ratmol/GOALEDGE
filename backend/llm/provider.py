"""
LLM provider selection with automatic fallback.

The Gaffer works locally (Ollama) AND when deployed (free hosted open models via
any OpenAI-compatible API: Cerebras, Groq, OpenRouter, ...).

Fallback: you can configure a CHAIN of hosted models. If one 404s (bad slug),
rate-limits, or errors, the next is tried automatically — so users never hit a
dead analyst.

Env (simplest — single provider):
  LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Env (chain / multi-provider fallback) — JSON list, highest priority first:
  LLM_PROVIDERS='[
    {"key_env":"CEREBRAS_API_KEY","base":"https://api.cerebras.ai/v1","model":"llama-3.3-70b"},
    {"key_env":"CEREBRAS_API_KEY","base":"https://api.cerebras.ai/v1","model":"llama3.1-8b"},
    {"key_env":"OPENROUTER_API_KEY","base":"https://openrouter.ai/api/v1","model":"meta-llama/llama-3.3-70b-instruct:free"}
  ]'
Each entry's key_env names the env var holding that provider's key. Entries whose
key is missing are skipped, so you can set only the ones you have.

Selection: hosted chain (any key present) -> local Ollama -> NullLLM (the analyst
then returns a deterministic, model-only summary; nothing breaks).
"""
from __future__ import annotations

import json
import os

import requests

from backend.llm.ollama_client import OllamaClient

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.cerebras.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b")

# Well-known providers auto-detected from their key env var — so just setting
# e.g. CEREBRAS_API_KEY works with NO LLM_PROVIDERS needed. Each provides an
# ordered model list (fast fallback within the provider).
KNOWN_PROVIDERS = [
    ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1",
     ["llama-3.3-70b", "llama3.1-8b"]),
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1",
     ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]),
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
     ["meta-llama/llama-3.3-70b-instruct:free"]),
    ("TOGETHER_API_KEY", "https://api.together.xyz/v1",
     ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"]),
]


class HostedLLM:
    """OpenAI-compatible client that tries a chain of (key, base, model)
    endpoints and returns the first one that answers."""
    supports_tools = False   # single-shot RAG path (robust across providers)

    def __init__(self, endpoints: list[dict]):
        # endpoints: [{"key","base","model"}], already filtered to ones with keys
        self.endpoints = endpoints

    def available(self) -> bool:
        return bool(self.endpoints)

    @property
    def model(self) -> str:
        return self.endpoints[0]["model"] if self.endpoints else "none"

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for ep in self.endpoints:
            try:
                r = requests.post(
                    f"{ep['base'].rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {ep['key']}"},
                    json={"model": ep["model"], "messages": messages,
                          "temperature": 0.4},
                    timeout=60)
                # 404 (bad model), 429 (rate limit), 5xx -> try next endpoint
                if r.status_code >= 400:
                    last_err = f"{ep['model']}: HTTP {r.status_code}"
                    continue
                return r.json()["choices"][0]["message"]["content"].strip()
            except requests.RequestException as e:
                last_err = f"{ep['model']}: {type(e).__name__}"
                continue
        # every endpoint failed
        raise RuntimeError(f"All LLM endpoints failed ({last_err})")


class NullLLM:
    supports_tools = False
    model = "none"

    def available(self) -> bool:
        return False

    def generate(self, prompt: str, system: str | None = None) -> str:
        return ""


def _endpoints_from_env() -> list[dict]:
    """Build the ordered endpoint chain from env, skipping any without a key."""
    eps: list[dict] = []

    # 1) explicit chain via LLM_PROVIDERS (JSON)
    raw = os.getenv("LLM_PROVIDERS", "").strip()
    if raw:
        try:
            for item in json.loads(raw):
                key = os.getenv(item.get("key_env", ""), "").strip()
                if key and item.get("base") and item.get("model"):
                    eps.append({"key": key, "base": item["base"],
                                "model": item["model"]})
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[provider] bad LLM_PROVIDERS json: {e}")

    # 2) simple single provider via LLM_API_KEY (appended as a fallback tier)
    simple_key = os.getenv("LLM_API_KEY", "").strip()
    if simple_key:
        eps.append({"key": simple_key, "base": LLM_BASE_URL, "model": LLM_MODEL})


    # 3) auto-detect well-known provider keys (no LLM_PROVIDERS required)
    if not eps:
        for key_env, base, models in KNOWN_PROVIDERS:
            k = os.getenv(key_env, "").strip()
            if k:
                for m in models:
                    eps.append({"key": k, "base": base, "model": m})

    return eps


def get_llm():
    eps = _endpoints_from_env()
    if eps:
        return HostedLLM(eps)
    oc = OllamaClient()
    if oc.available():
        return oc
    return NullLLM()


def describe(client) -> str:
    if isinstance(client, HostedLLM):
        chain = " -> ".join(e["model"] for e in client.endpoints)
        return f"hosted[{len(client.endpoints)}]:{chain}"
    if isinstance(client, OllamaClient):
        return f"ollama:{client.model}"
    return "none"


def diagnose() -> dict:
    """Safe introspection for /health/llm — reports which env vars are SEEN
    (booleans only, never values) and how the endpoint chain parsed."""
    raw = os.getenv("LLM_PROVIDERS", "").strip()
    info = {
        "env_seen": {
            "LLM_PROVIDERS": bool(raw),
            "LLM_API_KEY": bool(os.getenv("LLM_API_KEY", "").strip()),
            "CEREBRAS_API_KEY": bool(os.getenv("CEREBRAS_API_KEY", "").strip()),
            "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
            "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY", "").strip()),
            "TOGETHER_API_KEY": bool(os.getenv("TOGETHER_API_KEY", "").strip()),
        },
        "providers_json_valid": None,
        "providers_entries": 0,
        "usable_endpoints": 0,
        "missing_keys": [],
        "notes": [],
    }
    if raw:
        try:
            items = json.loads(raw)
            info["providers_json_valid"] = True
            info["providers_entries"] = len(items)
            for it in items:
                ke = it.get("key_env", "")
                if os.getenv(ke, "").strip():
                    info["usable_endpoints"] += 1
                elif ke not in info["missing_keys"]:
                    info["missing_keys"].append(ke)
        except Exception as e:
            info["providers_json_valid"] = False
            info["notes"].append(f"LLM_PROVIDERS is not valid JSON: {e}")
    eps = _endpoints_from_env()
    info["total_endpoints"] = len(eps)
    info["models"] = [e["model"] for e in eps]
    return info


if __name__ == "__main__":
    import types
    # chain built from env, skipping missing keys
    os.environ["LLM_PROVIDERS"] = json.dumps([
        {"key_env": "K_MISSING", "base": "https://a/v1", "model": "m-a"},
        {"key_env": "K_PRESENT", "base": "https://b/v1", "model": "m-b"},
    ])
    os.environ["K_PRESENT"] = "kb"
    os.environ.pop("LLM_API_KEY", None)
    import backend.llm.provider as P
    c = P.get_llm()
    assert isinstance(c, P.HostedLLM) and len(c.endpoints) == 1, c.endpoints
    assert c.endpoints[0]["model"] == "m-b"
    print("chain skips missing keys OK:", P.describe(c))

    # fallback: first endpoint 404s, second succeeds
    c2 = P.HostedLLM([{"key": "k1", "base": "https://x/v1", "model": "bad"},
                      {"key": "k2", "base": "https://y/v1", "model": "good"}])
    seq = []
    def fake_post(url, headers=None, json=None, timeout=None):
        m = json["model"]; seq.append(m)
        code = 404 if m == "bad" else 200
        return types.SimpleNamespace(
            status_code=code, raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "ok from good"}}]})
    P.requests.post = fake_post
    out = c2.generate("hi", system="s")
    assert out == "ok from good" and seq == ["bad", "good"], (out, seq)
    print("404 fallback OK: tried", seq, "->", out)
