"""
Ollama client — local, free LLM. Supports plain generation AND tool-calling
chat (/api/chat) so the analyst can call the match simulator as a tool.

The LLM never invents predictions — it calls the model's tools and explains the
numbers they return.
"""
from __future__ import annotations

import os

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model

    def available(self) -> bool:
        try:
            requests.get(f"{self.host}/api/tags", timeout=2)
            return True
        except requests.RequestException:
            return False

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """One /api/chat round. Returns the assistant message dict, which may
        contain `tool_calls`."""
        payload = {"model": self.model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("message", {})


SYSTEM_PROMPT = (
    "You are a football analyst who writes concise, calibrated match previews. "
    "You are given pre-computed win/draw/loss probabilities and key features. "
    "Explain WHY the numbers look this way. Never invent new probabilities; "
    "report the ones given. Keep it under 150 words."
)


def write_preview(client: OllamaClient, prompt: str) -> str:
    if not client.available():
        return ("[Ollama not running — install from https://ollama.com and run "
                "`ollama pull llama3.1`. Showing the model's numbers only.]")
    return client.generate(prompt, system=SYSTEM_PROMPT)
