# The Gaffer — RAG analyst (works deployed, for free)

The analyst no longer depends on a local Ollama. It runs on a **free hosted
open-source model** when deployed, and uses **RAG** (retrieval-augmented
generation) to ground every answer in your data.

## How it works
1. **Provider layer** (`backend/llm/provider.py`) — picks the LLM by env:
   - `LLM_API_KEY` set → **hosted** open model via any OpenAI-compatible API
     (Groq / OpenRouter / Together). This is the deployed path.
   - else local **Ollama** if running.
   - else a deterministic, model-only summary (no LLM needed).
2. **RAG** (`backend/rag/`):
   - `corpus.py` builds a knowledge base: one note per team generated from your
     CSV-derived model (matches, form, competitions, stat profile) + concept
     notes (xG, corners, cards, Kelly, home advantage). This is how we "use the
     CSV data" — retrieved at query time, not trained into a model.
   - `retriever.py` is a pure-Python **BM25** retriever (no heavy deps, runs on
     Vercel).
3. **Analyst** (`backend/llm/analyst.py`) — for a question it simulates the
   fixture, retrieves the most relevant notes, injects both into one prompt, and
   generates a grounded answer. Response includes `sources` (what it used).

## Enable the deployed analyst (free)
1. Create a free key at **https://console.groq.com** (or OpenRouter).
2. Locally: put it in `.env` as `LLM_API_KEY=...`.
3. On Vercel: add `LLM_API_KEY` (and optionally `LLM_MODEL`) as an env var
   (standard Production env — free on Hobby), then redeploy.
4. Check `/health` → `"llm_ready": true`, `"llm_provider": "hosted:..."`.

Swap providers by changing `LLM_BASE_URL` + `LLM_MODEL` — any OpenAI-compatible
endpoint works. Local Ollama still works with no key.

## Upgrade path: semantic retrieval
BM25 is lexical. For semantic RAG, precompute embeddings offline, commit the
vectors, and at query time embed the question via a free embeddings API
(Jina/Cohere) and cosine-match. The `Retriever` interface stays the same — only
its `_score` changes.

## Not RAG, not the LLM: RL
Reinforcement learning is a **separate** feature (a staking/prediction policy
that learns and is benchmarked against a baseline). It does not power the chat.
Tracked separately.
