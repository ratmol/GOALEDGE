# How the RAG analyst works (plain-English walkthrough)

This explains every moving part of "Ask the Gaffer" — what each file and function
does, how a question flows through the system, and why you sometimes see the
"[Model summary]" fallback.

---

## The big idea

RAG = **R**etrieval **A**ugmented **G**eneration. Instead of training a language
model on your football data, we:

1. **Retrieve** the most relevant facts from a knowledge base at question time.
2. **Augment** the prompt by injecting those facts + a live match simulation.
3. **Generate** a grounded answer with a free hosted open-source LLM.

So the LLM never guesses stats — it's handed the real numbers and told to explain
them. That's the whole trick.

---

## The pieces (files & their functions)

### 1. `backend/rag/corpus.py` — the knowledge base
Turns your data into little text "documents" the analyst can search.

- **`CONCEPTS`** — 8 hand-written notes explaining football/betting ideas
  (xG, corners, cards, Kelly staking, form, home advantage, recovery,
  competition weighting).
- **`team_docs(tm)`** — for every team in your data, auto-writes a note from the
  fitted model: how many matches, attacking/defensive strength, recent form,
  last-5 results, and (if available) its real stat profile. This is how the CSV
  data becomes retrievable knowledge.
- **`_strength_word(x)`** — converts a raw model number into a readable word
  ("very strong", "average", "weak") for those notes.
- **`build_corpus(tm)`** — concepts + team docs = the full corpus (~328 docs).

### 2. `backend/rag/retriever.py` — the search engine
Finds the few documents most relevant to a question. Uses **BM25**, a classic
keyword-ranking algorithm (no heavy AI libraries, so it runs on Vercel).

- **`tokenize(text)`** — splits text into lowercase words.
- **`Retriever.__init__`** — pre-computes word frequencies and rarity (IDF) for
  every doc, once, so searches are fast.
- **`Retriever._score(query, i)`** — the BM25 formula: rewards docs that contain
  the query's words, especially rare/distinctive ones.
- **`Retriever.search(query, k)`** — returns the top-k matching docs.
- **`build_retriever(tm)`** — builds the corpus and wraps it in a Retriever.

### 3. `backend/llm/provider.py` — the LLM selector (with fallback)
Decides *which* language model to talk to, and fails over if one breaks.

- **`HostedLLM`** — talks to any OpenAI-compatible API (Cerebras, Groq,
  OpenRouter). Holds a *chain* of models; `generate()` tries the first, and on a
  404 / rate-limit / error automatically tries the next — so users never hit a
  dead analyst.
- **`NullLLM`** — a stand-in used when no LLM is configured. `available()` is
  False, which triggers the deterministic fallback (the "[Model summary]" you saw).
- **`_endpoints_from_env()`** — reads `LLM_PROVIDERS` (the JSON chain) and/or
  `LLM_API_KEY` from the environment and builds the ordered list of models,
  skipping any whose key is missing.
- **`get_llm()`** — the selector: hosted chain if any key is set → else local
  Ollama if running → else NullLLM.
- **`describe(client)`** — a human-readable label shown at `/health`
  (e.g. `hosted[3]:llama-3.3-70b -> ...`).

### 4. `backend/llm/analyst.py` — The Gaffer
Ties retrieval + simulation + the LLM together.

- **`Analyst.retrieve(question, home, away)`** — asks the Retriever for the top
  notes relevant to the question and the two teams.
- **`Analyst._single_shot(...)`** — the main RAG path: runs the match simulation,
  retrieves notes, injects both into ONE prompt, and calls the LLM once.
  Returns the answer plus a `sources` list. Robust across all hosted providers.
- **`Analyst._compact_sim(d)`** — trims the big simulation output down to the key
  numbers, so the prompt stays small and cheap.
- **`Analyst._tool_loop(...)`** — an alternative path used only with local Ollama,
  where the model calls the simulator itself as a "tool".
- **`Analyst._fallback(...)`** — when NO LLM is available, returns a plain,
  deterministic summary straight from the simulator (no AI writing). This is the
  "[Model summary — set LLM_API_KEY ...]" message.
- **`Analyst.ask(question, home, away)`** — the entry point. Picks the path:
  no LLM → fallback; Ollama → tool loop; hosted → single-shot RAG.

### 5. `backend/api/main.py` — the wiring
- Builds `get_llm()` and `build_retriever(...)` at startup and hands them to the
  `Analyst`.
- **`POST /analyst/ask`** — receives `{question, home, away}`, calls
  `analyst.ask(...)`, returns `{answer, sources, ollama}`.
- **`GET /health`** — reports `llm_ready` and `llm_provider` so you can see at a
  glance whether the LLM is wired up.

---

## What happens when you click "Ask" (the flow)

```
You type a question + a fixture is selected (e.g. Spain vs England)
        │
        ▼
POST /analyst/ask  →  Analyst.ask(...)
        │
        ├─ is an LLM available? (get_llm)
        │      NO  → _fallback()  →  deterministic "[Model summary]"  ← the screenshot
        │      YES → _single_shot():
        │               1. simulate(home, away)         → real match numbers
        │               2. retrieve(question, teams)    → top notes from corpus
        │               3. inject both into one prompt
        │               4. HostedLLM.generate()         → grounded answer
        │                     (tries model 1 → 2 → 3 on failure)
        ▼
{ answer, sources }  →  shown in the UI
```

---

## Why the screenshot shows "[Model summary … set LLM_API_KEY]"

That yellow-warning state means **no LLM was available where the app is running**,
so it used the deterministic fallback. It still fetched real numbers (note the
`sources: model:Spain-vs-England`), but there was no language model to turn them
into a conversational answer.

It happens when `get_llm()` returned `NullLLM`, i.e. it found **no** LLM config.
Check, in order:

1. **Where is this running?** Open `/health`.
   - `"llm_ready": false, "llm_provider": "none"` → config isn't reaching the app.
2. **Local run:** your `.env` needs `CEREBRAS_API_KEY`, `OPENROUTER_API_KEY`, AND
   `LLM_PROVIDERS` (the JSON chain). Keys alone do nothing without `LLM_PROVIDERS`.
3. **Live (Vercel):** all three must be set as **Production** env vars, the code
   with `provider.py`'s chain support must be pushed, and you must **redeploy**
   after adding the vars (env changes need a fresh deploy).
4. **Bad JSON:** if `LLM_PROVIDERS` was pasted with extra quotes or broken, it's
   ignored → fallback. Paste it as raw JSON.
5. **Model slug wrong:** if a model name 404s, the chain rolls to the next; if all
   fail, you get fallback. Confirm current slugs on the provider's models page.

The single most useful check is `/health` → `llm_provider`. If it says
`hosted[...]`, the keys are wired and any remaining issue is a bad model slug. If
it says `none`, the environment variables aren't being seen.
