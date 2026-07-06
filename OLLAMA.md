# The Gaffer — local llama analyst (Ollama)

GoalEdge talks to a **free, local** LLM via [Ollama](https://ollama.com). The LLM
never invents numbers — it **calls the match simulator as a tool**, gets real
output back, and explains it. Everything runs on your machine, so there's no API
cost and the daily/scout features are effectively free.

The whole app runs **without** Ollama (you just get the model's numbers and a
deterministic offline summary). Install it to unlock conversational analysis.

## 1. Install Ollama
- Download from **https://ollama.com** (Windows/Mac/Linux) and install.
- It runs a local server at `http://localhost:11434` automatically.

## 2. Pull a tool-capable model
Tool-calling needs a model that supports it. Recommended:

```bash
ollama pull llama3.1        # 8B, supports tools — good default
# alternatives that support tools:
# ollama pull qwen2.5        ollama pull mistral-nemo
```

Set your choice in `.env` (defaults to `llama3.1`):
```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

## 3. Use it
Start the API (`uvicorn backend.api.main:app --reload`) and open the dashboard.

- **Ask the Gaffer** box: type a question like
  - "Why would Spain dominate corners vs Italy?"
  - "Is Brazil vs Argentina a low-scoring game?"
  - "Who's in better form, France or Portugal?"
  The model calls tools (`simulate_match`, `team_form`, `list_teams`) and answers
  from the real numbers. The panel shows which tools it used.
- **Scout report** button: a written match read generated from the simulation.

## 4. How it works (under the hood)
- `backend/llm/ollama_client.py` — `chat()` uses Ollama's `/api/chat` with tools.
- `backend/llm/analyst.py` — defines the tools, runs the call loop, dispatches
  tool calls to the real `MatchSimulator`, and token-trims tool output (cheap +
  fast). Falls back to a deterministic summary when Ollama is offline.
- `POST /analyst/ask` — `{ "question": "...", "home": "Spain", "away": "Italy" }`
  → `{ answer, tools_used, ollama }`.

## 5. Verify
With Ollama running:
```bash
curl -s http://localhost:8000/health        # "ollama": true
curl -s -X POST http://localhost:8000/analyst/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How does Spain vs Italy look?","home":"Spain","away":"Italy"}'
```

The tool-call loop itself is unit-tested offline (mock model):
```bash
python -m backend.llm.analyst
```

## Cost note
Local inference = **no token cost**. That's why scheduled daily previews and the
analyst are cheap by design. A hosted API can be swapped in later behind the same
`OllamaClient` interface if you ever want bigger models.
