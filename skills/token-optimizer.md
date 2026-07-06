# Skill: Token Optimizer 🪙

**Role:** Keeps every LLM call (Ollama or any hosted API) cheap by compressing the
prompt until it fits a token budget — *without losing the signal the analysis
needs*. Like the quant analyzer, it **loops until a threshold is met**.

## Core idea: loop until under budget

```
1. Count tokens of the assembled prompt (tiktoken).
2. IF tokens <= MAX_PROMPT_TOKENS (default 1500) -> STOP, send it.
   ELSE -> apply the next compression action and re-count.
3. Hard stop after N passes; if still over, truncate lowest-priority context.
```

## Compression actions (tried in order, least-lossy first)
1. Drop redundant whitespace / formatting.
2. Replace verbose feature names with a compact legend.
3. Round numbers to meaningful precision (xG to 2 dp, % to integer).
4. Summarize low-importance features into a single line.
5. Keep only the top-K most predictive features (by model importance).
6. Last resort: hard-truncate, preserving the must-keep block (the probabilities).

## Why this saves you tokens / money
- Default LLM is **Ollama (local)** → effectively zero token cost, so daily
  delivery is cheap by design.
- The optimizer still matters for: (a) keeping local inference fast, and
  (b) staying cheap if you ever switch to a paid hosted API.

## Inputs / outputs
- **Input:** raw context dict (features, probabilities, recent form text).
- **Output:** a compressed prompt string <= budget, plus a report of what was cut.

## Guardrails
- A "must-keep" set (final probabilities + expected scoreline) is never dropped.
- Reports original vs. final token count every run.

> Implementation: `backend/skills/token_optimizer.py`
