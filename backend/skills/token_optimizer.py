"""
Token Optimizer — compress an LLM prompt until it fits a token budget.

Loops through compression actions (least-lossy first) until token count is at or
below MAX_PROMPT_TOKENS. A "must-keep" block (the final probabilities) is never
dropped. Returns the compressed prompt plus a report.

Falls back to a word-based token estimate if tiktoken is unavailable.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "1500"))

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover - fallback
    def count_tokens(text: str) -> int:
        # ~0.75 words per token heuristic.
        return int(len(re.findall(r"\S+", text)) / 0.75)


@dataclass
class OptimizeResult:
    prompt: str
    original_tokens: int
    final_tokens: int
    actions_applied: list[str]
    under_budget: bool


def _collapse_whitespace(t: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", t)).strip()


def _round_numbers(t: str) -> str:
    return re.sub(r"\d+\.\d{3,}", lambda m: f"{float(m.group()):.2f}", t)


class TokenOptimizer:
    def __init__(self, budget: int = MAX_PROMPT_TOKENS, max_passes: int = 6):
        self.budget = budget
        self.max_passes = max_passes

    def optimize(self, context: str, must_keep: str = "") -> OptimizeResult:
        original = count_tokens(must_keep + context)
        applied: list[str] = []
        text = context

        actions = [
            ("collapse_whitespace", _collapse_whitespace),
            ("round_numbers", _round_numbers),
            ("drop_blank_lines", lambda t: "\n".join(
                l for l in t.splitlines() if l.strip())),
            ("keep_top_lines", self._keep_top_lines),
        ]

        for name, fn in actions:
            if count_tokens(must_keep + text) <= self.budget:
                break
            text = fn(text)
            applied.append(name)

        # Last resort: hard-truncate the context, never the must-keep block.
        if count_tokens(must_keep + text) > self.budget:
            text = self._truncate_to_budget(text, must_keep)
            applied.append("hard_truncate")

        final_prompt = (must_keep + "\n" + text).strip() if must_keep else text
        final = count_tokens(final_prompt)
        return OptimizeResult(final_prompt, original, final, applied,
                              final <= self.budget)

    def _keep_top_lines(self, t: str) -> str:
        """Prioritize lines that look feature/number-bearing."""
        lines = [l for l in t.splitlines() if l.strip()]
        scored = sorted(lines, key=lambda l: bool(re.search(r"\d", l)), reverse=True)
        return "\n".join(scored[: max(5, len(scored) // 2)])

    def _truncate_to_budget(self, t: str, must_keep: str) -> str:
        words = t.split()
        while words and count_tokens(must_keep + " ".join(words)) > self.budget:
            words = words[: int(len(words) * 0.9)] or words[:-1]
        return " ".join(words)


if __name__ == "__main__":
    must = "PROBS win=0.61 draw=0.22 loss=0.17 | xScore 2-1"
    blob = ("\n".join(f"feature_{i} = {i + 0.123456789}" for i in range(400)))
    opt = TokenOptimizer(budget=200).optimize(blob, must_keep=must)
    print(f"original={opt.original_tokens} final={opt.final_tokens} "
          f"under_budget={opt.under_budget} actions={opt.actions_applied}")
