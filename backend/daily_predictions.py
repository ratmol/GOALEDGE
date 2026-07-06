"""
Daily predictions job — writes a markdown digest of upcoming match predictions.

Uses Phase1Model when trained (phase1.pkl present), falls back to Elo baseline.
Run manually:   python backend/daily_predictions.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.llm.ollama_client import OllamaClient, write_preview
from backend.skills.token_optimizer import TokenOptimizer

SAMPLE_FIXTURES = [
    ("Argentina", "Brazil"),
    ("France", "England"),
    ("Spain", "Germany"),
    ("Netherlands", "Portugal"),
    ("Morocco", "Croatia"),
]


def _get_predictor():
    from backend.models.phase1 import Phase1Model, MODEL_PATH
    if MODEL_PATH.exists():
        m = Phase1Model.load()
        return m.predict, "phase1"
    from backend.models.elo import EloModel
    elo = EloModel()
    return lambda h, a, neutral=True: elo.win_draw_loss(h, a, neutral), "elo"


def run(out_dir: Path) -> Path:
    predict_fn, model_name = _get_predictor()
    ollama, opt = OllamaClient(), TokenOptimizer()
    lines = [
        f"# GoalEdge Predictions — {date.today():%A %d %B %Y}",
        f"> Model: **{model_name}**\n",
    ]

    for home, away in SAMPLE_FIXTURES:
        p = predict_fn(home, away, neutral=True)
        must = (f"PROBS {home} win={p['win']:.2f} draw={p['draw']:.2f} "
                f"{away} win={p['loss']:.2f}")
        ctx = (f"Match: {home} vs {away} at a neutral World Cup venue.\n"
               f"Explain these probabilities concisely.")
        preview = write_preview(ollama, opt.optimize(ctx, must_keep=must).prompt)

        fav = max(
            [(home, p['win']), ('Draw', p['draw']), (away, p['loss'])],
            key=lambda x: x[1]
        )
        lines += [
            f"## ⚽ {home} vs {away}",
            f"| Outcome | Probability |",
            f"|---------|-------------|",
            f"| {home} Win | **{p['win']:.1%}** |",
            f"| Draw | **{p['draw']:.1%}** |",
            f"| {away} Win | **{p['loss']:.1%}** |",
            f"\n**Tip:** {fav[0]} ({fav[1]:.1%})\n",
            f"{preview}\n",
            "---\n",
        ]

    out = out_dir / f"predictions_{date.today():%Y%m%d}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[daily] Wrote {out}")
    return out


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
