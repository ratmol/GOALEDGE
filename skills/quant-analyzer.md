# Skill: Quant Analyzer 📈

**Role:** A mathematically brilliant quantitative analyst. Given the prepared
dataset, it builds and *iteratively refines* match-outcome predictions until they
hit a calibration quality threshold — it does **not** stop after one pass.

## Core idea: loop until threshold

This skill is not a one-shot instruction. It runs a refinement loop:

```
1. Build features + fit the current model (Poisson + XGBoost ensemble).
2. Evaluate on a held-out / walk-forward validation set.
3. Score calibration with Brier score and log-loss.
4. IF Brier <= TARGET_BRIER (default 0.20)  -> STOP, ship the model.
   ELSE -> apply the next improvement action and repeat.
5. Hard stop after MAX_QUANT_ITERATIONS to avoid runaway loops.
```

## Improvement actions (tried in order each iteration)
1. Add/refresh rolling-form windows (3/5/10 matches).
2. Add recovery features: days rest, minutes load, travel congestion.
3. Add **team chemistry** features (see `features/chemistry.py`): squad stability,
   minutes played together, lineup continuity vs. previous match.
4. Retune XGBoost (depth, learning rate, regularization) via small grid.
5. Recalibrate probabilities (Platt / isotonic).
6. Blend Poisson and XGBoost weights to minimize validation log-loss.

## Inputs / outputs
- **Input:** unified match dataframe (club + league + international).
- **Output:** fitted model, calibration report, and a per-match probability
  distribution {win, draw, loss} + expected scoreline.

## Guardrails
- Never reports a point prediction — always a calibrated probability.
- Logs every iteration's score so the loop is auditable.
- If threshold is not reached, returns the **best** iteration, clearly flagged.

> Implementation: `backend/skills/quant_analyzer.py`
