"""
Retrain the Phase-1 model (Poisson + XGBoost + quant loop) and save to disk.
Run from the project root:  python train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.models.phase1 import run_quant_loop

if __name__ == "__main__":
    print("[train] Starting Phase-1 training...")
    model = run_quant_loop()
    print("[train] Done. Model saved to backend/models/phase1.pkl")
    print("[train] Sample predictions:")
    for home, away in [("Argentina", "Brazil"), ("Germany", "France"), ("England", "Spain")]:
        p = model.predict(home, away, neutral=True)
        print(f"  {home} vs {away}: win={p['win']:.3f} draw={p['draw']:.3f} loss={p['loss']:.3f}")
