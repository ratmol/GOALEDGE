"""
Match Simulator — the full-stats engine (v2).

Simulates an ENTIRE match via Monte Carlo: goals, xG, shots, shots on target,
possession, corners, fouls, yellow/red cards, offsides + markets. v2 adds
weighted goal model, recent form, recovery, and pluggable team profiles.
"""
from __future__ import annotations

import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RATES = {
    "goals": 1.35, "shots": 12.0, "shots_on_target": 4.3, "corners": 5.0,
    "fouls": 13.0, "yellows": 1.9, "reds": 0.08, "offsides": 2.2,
}
LEAGUE_AVG_LAMBDA = RATES["goals"]
_DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass
class TeamLine:
    name: str
    xg: float; shots: float; sot: float; possession: float
    corners: float; fouls: float; yellows: float; reds: float; offsides: float


class TeamModel:
    def __init__(self):
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.intercept: float = float(np.log(LEAGUE_AVG_LAMBDA))
        self.home_adv: float = 0.25
        self.form: dict[str, float] = {}
        self.last5: dict[str, list[str]] = {}
        self.played: dict[str, int] = {}
        self.profiles: dict[str, dict] = {}
        self.competitions: list[str] = []
        self.n_matches: int = 0

    def fit(self, df) -> "TeamModel":
        self.n_matches = len(df)
        self.competitions = sorted(set(df["tournament"].astype(str)))
        w = df["weight"].to_numpy(dtype=float) if "weight" in df else np.ones(len(df))
        gf_w, ga_w, wsum = defaultdict(float), defaultdict(float), defaultdict(float)
        hg = ag = hn = 0.0
        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            hs, as_ = int(r.home_score), int(r.away_score)
            wi = float(w[i])
            gf_w[h] += wi * hs; ga_w[h] += wi * as_; wsum[h] += wi
            gf_w[a] += wi * as_; ga_w[a] += wi * hs; wsum[a] += wi
            self.played[h] = self.played.get(h, 0) + 1
            self.played[a] = self.played.get(a, 0) + 1
            if not bool(getattr(r, "neutral", True)):
                hg += wi * hs; ag += wi * as_; hn += wi
        avg = max(sum(gf_w.values()) / max(sum(wsum.values()), 1e-9), 0.5)
        self.intercept = float(np.log(avg))
        for t in wsum:
            scored = gf_w[t] / wsum[t]; conceded = ga_w[t] / wsum[t]
            self.attack[t] = float(np.log(max(scored, 0.15) / avg))
            self.defence[t] = float(np.log(avg / max(conceded, 0.15)))
        if hn > 0:
            self.home_adv = float(np.clip(np.log((hg / hn) / max(ag / hn, 0.1)), 0.0, 0.6))
        self._fit_form(df)
        self._load_profiles()
        return self

    def _fit_form(self, df, k: int = 6) -> None:
        recent = defaultdict(lambda: deque(maxlen=k))
        results = defaultdict(lambda: deque(maxlen=5))
        gd_all = defaultdict(list)
        for r in df.sort_values("date").itertuples(index=False):
            h, a = r.home_team, r.away_team
            hs, as_ = int(r.home_score), int(r.away_score)
            recent[h].append(hs - as_); recent[a].append(as_ - hs)
            gd_all[h].append(hs - as_); gd_all[a].append(as_ - hs)
            results[h].append("W" if hs > as_ else "D" if hs == as_ else "L")
            results[a].append("W" if as_ > hs else "D" if as_ == hs else "L")
        for t in gd_all:
            base = float(np.mean(gd_all[t])) if gd_all[t] else 0.0
            rec = float(np.mean(recent[t])) if recent[t] else 0.0
            self.form[t] = float(np.clip(rec - base, -2.0, 2.0))
            self.last5[t] = list(results[t])

    def _load_profiles(self) -> None:
        path = _DATA / "team_profiles.csv"
        if not path.exists():
            return
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader((ln for ln in f if not ln.startswith("#"))):
                try:
                    self.profiles[row["team"]] = {
                        "shots": float(row["avg_shots"]), "sot": float(row["avg_sot"]),
                        "corners": float(row["avg_corners"]), "fouls": float(row["avg_fouls"]),
                        "yellows": float(row["avg_yellows"]), "offsides": float(row["avg_offsides"]),
                        "possession_bias": float(row["possession_bias"]),
                    }
                except (KeyError, ValueError):
                    continue

    @property
    def teams(self) -> list[str]:
        return sorted(self.played)


class MatchSimulator:
    def __init__(self, model: TeamModel, n_sims: int = 20000, seed: int = 7):
        self.tm = model; self.n = n_sims; self.rng = np.random.default_rng(seed)

    def _base_lambdas(self, home, away, neutral):
        tm = self.tm; ha = 0.0 if neutral else tm.home_adv
        lam_h = np.exp(tm.intercept + tm.attack.get(home, 0.0) - tm.defence.get(away, 0.0) + ha)
        lam_a = np.exp(tm.intercept + tm.attack.get(away, 0.0) - tm.defence.get(home, 0.0))
        return float(lam_h), float(lam_a)

    @staticmethod
    def _recovery_mult(rest_days):
        return float(1.0 - 0.04 * max(0.0, 4.0 - rest_days))

    def adjusted_lambdas(self, home, away, neutral, rest_home=4.0, rest_away=4.0):
        lam_h, lam_a = self._base_lambdas(home, away, neutral)
        fh, fa = self.tm.form.get(home, 0.0), self.tm.form.get(away, 0.0)
        lam_h *= float(np.exp(np.clip(0.12 * fh - 0.08 * fa, -0.5, 0.5)))
        lam_a *= float(np.exp(np.clip(0.12 * fa - 0.08 * fh, -0.5, 0.5)))
        lam_h *= self._recovery_mult(rest_home); lam_a *= self._recovery_mult(rest_away)
        return max(lam_h, 0.05), max(lam_a, 0.05)

    def _possession(self, lam_h, lam_a, home, away):
        ph = 50.0 + 14.0 * np.tanh(lam_h - lam_a)
        pr_h = self.tm.profiles.get(home, {}).get("possession_bias")
        pr_a = self.tm.profiles.get(away, {}).get("possession_bias")
        if pr_h and pr_a:
            ph = 0.5 * ph + 0.5 * (50.0 + (pr_h - pr_a))
        ph = float(np.clip(ph, 28.0, 72.0))
        return ph, 100.0 - ph

    def _line(self, name, lam, possession):
        prof = self.tm.profiles.get(name)
        att = lam / LEAGUE_AVG_LAMBDA; poss_f = possession / 50.0
        if prof:
            scale = 0.55 + 0.45 * att
            shots = prof["shots"] * scale; sot = prof["sot"] * scale
            corners = prof["corners"] * scale
            fouls = prof["fouls"] * (1.05 - 0.05 * poss_f)
            yellows = prof["yellows"] * (1.05 - 0.05 * poss_f)
            offsides = prof["offsides"] * (0.6 + 0.4 * att)
        else:
            shots = RATES["shots"] * (0.6 * att + 0.4 * poss_f)
            sot = max(0.5, shots * RATES["shots_on_target"] / RATES["shots"])
            corners = RATES["corners"] * (0.65 * att + 0.35 * poss_f)
            chase = 50.0 / max(possession, 25.0)
            fouls = RATES["fouls"] * (0.7 + 0.3 * chase)
            yellows = RATES["yellows"] * (0.7 + 0.3 * chase)
            offsides = RATES["offsides"] * (0.5 + 0.5 * att)
        reds = RATES["reds"] * (1.1 - 0.1 * poss_f)
        return TeamLine(name, round(lam, 2), shots, max(0.4, sot), possession,
                        corners, fouls, yellows, reds, offsides)

    def simulate(self, home, away, neutral=True, rest_home=4.0, rest_away=4.0):
        lam_h, lam_a = self.adjusted_lambdas(home, away, neutral, rest_home, rest_away)
        ph, pa = self._possession(lam_h, lam_a, home, away)
        lh, la = self._line(home, lam_h, ph), self._line(away, lam_a, pa)
        n, rng = self.n, self.rng
        def P(mean): return rng.poisson(max(mean, 1e-6), n)
        gh, ga = P(lam_h), P(lam_a)
        sh, sa = P(lh.shots), P(la.shots)
        soth, sota = np.minimum(sh, P(lh.sot)), np.minimum(sa, P(la.sot))
        ch, ca = P(lh.corners), P(la.corners)
        fh, fa = P(lh.fouls), P(la.fouls)
        yh, ya = P(lh.yellows), P(la.yellows)
        rh, ra = P(lh.reds), P(la.reds)
        oh, oa = P(lh.offsides), P(la.offsides)
        tg, tc, tcards = gh + ga, ch + ca, yh + ya + rh + ra
        def pct(m): return round(float(np.mean(m)) * 100, 1)
        def st(a, b): return {"home": round(float(a.mean()), 1), "away": round(float(b.mean()), 1)}
        scores, counts = np.unique(np.stack([gh, ga], 1), axis=0, return_counts=True)
        order = np.argsort(-counts)[:5]
        top = [{"score": f"{int(scores[i][0])}-{int(scores[i][1])}",
                "prob": round(float(counts[i] / n) * 100, 1)} for i in order]
        return {
            "fixture": {"home": home, "away": away, "neutral": neutral},
            "expected_goals": {"home": round(lam_h, 2), "away": round(lam_a, 2)},
            "form": {
                "home": {"rating": round(self.tm.form.get(home, 0.0), 2), "last5": self.tm.last5.get(home, [])},
                "away": {"rating": round(self.tm.form.get(away, 0.0), 2), "last5": self.tm.last5.get(away, [])},
            },
            "recovery": {"home_rest_days": rest_home, "away_rest_days": rest_away},
            "outcome": {"home_win": pct(gh > ga), "draw": pct(gh == ga), "away_win": pct(gh < ga)},
            "scorelines": top,
            "team_stats": {
                "possession": {"home": round(ph, 1), "away": round(pa, 1)},
                "expected_goals": {"home": round(lam_h, 2), "away": round(lam_a, 2)},
                "shots": st(sh, sa), "shots_on_target": st(soth, sota),
                "corners": st(ch, ca), "fouls": st(fh, fa),
                "yellow_cards": st(yh, ya),
                "red_cards": {"home": round(float(rh.mean()), 2), "away": round(float(ra.mean()), 2)},
                "offsides": st(oh, oa),
            },
            "markets": {
                "total_goals_avg": round(float(tg.mean()), 2),
                "over_1_5_goals": pct(tg > 1.5), "over_2_5_goals": pct(tg > 2.5),
                "over_3_5_goals": pct(tg > 3.5), "btts_yes": pct((gh > 0) & (ga > 0)),
                "home_clean_sheet": pct(ga == 0), "away_clean_sheet": pct(gh == 0),
                "home_win_to_nil": pct((gh > ga) & (ga == 0)),
                "total_corners_avg": round(float(tc.mean()), 1),
                "over_9_5_corners": pct(tc > 9.5), "over_10_5_corners": pct(tc > 10.5),
                "total_cards_avg": round(float(tcards.mean()), 1),
                "over_3_5_cards": pct(tcards > 3.5), "red_card_shown": pct((rh + ra) > 0),
            },
            "data_coverage": {
                "matches": self.tm.n_matches, "competitions": self.tm.competitions,
                "home_matches_in_data": self.tm.played.get(home, 0),
                "away_matches_in_data": self.tm.played.get(away, 0),
                "profile_home": home in self.tm.profiles, "profile_away": away in self.tm.profiles,
            },
            "n_sims": n,
        }


def build_default_simulator(n_sims: int = 20000):
    from backend.data.loader import load_matches
    tm = TeamModel().fit(load_matches(verbose=False))
    return MatchSimulator(tm, n_sims=n_sims)


if __name__ == "__main__":
    import json
    sim = build_default_simulator(15000)
    print(json.dumps(sim.simulate("Spain", "England", neutral=True), indent=2))
