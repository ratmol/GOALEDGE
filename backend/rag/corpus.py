"""
RAG corpus — the knowledge The Gaffer retrieves from.

Two sources, both free and offline:
  1. Per-team notes generated from the fitted TeamModel (your CSV-derived data):
     matches played, competitions, recent form, and — where available — real
     stat profile. This is how we "use the CSV data" the RAG way: retrieved at
     query time, not baked into a trained model.
  2. Static concept notes (xG, corners, cards, Kelly staking, home advantage,
     the competitions) so the analyst can answer conceptual questions too.

A document is just {"id", "text"}.
"""
from __future__ import annotations


CONCEPTS = [
    {"id": "concept:xg",
     "text": "Expected goals (xG) measures the quality of chances a team "
             "creates. It is a better predictor of future results than actual "
             "goals because it strips out finishing luck. GoalEdge's model "
             "estimates each team's xG per match; higher xG means more and "
             "better chances."},
    {"id": "concept:corners",
     "text": "Corners correlate with territorial dominance and attacking "
             "pressure. A team that has more possession and more shots usually "
             "wins more corners. Average is about 5 corners per team per match, "
             "around 10 total."},
    {"id": "concept:cards",
     "text": "Yellow and red cards rise with match intensity and with chasing "
             "the game. The team with less possession tends to commit more "
             "fouls and pick up more cards. Derbies and knockout games see more "
             "cards. Average is roughly 1.9 yellows per team per match."},
    {"id": "concept:kelly",
     "text": "Value betting compares the model's probability to the bookmaker's "
             "odds. Edge is model probability minus the fair (de-vigged) "
             "implied probability. The Kelly criterion sizes a stake by edge "
             "and odds; fractional Kelly (for example a quarter) reduces "
             "variance. Only bet when expected value is positive."},
    {"id": "concept:form",
     "text": "Recent form is how a team has played lately versus its own "
             "baseline. GoalEdge's form index is positive when a side is "
             "outperforming its usual goal difference and negative when "
             "underperforming; it nudges expected goals up or down."},
    {"id": "concept:home",
     "text": "Home advantage is worth roughly a quarter to a third of a goal in "
             "expected-goal terms. World Cup and continental tournament matches "
             "are usually at neutral venues, so home advantage is removed except "
             "for the host nation."},
    {"id": "concept:recovery",
     "text": "Recovery and rest matter in congested tournaments. A team with "
             "fewer rest days before a match tends to produce slightly less, "
             "which GoalEdge models as a small fatigue penalty on expected "
             "goals when rest drops below four days."},
    {"id": "concept:competitions",
     "text": "GoalEdge weights matches by competition and recency: friendlies "
             "count least, then qualifiers and Nations League, then the World "
             "Cup, Euros and Copa America count most. Recent games matter more "
             "than old ones (three-year half-life)."},
]


def _strength_word(x: float) -> str:
    if x > 0.35:
        return "very strong"
    if x > 0.12:
        return "strong"
    if x > -0.12:
        return "average"
    if x > -0.35:
        return "weak"
    return "very weak"


def team_docs(tm) -> list[dict]:
    docs = []
    for team in tm.teams:
        played = tm.played.get(team, 0)
        form = tm.form.get(team, 0.0)
        last5 = "".join(tm.last5.get(team, [])) or "n/a"
        atk = _strength_word(tm.attack.get(team, 0.0))
        dfn = _strength_word(tm.defence.get(team, 0.0))
        form_word = ("in good form" if form > 0.25 else
                     "in poor form" if form < -0.25 else "in steady form")
        parts = [
            f"{team} national team.",
            f"Appears in {played} matches in the dataset.",
            f"Attacking strength is {atk}; defensive solidity is {dfn}.",
            f"Recently {form_word} (form index {round(form, 2)}, last five {last5}).",
        ]
        prof = tm.profiles.get(team)
        if prof:
            parts.append(
                f"Typical match profile: about {prof['shots']:.0f} shots, "
                f"{prof['corners']:.0f} corners, {prof['yellows']:.1f} yellow "
                f"cards, and {prof['possession_bias']:.0f}% possession.")
        docs.append({"id": f"team:{team}", "text": " ".join(parts)})
    return docs


def build_corpus(tm) -> list[dict]:
    return CONCEPTS + team_docs(tm)


if __name__ == "__main__":
    from backend.models.match_simulator import build_default_simulator
    tm = build_default_simulator(2000).tm
    corpus = build_corpus(tm)
    print(f"corpus size: {len(corpus)} docs ({len(CONCEPTS)} concepts + "
          f"{len(corpus)-len(CONCEPTS)} teams)")
    print("sample team doc:", next(d for d in corpus if d["id"].startswith("team:"))["text"][:180])
