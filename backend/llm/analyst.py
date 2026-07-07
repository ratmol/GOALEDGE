"""
The Gaffer — a RAG football analyst that works locally AND when deployed.

Two answer paths, chosen by the LLM client's capability:
  * Hosted open model (Groq/OpenRouter, no tool-calling): single-shot RAG —
    we simulate the fixture, RETRIEVE relevant knowledge (team + concept docs)
    from the corpus, inject both into one prompt, and generate. Robust across
    providers; this is the deployable path.
  * Local Ollama (supports tools): multi-turn tool-calling loop where the model
    calls the simulator itself.

Offline (no LLM configured): a deterministic, model-only summary.

The LLM never invents stats — it is grounded in simulator output + retrieved
facts.
"""
from __future__ import annotations

import json

from backend.skills.token_optimizer import TokenOptimizer

TOOLS = [
    {"type": "function", "function": {
        "name": "simulate_match",
        "description": "Run the Monte-Carlo match model for two teams: outcome "
                       "probabilities, expected goals, projected stats and markets.",
        "parameters": {"type": "object", "properties": {
            "home": {"type": "string"}, "away": {"type": "string"},
            "neutral": {"type": "boolean"}, "rest_home": {"type": "number"},
            "rest_away": {"type": "number"}}, "required": ["home", "away"]}}},
    {"type": "function", "function": {
        "name": "team_form",
        "description": "Recent form index and last-5 results for a team.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"}}, "required": ["team"]}}},
]

SYSTEM = (
    "You are 'The Gaffer', a sharp football analyst who reads matches like a "
    "tactician and prices them like a trader. Ground every claim in the DATA and "
    "NOTES provided (simulator output + retrieved facts). Never invent numbers; "
    "cite the ones given. If something isn't covered, say so plainly. Be concise "
    "and concrete, <=170 words."
)


class Analyst:
    def __init__(self, simulator, client, retriever=None, max_rounds: int = 4):
        self.sim = simulator
        self.client = client
        self.retriever = retriever
        self.max_rounds = max_rounds
        self.opt = TokenOptimizer(budget=600)

    # -- shared helpers ------------------------------------------------------
    def _compact_sim(self, d: dict) -> dict:
        ts, m = d["team_stats"], d["markets"]
        return {
            "fixture": d["fixture"], "outcome": d["outcome"],
            "expected_goals": d["expected_goals"],
            "form": {"home": d["form"]["home"]["rating"],
                     "away": d["form"]["away"]["rating"]},
            "possession": ts["possession"], "shots": ts["shots"],
            "corners": ts["corners"], "yellow_cards": ts["yellow_cards"],
            "markets": {k: m[k] for k in
                        ("total_goals_avg", "over_2_5_goals", "btts_yes",
                         "total_corners_avg", "over_10_5_corners",
                         "total_cards_avg")},
            "top_scoreline": d["scorelines"][0] if d["scorelines"] else None,
        }

    def _valid(self, home, away) -> bool:
        return bool(home and away and home != away
                    and home in self.sim.tm.teams and away in self.sim.tm.teams)

    def retrieve(self, question, home, away, k: int = 3) -> list[dict]:
        if self.retriever is None:
            return []
        q = " ".join(x for x in (question, home, away) if x)
        return self.retriever.search(q, k=k)

    # -- RAG single-shot (hosted / deployable path) -------------------------
    def _single_shot(self, question, home, away) -> dict:
        ctx, sources = [], []
        if self._valid(home, away):
            d = self.sim.simulate(home, away)
            ctx.append("MATCH DATA: " + json.dumps(self._compact_sim(d)))
            sources.append(f"model:{home}-vs-{away}")
        for hit in self.retrieve(question, home, away):
            ctx.append("NOTE: " + hit["text"])
            sources.append(hit["id"])
        context = "\n\n".join(ctx) if ctx else "(No specific fixture selected.)"
        prompt = f"{context}\n\nUser question: {question}"
        try:
            answer = self.client.generate(prompt, system=SYSTEM)
        except Exception as e:
            # LLM is configured but the call failed (bad model slug, quota, auth).
            # Degrade to the deterministic summary and surface the real reason.
            fb = self._fallback(question, home, away)
            fb["ollama"] = True
            fb["llm_error"] = str(e)[:300]
            return fb
        return {"answer": (answer or "").strip(), "sources": sources,
                "tools_used": ["retrieve"], "ollama": True}

    # -- Ollama tool-calling loop -------------------------------------------
    def _dispatch(self, name, args) -> str:
        try:
            if name == "simulate_match":
                d = self.sim.simulate(args["home"], args["away"],
                                      bool(args.get("neutral", True)),
                                      float(args.get("rest_home", 4)),
                                      float(args.get("rest_away", 4)))
                payload = self._compact_sim(d)
            elif name == "team_form":
                t = args["team"]
                payload = {"team": t,
                           "form_index": round(self.sim.tm.form.get(t, 0.0), 2),
                           "last5": self.sim.tm.last5.get(t, [])}
            else:
                payload = {"error": f"unknown tool {name}"}
        except Exception as e:
            payload = {"error": str(e)}
        return self.opt.optimize(json.dumps(payload)).prompt

    def _tool_loop(self, question, home, away) -> dict:
        ctx = question + (f"\n(Fixture in UI: {home} vs {away}.)"
                          if home and away else "")
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": ctx}]
        used = []
        for _ in range(self.max_rounds):
            msg = self.client.chat(messages, tools=TOOLS)
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return {"answer": (msg.get("content") or "").strip(),
                        "tools_used": used, "sources": [], "ollama": True}
            for tc in calls:
                fn = tc.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                messages.append({"role": "tool", "content": self._dispatch(name, args)})
                used.append({"tool": name, "args": args})
        return {"answer": (messages[-1].get("content") or "Couldn't finish.").strip(),
                "tools_used": used, "sources": [], "ollama": True}

    # -- offline fallback ---------------------------------------------------
    def _fallback(self, question, home, away) -> dict:
        if self._valid(home, away):
            d = self.sim.simulate(home, away)
            o, m, ts = d["outcome"], d["markets"], d["team_stats"]
            ans = (f"[Model summary — set LLM_API_KEY for a free hosted analyst, "
                   f"or run Ollama locally] {home} {o['home_win']}% / draw "
                   f"{o['draw']}% / {away} {o['away_win']}%. xG "
                   f"{d['expected_goals']['home']}-{d['expected_goals']['away']}, "
                   f"possession {ts['possession']['home']}-{ts['possession']['away']}, "
                   f"corners {ts['corners']['home']}-{ts['corners']['away']}. "
                   f"Total goals ~{m['total_goals_avg']}, BTTS {m['btts_yes']}%.")
            return {"answer": ans, "tools_used": ["simulate_match"],
                    "sources": [f"model:{home}-vs-{away}"], "ollama": False}
        return {"answer": "The analyst model isn't configured. Set LLM_API_KEY "
                "(a free hosted model — see RAG.md) or run Ollama locally to chat "
                "with The Gaffer. You can still use Run model / Scout report.",
                "tools_used": [], "sources": [], "ollama": False}

    # -- entry point --------------------------------------------------------
    def ask(self, question, home=None, away=None) -> dict:
        if not self.client.available():
            return self._fallback(question, home, away)
        if getattr(self.client, "supports_tools", False):
            return self._tool_loop(question, home, away)
        return self._single_shot(question, home, away)


if __name__ == "__main__":
    from backend.models.match_simulator import build_default_simulator
    from backend.rag.retriever import build_retriever
    sim = build_default_simulator(6000)
    retr = build_retriever(sim.tm)

    class HostedMock:            # supports_tools defaults False
        def available(self): return True
        def generate(self, prompt, system=None):
            has = "MATCH DATA" in prompt and "NOTE:" in prompt
            return f"(grounded={has}) prompt had {len(prompt)} chars"

    a = Analyst(sim, HostedMock(), retriever=retr)
    out = a.ask("Will there be lots of corners in Spain vs Italy?", "Spain", "Italy")
    print("single-shot sources:", out["sources"])
    print("single-shot answer:", out["answer"])

    class Off:
        supports_tools = False
        def available(self): return False
    print("fallback:", Analyst(sim, Off(), retriever=retr).ask("preview", "Brazil", "Panama")["answer"][:90])
