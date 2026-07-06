"""
The Gaffer — a tool-calling football analyst on top of local Ollama.

You ask a natural-language question ("Why would Spain dominate corners vs Italy?",
"Is Brazil-Argentina a low-scoring game?"). The LLM decides which tools to call
on the REAL match simulator, gets actual numbers back, and explains them. The
model never invents stats — it must call the simulator.

Tools exposed to the model:
  - simulate_match(home, away, neutral, rest_home, rest_away)
  - team_form(team)
  - list_teams()

Works offline too: if Ollama isn't running, ask() falls back to a deterministic
summary straight from the simulator (when two teams are identifiable).
"""
from __future__ import annotations

import json

from backend.skills.token_optimizer import TokenOptimizer

TOOLS = [
    {"type": "function", "function": {
        "name": "simulate_match",
        "description": "Run the Monte-Carlo match model for two international "
                       "teams and return outcome probabilities, expected goals, "
                       "projected stats (possession, shots, corners, cards) and "
                       "betting markets.",
        "parameters": {"type": "object", "properties": {
            "home": {"type": "string", "description": "Home/first team"},
            "away": {"type": "string", "description": "Away/second team"},
            "neutral": {"type": "boolean", "description": "Neutral venue (default true)"},
            "rest_home": {"type": "number", "description": "Home days rest (default 4)"},
            "rest_away": {"type": "number", "description": "Away days rest (default 4)"},
        }, "required": ["home", "away"]}}},
    {"type": "function", "function": {
        "name": "team_form",
        "description": "Recent form index and last-5 results (W/D/L) for a team.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"}}, "required": ["team"]}}},
    {"type": "function", "function": {
        "name": "list_teams",
        "description": "List the teams available in the data.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = (
    "You are 'The Gaffer', a sharp football analyst who reads matches like a "
    "tactician and prices them like a trader. You have tools backed by a real "
    "Monte-Carlo match model. ALWAYS call a tool to get numbers before quoting "
    "any stat or probability — never invent figures. Be concise and concrete, "
    "cite the numbers the tools return, and give a clear read. <=160 words."
)


class Analyst:
    def __init__(self, simulator, client, max_rounds: int = 4):
        self.sim = simulator
        self.client = client
        self.max_rounds = max_rounds
        self.opt = TokenOptimizer(budget=600)

    # -- tool implementations ------------------------------------------------
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

    def _dispatch(self, name: str, args: dict) -> str:
        try:
            if name == "simulate_match":
                d = self.sim.simulate(
                    args["home"], args["away"],
                    bool(args.get("neutral", True)),
                    float(args.get("rest_home", 4)),
                    float(args.get("rest_away", 4)))
                payload = self._compact_sim(d)
            elif name == "team_form":
                t = args["team"]
                payload = {"team": t,
                           "form_index": round(self.sim.tm.form.get(t, 0.0), 2),
                           "last5": self.sim.tm.last5.get(t, []),
                           "matches_in_data": self.sim.tm.played.get(t, 0)}
            elif name == "list_teams":
                payload = {"teams": self.sim.tm.teams}
            else:
                payload = {"error": f"unknown tool {name}"}
        except Exception as e:
            payload = {"error": str(e)}
        text = json.dumps(payload)
        return self.opt.optimize(text).prompt   # token-trim tool output

    # -- offline deterministic fallback -------------------------------------
    def _fallback(self, question: str, home, away) -> dict:
        if home and away and home != away:
            d = self.sim.simulate(home, away)
            o, m = d["outcome"], d["markets"]
            ts = d["team_stats"]
            ans = (f"[Offline summary — install Ollama for full analysis] "
                   f"{home} {o['home_win']}% / draw {o['draw']}% / "
                   f"{away} {o['away_win']}%. Expected goals "
                   f"{d['expected_goals']['home']}-{d['expected_goals']['away']}, "
                   f"possession {ts['possession']['home']}-{ts['possession']['away']}, "
                   f"corners {ts['corners']['home']}-{ts['corners']['away']}. "
                   f"Total goals ~{m['total_goals_avg']}, BTTS {m['btts_yes']}%, "
                   f"over 10.5 corners {m['over_10_5_corners']}%.")
            return {"answer": ans, "tools_used": ["simulate_match"],
                    "ollama": False}
        return {"answer": "Ollama isn't running. Install it from "
                "https://ollama.com and run `ollama pull llama3.1` to chat with "
                "The Gaffer. You can still use Run model / Scout report.",
                "tools_used": [], "ollama": False}

    # -- main entry ----------------------------------------------------------
    def ask(self, question: str, home=None, away=None) -> dict:
        if not self.client.available():
            return self._fallback(question, home, away)

        ctx = question
        if home and away:
            ctx += f"\n(Current fixture in the UI: {home} vs {away}.)"
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": ctx}]
        tools_used = []

        for _ in range(self.max_rounds):
            msg = self.client.chat(messages, tools=TOOLS)
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return {"answer": (msg.get("content") or "").strip(),
                        "tools_used": tools_used, "ollama": True}
            for tc in calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                result = self._dispatch(name, args)
                tools_used.append({"tool": name, "args": args})
                messages.append({"role": "tool", "content": result})

        return {"answer": (messages[-1].get("content") or
                           "I gathered the data but couldn't finish the read.").strip(),
                "tools_used": tools_used, "ollama": True}


if __name__ == "__main__":
    # Verify the tool-call loop with a MOCK client (no live Ollama needed).
    from backend.models.match_simulator import build_default_simulator

    class MockClient:
        def __init__(self): self.n = 0
        def available(self): return True
        def chat(self, messages, tools=None):
            self.n += 1
            if self.n == 1:  # first round: ask to simulate
                return {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {
                            "name": "simulate_match",
                            "arguments": {"home": "Spain", "away": "Italy"}}}]}
            # second round: model has tool result, writes answer
            last_tool = [m for m in messages if m.get("role") == "tool"][-1]
            return {"role": "assistant",
                    "content": "Tool returned: " + last_tool["content"][:120]}

    sim = build_default_simulator(8000)
    a = Analyst(sim, MockClient())
    out = a.ask("How does Spain vs Italy look?", "Spain", "Italy")
    print("TOOLS USED:", out["tools_used"])
    print("ANSWER:", out["answer"][:200])
    print("--- offline fallback ---")
    a2 = Analyst(sim, type("Off", (), {"available": lambda s: False})())
    print(a2.ask("preview?", "Brazil", "Panama")["answer"][:160])
