# Product

## Register

product

## Users

**Primary — the sharp:** Value/sharp bettors who care about EV, Kelly sizing, de-vigged odds, and calibrated probability. In "work mode": analytical, time-pressured, hunting edge over the market. They trust numbers over narrative but want the narrative when it explains the model. The Gaffer is their analyst-in-residence.

**Secondary — the newcomer (hybrid goal):** Football fans who are curious but NOT yet fluent in EV/Kelly. They must feel *welcomed*, not intimidated, on arrival — the front door (header, hero, first-run, empty states) speaks plain human language and invites a first prediction, then progressively reveals the sharp tools. The depth is earned, not dumped on them. Nobody bounces because the app looked like a Bloomberg terminal on load.

## Product Purpose

GoalEdge prices full matches — not just win/draw/loss, but goals, expected goals, shots, corners, cards, scorelines — via a 20,000-run Monte Carlo engine and a Dixon-Coles/XGBoost blend. It surfaces calibrated probabilities, de-vigged odds, expected value, and fractional Kelly stakes against live market odds (The Odds API). The LLM scout report (Ollama, local) wraps the numbers in The Gaffer's voice. Success looks like a bettor trusting GoalEdge's EV edge over a bookmaker's implied odds and winning with discipline over time.

## Brand Personality

Bold, Confident, Expert. The Gaffer energy: opinionated, authoritative, a pundit who also builds models. Not the humble academic tool, not the flashy gambling site — the edge-finding desk of a quant who watches football obsessively. Premium analytics with a personality.

## Anti-references

- **Generic SaaS dashboard**: Beige/cream cards, hero metrics with gradient text, Tailwind defaults, identical card grids. The visual grammar of every AI-generated B2B tool.
- **Consumer sports app**: Round team logos, giant photography, ESPN-style palette. Too casual, too marketing-first.
- **Neon gambling site**: Promotional, loud, trust-eroding. Lacks analytical credibility.
- **Plain academic tool**: FBref-style raw tables with no visual identity. The data is right but the experience is hostile.

## Design Principles

1. **Probabilities, not prophecies.** Every number is a calibrated distribution, never a verdict. The design should make uncertainty legible, not hide it.
2. **Edge is the product.** Surface EV and Kelly sizing as primary outputs — not just "who wins." Design hierarchy should make the bet-relevant numbers immediately findable.
3. **The Gaffer speaks.** The scout report and persona voice are part of the UX. Design should give the LLM output the authority of a real analyst briefing, not a chatbot reply.
4. **Dark and earned.** Dark theme is the correct choice for a tool used in analysis mode — not an aesthetic default. Every pixel should feel intentional, not generated.
5. **Speed before decoration.** The sharp wants the number now. Performance and information density beat embellishment.
6. **Welcoming front door, sharp back room (hybrid).** The entry layer — hero, first prediction, empty/loading states, the Gaffer's intro — is warm, plain-spoken, and inviting so a newcomer feels at home. The analysis surfaces (probabilities, EV, Kelly, value panel) stay dense, dark, and precise. Warmth lives at the edges; rigor lives at the core. One product, two registers, no dumbing-down of the math.

## Accessibility & Inclusion

Best effort: WCAG AA contrast ratios for readable text, keyboard-navigable core flows. No formal compliance requirement, but don't create barriers. Dark palette must maintain ≥4.5:1 for body text.
