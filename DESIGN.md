# Design

## Visual Theme

**Mood:** Late-night trading desk during a Champions League knockout — all numbers, green-light monitors, gold positions on a dark field.

**Color strategy:** Committed. The deep tactical green carries 30–40% of the surface (primary actions, key metrics, probability bars). Gold is the value signal — EV positive, winner badge, accent highlight. Dark is absolute; no warmth-tinted neutrals.

**Register:** Product. Design serves the prediction engine, not the brand shell — except on the **welcome layer** (nav, hero, first-run/empty states, the Gaffer's intro), which is intentionally warmer and more human so newcomers feel invited, not intimidated (see Product principle 6, "welcoming front door, sharp back room").

**Warmth budget (hybrid):** The welcome layer MAY use gold (`--color-accent`) more generously, plain-language welcoming copy, a touch more breathing room, and one inviting moment (a warm floodlit hero glow is acceptable here). The analysis core (probability bars, EV/Kelly, value panel, stat tables) stays strictly dark, dense, and precise — no warm neutrals there. Warmth at the edges, rigor at the core.

---

## Color Palette

```css
:root {
  /* Backgrounds */
  --color-bg:        oklch(0.09 0.000 0);      /* near-pure black — analysis desk */
  --color-surface:   oklch(0.14 0.018 254);    /* dark blue-black panel */
  --color-surface-2: oklch(0.19 0.020 254);    /* elevated surface, nested cards */
  --color-border:    oklch(0.28 0.020 254);    /* subtle structural line */
  --color-border-2:  oklch(0.22 0.015 254);    /* deeper separator */

  /* Typography */
  --color-ink:       oklch(0.94 0.008 254);    /* near-white body text (≥7:1 on bg) */
  --color-muted:     oklch(0.62 0.018 254);    /* secondary text (≥3.5:1 on bg) */
  --color-faint:     oklch(0.42 0.014 254);    /* labels, captions */

  /* Brand */
  --color-primary:   oklch(0.54 0.185 145);    /* tactical green — primary actions */
  --color-primary-d: oklch(0.38 0.160 145);    /* darker green — pressed state, fills */
  --color-accent:    oklch(0.72 0.190 78);     /* value gold — EV+, winner badge, cta */

  /* Semantic / Status */
  --color-win:       oklch(0.60 0.190 142);    /* win probability bar */
  --color-draw:      oklch(0.72 0.185 78);     /* draw bar — same as accent */
  --color-loss:      oklch(0.55 0.220 25);     /* loss / negative EV */
  --color-ev-pos:    oklch(0.72 0.190 78);     /* EV positive → gold */
  --color-ev-neg:    oklch(0.55 0.220 25);     /* EV negative → red-orange */
  --color-neutral:   oklch(0.60 0.025 254);    /* neutral / no-signal */
}
```

---

## Typography

**Heading font:** `"Inter"`, weight 700–900. Compressed, high-contrast numerals. Fall back to `system-ui`.

**Body font:** `"Inter"`, weight 400–500. Tight line-height (1.5) for data-dense layouts.

**Monospace (numbers, odds, tokens):** `"JetBrains Mono"`, `"Fira Code"`, `monospace`. Use for: probabilities, odds values, Kelly stakes, token counts. Tabular numerals (`font-variant-numeric: tabular-nums`) everywhere numbers appear.

**Scale:**
| Role | Size | Weight | Notes |
|---|---|---|---|
| Display / hero | `clamp(2rem, 4vw, 3.5rem)` | 900 | The Gaffer header only |
| h1 | `clamp(1.4rem, 2.5vw, 2rem)` | 700 | Match title |
| h2 | `1.1rem` | 700 | Section headers |
| Body | `0.95rem` | 400 | Prose / scout report |
| Label | `0.75rem` | 600 | Form labels — do NOT uppercase unless it's a badge/status pill |
| Mono / stat | `0.875rem–1rem` | 500–600 | Odds, probabilities, EV |
| Caption | `0.7rem` | 400 | Token count, metadata |

**Line lengths:** Cap prose (scout report) at 65ch. Data columns are exempt.

**text-wrap:** `balance` on h1–h2, `pretty` on scout report paragraphs.

---

## Spacing & Layout

**Base unit:** `0.25rem` (4px).

**Spacing scale:** 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64px — use deliberately, not uniformly.

**App shell:** Single-column centered container, max-width `680px` (prediction tool). Full-width on data tables / analysis panels.

**Panel structure:**
- Cards: `border-radius: 12px`, `border: 1px solid var(--color-border)`
- Inner padding: `1.25rem` (default) / `1.75rem` (hero panel)
- Nested panels: use `--color-surface-2` and a tighter border

**Vertical rhythm:** Vary spacing. Not every gap is `1.2rem`. Use contrast between tight data sections and breathing room around key outputs.

---

## Components

### PredictPanel

- Team selectors: full-width `<select>` in dark surface. Remove ALL-CAPS uppercase labels — use sentence-case at `0.8rem` weight 500.
- "vs" separator: larger, `var(--color-muted)`, centered vertically.
- Neutral venue toggle: custom checkbox using `accent-color: var(--color-primary)`.
- Buttons: `Predict Outcome` — solid green (`var(--color-primary)`), white text. `Get Analysis` — solid gold (`var(--color-accent)`), dark ink text.
- No emoji in button labels (⚽ 🔍 read as casual). Replace with SVG icons or bare text.

### ResultCard

- Match header: `h2` at `1.15rem 700`, home vs away with muted "vs".
- Model badge: pill with `border-radius: 99px`, `0.7rem 700 uppercase`. Acceptable as a badge (not an eyebrow above a heading).
- Probability bars: `height: 8px` (not 12 — more precise). Animate with `transition: width 0.5s cubic-bezier(0.16, 1, 0.3, 1)`.
- "Most likely" row: use a muted background inner box (current approach is correct).
- Scout report / AI preview: render in a distinct `<blockquote>`-style section. The Gaffer's voice deserves a visual container — not just `color: var(--muted)`. Use a subtle left rule in `var(--color-primary)` at `2px` width. Do NOT use a wide left-border stripe (impeccable bans `border-left` accents > 1px as design anti-pattern) — use a `box-shadow: inset 2px 0 0 var(--color-primary)` trick instead.

### Probability Bars

```
[Label]                [XX.X%]
[==================          ]  8px height, radius 99
```
- Bar track: `var(--color-surface-2)`
- Win fill: `var(--color-win)`
- Draw fill: `var(--color-draw)`
- Loss fill: `var(--color-loss)`

### Value / EV Indicators

- EV+ values: `var(--color-ev-pos)` with a subtle `↑` glyph
- EV− values: `var(--color-ev-neg)` with a `↓` glyph
- Kelly stake: monospace, gold, prominent sizing

---

## Motion

- Probability bar reveal: `width 0s → 100%` on mount with `transition: width 0.5s cubic-bezier(0.16, 1, 0.3, 1)`. Stagger bars by 80ms.
- Loading state: single-line skeleton shimmer on the result area — `background: linear-gradient(90deg, surface, surface-2, surface)` animated.
- `@media (prefers-reduced-motion: reduce)`: disable all transitions, instant renders.

---

## Iconography

- No emoji in functional UI. Reserve emoji for the scout report text output (The Gaffer's voice can use them; the chrome cannot).
- SVG icons only: Heroicons or Lucide. 16×16 in labels, 20×20 in buttons.
- Football-specific: ⚽ emoji acceptable ONLY in The Gaffer header / persona block, never in a button or label.

---

## Anti-patterns to avoid

- `border-left` > 1px as a colored accent stripe on cards or callouts
- Gradient text (`background-clip: text`)
- ALL-CAPS tracking labels above every section (one badge is fine; repeated eyebrows are AI grammar)
- Identical card grids with icon + heading + text
- `z-index: 999` or arbitrary z-index values
