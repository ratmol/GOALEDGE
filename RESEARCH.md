# GoalEdge Research — What Works, What Doesn't, and What the Market Already Knows

*Compiled 2026-07-05, mid-tournament (World Cup 2026, Round of 16 in progress). This is the thinking behind the v3 rebuild, written from the perspective of someone actually putting money on matches.*

## 1. The uncomfortable starting point: the market is the best model

Every serious study of football betting reaches the same conclusion: the closing odds at a sharp bookmaker (Pinnacle being the reference) are, after removing the margin, the most accurate publicly available probability estimate for a match. Academic models — Poisson, Dixon-Coles, Elo variants, gradient boosting, deep nets — generally match but rarely beat the closing line over large samples. Kuypers, Constantinou & Fenton, and the annual "Machine Learning for Soccer" Kaggle post-mortems all point the same way.

What this means for GoalEdge is not "give up". It means three things. First, the model's job is to find the *few* matches where the price is wrong, not to predict every match — most matches should produce a NO BET verdict, and passing is a position. Second, the correct benchmark for the model is never accuracy ("did I pick the winner?") but calibration (Brier/log-loss) and, ultimately, whether bets flagged as +EV beat the closing line. Beating the closing line consistently — even if individual bets lose — is the single best-established predictor of long-term profit. Third, edges are structurally more likely where books are less efficient: tournament football with short pricing windows, low-liquidity markets (corners, cards), and moments when public money floods one side (host nations, Brazil, England — recreational money distorts those prices, which is why the *fade-the-public* effect exists at all).

International tournaments are actually one of the better hunting grounds: teams play rarely, sample sizes are thin, squads change between windows, and books have less data than they do for the Premier League. The same thinness that makes our model uncertain also makes theirs.

## 2. What actually predicts international matches

Team strength dominates everything else, and the best cheap estimator is a goal-based rating fit over many years of matches, not a form table over five games. The two workhorses that have survived thirty years of scrutiny are the Elo family (the World Football Elo Ratings outperform the official FIFA ranking as a predictor) and the Dixon-Coles (1997) bivariate Poisson with a low-score correction. Everything else — xG models, player ratings, market-value aggregates (the "Transfermarkt model") — adds a little on top when the data is fresh, but strength ratings do the heavy lifting.

Draws are where naive models bleed money. An independent-Poisson model underprices 0-0 and 1-1, which is precisely why Dixon-Coles introduced the rho correction. Roughly a quarter of international matches end level after 90 minutes; a model that systematically prices draws at 20% when the true rate is 26% donates money on every 1X2 bet placed around it. v3 fits rho by maximum likelihood.

Recency and competition weighting matter. A friendly tells you less than a qualifier, which tells you less than a knockout match; a result from 2014 tells you almost nothing about 2026. Dixon-Coles themselves used exponential time decay, and every serious implementation since has kept it. The loader already applies both weights; v3 finally feeds them into the likelihood rather than ignoring them.

Home advantage is real but misapplied at a "neutral" World Cup. This tournament is hosted by the USA, Mexico and Canada — their matches are *not* neutral, and altitude (Mexico City), travel distance between host cities, and crowd composition are live variables. Rest days matter modestly (the literature converges on a small but real congestion effect below four days' rest); v3 carries rest into the features rather than pretending everyone had a week.

Things that sound smart but don't reliably pay: head-to-head records (thin samples, mostly noise once strength is accounted for — kept as a minor feature only), "momentum" narratives beyond what shows up in recent goal difference, motivational stories, and star-player counting. Injuries and suspensions *do* matter but they are largely priced in within minutes of the news — a model without a squad feed cannot beat the market on them, so v3 flags any suspiciously large edge (>15%) as "probably news you don't have" rather than a bet.

## 3. What was wrong with v1/v2, honestly

The training set was 264 matches. Poisson MLE over ~50 teams needs thousands of matches to distinguish attack from defence strength; on 264 the parameters are mostly prior. This was the single biggest flaw, and the reason the fix (the full open dataset of international results, ~48,000 matches, filtered to the modern era) is a one-command script rather than an optional extra.

The validation split was random, not chronological. Elo and form features computed over the whole history leak the future into the past; a random split then "validates" on matches the features have already seen around. Reported Brier scores were flattering fiction. v3 validates strictly out-of-time on the most recent 20%.

The inference path was broken. `Phase1Model.predict()` fed the XGBoost half a vector of 0.5s plus ratings from a *freshly constructed, unfitted* Elo — every team at 1500. Half of every blended prediction was noise. Similarly the API's Elo fallback shipped unfitted. Both fixed.

There was no concept of price. A prediction app for gamblers that never touches odds is a horoscope with decimals. Everything in section 4 exists because of this.

The Monte Carlo simulator's secondary stats (corners, cards, fouls) are modelled from league-average rates and a handful of team profiles, not fitted from data. They are labelled as such in the API now; treat corner/card "edges" as entertainment until real per-team stats are wired in.

## 4. The gambler's toolkit (what v3 adds)

De-vig before comparing. Bookmaker odds embed a margin (overround) of 2–7%; comparing model probability against raw 1/odds overstates your edge by that margin. v3 normalises implied probabilities (proportional method) and reports the book's margin so you can see what you're paying.

Expected value at the price you can actually get. Consensus (median) across books is the sanity check; the *best* available price is what EV should be computed at. The Odds API integration returns both.

Fractional Kelly staking. Full Kelly is optimal only if your probabilities are exactly right; they aren't, and overbetting an overestimated edge is how bankrolls die. Quarter-Kelly is the standard compromise (half the growth, a quarter of the variance), with a hard cap of 5% of bankroll per bet. Both configurable in `.env`.

Edge thresholds and humility rules. Nothing is flagged below a 3% edge (inside model error). Edges above 15% are flagged *against* — at that size the more likely explanation is that the book knows something the model doesn't (lineup news, injury, motivation). Long-shot edges (model prob <10%) get a warning: calibration is weakest in the tails, and favourite–longshot bias means the tails are exactly where books pad their margin most.

The discipline that no code can enforce: record every bet with the closing odds, and judge yourself on closing-line value, not short-term profit. Fifty bets tell you almost nothing (variance swamps a 3% edge); five hundred begin to. If your flagged bets don't beat the closing line after a few hundred samples, the model has no edge — stop.

## 5. Bigger-picture items (the "am I missing something?" list)

The tournament is live *now* — Round of 16, July 2026. A World Cup model whose data ends in 2022 is answering last cycle's question. v3 bundles the 2026 knockout results so far and the remaining fixtures, and the download script refreshes everything. After each matchday, re-run `python scripts/update` (or just retrain) — in-tournament form is the freshest signal you have.

Draws-then-shootouts: knockout 1X2 markets settle on 90 minutes. The model correctly treats Germany–Paraguay as a draw even though Paraguay advanced. Don't confuse "to qualify" markets with 1X2 — they're different prices.

Sample-size reality: even with the full dataset, a 3–5% ROI on flagged bets is an excellent outcome, achieved by maybe the top few percent of bettors. Anyone promising more per bet is selling something.

Legality and safety: sports betting rules vary by jurisdiction, and staking software doesn't change the house's legal edge where betting is restricted. Set a bankroll you can lose entirely, never chase, and treat the NO BET verdict as the model's most common and most valuable output.

## 6. Data sources used or wired in

Match history: martj42/international_results (GitHub, ~48k matches, CC0) via `scripts/download_data.py`. Live/official results: football-data.org free tier (key in `.env`). Live odds: the-odds-api.com free tier, sport key `soccer_fifa_world_cup`, h2h market, EU+US regions. WC 2026 tournament state compiled 2026-07-05 from ESPN, Yahoo Sports, Al Jazeera, FOX Sports and CNN coverage of the Round of 32 and Round of 16.
