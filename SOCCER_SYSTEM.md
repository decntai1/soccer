# Soccer system: status (soccer-0.1, free-plan ingest, 2026-11-20)

Built from `SOCCER_MODEL_SPEC.md`. **Nothing here has touched live data.** The build machine could not reach
API-Football or football-data.co.uk, so every test ran against simulated responses. Treat all of it as unproven
until the first-run checklist below is done.

## Files

| File | What it does |
|---|---|
| `soccer_ingest.py` | API-Football -> CSV tables, DuckDB, relationships JSON, ProLyt upload. Budgeted, cached, three sanity checks. |
| `soccer_model.py` | Dixon-Coles goals, first-half share, negative-binomial corners/cards/shots, referee factor, player rates. |
| `soccer_system.py` | Board reader, de-margining, model blend, grading, weights, slips, four votes, bandit 2, report. |
| `soccer_bandit.py` | Bandit 1: Thompson-sampling backtest on football-data.co.uk closing prices, per season, with baselines. |
| `soccer_selftest.py`, `soccer_fake.py` | Offline tests and the simulated API / board they run on. |
| `.github/workflows/soccer.yml` | Daily ingest + grade, pricing on board push, manual backtest and lineups jobs. |

## The free plan (tested 2026-11-20 with your key)

| Works | Refused |
|---|---|
| `fixtures?date=` for yesterday, today, tomorrow (109 games returned for 19 Nov) | any `season=2025` or `2026` (only 2022-2024) |
| `fixtures/events?fixture=` for a current-season game from yesterday | `ids=` (batching) |
| | dates outside D-1..D+1 |

API-Football's helper said current-season games are refused even inside the window. The live test says
otherwise. The first real run settles it: a refused game prints `REFUSED BY PLAN`, nothing is guessed.

So the ingest now does, per run: 3 date calls, then 1 `fixtures?id=` call per finished game of our six leagues
(if that call turns out not to carry events and statistics, it switches to 4 calls per game and says so). Busy
Saturday: ~50 games, ~55 calls. **A game whose day leaves the window unfetched is gone for good on this plan**:
the workflow runs at 05:23 and 23:41 UTC so one failed run loses nothing, and the log lists `LOST` games.

Everything before the first run (this season since August, and the two before) comes from football-data.co.uk:
results, half-time, shots, corners, cards, referee (England). Its team names are mapped to API team ids, by results
where both sources hold a game and by name otherwise. Check the `model history: ... teams mapped X of Y` line of
the first `price`. Fix misses in `data/soccer_team_map.csv` (`fd_name,team_id`).

Player lines need API games: the player model and scorer grading start from the first run, with no history.

## Setup

Repo `decntai1/soccer`, secret `APIFOOTBALL_KEY`. Actions -> soccer -> Run workflow, job `daily`. Boards go to
`boards/soccer/<UTC date of kick-off>/*.net.json`.

## First-run checklist

| # | Task | What must be true |
|---|---|---|
| 1 | `daily` run | `plan: free`, no `REFUSED BY PLAN`, the three `Sanity` lines OK, no `WARNING` on league ids |
| 2 | First `price` | `teams mapped` close to all; `unread` list sent to me |
| 3 | `backtest` job | `results/bandit1_report.md` read (no API requests) |
| 4 | One matchday priced and graded | `handcheck` on five results, then `handcheck --confirm` |

## Tested (simulated data only)

- `selftest`: 45 checks. New for the free plan: only date and id calls, never season/ids; 3 date calls on a repeat
  run; LOST games reported and not requested; the 4-call fallback; coverage inferred and marked as inferred; all
  teams mapped from football-data names; the full price/grade loop with football-data as the model's history.
- `soccer_bandit.py simulate`: 7 checks, unchanged.

## Not tested, in order of risk

1. Whether the free plan serves our six leagues' current games through `fixtures?id=` (tested: Brazil's Serie A).
2. The board reader: Hungarian soccer market names are still the spec's guesses.
3. vegas.hu settlement rules (`RULES`): unset; rows that depend on them are not graded.
4. football-data columns for 2. Bundesliga, Serie B and Segunda: shots/corners/cards may be missing there.
5. The team-name map on real names (tested on simulated names and on 35 real pairs by hand).
6. ProLyt upload: never executed.

## Things the simulations already say

- **Bandit 1 has little power.** With home wins priced 4 points too low (always-home earns +2.8%), 480 games a
  season, the bandit still lost 5.9% on 1X2 (3.3% when carrying what it learned between seasons) and was in profit
  in 1 of 6 seasons either way. It only found a 10-point error. Real seasons have 2,800 games, so it will do
  somewhat better, but a "no edge" result from it means "no huge edge", not "no edge".
- **Slips start deep in the red.** On a simulated board with a 6% margin a 3x slip came out at about -21%
  expected return and a 6x at about -30%.
- **The model is noisy early.** Fitted on 15 games a team it was off by 12 points on home-win chance on average.
  Its weight is 0 until held-out games give it t > 2.

## Where I departed from the spec

- League ids and coverage come from one `leagues?season=S` call per season instead of one per country.
- Sleep stays at 7 s; API-Football's own guide gives the free plan 10 requests a minute.
- A player line with no team tag is still read if the name matches exactly one player in the two squads;
  otherwise it is typed None as the spec says.
- Side legs (1X2, handicap) are allowed only on a team the book rates 60%+ to win; draw-no-bet is never a leg.
- Bandit 2: all three strategies' slips are built and graded on paper every matchday, so every arm learns every
  matchday; the bandit's pick is only marked as the one that would be staked.
- The fourth vote needs `data/soccer_outside.csv` (`fixture_id,type,side`). Without it the most a leg gets is 3.
- Bandit 1 adds a no-exploration run and a `--carry` option as diagnostics. Default is the spec: fresh prior
  each season.
- `soccer_ingest.py` refuses to overwrite a season's tables with fewer games than they already hold (lost cache).
