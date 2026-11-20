# Soccer system: status (soccer-0.1, built 2026-11-20)

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

## Setup (GitHub Actions, same pattern as the NHL repo)

1. New repo, upload everything including `.github/workflows/soccer.yml` and `.gitignore`.
2. Settings -> Secrets -> Actions: `APIFOOTBALL_KEY` (and `PROLYT_API_KEY` if wanted). The key goes nowhere else.
3. Actions -> soccer -> Run workflow, job `daily`. First run: about 30 requests for the current season so far.
4. Actions -> soccer -> Run workflow, job `backtest`. No API calls; about 10 minutes.
5. Boards go to `boards/soccer/<UTC date of kick-off>/*.net.json`. The push prices them and builds paper slips.

Local: `pip install pandas numpy scipy duckdb tabulate`, `export APIFOOTBALL_KEY=...`, then the same commands.

## First-run checklist (spec section 8)

| # | Task | Command | What must be true |
|---|---|---|---|
| 1 | League ids + coverage | `soccer_ingest.py run` | No `ERROR`/`WARNING` id lines; `data/soccer_2026_coverage.csv` read by eye |
| 2 | Current season | same run | All three `Sanity` lines say OK. No `REFUSED BY PLAN` line |
| 3 | Bandit 1 | `soccer_bandit.py download`, then `backtest` | Read `results/bandit1_report.md`; check `closing prices` is near 100% for recent seasons |
| 4 | First real board | `soccer_system.py unread --boards DIR` | Send me the output: the market patterns get fixed from it |
| 5 | One matchday | `price`, next day `grade`, then `handcheck` | Five printed results match the match reports -> `handcheck --confirm` |
| 6 | Slips | `slips` | Only after 5. Paper only |

Until step 5 is confirmed, `report` prints no numbers.

## Tested (all on simulated data)

- `soccer_system.py selftest`: 37 checks pass. Request budget (stops at 5 left, 20 ids a call, no finished game
  fetched twice, plan refusal reported), keys verified before being declared, the three sanity checks under both
  ways the API might list own goals, 59 hand-written settlement cases (quarter lines, pushes, halves, own goals,
  did-not-play, unconfirmed rules), de-margining back to the simulation's true prices, the one-way cap, slip rules,
  weights staying at 0.
- `soccer_bandit.py simulate`: against a book that is exactly right the bandit loses the margin (-4.4% a bet) and
  is not "in profit in most seasons"; against a book that prices home wins 10 points too low it finds it.

## Not tested, in order of risk

1. **Free-plan season access.** API-Football says the free plan covers "recent seasons" and paid plans unlock
   deeper history. If the key cannot read 2026, 2025 or 2024, the run prints `REFUSED BY PLAN` with the API's own
   message and skips that season. If 2026 is refused, the live model has no data and the cheapest fix is the
   $19/month plan or fitting the team model from football-data.co.uk (not built).
2. **API field names** are from documentation and memory, not a live response. Wrong names show up as failed
   sanity checks or empty stat columns, not as silent bad grades.
3. **The board reader.** The capture layout (markets / odds / childMarkets / desktopOddIds / sv / competitorId) is
   taken from the NHL captures. The Hungarian soccer market names are the spec's guesses. Expect a long `unread`
   list on the first real board.
4. **vegas.hu rules** (`RULES` in `soccer_system.py`): second yellow, coach cards, booking points, first scorer
   after an own goal, late substitutes. All unset. A row whose result depends on an unset rule is not graded.
   Quarter-line stake splitting is assumed.
5. **Team names** on the board against the API's: matched by date and fuzzy name; no unique match = game kept,
   not graded. Fix misses in `data/soccer_aliases.csv` (`board_name,team_id`).
6. **football-data.co.uk columns**: names taken from published listings of the files, both the old (`BbAv*`) and
   new (`AvgC*`, `AHCh`) layouts. Check the price-source column in the backtest report.
7. **ProLyt upload**: written to the spec (`POST {url}/upload`, `X-API-Key`, `overwrite=1`), never executed.

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
